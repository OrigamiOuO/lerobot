#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain MLP Policy.

Shares the same observation conditioning pipeline as PretrainDiffusionPolicy
(state_mlp + pretrained sparse_pc encoder), but replaces the iterative diffusion
denoising UNet / Transformer Decoder with a simple multi-layer MLP that directly
regresses the entire action chunk.

This is the simplest possible ablation baseline: an MLP conditioned on the global
feature vector, outputting all future actions in one forward pass.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrain_mlp.configuration_pretrain_mlp import PretrainMLPConfig
from lerobot.policies.pretrain_diffusion.pretrain_encoder_v2.temporal_tactile_encoder_v2 import (
    PretrainedSparsePCEncoderV2,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION


OBS_STATE = "observation.state"
OBS_STATE_VELOCITY = "observation.state_velocity"
OBS_TACTILE = "observation.tactile"
OBS_FSR = "observation.fsr"
OBS_SPARSE_PC = "observation.sparse_pc"


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    if activation == "silu":
        return F.silu
    raise RuntimeError(f"activation should be relu/gelu/glu/leaky_relu/silu, not {activation}.")


class PretrainMLPActionHead(nn.Module):
    """A large multi-layer MLP that regresses the full action chunk from global conditioning.

    Architecture:
        Linear(global_cond_dim, hidden[0]) -> Activation
        -> Linear(hidden[0], hidden[1]) -> Activation
        -> ...
        -> Linear(hidden[-1], horizon * action_dim)
        -> Reshape to (B, horizon, action_dim)
    """

    def __init__(self, config: PretrainMLPConfig):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.global_cond_dim
        activation_fn = _get_activation_fn(config.mlp_activation)

        for hidden_dim in config.mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True) if config.mlp_activation == "relu" else nn.Identity())
            in_dim = hidden_dim

        # Replace the last generic activation with the correct one if not relu
        if config.mlp_activation != "relu":
            # pop the last relu and replace
            layers.pop()
            layers.append(nn.Linear(in_dim, config.mlp_hidden_dims[-1]))
            # For non-relu activations, we use the proper activation
            if config.mlp_activation == "gelu":
                layers.append(nn.GELU())
            elif config.mlp_activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            elif config.mlp_activation == "silu":
                layers.append(nn.SiLU(inplace=True))
            else:
                # glu is not used as a pointwise activation in hidden layers
                layers.append(nn.ReLU(inplace=True))
            in_dim = config.mlp_hidden_dims[-1]

        self.hidden_layers = nn.Sequential(*layers)

        # Final projection to (horizon * action_dim).
        action_dim = config.action_feature.shape[0] if config.action_feature is not None else config.action_dim
        self.output_dim = config.horizon * action_dim
        self.final_layer = nn.Linear(in_dim, self.output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_cond: Tensor) -> Tensor:
        """Forward pass.

        Args:
            global_cond: (B, global_cond_dim) global conditioning vector.

        Returns:
            actions: (B, horizon, action_dim) predicted action chunk.
        """
        batch_size = global_cond.shape[0]
        x = self.hidden_layers(global_cond)  # (B, last_hidden_dim)
        x = self.final_layer(x)              # (B, horizon * action_dim)
        actions = x.view(batch_size, self.config.horizon, -1)
        return actions


class PretrainMLPModel(nn.Module):
    """MLP-style action generator that shares the same observation conditioning
    pipeline as PretrainDiffusionModel / PretrainACTModel.

    Architecture:
    1. state_mlp: fuses (state, velocity, tactile, fsr) -> state_latent (B, T, 512)
    2. sparse_pc_encoder: encodes sparse_pc -> sparse_latent (B, T, 384)
    3. global_cond = concat(mean(state_latent), mean(sparse_latent)) -> (B, global_cond_dim)
    4. Large MLP action head -> action chunk (B, horizon, action_dim)
    """

    def __init__(self, config: PretrainMLPConfig):
        super().__init__()
        self.config = config

        # Same state fusion MLP as pretrain_diffusion / pretrain_act.
        self.state_mlp = nn.Sequential(
            nn.Linear(config.effective_state_fused_input_dim, config.state_mlp_hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(config.state_mlp_hidden_dim1, config.state_mlp_hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(config.state_mlp_hidden_dim2, config.state_mlp_output_dim),
            nn.ReLU(inplace=True),
        )

        # Same pretrained sparse PC encoder.
        if config.ablation_mode == "full":
            self.sparse_pc_encoder = PretrainedSparsePCEncoderV2(
                checkpoint_path=config.resolve_pretrained_encoder_ckpt_path(),
                max_seq_len=config.pretrained_encoder_max_seq_len,
                num_points=config.sparse_pc_num_points,
                point_dim=config.sparse_pc_point_dim,
                embed_dim=config.pretrained_encoder_embed_dim,
                freeze=config.freeze_pretrained_encoder,
            )
        else:
            self.sparse_pc_encoder = None

        # MLP action head.
        self.action_head = PretrainMLPActionHead(config)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Identical to PretrainDiffusionModel / PretrainACTModel._prepare_global_conditioning."""
        parts = [batch[OBS_STATE], batch[OBS_STATE_VELOCITY]]
        if self.config.ablation_mode in ("full", "no_fsr"):
            parts.append(batch[OBS_TACTILE])
        if self.config.ablation_mode in ("full", "no_twintac"):
            parts.append(batch[OBS_FSR])

        fused_state = torch.cat(parts, dim=-1)
        expected = self.config.effective_state_fused_input_dim
        if fused_state.shape[-1] != expected:
            raise ValueError(
                f"Expected fused state dim {expected} (ablation_mode={self.config.ablation_mode!r}), "
                f"got {fused_state.shape[-1]}"
            )

        state_latent = self.state_mlp(fused_state).mean(dim=1)

        if self.config.ablation_mode == "full":
            sparse_latent = self.sparse_pc_encoder(batch[OBS_SPARSE_PC]).mean(dim=1)
            return torch.cat([state_latent, sparse_latent], dim=-1)

        return state_latent

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate action chunk via MLP action head.

        Args:
            batch: Observation batch with keys like observation.state, etc.

        Returns:
            actions: (B, horizon, action_dim) predicted action chunk.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected n_obs_steps={self.config.n_obs_steps} but got batch with "
                f"observation.state shape {batch[OBS_STATE].shape} (n_obs_steps={n_obs_steps}). "
                f"Check that --policy.n_obs_steps matches the dataset delta_timestamps."
            )

        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # MLP action head directly regresses the entire action chunk.
        actions = self.action_head(global_cond)  # (B, horizon, action_dim)

        return actions

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass for training. Returns predicted actions."""
        return self.generate_actions(batch)


class PretrainMLPPolicy(PreTrainedPolicy):
    """MLP-style policy conditioned on fused state features and pretrained sparse_pc latent.

    Shares the same observation encoding as PretrainDiffusionPolicy / PretrainACTPolicy
    but uses a simple multi-layer MLP for action generation. This is the simplest possible
    ablation baseline.
    """

    config_class = PretrainMLPConfig
    name = "pretrain_mlp"

    def __init__(self, config: PretrainMLPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.model = PretrainMLPModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        for key in self.config.active_obs_keys:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Training forward pass.

        Returns:
            loss: MSE loss between predicted and ground-truth actions.
            None: no auxiliary loss dict (for compatibility).
        """
        required_keys = set(self.config.active_obs_keys) | {ACTION, "action_is_pad"}
        assert set(batch).issuperset(required_keys)

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        if horizon != self.config.horizon:
            raise ValueError(
                f"Expected horizon={self.config.horizon} but got action shape {batch[ACTION].shape} "
                f"(horizon={horizon})."
            )
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected n_obs_steps={self.config.n_obs_steps} but got observation.state shape "
                f"{batch[OBS_STATE].shape} (n_obs_steps={n_obs_steps}). "
                f"Check that --policy.n_obs_steps matches the dataset delta_timestamps."
            )

        pred = self.model(batch)

        loss = F.mse_loss(pred, batch[ACTION], reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean(), None
