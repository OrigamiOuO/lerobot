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

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
from lerobot.policies.pretrain_diffusion.configuration_pretrain_diffusion import PretrainDiffusionConfig
from lerobot.policies.pretrain_diffusion.pretrain_encoder.temporal_tactile_encoder import (
    PretrainedSparsePCEncoder,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION


OBS_STATE = "observation.state"
OBS_STATE_VELOCITY = "observation.state_velocity"
OBS_TACTILE = "observation.tactile"
OBS_FSR = "observation.fsr"
OBS_SPARSE_PC = "observation.sparse_pc"


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


class PretrainDiffusionPolicy(PreTrainedPolicy):
    """Diffusion policy conditioned on fused state features and pretrained sparse_pc latent."""

    config_class = PretrainDiffusionConfig
    name = "pretrain_diffusion"

    def __init__(self, config: PretrainDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = PretrainDiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE_VELOCITY: deque(maxlen=self.config.n_obs_steps),
            OBS_TACTILE: deque(maxlen=self.config.n_obs_steps),
            OBS_FSR: deque(maxlen=self.config.n_obs_steps),
            OBS_SPARSE_PC: deque(maxlen=self.config.n_obs_steps),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        loss = self.diffusion.compute_loss(batch)
        return loss, None


class PretrainDiffusionModel(nn.Module):
    def __init__(self, config: PretrainDiffusionConfig):
        super().__init__()
        self.config = config

        self.state_mlp = nn.Sequential(
            nn.Linear(config.state_fused_input_dim, config.state_mlp_hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(config.state_mlp_hidden_dim1, config.state_mlp_hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(config.state_mlp_hidden_dim2, config.state_mlp_output_dim),
            nn.ReLU(inplace=True),
        )

        self.sparse_pc_encoder = PretrainedSparsePCEncoder(
            checkpoint_path=config.resolve_pretrained_encoder_ckpt_path(),
            seq_len=config.n_obs_steps,
            num_points=config.sparse_pc_num_points,
            in_dim=config.sparse_pc_point_dim,
            embed_dim=config.pretrained_encoder_embed_dim,
            freeze=config.freeze_pretrained_encoder,
        )

        global_cond_dim = config.state_mlp_output_dim + config.pretrained_encoder_embed_dim
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        state_velocity = batch[OBS_STATE_VELOCITY]
        tactile = batch[OBS_TACTILE]
        fsr = batch[OBS_FSR]
        sparse_pc = batch[OBS_SPARSE_PC]

        fused_state = torch.cat([state, state_velocity, tactile, fsr], dim=-1)
        if fused_state.shape[-1] != self.config.state_fused_input_dim:
            raise ValueError(
                f"Expected fused state dim {self.config.state_fused_input_dim}, got {fused_state.shape[-1]}"
            )

        state_latents = self.state_mlp(fused_state)
        state_latent = state_latents.mean(dim=1)

        sparse_tokens = self.sparse_pc_encoder(sparse_pc)
        sparse_latent = sparse_tokens.mean(dim=1)

        return torch.cat([state_latent, sparse_latent], dim=-1)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        required_keys = {
            OBS_STATE,
            OBS_STATE_VELOCITY,
            OBS_TACTILE,
            OBS_FSR,
            OBS_SPARSE_PC,
            ACTION,
            "action_is_pad",
        }
        assert set(batch).issuperset(required_keys)

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
