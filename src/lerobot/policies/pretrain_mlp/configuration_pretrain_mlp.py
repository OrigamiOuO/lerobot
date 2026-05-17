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
"""Pretrain MLP Policy Configuration.

Shares the same observation conditioning pipeline as PretrainDiffusion / PretrainACT
(state_mlp + pretrained sparse_pc encoder), but replaces the action generator with
a simple multi-layer MLP that directly regresses the entire action chunk.

This serves as the simplest possible ablation baseline for comparing action generation
heads (Diffusion / Transformer Decoder / MLP).
"""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("pretrain_mlp")
@dataclass
class PretrainMLPConfig(PreTrainedConfig):
    """MLP-style policy using a pretrained sparse point-cloud temporal encoder.

    Same observation conditioning as PretrainDiffusion / PretrainACT:
    1. state_fused = concat(state, state_velocity, tactile, fsr)  # expected 88-dim
    2. state_fused is projected with a 3-layer MLP: 88 -> 128 -> 256 -> 512
    3. sparse_pc is encoded by the pretrained temporal transformer to a 384-dim latent
    4. final conditioning is concat(state_latent_512, sparse_latent_384) -> 896-dim

    The action generator is a simple multi-layer MLP that takes the global conditioning
    vector and directly regresses the full action chunk (horizon x action_dim).
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    horizon: int = 8
    n_action_steps: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Feature keys.
    state_feature_key: str = "observation.state"
    state_velocity_feature_key: str = "observation.state_velocity"
    tactile_feature_key: str = "observation.tactile"
    fsr_feature_key: str = "observation.fsr"
    sparse_pc_feature_key: str = "observation.sparse_pc"

    # Privileged feature keys (explicitly unused).
    hand_pc_feature_key: str = "observation.hand_pc"
    cube_pos_feature_key: str = "cube_pos"

    # Ablation mode for baseline experiments.
    # "full"       : all modalities (default)
    # "no_tactile" : remove tactile + FSR + sparse_pc (state + velocity only)
    # "no_twintac" : remove TwinTac tactile (first 32-dim) + sparse_pc, keep FSR
    # "no_fsr"     : remove FSR (last 12-dim) + sparse_pc, keep TwinTac tactile
    ablation_mode: str = "full"

    # Per-modality dims used to derive effective MLP input when ablating.
    tactile_dim: int = 32  # dim of observation.tactile  (TwinTac)
    fsr_dim: int = 12      # dim of observation.fsr

    # State fusion MLP dimensions.
    state_fused_input_dim: int = 88
    state_mlp_hidden_dim1: int = 128
    state_mlp_hidden_dim2: int = 256
    state_mlp_output_dim: int = 512

    # Pretrained sparse point-cloud encoder configuration.
    sparse_pc_num_points: int = 44
    sparse_pc_point_dim: int = 4
    pretrained_encoder_embed_dim: int = 384
    pretrained_encoder_max_seq_len: int = 16
    pretrained_encoder_ckpt_path: str = (
        "src/lerobot/policies/pretrain_diffusion/pretrain_encoder_v2/latest_geometry_ckpt.pth"
    )
    freeze_pretrained_encoder: bool = True

    # --- MLP-specific action head config ---
    # A large MLP that directly regresses the action chunk.
    action_dim: int = 8   # Action dimension; overridden by output_features when loading from dataset.
    mlp_hidden_dims: list = field(
        default_factory=lambda: [2048, 2048, 2048, 2048]
    )
    """Hidden layer dimensions for the action generation MLP.
    The MLP will have len(mlp_hidden_dims) hidden layers, each followed by ReLU.
    The final layer projects to (horizon * action_dim) and is reshaped."""

    mlp_activation: str = "relu"

    # Loss computation.
    do_mask_loss_for_padding: bool = False

    # Training presets.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        supported_ablation_modes = ["full", "no_tactile", "no_twintac", "no_fsr"]
        if self.ablation_mode not in supported_ablation_modes:
            raise ValueError(
                f"`ablation_mode` must be one of {supported_ablation_modes}. Got {self.ablation_mode}."
            )

    @property
    def effective_state_fused_input_dim(self) -> int:
        """Actual MLP input dim after removing modalities according to ablation_mode."""
        if self.ablation_mode == "full":
            return self.state_fused_input_dim
        elif self.ablation_mode == "no_tactile":
            return self.state_fused_input_dim - self.tactile_dim - self.fsr_dim
        elif self.ablation_mode == "no_twintac":
            return self.state_fused_input_dim - self.tactile_dim
        elif self.ablation_mode == "no_fsr":
            return self.state_fused_input_dim - self.fsr_dim
        return self.state_fused_input_dim

    @property
    def global_cond_dim(self) -> int:
        """Total conditioning dim fed into the MLP action head."""
        if self.ablation_mode == "full":
            return self.state_mlp_output_dim + self.pretrained_encoder_embed_dim
        return self.state_mlp_output_dim

    @property
    def active_obs_keys(self) -> list[str]:
        """Observation keys actually used by this ablation variant."""
        keys = [self.state_feature_key, self.state_velocity_feature_key]
        if self.ablation_mode in ("full", "no_fsr"):
            keys.append(self.tactile_feature_key)
        if self.ablation_mode in ("full", "no_twintac"):
            keys.append(self.fsr_feature_key)
        if self.ablation_mode == "full":
            keys.append(self.sparse_pc_feature_key)
        return keys

    @property
    def state_feature_keys(self) -> list[str]:
        """Expose the required observation keys as a list."""
        return self.active_obs_keys

    def resolve_pretrained_encoder_ckpt_path(self) -> Path:
        path = Path(self.pretrained_encoder_ckpt_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
