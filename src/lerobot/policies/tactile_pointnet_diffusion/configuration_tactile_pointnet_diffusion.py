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
"""Configuration for TactilePointnetDiffusion policy.

Architecture:
    PointNet Encoder → processes hand_pc point cloud (1352×7)
    Multi-Modal Transformer Encoder → fuses state, state_velocity, tactile, and PointNet features
    Conditional UNet → diffusion-based action denoising conditioned on Transformer output
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


OBS_HAND_PC = "observation.hand_pc"
OBS_SPARSE_PC = "observation.sparse_pc"


@PreTrainedConfig.register_subclass("tactile_pointnet_diffusion")
@dataclass
class TactilePointnetDiffusionConfig(PreTrainedConfig):
    """Diffusion policy with PointNet + Transformer encoder for tactile/state fusion.

    This policy removes image processing entirely and instead uses:
    1. PointNet to encode hand point cloud data (observation.hand_pc)
    2. A Transformer encoder to fuse multiple modalities:
       - observation.state (joint positions)
       - observation.state_velocity (joint velocities)
       - tactile raw data (observation.tactile + observation.fsr concatenated)
       - PointNet-encoded hand point cloud features
    3. The Transformer output conditions a 1D UNet for diffusion-based action generation.
    """

    # ===== Inputs / output structure =====
    n_obs_steps: int = 4
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ===== Feature keys =====
    # State features: each becomes a separate token type in the Transformer
    state_feature_key: str = "observation.state"
    state_velocity_feature_key: str = "observation.state_velocity"

    # Tactile features: concatenated into a single 44-dim vector per frame
    tactile_feature_keys: list[str] = field(
        default_factory=lambda: [
            "observation.tactile",
            "observation.fsr",
        ]
    )

    # Point cloud feature processed by PointNet
    pointcloud_feature_key: str = OBS_HAND_PC

    # Optional sparse point cloud feature encoded and concatenated to state
    include_sparse_pc_in_state: bool = True
    sparse_pointcloud_feature_key: str = OBS_SPARSE_PC

    # Optional: cube_pos as additional token (default off — see design decisions)
    include_cube_pos: bool = False
    cube_pos_key: str = "cube_pos"

    # The original implementation doesn't sample frames for the last 7 steps
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ===== PointNet Architecture =====
    pointnet_input_dim: int = 7  # xyz + fx,fy,fz + f_magnitude
    pointnet_hidden_dims: tuple[int, ...] = (64, 128, 256)
    pointnet_output_dim: int = 256
    pointnet_use_batch_norm: bool = True

    # ===== Sparse Point Cloud Encoder =====
    sparse_pc_hidden_dims: tuple[int, ...] = (128, 64)
    sparse_pc_output_dim: int = 32

    # ===== Transformer Encoder Architecture =====
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.1
    # How to aggregate Transformer output tokens into a single conditioning vector:
    #   "mean": mean-pool all output tokens → (B, d_model)
    #   "cls":  prepend a learnable [CLS] token, use its output → (B, d_model)
    #   "flatten": flatten all output tokens → (B, num_tokens * d_model)
    transformer_aggregation: str = "mean"

    # ===== UNet Architecture =====
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ===== Noise Scheduler =====
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ===== Inference =====
    num_inference_steps: int | None = None

    # ===== Loss =====
    do_mask_loss_for_padding: bool = False

    # ===== Training presets =====
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multniiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        supported_aggregations = ["mean", "cls", "flatten"]
        if self.transformer_aggregation not in supported_aggregations:
            raise ValueError(
                f"`transformer_aggregation` must be one of {supported_aggregations}. "
                f"Got {self.transformer_aggregation}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def get_all_feature_keys(self) -> list[str]:
        """Return all observation feature keys used by this policy."""
        keys = [self.state_feature_key, self.state_velocity_feature_key]
        keys.extend(self.tactile_feature_keys)
        keys.append(self.pointcloud_feature_key)
        if self.include_sparse_pc_in_state:
            keys.append(self.sparse_pointcloud_feature_key)
        if self.include_cube_pos:
            keys.append(self.cube_pos_key)
        return keys

    def validate_features(self) -> None:
        """Validate that required features are present in input_features."""
        required_keys = [
            self.state_feature_key,
            self.state_velocity_feature_key,
            self.pointcloud_feature_key,
        ]
        # At least one tactile key should be present
        has_tactile = any(k in self.input_features for k in self.tactile_feature_keys)

        for key in required_keys:
            if key not in self.input_features:
                raise ValueError(
                    f"Required feature '{key}' not found in input_features. "
                    f"Available: {list(self.input_features.keys())}"
                )

        if not has_tactile:
            raise ValueError(
                f"At least one tactile feature from {self.tactile_feature_keys} "
                f"must be in input_features."
            )

        if self.include_cube_pos and self.cube_pos_key not in self.input_features:
            raise ValueError(
                f"include_cube_pos=True but '{self.cube_pos_key}' not in input_features."
            )

        if self.include_sparse_pc_in_state and self.sparse_pointcloud_feature_key not in self.input_features:
            raise ValueError(
                "include_sparse_pc_in_state=True but "
                f"'{self.sparse_pointcloud_feature_key}' not in input_features."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
