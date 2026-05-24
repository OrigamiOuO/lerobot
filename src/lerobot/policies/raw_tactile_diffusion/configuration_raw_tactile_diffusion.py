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
"""Configuration for RawTactileDiffusion policy.

Architecture:
    Multi-modal state features (state, state_velocity, tactile, fsr) are concatenated,
    passed through an MLP encoder to produce a global conditioning vector,
    which conditions a 1D Conditional UNet for diffusion-based action denoising.

No image input or point cloud processing is used.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("raw_tactile_diffusion")
@dataclass
class RawTactileDiffusionConfig(PreTrainedConfig):
    """Diffusion policy using only raw tactile and state observations (no images, no point cloud).

    This policy concatenates multiple state observation modalities into a single composite
    state vector, encodes it through an MLP, and uses the resulting embedding as global
    conditioning for a 1D Conditional UNet diffusion model.

    Key features:
    - Uses observation.state, observation.state_velocity, observation.tactile, observation.fsr
    - Optional: cube_pos can be included as additional state feature
    - No image observations required
    - No point cloud processing (no PointNet)
    - Simple MLP-based state encoding instead of Transformer cross-attention
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
    # State features to concatenate (in order)
    state_feature_keys: list[str] = field(
        default_factory=lambda: [
            OBS_STATE,  # "observation.state" - joint positions
            "observation.state_velocity",  # joint velocities
            "observation.tactile",  # tactile sensor readings
            "observation.fsr",  # force-sensitive resistor readings
        ]
    )

    # Optional: include cube_pos as additional state feature
    include_cube_pos: bool = False
    cube_pos_key: str = "cube_pos"

    # The original implementation doesn't sample frames for the last 7 steps
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ===== MLP State Encoder Architecture =====
    # Hidden dimensions of the MLP that encodes the composite state vector.
    # The MLP takes (composite_state_dim * n_obs_steps) as input and produces
    # state_embed_dim as output for UNet conditioning.
    state_encoder_hidden_dims: tuple[int, ...] = (256, 256)
    state_embed_dim: int = 256

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
    do_mask_loss_for_padding: bool = True

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
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
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
        keys = list(self.state_feature_keys)
        if self.include_cube_pos:
            keys.append(self.cube_pos_key)
        return keys

    def validate_features(self) -> None:
        """Validate that required features are present in input_features."""
        has_any_state = any(key in self.input_features for key in self.state_feature_keys)

        if not has_any_state:
            raise ValueError(
                f"At least one state feature from {self.state_feature_keys} "
                f"must be in input_features. Available: {list(self.input_features.keys())}"
            )

        # Validate that tactile is present (since this policy is designed for tactile use)
        if "observation.tactile" not in self.input_features:
            raise ValueError(
                "This policy requires 'observation.tactile' in input_features, "
                f"but it was not found. Available: {list(self.input_features.keys())}"
            )

        if self.include_cube_pos and self.cube_pos_key not in self.input_features:
            raise ValueError(
                f"include_cube_pos=True but '{self.cube_pos_key}' not in input_features. "
                f"Available: {list(self.input_features.keys())}"
            )

    def compute_composite_state_dim(self) -> int:
        """Compute total dimension of concatenated state features."""
        if not hasattr(self, "input_features") or not self.input_features:
            raise ValueError(
                "input_features not yet populated. This should be called after feature "
                "initialization by the factory."
            )

        total_dim = 0
        for key in self.state_feature_keys:
            if key in self.input_features:
                feature = self.input_features[key]
                if isinstance(feature.shape, (list, tuple)):
                    dim = 1
                    for s in feature.shape:
                        dim *= s
                    total_dim += dim

        if self.include_cube_pos and self.cube_pos_key in self.input_features:
            feature = self.input_features[self.cube_pos_key]
            if isinstance(feature.shape, (list, tuple)):
                dim = 1
                for s in feature.shape:
                    dim *= s
                total_dim += dim

        return total_dim

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
