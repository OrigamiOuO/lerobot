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
"""Configuration for DiffusionBaselinePolicy with multi-modal state concatenation support."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("diffusion_baseline")
@dataclass
class DiffusionBaselineConfig(PreTrainedConfig):
    """Diffusion policy with multi-modal state features concatenation support.
    
    This variant extends the standard Diffusion policy to handle datasets with multiple state modalities
    (e.g., proprioceptive state, tactile sensing, FSR pressure). These features are automatically
    concatenated along the feature dimension to form a composite state representation.
    
    Key difference from standard Diffusion:
    - Supports concatenation of multiple observation.* state features (e.g., observation.state,
      observation.tactile, observation.fsr, observation.state_velocity)
    - The total state dimension is automatically computed as the sum of all state feature dimensions
    - All state features are processed together through the conditioning network
    
    Example dataset features:
        - observation.state: (22,)       # joint positions
        - observation.tactile: (32,)     # tactile sensors
        - observation.fsr: (12,)         # force-sensitive resistors
        - Total concatenated state: (66,)
    
    The policy uses a conditional UNet architecture with:
    - Multi-modal state observation concatenation as global conditioning
    - RGB image encoding using a ResNet backbone with spatial softmax
    - FiLM (Feature-wise Linear Modulation) for timestep conditioning
    - DDPM or DDIM noise scheduling for diffusion process
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    horizon: int = 8
    n_action_steps: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Multi-state feature concatenation.
    # List of observation state feature keys to concatenate (in order).
    # These will be concatenated and treated as a single composite state.
    state_feature_keys: list[str] = field(
        default_factory=lambda: [
            OBS_STATE,  # "observation.state" - always included as primary state
            "observation.state_velocity",
            "observation.tactile",
            "observation.fsr",
        ]
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

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
        # U-Net downsamples by 2 with each stage.
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

    def compute_composite_state_dim(self) -> int:
        """
        Compute the total dimension of the concatenated state features.
        
        Returns:
            Total state dimension after concatenation. For example, if we have:
            - observation.state: 22
            - observation.tactile: 32
            - observation.fsr: 12
            Total would be 66.
        """
        if not hasattr(self, 'input_features') or not self.input_features:
            raise ValueError(
                "input_features not yet populated. This should be called after feature "
                "initialization by the factory."
            )
        
        total_dim = 0
        for key in self.state_feature_keys:
            if key in self.input_features:
                feature = self.input_features[key]
                # Feature shape is already 1D for state features (after flattening multi-frame)
                if isinstance(feature.shape, (list, tuple)):
                    dim = 1
                    for s in feature.shape:
                        dim *= s
                    total_dim += dim
        
        return total_dim

    def validate_features(self) -> None:
        """Validate that we have at least one state feature or image/env features."""
        has_image = bool(self.image_features)
        has_env = self.env_state_feature is not None
        
        # Check for any state features in the configured state_feature_keys
        has_any_state = any(
            key in self.input_features for key in self.state_feature_keys
        )
        
        if not has_image and not has_env and not has_any_state:
            raise ValueError(
                "You must provide at least one of: image features, environment state, "
                "or state features (observation.state, observation.tactile, etc.)."
            )

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
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
