#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
# Modified for Diffusion-Hao tactile adaptation.
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
"""Configuration class for DiffusionHaoPolicy with tactile sensor support."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion_hao")
@dataclass
class DiffusionHaoConfig(PreTrainedConfig):
    """Configuration class for DiffusionHaoPolicy with tactile sensor support.

    Extends the original Diffusion Policy to handle:
    - Standard images (global, inhand, tac_raw) via shared ResNet backbone
    - Tactile depth + normal (4-channel) via independent ResNet backbone
    - Tactile marker displacement fused with robot state

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model action prediction size.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        tactile_vision_backbone: ResNet variant for 4-channel tactile data.
        tactile_backbone_in_channels: Input channels for tactile backbone (depth=1 + normal=3 = 4).
        tactile_marker_embed_dim: Output dimension for tactile marker encoder.
    """

    # Inputs / output structure.
    n_obs_steps: int = 8
    horizon: int = 16
    n_action_steps: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "TACTILE": NormalizationMode.MEAN_STD,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps.
    drop_n_last_frames: int = 7

    # === Vision backbone (for standard images and tac_raw) ===
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # === Tactile vision backbone (for tac_depth + tac_normal 4-channel data) ===
    tactile_vision_backbone: str = "resnet18"
    tactile_backbone_in_channels: int = 4  # depth(1) + normal(3)
    tactile_pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    tactile_use_group_norm: bool = False  # Use BatchNorm for pretrained
    tactile_spatial_softmax_num_keypoints: int = 32
    tactile_crop_shape: tuple[int, int] | None = (84, 84)

    # === Tactile marker encoder ===
    tactile_marker_input_dim: int = 70  # 35 markers × 2 coordinates
    tactile_marker_embed_dim: int = 16  # Output dimension (similar to state_dim)

    # === Unet ===
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # === Noise scheduler ===
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # === Inference ===
    num_inference_steps: int | None = None

    # === Loss computation ===
    do_mask_loss_for_padding: bool = False

    # === Training presets ===
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

        if not self.tactile_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_vision_backbone` must be one of the ResNet variants. "
                f"Got {self.tactile_vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.prediction_type}."
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
                "The horizon should be an integer multiple of the downsampling factor "
                f"(which is determined by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
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

    def validate_features(self) -> None:
        """Validate that all required features are present."""
        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        "but we expect all image shapes to match."
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

    # === Tactile feature properties ===

    @property
    def tactile_depth_features(self) -> dict[str, PolicyFeature]:
        """Return features for tactile depth data.

        Supports both legacy keys (``observation.tac_depth.*``) and
        video/image-stream keys (``observation.images.tac_depth.*``).
        """
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_depth.") or key.startswith("observation.images.tac_depth.")
        }

    @property
    def tactile_normal_features(self) -> dict[str, PolicyFeature]:
        """Return features for tactile normal data.

        Supports both legacy keys (``observation.tac_normal.*``) and
        video/image-stream keys (``observation.images.tac_normal.*``).
        """
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_normal.") or key.startswith("observation.images.tac_normal.")
        }

    @property
    def tactile_marker_features(self) -> dict[str, PolicyFeature]:
        """Return features for tactile marker displacement data."""
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_marker_displacement.")
        }

    @property
    def has_tactile_vision(self) -> bool:
        """Check if tactile vision data (depth + normal) is present."""
        return len(self.tactile_depth_features) > 0 and len(self.tactile_normal_features) > 0

    @property
    def has_tactile_marker(self) -> bool:
        """Check if tactile marker displacement data is present."""
        return len(self.tactile_marker_features) > 0
