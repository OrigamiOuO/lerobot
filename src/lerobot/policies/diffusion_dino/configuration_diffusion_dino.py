#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
# Modified for Diffusion-Dino: DINOv2-based Diffusion Policy with tactile sensor support.
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
"""Configuration class for DiffusionDinoPolicy with DINOv2 vision backbone and tactile sensor support."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig

# DINOv2 model variant → embedding dimension mapping
DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


@PreTrainedConfig.register_subclass("diffusion_dino")
@dataclass
class DiffusionDinoConfig(PreTrainedConfig):
    """Configuration class for DiffusionDinoPolicy with DINOv2 vision backbone.

    Extends the Diffusion-Hao architecture by replacing the ResNet RGB vision
    encoder with a frozen DINOv2 ViT encoder.  Patch tokens are reshaped into
    a 2D feature map and pooled via SpatialSoftmax to preserve spatial information.

    Tactile encoders (raw, fused, marker) remain identical to diffusion_hao.

    Args:
        vision_backbone: DINOv2 model variant name (e.g. ``dinov2_vits14``).
        freeze_vision_backbone: Whether to freeze the DINOv2 backbone weights.
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model action prediction size.
        n_action_steps: The number of action steps to run in the environment for one invocation.
    """

    # Whether to use tactile sensor data. Set to False for ablation without tactile.
    use_tactile: bool = True

    # Inputs / output structure.
    n_obs_steps: int = 4
    horizon: int = 8
    n_action_steps: int = 4

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

    # === DINOv2 Vision backbone (for standard RGB images) ===
    vision_backbone: str = "dinov2_vits14"
    freeze_vision_backbone: bool = True
    crop_shape: tuple[int, int] | None = (224, 224)
    crop_is_random: bool = True
    # DINOv2 expects 224×224 input (must be divisible by patch_size=14)
    resize_shape: tuple[int, int] | None = (224, 224)
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # ### Tactile info config ###
    tactile_use_group_norm: bool = False
    tactile_spatial_softmax_num_keypoints: int = 32
    tactile_crop_shape: tuple[int, int] | None = (480, 480)
    tactile_resize_shape: tuple[int, int] | None = (224, 224)

    # === Tactile raw image backbone (for tac_raw 3-channel data, still ResNet) ===
    tactile_raw_backbone: str = "resnet18"
    tactile_raw_backbone_in_channels: int = 3
    tactile_raw_pretrained_backbone_weights: str | None = None

    # === Tactile vision backbone (for tac_depth + tac_normal 4-channel data, still ResNet) ===
    tactile_fused_backbone: str = "resnet18"
    tactile_fused_backbone_in_channels: int = 4
    tactile_fused_pretrained_backbone_weights: str | None = None

    # === Tactile marker encoder ===
    tactile_marker_input_dim: int = 70
    tactile_marker_embed_dim: int = 16

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

    @property
    def dino_embed_dim(self) -> int:
        """Return the embedding dimension for the configured DINOv2 variant."""
        return DINOV2_EMBED_DIMS[self.vision_backbone]

    @property
    def dino_patch_size(self) -> int:
        """DINOv2 patch size (always 14 for all official variants)."""
        return 14

    def __post_init__(self):
        super().__post_init__()

        if self.vision_backbone not in DINOV2_EMBED_DIMS:
            raise ValueError(
                f"`vision_backbone` must be one of {list(DINOV2_EMBED_DIMS.keys())}. "
                f"Got {self.vision_backbone}."
            )

        # Validate that resize_shape is divisible by patch_size
        if self.resize_shape is not None:
            ps = self.dino_patch_size
            if self.resize_shape[0] % ps != 0 or self.resize_shape[1] % ps != 0:
                raise ValueError(
                    f"`resize_shape` must be divisible by DINOv2 patch_size={ps}. "
                    f"Got {self.resize_shape}."
                )

        if not self.tactile_raw_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_raw_backbone` must be one of the ResNet variants. "
                f"Got {self.tactile_raw_backbone}."
            )

        if not self.tactile_fused_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_fused_backbone` must be one of the ResNet variants. "
                f"Got {self.tactile_fused_backbone}."
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
        if not self.use_tactile and self.input_features:
            tactile_prefixes = (
                "observation.images.tac_raw.",
                "observation.tac_raw.",
                "observation.images.tac_depth.",
                "observation.tac_depth.",
                "observation.images.tac_normal.",
                "observation.tac_normal.",
                "observation.tac_marker_displacement.",
            )
            self.input_features = {
                k: v for k, v in self.input_features.items()
                if not any(k.startswith(p) for p in tactile_prefixes)
            }

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

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

    # === Tactile feature properties (identical to diffusion_hao) ===

    @property
    def tactile_raw_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.images.tac_raw.") or key.startswith("observation.tac_raw.")
        }

    @property
    def tactile_depth_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_depth.") or key.startswith("observation.images.tac_depth.")
        }

    @property
    def tactile_normal_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_normal.") or key.startswith("observation.images.tac_normal.")
        }

    @property
    def tactile_marker_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if key.startswith("observation.tac_marker_displacement.")
        }

    @property
    def has_tactile_raw(self) -> bool:
        return len(self.tactile_raw_features) > 0

    @property
    def has_tactile_depth(self) -> bool:
        return len(self.tactile_depth_features) > 0

    @property
    def has_tactile_normal(self) -> bool:
        return len(self.tactile_normal_features) > 0

    @property
    def has_tactile_fused(self) -> bool:
        return len(self.tactile_depth_features) > 0 and len(self.tactile_normal_features) > 0

    @property
    def has_tactile_marker(self) -> bool:
        return len(self.tactile_marker_features) > 0
