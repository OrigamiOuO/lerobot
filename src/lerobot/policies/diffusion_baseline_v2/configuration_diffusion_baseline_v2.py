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
"""Configuration for DiffusionBaselineV2Policy — baseline + sparse_pc encoding."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("diffusion_baseline_v2")
@dataclass
class DiffusionBaselineV2Config(PreTrainedConfig):
    """Diffusion policy with multi-modal state + sparse_pc encoding.

    Extends DiffusionBaseline by additionally encoding ``observation.sparse_pc``
    (a 44×4 point cloud) through a lightweight encoder and injecting the result
    into the global conditioning of the UNet.
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
    # NOTE: This is stored as a private field; access via the state_feature_keys property
    # which conditionally appends sparse_pc_feature_key for column filtering.
    _state_feature_keys: list[str] = field(
        default_factory=lambda: [
            OBS_STATE,
            "observation.state_velocity",
            "observation.tactile",
            "observation.fsr",
        ],
        repr=False,
    )

    @property
    def state_feature_keys(self) -> list[str]:
        """Return keys for observation filtering and state concatenation.
        
        When ``include_sparse_pc_in_cond`` is True, ``sparse_pc_feature_key``
        is appended so that the data loader does NOT filter it out
        (``filter_unused_dataset_columns``).  The model's
        ``_concatenate_state_features`` will skip it because it is encoded
        separately by the sparse PC encoder.
        """
        keys = list(self._state_feature_keys)
        if self.include_sparse_pc_in_cond:
            keys.append(self.sparse_pc_feature_key)
        return keys

    @state_feature_keys.setter
    def state_feature_keys(self, value: list[str]) -> None:
        # Allow setting via CLI / draccus by delegating to the private field.
        self._state_feature_keys = value

    drop_n_last_frames: int = 7

    # --- Sparse point cloud encoding ---
    sparse_pc_feature_key: str = "observation.sparse_pc"
    """Key for the sparse point cloud feature in the batch dict."""
    include_sparse_pc_in_cond: bool = True
    """Whether to encode sparse_pc and include it in the UNet global conditioning."""
    sparse_pc_embed_dim: int = 64
    """Output dimension of the sparse PC encoder (added to global_cond_dim)."""
    sparse_pc_num_points: int = 44
    sparse_pc_point_dim: int = 4

    # Architecture / modeling.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Unet.
    down_dims: tuple[int, ...] = (256, 512)
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

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor. "
                f"Got {self.horizon=} and {self.down_dims=}"
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
        total_dim = 0
        # Use _state_feature_keys to exclude sparse_pc (handled by separate encoder)
        for key in self._state_feature_keys:
            if key in self.input_features:
                feature = self.input_features[key]
                if isinstance(feature.shape, (list, tuple)):
                    dim = 1
                    for s in feature.shape:
                        dim *= s
                    total_dim += dim
        return total_dim

    def validate_features(self) -> None:
        has_image = bool(self.image_features)
        has_env = self.env_state_feature is not None
        # Use _state_feature_keys to avoid counting sparse_pc
        has_any_state = any(
            key in self.input_features for key in self._state_feature_keys
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
                        f"but image '{key}' has shape {image_ft.shape}."
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
