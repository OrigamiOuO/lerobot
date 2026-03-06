#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Modified for ACT-Hao tactile adaptation.
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
"""Configuration class for ACT-Hao policy with tactile sensor support."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_hao")
@dataclass
class ACTHaoConfig(PreTrainedConfig):
    """Configuration class for the ACT-Hao policy with tactile sensor support.

    Extends the original ACT policy to handle:
    - Standard images (global, inhand, tac_raw) via shared ResNet backbone
    - Tactile depth + normal (4-channel) via independent ResNet backbone
    - Tactile marker displacement fused with robot state

    Args:
        n_obs_steps: Number of observation steps.
        chunk_size: Action chunk size.
        n_action_steps: Number of action steps per policy invocation.
        vision_backbone: ResNet variant for standard images.
        tactile_vision_backbone: ResNet variant for 4-channel tactile data.
        tactile_backbone_in_channels: Input channels for tactile backbone (depth=1 + normal=3 = 4).
        dim_model: Transformer hidden dimension.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone (for standard images and tac_raw).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Tactile vision backbone (for tac_depth + tac_normal 4-channel data).
    tactile_vision_backbone: str = "resnet18"
    tactile_backbone_in_channels: int = 4  # depth(1) + normal(3)
    tactile_pretrained_backbone_weights: str | None = None
    tactile_replace_final_stride_with_dilation: int = False

    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if not self.tactile_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_vision_backbone` must be a ResNet variant. Got {self.tactile_vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})"
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    # --- Tactile feature properties ---

    @property
    def tactile_depth_features(self) -> dict:
        """Return all tac_depth features."""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if "tac_depth" in k}

    @property
    def tactile_normal_features(self) -> dict:
        """Return all tac_normal features."""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if "tac_normal" in k}

    @property
    def tactile_marker_features(self) -> dict:
        """Return all tac_marker_displacement features."""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if "tac_marker" in k}

    @property
    def has_tactile_vision(self) -> bool:
        """Whether there is tactile depth+normal data that needs a vision backbone."""
        return bool(self.tactile_depth_features) and bool(self.tactile_normal_features)

    # --- Standard config methods ---

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.has_tactile_vision:
            raise ValueError(
                "You must provide at least one of: image features, environment state, "
                "or tactile vision features."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
