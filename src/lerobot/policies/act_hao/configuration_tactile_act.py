#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("tactile_act_hao")
@dataclass
class TactileACTConfig(PreTrainedConfig):
    """Configuration class for Tactile ACT policy with GelSight tactile sensor support.

    Supports four types of tactile data from GelSight sensors (individually toggleable
    for ablation experiments):
        - Raw tactile RGB image (3ch): dedicated ResNet backbone (unfrozen BN).
        - Depth map (1ch, image-like): ResNet backbone (unfrozen BN).
        - Normal map (3ch, image-like): ResNet backbone (unfrozen BN).
        - Marker displacement (35 markers × 2D coordinates): MLP encoder → 1D token.

    When both depth and normal are enabled, they are fused into a single 4-channel
    input to a shared ResNet backbone. When only one is enabled, the backbone uses
    the corresponding number of input channels (1 or 3).

    Additionally supports:
        - Camera RGB images (processed by a separate ResNet backbone with FrozenBatchNorm2d).
        - Robot proprioceptive state (joint positions).

    The tactile backbone uses standard (unfrozen) BatchNorm2d for full fine-tuning,
    while the camera backbone uses FrozenBatchNorm2d following the original ACT design.
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
        }
    )

    # Architecture.
    # Vision backbone (for camera RGB images, with FrozenBatchNorm2d).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Backbone freezing: when True, freeze ALL backbone conv/linear weights
    # (only the 1×1 projection layers and Transformer are trained).
    # Recommended for small datasets (<5000 frames) to prevent overfitting.
    freeze_backbone: bool = False
    freeze_tactile_backbone: bool = False
    freeze_tactile_raw_backbone: bool = False

    # Tactile image features: depth (1ch) and/or normal (3ch) → ResNet (unfrozen BN).
    # Both can be independently toggled for ablation experiments.
    # When both enabled: fused 4ch input. When one enabled: 1ch or 3ch input.
    use_tactile_depth: bool = True
    use_tactile_normal: bool = True
    tactile_vision_backbone: str = "resnet18"
    pretrained_tactile_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # Tactile raw RGB image features: 3ch ResNet (unfrozen BN).
    # The raw tactile image is processed by a dedicated backbone, separate from camera images.
    use_tactile_raw_image: bool = True
    tactile_raw_image_key: str = "observation.images.tac_raw.tac1"
    tactile_raw_vision_backbone: str = "resnet18"
    pretrained_tactile_raw_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    optimizer_lr_tactile_raw_backbone: float = 1e-5

    # Tactile marker displacement features: (35, 2) → flatten → MLP → independent token.
    use_tactile_marker: bool = True
    tactile_marker_input_dim: int = 70   # 35 markers * 2 (x, y)
    tactile_marker_hidden_dim: int = 128

    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_lr_tactile_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if (self.use_tactile_depth or self.use_tactile_normal) and not self.tactile_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_vision_backbone` must be one of the ResNet variants. Got {self.tactile_vision_backbone}."
            )
        if self.use_tactile_raw_image and not self.tactile_raw_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_raw_vision_backbone` must be one of the ResNet variants. Got {self.tactile_raw_vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        """Validate that at least one meaningful input feature is provided."""
        has_vision = bool(self.image_features)
        has_env = self.env_state_feature is not None
        has_state = self.robot_state_feature is not None
        has_tactile = self.use_tactile_depth or self.use_tactile_normal or self.use_tactile_marker or self.use_tactile_raw_image

        if not has_vision and not has_env and not has_state and not has_tactile:
            raise ValueError(
                "You must provide at least one of: image features, environment state, "
                "robot state, or tactile features among the inputs."
            )

    @property
    def observation_delta_indices(self) -> list | None:
        if self.n_obs_steps <= 1:
            return None
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
