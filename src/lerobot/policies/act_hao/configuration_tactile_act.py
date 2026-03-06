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


@PreTrainedConfig.register_subclass("act_hao")
@dataclass
class TactileACTConfig(PreTrainedConfig):
    """Configuration class for Tactile ACT policy with GelSight tactile sensor support.

    Supports three types of tactile data from GelSight sensors:
        - Depth map (1-channel, image-like) and Normal map (3-channel, image-like):
          Fused as a 4-channel input to a dedicated ResNet backbone (方案A).
        - Marker displacement (35 markers × 2D coordinates):
          Encoded as an independent 1D token in the transformer encoder (方案B).

    Additionally supports:
        - Camera RGB images (processed by a separate ResNet backbone with FrozenBatchNorm2d).
        - Robot proprioceptive state (joint positions).

    The tactile backbone uses standard (unfrozen) BatchNorm2d for full fine-tuning,
    while the camera backbone uses FrozenBatchNorm2d following the original ACT design.

    Args:
        n_obs_steps: Number of observation steps. Fixed to 1 (single frame) for this version.
        chunk_size: Action chunk size for prediction.
        n_action_steps: Number of action steps to execute per policy invocation.
        vision_backbone: ResNet variant for camera image encoding.
        pretrained_backbone_weights: Pretrained weights for camera backbone.
        tactile_vision_backbone: ResNet variant for tactile image (depth+normal) encoding.
        pretrained_tactile_backbone_weights: Pretrained weights for tactile backbone.
        use_tactile_image_features: Whether to use depth+normal tactile images.
        use_tactile_marker: Whether to use marker displacement features.
        tactile_marker_input_dim: Flattened dimension of marker displacement (35*2=70).
        tactile_marker_hidden_dim: Hidden dimension for marker MLP encoder.
        optimizer_lr_tactile_backbone: Learning rate for the tactile backbone.
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

    # Tactile image features: depth (1ch) + normal (3ch) → 4ch ResNet (unfrozen BN).
    use_tactile_image_features: bool = True
    tactile_vision_backbone: str = "resnet18"
    pretrained_tactile_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

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
        if self.use_tactile_image_features and not self.tactile_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_vision_backbone` must be one of the ResNet variants. Got {self.tactile_vision_backbone}."
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
        has_tactile = self.use_tactile_image_features or self.use_tactile_marker

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
