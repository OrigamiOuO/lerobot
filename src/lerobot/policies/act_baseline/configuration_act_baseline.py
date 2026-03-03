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
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("act_baseline")
@dataclass
class ACTBaselineConfig(PreTrainedConfig):
    """ACT policy with multi-modal state features concatenation support.
    
    This variant extends the standard ACT policy to handle datasets with multiple state modalities
    (e.g., proprioceptive state, tactile sensing, FSR pressure). These features are automatically
    concatenated along the feature dimension to form a composite state representation.
    
    Key difference from standard ACT:
    - Supports concatenation of multiple observation.* state features (e.g., observation.state,
      observation.tactile, observation.fsr, observation.state_velocity)
    - The total state dimension is automatically computed as the sum of all state feature dimensions
    - All state features are automatically normalized together
    
    Example dataset features:
        - observation.state: (22,)       # joint positions
        - observation.tactile: (32,)     # tactile sensors
        - observation.fsr: (12,)         # force-sensitive resistors
        - Total concatenated state: (66,)
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

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    
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

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
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
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

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

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
