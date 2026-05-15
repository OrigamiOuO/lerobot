#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
# Modified for Diffusion-Sparsh tactile adaptation.
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
"""Configuration class for DiffusionSparshPolicy with tactile sensor support."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig



TACTILE_RAW_PREFIXES = (
    "observation.images.tac_raw.",
    "observation.tac_raw.",
)
TACTILE_DEPTH_PREFIXES = (
    "observation.images.tac_depth.",
    "observation.tac_depth.",
)
TACTILE_NORMAL_PREFIXES = (
    "observation.images.tac_normal.",
    "observation.tac_normal.",
)
TACTILE_MARKER_PREFIXES = (
    "observation.tac_marker_displacement.",
)
TACTILE_IMAGE_PREFIXES = (
    *TACTILE_RAW_PREFIXES,
    *TACTILE_DEPTH_PREFIXES,
    *TACTILE_NORMAL_PREFIXES,
)
TACTILE_PREFIXES = (
    *TACTILE_IMAGE_PREFIXES,
    *TACTILE_MARKER_PREFIXES,
)
WRIST_RGB_KEYWORDS = (
    "wrist",
    "inhand",
    "in_hand",
    "hand_eye",
    "handeye",
    "ee_camera",
    "end_effector",
)


def _starts_with_any(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def _is_tactile_feature_key(key: str) -> bool:
    return _starts_with_any(key, TACTILE_PREFIXES)


def _is_tactile_image_feature_key(key: str) -> bool:
    return _starts_with_any(key, TACTILE_IMAGE_PREFIXES)


def _is_wrist_rgb_feature_key(key: str) -> bool:
    """Return whether a non-tactile RGB key should be treated as wrist/in-hand RGB.

    Keys not matching these wrist/in-hand keywords are treated as global/external
    RGB. This keeps common keys such as ``observation.images.global`` and
    ``observation.images.front`` under ``use_global_rgb``.
    """
    key_lower = key.lower()
    return any(keyword in key_lower for keyword in WRIST_RGB_KEYWORDS)


@PreTrainedConfig.register_subclass("diffusion_sparsh")
@dataclass
class DiffusionSparshConfig(PreTrainedConfig):
    """Configuration class for DiffusionSparshPolicy with tactile sensor support.

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

    # Backward-compatible master switch for all tactile streams.
    # If False, tac RGB, tac fusion and marker motion are all ignored by the model,
    # but `input_features` is kept unchanged so the dataset can still contain them.
    use_tactile: bool = True

    # === Modality switches for ablation ===
    # The dataset may contain all five streams. These flags only decide whether
    # each stream is packed/encoded into the policy condition during training and inference.
    use_global_rgb: bool = True
    use_wrist_rgb: bool = True
    use_tac_rgb: bool = True
    use_tac_fusion: bool = True
    use_marker_motion: bool = True

    # Inputs / output structure.
    n_obs_steps: int = 2
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

    # === Vision backbone (for standard images and tac_raw) ===
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (480, 480)
    crop_is_random: bool = True
    resize_shape: tuple[int, int] | None = (224, 224)
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # ### Tactile info config ###
    tactile_use_group_norm: bool = False  # Use BatchNorm for pretrained
    tactile_spatial_softmax_num_keypoints: int = 32
    tactile_crop_shape: tuple[int, int] | None = (480, 480)
    tactile_resize_shape: tuple[int, int] | None = (224, 224)

    # === Tactile raw image backbone (for tac_raw 3-channel data) ===
    tactile_raw_backbone: str = "resnet18"
    tactile_raw_backbone_in_channels: int = 3  # raw tactile images typically have 3 channels (e.g. RGB or grayscale repeated)
    tactile_raw_pretrained_backbone_weights: str | None = None

    # === Tactile raw encoder backend ===
    # "resnet" keeps the original baseline behavior. "sparsh" replaces the
    # tactile raw ResNet with a pretrained Sparsh ViT encoder loaded from
    # Hugging Face. The rest of the policy is unchanged.
    tactile_raw_encoder_type: str = "sparsh"  # one of ["resnet", "sparsh"]

    # === Sparsh pretrained tactile encoder ===
    # Default uses the public Sparsh DINO base checkpoint. This checkpoint expects
    # two tactile RGB frames concatenated along channels: I_t + I_{t-k} -> 6ch.
    sparsh_model_name: str = "facebook/sparsh-dino-base"
    # Use the official .ckpt by default because the official Sparsh loader expects
    # a checkpoint with a top-level ``model`` dict. The loader below also accepts
    # .safetensors if you intentionally switch to it.
    sparsh_checkpoint_filename: str = "dino_vitbase.ckpt"
    # Optional path to a local clone of facebookresearch/sparsh. If None, the
    # code assumes `tactile_ssl` is already importable through PYTHONPATH or
    # `pip install -e /path/to/sparsh`.
    sparsh_repo_path: str | None = None
    # Official Sparsh ViT builder settings. These mirror config/experiment/dino_vit.yaml
    # and config/task/* downstream configs in the Sparsh repository.
    sparsh_model_size: str = "base"  # one of ["tiny", "small", "base", "large"]
    sparsh_ssl_name: str = "dino"  # one of ["dino", "ijepa", "mae", "dinov2"]
    sparsh_input_channels: int = 6
    sparsh_patch_size: int = 16
    sparsh_feature_dim: int = 768
    sparsh_projection_dim: int = 128
    sparsh_num_register_tokens: int = 1
    sparsh_pos_embed_fn: str = "sinusoidal"
    # Official Sparsh DINO/I-JEPA pretraining uses img_size=[320, 240].
    sparsh_input_size: tuple[int, int] = (320, 240)
    # How to turn official patch-token output (B, N, D) into a vector.
    sparsh_pooling: str = "mean_patch"  # one of ["mean_patch", "first_patch"]
    sparsh_frozen: bool = True
    # With n_obs_steps=2, use stride=1. If you later set n_obs_steps>=6, set
    # sparsh_temporal_stride=5 to match Sparsh's original temporal pairing.
    sparsh_temporal_stride: int = 1
    # If raw images are still uint8-like [0,255], divide by 255 before Sparsh.
    # If LeRobot normalization has already produced mean/std-normalized tensors,
    # this branch leaves them unchanged.
    sparsh_auto_rescale_uint8: bool = True
    # If True, keep Sparsh in eval mode even when the policy is in train mode.
    sparsh_keep_eval_when_frozen: bool = True

    # TODO: add different encoder for tactile depth and tactile normal instead of fusing them as 4-channel input to a single backbone
    # === Tactile vision backbone (for tac_depth + tac_normal 4-channel data) ===
    tactile_fused_backbone: str = "resnet18"
    tactile_fused_backbone_in_channels: int = 4  # depth(1) + normal(3)
    tactile_fused_pretrained_backbone_weights: str | None = None

    # === Tactile marker encoder ===
    tactile_marker_input_dim: int = 70  # 35 markers × 2 coordinates
    tactile_marker_embed_dim: int = 16  # Output dimension (similar to state_dim)

    # Optional per-modality projection before concatenating global_cond.
    # This keeps high-dimensional tactile_fused features from dominating RGB/state.
    # Set to None to preserve the original raw feature dimensions.
    modality_projection_dim: int | None = 128
    project_state_condition: bool = False

    # === Optional multi-modal consensus MoE ===
    use_modal_moe: bool = False
    moe_num_experts: int = 2
    moe_hidden_dim: int = 256
    moe_dropout: float = 0.1
    moe_routing_dropout: float = 0.1
    moe_topk: int = 2
    modal_moe_debug_inference: bool = True
    modal_moe_debug_every_n_calls: int = 5
    modal_moe_debug_print_full_weights: bool = True

    # === Optional denoiser-level MoE (dp_unets_spec-style) ===
    # Build multiple denoiser experts per non-state modality and combine their
    # predictions with learned routing weights.
    # Start from the single-denoiser baseline by default. Enable this only after
    # tactile-DiT baselines are stable and interpretable.
    use_denoiser_moe: bool = False
    # If enabled with semantic grouping, one module per group corresponds to:
    # visual expert, tactile expert, and optional other expert.
    denoiser_num_modules: int = 1
    denoiser_composition_strategy: str = "soft_gating"  # one of ["soft_gating", "hard_routing", "topk_moe"]
    denoiser_topk: int = 1
    denoiser_grouping_strategy: str = "semantic"  # one of ["modality", "semantic"]
    denoiser_moe_debug_inference: bool = True
    denoiser_moe_debug_every_n_calls: int = 5
    denoiser_moe_debug_print_full_weights: bool = True

    denoiser_type: str = "dit"  # one of ["unet", "dit"]
    # === Unet ===
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # === DiT ===
    # Smaller default DiT for small robot/tactile datasets. Scale back up after
    # the simple baseline is stable.
    dit_d_model: int = 256
    dit_nhead: int = 8
    dit_num_layers: int = 4
    dit_dim_feedforward: int = 1024
    dit_dropout: float = 0.1

    # === Noise scheduler ===
    noise_scheduler_type: str = "DDIM"
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

        supported_tactile_raw_encoder_types = ["resnet", "sparsh"]
        if self.tactile_raw_encoder_type not in supported_tactile_raw_encoder_types:
            raise ValueError(
                "`tactile_raw_encoder_type` must be one of "
                f"{supported_tactile_raw_encoder_types}. Got {self.tactile_raw_encoder_type}."
            )

        if self.tactile_raw_encoder_type == "resnet" and not self.tactile_raw_backbone.startswith("resnet"):
            raise ValueError(
                f"`tactile_raw_backbone` must be one of the ResNet variants when "
                f"`tactile_raw_encoder_type='resnet'`. Got {self.tactile_raw_backbone}."
            )

        if self.tactile_raw_encoder_type == "sparsh":
            if self.sparsh_input_channels != 6:
                raise ValueError(
                    "Sparsh expects two RGB tactile frames concatenated "
                    f"as 6 channels. Got {self.sparsh_input_channels}."
                )
            supported_sparsh_sizes = ["tiny", "small", "base", "large"]
            if self.sparsh_model_size not in supported_sparsh_sizes:
                raise ValueError(
                    f"`sparsh_model_size` must be one of {supported_sparsh_sizes}. "
                    f"Got {self.sparsh_model_size}."
                )
            supported_ssl_names = ["dino", "ijepa", "mae", "dinov2"]
            if self.sparsh_ssl_name not in supported_ssl_names:
                raise ValueError(
                    f"`sparsh_ssl_name` must be one of {supported_ssl_names}. "
                    f"Got {self.sparsh_ssl_name}."
                )
            supported_pooling = ["mean_patch", "first_patch"]
            if self.sparsh_pooling not in supported_pooling:
                raise ValueError(
                    f"`sparsh_pooling` must be one of {supported_pooling}. "
                    f"Got {self.sparsh_pooling}."
                )
            if self.sparsh_projection_dim < 1:
                raise ValueError(
                    f"`sparsh_projection_dim` must be positive. Got {self.sparsh_projection_dim}."
                )
            if self.sparsh_patch_size < 1:
                raise ValueError(
                    f"`sparsh_patch_size` must be positive. Got {self.sparsh_patch_size}."
                )
            if self.sparsh_temporal_stride < 1:
                raise ValueError(
                    f"`sparsh_temporal_stride` must be >= 1. Got {self.sparsh_temporal_stride}."
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

        supported_denoiser_types = ["unet", "dit"]
        if self.denoiser_type not in supported_denoiser_types:
            raise ValueError(
                f"`denoiser_type` must be one of {supported_denoiser_types}. "
                f"Got {self.denoiser_type}."
            )

        if self.dit_d_model % self.dit_nhead != 0:
            raise ValueError(
                "`dit_d_model` must be divisible by `dit_nhead`. "
                f"Got {self.dit_d_model=} and {self.dit_nhead=}."
            )

        if self.modality_projection_dim is not None and self.modality_projection_dim < 1:
            raise ValueError(
                "`modality_projection_dim` must be None or a positive integer. "
                f"Got {self.modality_projection_dim}."
            )

        if self.use_modal_moe:
            if self.moe_num_experts < 1:
                raise ValueError(f"`moe_num_experts` must be >= 1. Got {self.moe_num_experts}.")
            if self.moe_hidden_dim < 1:
                raise ValueError(f"`moe_hidden_dim` must be >= 1. Got {self.moe_hidden_dim}.")
            if not (0.0 <= self.moe_dropout < 1.0):
                raise ValueError(f"`moe_dropout` must be in [0, 1). Got {self.moe_dropout}.")
            if not (0.0 <= self.moe_routing_dropout < 1.0):
                raise ValueError(
                    f"`moe_routing_dropout` must be in [0, 1). Got {self.moe_routing_dropout}."
                )
            if self.moe_topk < 1:
                raise ValueError(f"`moe_topk` must be >= 1. Got {self.moe_topk}.")
            if self.modal_moe_debug_every_n_calls < 1:
                raise ValueError(
                    "`modal_moe_debug_every_n_calls` must be >= 1. "
                    f"Got {self.modal_moe_debug_every_n_calls}."
                )

        if self.use_denoiser_moe:
            if self.denoiser_num_modules < 1:
                raise ValueError(
                    f"`denoiser_num_modules` must be >= 1. Got {self.denoiser_num_modules}."
                )

            supported_comp = ["soft_gating", "hard_routing", "topk_moe"]
            if self.denoiser_composition_strategy not in supported_comp:
                raise ValueError(
                    "`denoiser_composition_strategy` must be one of "
                    f"{supported_comp}. Got {self.denoiser_composition_strategy}."
                )

            if self.denoiser_topk < 1:
                raise ValueError(f"`denoiser_topk` must be >= 1. Got {self.denoiser_topk}.")

            if self.denoiser_moe_debug_every_n_calls < 1:
                raise ValueError(
                    "`denoiser_moe_debug_every_n_calls` must be >= 1. "
                    f"Got {self.denoiser_moe_debug_every_n_calls}."
                )

            supported_grouping = ["modality", "semantic"]
            if self.denoiser_grouping_strategy not in supported_grouping:
                raise ValueError(
                    "`denoiser_grouping_strategy` must be one of "
                    f"{supported_grouping}. Got {self.denoiser_grouping_strategy}."
                )

        # Tactile fused stream expects one-to-one depth/normal sensor pairing.
        depth_count = len(self.tactile_depth_features)
        normal_count = len(self.tactile_normal_features)
        if depth_count != normal_count:
            raise ValueError(
                "Tactile depth/normal feature count mismatch: "
                f"got {depth_count} depth streams vs {normal_count} normal streams. "
                "Please provide paired depth+normal keys per tactile sensor."
            )

        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # Transformer/DiT denoisers do not impose this constraint.
        if self.denoiser_type == "unet":
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
        """Validate that all required features are present.

        Unlike the original Diffusion Policy, tactile raw/depth/normal streams may
        also appear as image-like features. They are intentionally excluded from
        the standard RGB image shape check because they are processed by separate
        tactile branches.
        """
        # Do not mutate `input_features` for ablation. The dataset can keep all
        # streams; modality switches only affect which features are used by the model.
        rgb_image_features = self.enabled_rgb_image_features

        if self.crop_shape is not None:
            for key, image_ft in rgb_image_features.items():
                # Support both channel-first (C,H,W) and channel-last (H,W,C).
                if image_ft.shape[0] in (1, 3, 4):
                    height, width = image_ft.shape[1], image_ft.shape[2]
                else:
                    height, width = image_ft.shape[0], image_ft.shape[1]
                if self.crop_shape[0] > height or self.crop_shape[1] > width:
                    raise ValueError(
                        f"`crop_shape` should fit within RGB image shapes. Got {self.crop_shape} "
                        f"for `{key}` with shape {image_ft.shape}."
                    )

        if len(rgb_image_features) > 0:
            first_image_key, first_image_ft = next(iter(rgb_image_features.items()))
            for key, image_ft in rgb_image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        "but we expect enabled standard RGB image shapes to match. Tactile image-like "
                        "features are excluded from this check."
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

    # === Active feature properties used by the policy ===

    @property
    def rgb_image_features(self) -> dict[str, PolicyFeature]:
        """Return all non-tactile RGB image features present in the dataset config."""
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.image_features.items()
            if not _is_tactile_feature_key(key)
        }

    @property
    def global_rgb_features(self) -> dict[str, PolicyFeature]:
        """Return external/global RGB streams.

        Non-tactile RGB keys that are not recognized as wrist/in-hand cameras are
        treated as global RGB, so keys such as ``observation.images.global`` or
        ``observation.images.front`` are covered by ``use_global_rgb``.
        """
        return {
            key: ft
            for key, ft in self.rgb_image_features.items()
            if not _is_wrist_rgb_feature_key(key)
        }

    @property
    def wrist_rgb_features(self) -> dict[str, PolicyFeature]:
        """Return wrist/in-hand RGB streams."""
        return {
            key: ft
            for key, ft in self.rgb_image_features.items()
            if _is_wrist_rgb_feature_key(key)
        }

    @property
    def enabled_rgb_image_features(self) -> dict[str, PolicyFeature]:
        """Return standard RGB streams enabled for the current ablation setting.

        The original dataset feature order is preserved after filtering, which
        keeps camera packing deterministic across ablation runs.
        """
        return {
            key: ft
            for key, ft in self.rgb_image_features.items()
            if (self.use_wrist_rgb and _is_wrist_rgb_feature_key(key))
            or (self.use_global_rgb and not _is_wrist_rgb_feature_key(key))
        }

    @property
    def tactile_raw_features(self) -> dict[str, PolicyFeature]:
        """Return active raw tactile RGB image features."""
        if not (self.use_tactile and self.use_tac_rgb) or not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if _starts_with_any(key, TACTILE_RAW_PREFIXES)
        }

    @property
    def tactile_depth_features(self) -> dict[str, PolicyFeature]:
        """Return active tactile depth features used for tac_fusion.

        Supports both legacy keys (``observation.tac_depth.*``) and
        video/image-stream keys (``observation.images.tac_depth.*``).
        """
        if not (self.use_tactile and self.use_tac_fusion) or not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if _starts_with_any(key, TACTILE_DEPTH_PREFIXES)
        }

    @property
    def tactile_normal_features(self) -> dict[str, PolicyFeature]:
        """Return active tactile normal features used for tac_fusion.

        Supports both legacy keys (``observation.tac_normal.*``) and
        video/image-stream keys (``observation.images.tac_normal.*``).
        """
        if not (self.use_tactile and self.use_tac_fusion) or not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if _starts_with_any(key, TACTILE_NORMAL_PREFIXES)
        }

    @property
    def tactile_marker_features(self) -> dict[str, PolicyFeature]:
        """Return active marker displacement features."""
        if not (self.use_tactile and self.use_marker_motion) or not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if _starts_with_any(key, TACTILE_MARKER_PREFIXES)
        }

    @property
    def has_global_rgb(self) -> bool:
        """Check if global RGB is enabled and present."""
        return self.use_global_rgb and len(self.global_rgb_features) > 0

    @property
    def has_wrist_rgb(self) -> bool:
        """Check if wrist/in-hand RGB is enabled and present."""
        return self.use_wrist_rgb and len(self.wrist_rgb_features) > 0

    @property
    def has_tactile_raw(self) -> bool:
        """Check if raw tactile image data is enabled and present."""
        return len(self.tactile_raw_features) > 0

    @property
    def has_tactile_depth(self) -> bool:
        """Check if tactile depth data is enabled and present."""
        return len(self.tactile_depth_features) > 0

    @property
    def has_tactile_normal(self) -> bool:
        """Check if tactile normal data is enabled and present."""
        return len(self.tactile_normal_features) > 0

    @property
    def has_tactile_fused(self) -> bool:
        """Check if tactile fusion data (depth + normal) is enabled and present."""
        return len(self.tactile_depth_features) > 0 and len(self.tactile_normal_features) > 0

    @property
    def has_tactile_marker(self) -> bool:
        """Check if marker displacement data is enabled and present."""
        return len(self.tactile_marker_features) > 0
