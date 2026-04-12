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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("pretrain_diffusion")
@dataclass
class PretrainDiffusionConfig(PreTrainedConfig):
    """Diffusion policy using a pretrained sparse point-cloud temporal encoder.

    Observation conditioning is built as follows:
    1. state_fused = concat(state, state_velocity, tactile, fsr)  # expected 88-dim
    2. state_fused is projected with a 3-layer MLP: 88 -> 128 -> 256 -> 512
    3. sparse_pc is encoded by the pretrained temporal transformer to a 384-dim latent
    4. final conditioning is concat(state_latent_512, sparse_latent_384) -> 896-dim

    Privileged inputs (e.g. hand_pc and cube_pos) are ignored by this policy.
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    horizon: int = 8
    n_action_steps: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Feature keys.
    state_feature_key: str = "observation.state"
    state_velocity_feature_key: str = "observation.state_velocity"
    tactile_feature_key: str = "observation.tactile"
    fsr_feature_key: str = "observation.fsr"
    sparse_pc_feature_key: str = "observation.sparse_pc"

    # Privileged feature keys (explicitly unused).
    hand_pc_feature_key: str = "observation.hand_pc"
    cube_pos_feature_key: str = "cube_pos"

    # State fusion MLP dimensions.
    state_fused_input_dim: int = 88
    state_mlp_hidden_dim1: int = 128
    state_mlp_hidden_dim2: int = 256
    state_mlp_output_dim: int = 512

    # Pretrained sparse point-cloud encoder configuration.
    sparse_pc_num_points: int = 44
    sparse_pc_point_dim: int = 4
    pretrained_encoder_embed_dim: int = 384
    pretrained_encoder_max_seq_len: int = 16
    pretrained_encoder_ckpt_path: str = (
        "src/lerobot/policies/pretrain_diffusion/pretrain_encoder_v2/latest_geometry_ckpt.pth"
    )
    freeze_pretrained_encoder: bool = True

    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler.
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference.
    num_inference_steps: int | None = None

    # Loss computation.
    do_mask_loss_for_padding: bool = False

    # Training presets.
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

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def resolve_pretrained_encoder_ckpt_path(self) -> Path:
        path = Path(self.pretrained_encoder_ckpt_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path

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
        required_keys = [
            self.state_feature_key,
            self.state_velocity_feature_key,
            self.tactile_feature_key,
            self.fsr_feature_key,
            self.sparse_pc_feature_key,
        ]
        if not self.input_features:
            return

        for key in required_keys:
            if key not in self.input_features:
                raise ValueError(
                    f"Required feature '{key}' not found in input_features. "
                    f"Available: {list(self.input_features.keys())}"
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
