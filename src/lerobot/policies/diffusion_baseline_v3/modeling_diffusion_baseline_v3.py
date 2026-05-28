#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy V3 with PointNet-style Sparse PC Encoder.

Compared to V2, the V3 SparsePCEncoder uses:

- **Max-pooling** instead of mean-pooling — preserves the most salient
  per-point features rather than diluting them across all points
  (PointNet key insight).
- **Deeper per-point MLP** (4→64→128 vs 4→32→64) with LayerNorm for
  stable training.
- **Projection head** (128→128→embed_dim) after pooling for richer
  global feature learning.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion_baseline_v3.configuration_diffusion_baseline_v3 import DiffusionBaselineV3Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

OBS_SPARSE_PC = "observation.sparse_pc"


class DiffusionBaselineV3Policy(PreTrainedPolicy):
    """Diffusion Policy with PointNet-style sparse_pc encoder."""

    config_class = DiffusionBaselineV3Config
    name = "diffusion_baseline_v3"

    def __init__(
        self,
        config: DiffusionBaselineV3Config,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.composite_state_dim = self._compute_composite_state_dim()
        self._queues = None
        self.diffusion = DiffusionBaselineV3Model(config, composite_state_dim=self.composite_state_dim)
        self.reset()

    def _compute_composite_state_dim(self) -> int:
        """Compute total dimension of concatenated state features (excluding sparse_pc)."""
        total_dim = 0
        for key in self.config._state_feature_keys:
            if key in self.config.input_features:
                feature = self.config.input_features[key]
                dim = 1
                for s in feature.shape:
                    dim *= s
                total_dim += dim
        return total_dim if total_dim > 0 else 0

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        for key in self.config.state_feature_keys:
            if key in self.config.input_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

        if self.config.include_sparse_pc_in_cond:
            self._queues[OBS_SPARSE_PC] = deque(maxlen=self.config.n_obs_steps)

        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """Factory for noise scheduler instances."""
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionBaselineV3Model(nn.Module):
    """Diffusion model with composite state + PointNet-style sparse_pc conditioning."""

    def __init__(self, config: DiffusionBaselineV3Config, composite_state_dim: int = 0):
        super().__init__()
        self.config = config
        self.composite_state_dim = composite_state_dim

        global_cond_dim = composite_state_dim if composite_state_dim > 0 else 0

        # --- PointNet-style Sparse PC encoder ---
        if self.config.include_sparse_pc_in_cond:
            self.sparse_pc_encoder = SparsePCEncoderV3(
                num_points=self.config.sparse_pc_num_points,
                point_dim=self.config.sparse_pc_point_dim,
                hidden_dims=self.config.sparse_pc_hidden_dims,
                embed_dim=self.config.sparse_pc_embed_dim,
            )
            global_cond_dim += self.config.sparse_pc_embed_dim
        else:
            self.sparse_pc_encoder = None

        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionBaselineV3RgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionBaselineV3RgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionBaselineV3ConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _concatenate_state_features(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Concatenate state features (sparse_pc excluded — handled separately)."""
        state_parts = []
        skip_key = self.config.sparse_pc_feature_key
        for key in self.config.state_feature_keys:
            if key in batch and key != skip_key:
                state_parts.append(batch[key])

        if not state_parts:
            return None

        return torch.cat(state_parts, dim=-1)

    # ========= inference ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode features and concatenate into global conditioning vector."""
        batch_size = None
        n_obs_steps = None

        for key in self.config.state_feature_keys:
            if key in batch:
                batch_size = batch[key].shape[0]
                n_obs_steps = batch[key].shape[1]
                break

        if batch_size is None and self.sparse_pc_encoder is not None and OBS_SPARSE_PC in batch:
            batch_size = batch[OBS_SPARSE_PC].shape[0]
            n_obs_steps = batch[OBS_SPARSE_PC].shape[1]
        if batch_size is None and OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES].shape[0]
            n_obs_steps = batch[OBS_IMAGES].shape[1]
        elif batch_size is None and OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            n_obs_steps = batch[OBS_ENV_STATE].shape[1]

        global_cond_feats = []

        composite_state = self._concatenate_state_features(batch)
        if composite_state is not None:
            global_cond_feats.append(composite_state)

        # --- Encode sparse_pc with V3 encoder ---
        if self.sparse_pc_encoder is not None:
            if OBS_SPARSE_PC in batch:
                spc = batch[OBS_SPARSE_PC]  # (B, T, N, 4)
                B_sp, T_sp = spc.shape[:2]
                spc_flat = spc.reshape(B_sp * T_sp, *spc.shape[2:])  # (B*T, N, 4)
                spc_feat = self.sparse_pc_encoder(spc_flat)           # (B*T, embed_dim)
                spc_feat = spc_feat.reshape(B_sp, T_sp, -1)           # (B, T, embed_dim)
            else:
                device = next(self.sparse_pc_encoder.parameters()).device
                spc_feat = torch.zeros(
                    batch_size, n_obs_steps, self.config.sparse_pc_embed_dim,
                    device=device, dtype=torch.float32,
                )
            global_cond_feats.append(spc_feat)

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate actions from observations."""
        batch_size = None
        n_obs_steps = None

        for key in self.config.state_feature_keys:
            if key in batch:
                batch_size = batch[key].shape[0]
                n_obs_steps = batch[key].shape[1]
                break

        if batch_size is None and OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES].shape[0]
            n_obs_steps = batch[OBS_IMAGES].shape[1]
        elif batch_size is None and OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            n_obs_steps = batch[OBS_ENV_STATE].shape[1]

        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute diffusion loss."""
        assert ACTION in batch
        assert "action_is_pad" in batch

        n_obs_steps = None
        for key in self.config.state_feature_keys:
            if key in batch:
                n_obs_steps = batch[key].shape[1]
                break
        if n_obs_steps is None and OBS_IMAGES in batch:
            n_obs_steps = batch[OBS_IMAGES].shape[1]
        elif n_obs_steps is None and OBS_ENV_STATE in batch:
            n_obs_steps = batch[OBS_ENV_STATE].shape[1]

        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


# ============================================================================
# Helper Modules  (shared with V2 — renamed for V3 namespace)
# ============================================================================

class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax — keypoint extraction from 2D feature maps."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class DiffusionBaselineV3RgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector."""

    def __init__(self, config: DiffusionBaselineV3Config):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """Replace submodules matching a predicate with a function."""
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionBaselineV3ConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""

    def __init__(self, config: DiffusionBaselineV3Config, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)

        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


# ============================================================================
# V3 Sparse PC Encoder  (PointNet-style — key improvement over V2)
# ============================================================================

class SparsePCEncoderV3(nn.Module):
    """PointNet-style encoder for sparse point cloud (44×4).

    Key improvements over V2:

    1. **Max-pooling** instead of mean-pooling — preserves the most salient
       features across points rather than averaging them away.  This is
       PointNet's critical insight: max-pooling extracts the strongest
       activations, making the network robust to point ordering.

    2. **Deeper per-point MLP** — (4→64→128) vs (4→32→64) in V2, with
       LayerNorm for stable training.  The extra capacity allows the
       encoder to learn richer point-wise representations.

    3. **Projection head** — 128→128→embed_dim after pooling instead of
       directly projecting from the point-feature space.  This gives the
       encoder a dedicated "global reasoning" stage.

    Architecture::

        Per-point:  Linear(4→64)→LN→ReLU→Linear(64→128)→LN→ReLU
        Global:     Max-Pool over N points → (B, 128)
        Projection: Linear(128→128)→ReLU→Linear(128→embed_dim)

    Input:  (B, N, C)  where N=44, C=4 (xyz + force)
    Output: (B, embed_dim)  default embed_dim=64
    """

    def __init__(
        self,
        num_points: int = 44,
        point_dim: int = 4,
        hidden_dims: tuple[int, ...] = (64, 128),
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.embed_dim = embed_dim

        # Per-point MLP: deeper than V2 with LayerNorm for stability
        layers = []
        in_dim = point_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.point_mlp = nn.Sequential(*layers)
        self.point_feat_dim = hidden_dims[-1]  # e.g., 128

        # Global projection head after max-pooling
        # V2 had: Linear(64→embed_dim) directly
        # V3: 128→128→embed_dim for better global reasoning
        self.proj = nn.Sequential(
            nn.Linear(self.point_feat_dim, self.point_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.point_feat_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode sparse point cloud into a fixed-dim global feature.

        Args:
            x: (B, N, C) point cloud, e.g. (B, 44, 4).

        Returns:
            (B, embed_dim) global point cloud feature.
        """
        B = x.shape[0]
        # Flatten batch and point dims for per-point MLP
        x_flat = x.reshape(B * self.num_points, self.point_dim)  # (B*N, C)
        h = self.point_mlp(x_flat)                                # (B*N, point_feat_dim)
        h = h.reshape(B, self.num_points, self.point_feat_dim)    # (B, N, point_feat_dim)
        # Max-pool over points → selects most salient features (PointNet key idea)
        h, _ = h.max(dim=1)                                       # (B, point_feat_dim)
        # Projection head for richer global representation
        out = self.proj(h)                                        # (B, embed_dim)
        return out
