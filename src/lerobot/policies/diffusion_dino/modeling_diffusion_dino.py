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
"""Diffusion-Dino Policy: Diffusion Policy with DINOv2 vision backbone and tactile sensor support.

Replaces the ResNet RGB encoder from diffusion_hao with a DINOv2 ViT encoder.
Patch tokens are reshaped into a 2D feature map and pooled via SpatialSoftmax
to preserve spatial information for robot manipulation tasks.

Tactile encoders (raw, fused, marker) and the UNet remain identical to diffusion_hao.
"""

import math
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion_dino.configuration_diffusion_dino import DiffusionDinoConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Custom keys for tactile data
OBS_TAC_FUSED = "observation.tac_fused"
OBS_TAC_RAW = "observation.tac_raw"
OBS_TAC_MARKER = "observation.tac_marker"


class DiffusionDinoPolicy(PreTrainedPolicy):
    """Diffusion-Dino Policy with DINOv2 vision backbone and tactile sensor support."""

    config_class = DiffusionDinoConfig
    name = "diffusion_dino"

    def __init__(self, config: DiffusionDinoConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = DiffusionDinoModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.has_tactile_raw:
            self._queues[OBS_TAC_RAW] = deque(maxlen=self.config.n_obs_steps)
        if self.config.has_tactile_fused:
            self._queues[OBS_TAC_FUSED] = deque(maxlen=self.config.n_obs_steps)
        if self.config.has_tactile_marker:
            self._queues[OBS_TAC_MARKER] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self._prepare_batch(batch)
        loss = self.diffusion.compute_loss(batch)
        return loss, None

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare the batch: collect images, synthesize tactile 4-channel, flatten marker."""
        batch = dict(batch)

        depth_keys = list(self.config.tactile_depth_features.keys())
        normal_keys = list(self.config.tactile_normal_features.keys())
        tac_raw_keys = list(self.config.tactile_raw_features.keys())

        # 1. Collect standard image features (excluding tactile keys)
        if self.config.image_features:
            tactile_keys = set(depth_keys + normal_keys + tac_raw_keys)
            rgb_image_keys = [key for key in self.config.image_features if key not in tactile_keys]
            if len(rgb_image_keys) > 0:
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in rgb_image_keys], dim=-4)

        # 2. Collect tactile raw features
        if self.config.has_tactile_raw:
            if len(tac_raw_keys) > 0:
                tac_raw_images = torch.stack([batch[key] for key in tac_raw_keys], dim=-4)
                batch[OBS_TAC_RAW] = tac_raw_images

        # 3. Synthesize tac_depth + tac_normal → 4-channel tactile images
        if self.config.has_tactile_fused:
            if len(depth_keys) > 0 and len(normal_keys) > 0:
                fused_list = []
                for depth_idx in range(len(depth_keys)):
                    depth = batch[depth_keys[depth_idx]]
                    normal = batch[normal_keys[depth_idx]]

                    if depth.dim() >= 3:
                        if depth.shape[-3] in (1, 3):
                            depth = depth.mean(dim=-3, keepdim=True)
                        elif depth.shape[-1] in (1, 3):
                            depth = depth.mean(dim=-1, keepdim=True)

                    if normal.dim() >= 3 and normal.shape[-3] in (1, 3):
                        normal_for_concat = normal.movedim(-3, -1)
                    else:
                        normal_for_concat = normal

                    if depth.dim() >= 3 and depth.shape[-3] == 1:
                        depth_for_concat = depth.movedim(-3, -1)
                    else:
                        depth_for_concat = depth

                    tac_4ch = torch.cat([normal_for_concat, depth_for_concat], dim=-1)

                    if tac_4ch.dim() == 4:
                        tac_4ch = tac_4ch.permute(0, 3, 1, 2)
                    else:
                        tac_4ch = tac_4ch.permute(0, 1, 4, 2, 3)

                    fused_list.append(tac_4ch)

                if len(fused_list) > 1:
                    batch[OBS_TAC_FUSED] = torch.stack(fused_list, dim=-4)
                else:
                    batch[OBS_TAC_FUSED] = fused_list[0]

        # 4. Flatten tac_marker_displacement
        if self.config.has_tactile_marker:
            marker_keys = list(self.config.tactile_marker_features.keys())
            if len(marker_keys) > 0:
                marker_key = marker_keys[0]
                marker = batch[marker_key]
                batch[OBS_TAC_MARKER] = marker.flatten(start_dim=-2)

        return batch


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionDinoModel(nn.Module):
    """Core Diffusion model with DINOv2 vision backbone and tactile sensor support."""

    def __init__(self, config: DiffusionDinoConfig):
        super().__init__()
        self.config = config

        # Separate tactile keys from RGB keys
        self._tactile_depth_keys = list(self.config.tactile_depth_features.keys())
        self._tactile_normal_keys = list(self.config.tactile_normal_features.keys())
        self._tactile_raw_keys = list(self.config.tactile_raw_features.keys())

        tactile_keys = set(self._tactile_depth_keys + self._tactile_normal_keys + self._tactile_raw_keys)
        self._rgb_image_keys = [key for key in self.config.image_features if key not in tactile_keys]

        # Start with robot state dimension
        global_cond_dim = self.config.robot_state_feature.shape[0]

        # === DINOv2 RGB encoder ===
        if len(self._rgb_image_keys) > 0:
            num_images = len(self._rgb_image_keys)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [Dinov2RgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = Dinov2RgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        # === Environment state ===
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # === Tactile raw encoder (ResNet, unchanged from diffusion_hao) ===
        if self.config.has_tactile_raw:
            self.tactile_raw_encoder = TactileRawEncoder(config)
            global_cond_dim += self.tactile_raw_encoder.feature_dim

        # === Tactile fused encoder (ResNet, unchanged from diffusion_hao) ===
        if self.config.has_tactile_fused:
            self.tactile_fused_encoder = TactileFusedEncoder(config)
            global_cond_dim += self.tactile_fused_encoder.feature_dim

        # === Tactile marker encoder ===
        if self.config.has_tactile_marker:
            self.tactile_marker_encoder = TactileMarkerEncoder(
                input_dim=config.tactile_marker_input_dim,
                embed_dim=config.tactile_marker_embed_dim,
            )
            global_cond_dim += config.tactile_marker_embed_dim

        # === UNet ===
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        # === Noise scheduler ===
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
        """Encode all features and concatenate them for global conditioning."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        # === DINOv2 image features ===
        if len(self._rgb_image_keys) > 0:
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

        # === Environment state ===
        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # === Tactile raw features ===
        if self.config.has_tactile_raw:
            tac_raw = batch[OBS_TAC_RAW]
            if tac_raw.dim() == 5:
                tac_raw_features = self.tactile_raw_encoder(
                    einops.rearrange(tac_raw, "b s c h w -> (b s) c h w")
                )
                tac_raw_features = einops.rearrange(
                    tac_raw_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
                )
            else:
                num_raw = tac_raw.shape[2]
                tac_raw_per_camera = einops.rearrange(tac_raw, "b s n c h w -> n (b s) c h w")
                raw_features_list = [self.tactile_raw_encoder(images) for images in tac_raw_per_camera]
                raw_features_stacked = torch.cat(raw_features_list, dim=0)
                tac_raw_features = einops.rearrange(
                    raw_features_stacked, "(n b s) d -> b s (n d)",
                    b=batch_size, s=n_obs_steps, n=num_raw,
                )
            global_cond_feats.append(tac_raw_features)

        # === Tactile fused features ===
        if self.config.has_tactile_fused:
            tac_fused = batch[OBS_TAC_FUSED]
            if tac_fused.dim() == 5:
                tac_fused_flat = einops.rearrange(tac_fused, "b s c h w -> (b s) c h w")
                tac_fused_features = self.tactile_fused_encoder(tac_fused_flat)
                tac_fused_features = einops.rearrange(
                    tac_fused_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
                )
            else:
                num_fused = tac_fused.shape[2]
                tac_fused_per_sensor = einops.rearrange(tac_fused, "b s n c h w -> n (b s) c h w")
                fused_features_list = [
                    self.tactile_fused_encoder(images) for images in tac_fused_per_sensor
                ]
                fused_features_stacked = torch.cat(fused_features_list, dim=0)
                tac_fused_features = einops.rearrange(
                    fused_features_stacked, "(n b s) d -> b s (n d)",
                    b=batch_size, s=n_obs_steps, n=num_fused,
                )
            global_cond_feats.append(tac_fused_features)

        # === Tactile marker features ===
        if self.config.has_tactile_marker:
            tac_marker = batch[OBS_TAC_MARKER]
            tac_marker_features = self.tactile_marker_encoder(tac_marker)
            global_cond_feats.append(tac_marker_features)

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})

        n_obs_steps = batch[OBS_STATE].shape[1]
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


# =============================================================================
# DINOv2 RGB Encoder (replaces DiffusionRgbEncoder from diffusion_hao)
# =============================================================================


class Dinov2RgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector using DINOv2 ViT backbone.

    Patch tokens from DINOv2 are reshaped into a 2D spatial feature map, then
    pooled via SpatialSoftmax to extract spatial keypoint coordinates — preserving
    spatial information critical for robot manipulation tasks.
    """

    def __init__(self, config: DiffusionDinoConfig):
        super().__init__()

        # Crop preprocessing
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Resize preprocessing
        if config.resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.do_resize = False

        # Load DINOv2 backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", config.vision_backbone)
        self.patch_size = self.backbone.patch_size  # 14 for all dinov2 variants
        self.embed_dim = self.backbone.embed_dim

        # Compute grid size from resize_shape (or crop_shape as fallback)
        if config.resize_shape is not None:
            input_h, input_w = config.resize_shape
        elif config.crop_shape is not None:
            input_h, input_w = config.crop_shape
        else:
            # Fall back to first image feature shape
            first_ft = next(iter(config.image_features.values()))
            input_h, input_w = first_ft.shape[1], first_ft.shape[2]

        self.grid_h = input_h // self.patch_size
        self.grid_w = input_w // self.patch_size
        feature_map_shape = (self.embed_dim, self.grid_h, self.grid_w)

        # Optionally freeze the backbone
        self.frozen = config.freeze_vision_backbone
        if self.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # SpatialSoftmax pooling over the 2D feature map
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Crop
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        # Resize
        if self.do_resize:
            x = self.resize(x)

        # Extract DINOv2 patch tokens
        ctx = torch.no_grad() if self.frozen else nullcontext()
        with ctx:
            features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, embed_dim)

        # Reshape to 2D feature map: (B, embed_dim, grid_h, grid_w)
        B = patch_tokens.shape[0]
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, self.grid_h, self.grid_w)

        # Pool and project
        x = torch.flatten(self.pool(feature_map), start_dim=1)
        x = self.relu(self.out(x))
        return x


# =============================================================================
# Tactile encoders (identical to diffusion_hao — ResNet-based)
# =============================================================================


class TactileRawEncoder(nn.Module):
    """Encodes 3-channel tactile raw data into a feature vector using ResNet."""

    def __init__(self, config: DiffusionDinoConfig):
        super().__init__()
        self.config = config

        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.tactile_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        if config.tactile_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.tactile_resize_shape)
        else:
            self.do_resize = False

        backbone_model = getattr(torchvision.models, config.tactile_raw_backbone)(
            weights=config.tactile_raw_pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.tactile_use_group_norm:
            if config.tactile_raw_pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.tactile_raw_features.values())).shape
        if config.tactile_resize_shape is not None:
            final_shape_h_w = config.tactile_resize_shape
        elif config.tactile_crop_shape is not None:
            final_shape_h_w = config.tactile_crop_shape
        else:
            final_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *final_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.tactile_spatial_softmax_num_keypoints)
        self.feature_dim = config.tactile_spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.tactile_spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        if self.do_resize:
            x = self.resize(x)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


class TactileFusedEncoder(nn.Module):
    """Encodes 4-channel tactile data (depth + normal) into a feature vector using ResNet."""

    def __init__(self, config: DiffusionDinoConfig):
        super().__init__()
        self.config = config

        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.tactile_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        if config.tactile_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.tactile_resize_shape)
        else:
            self.do_resize = False

        backbone_model = getattr(torchvision.models, config.tactile_fused_backbone)(
            weights=config.tactile_fused_pretrained_backbone_weights
        )

        original_conv = backbone_model.conv1
        new_conv = nn.Conv2d(
            config.tactile_fused_backbone_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            if config.tactile_fused_pretrained_backbone_weights is not None:
                new_conv.weight[:, :3] = original_conv.weight[:, :3]
                for i in range(3, config.tactile_fused_backbone_in_channels):
                    new_conv.weight[:, i : i + 1] = original_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        backbone_model.conv1 = new_conv

        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.tactile_use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if config.has_tactile_fused:
            depth_features = config.tactile_depth_features
            if len(depth_features) > 0:
                first_depth_ft = next(iter(depth_features.values()))
                h, w = first_depth_ft.shape[0], first_depth_ft.shape[1]
            else:
                h, w = 480, 640
        else:
            h, w = 480, 640

        if config.tactile_resize_shape is not None:
            final_shape_h_w = config.tactile_resize_shape
        elif config.tactile_crop_shape is not None:
            final_shape_h_w = config.tactile_crop_shape
        else:
            final_shape_h_w = (h, w)
        dummy_shape = (1, config.tactile_fused_backbone_in_channels, *final_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.spatial_pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.tactile_spatial_softmax_num_keypoints
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = config.tactile_spatial_softmax_num_keypoints * 2 + feature_map_shape[0]
        self.out = nn.Linear(
            config.tactile_spatial_softmax_num_keypoints * 2 + feature_map_shape[0], self.feature_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        if self.do_resize:
            x = self.resize(x)

        x = self.backbone(x)

        keypoint_coords = torch.flatten(self.spatial_pool(x), start_dim=1)
        scale_features = torch.flatten(self.global_pool(x), start_dim=1)
        x = torch.cat([keypoint_coords, scale_features], dim=-1)
        x = self.relu(self.out(x))
        return x


class TactileMarkerEncoder(nn.Module):
    """Encodes flattened tactile marker displacement into a feature vector."""

    def __init__(self, input_dim: int = 70, embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


# =============================================================================
# Shared components (SpatialSoftmax, UNet, etc.)
# =============================================================================


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation for keypoint extraction."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
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


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""

    def __init__(self, config: DiffusionDinoConfig, global_cond_dim: int):
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


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
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
