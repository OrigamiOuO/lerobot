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
"""Diffusion-Sparsh Policy: Diffusion Policy with tactile sensor support.

Main design:
- Standard RGB images are encoded by the RGB image branch.
- Tactile raw images are encoded either by the original ResNet branch or by a pretrained Sparsh branch.
- Tactile depth and normal streams are explicitly paired by sensor suffix, then
  fused into a 4-channel normal+depth tensor and encoded by a tactile fused branch.
- Marker displacement is flattened and encoded by a lightweight MLP.
- All encoded features build a global condition for a UNet or DiT action denoiser.
- Optional denoiser-level MoE routes between semantic expert denoisers such as
  visual, tactile, and other experts.
"""

import math
import warnings
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

from lerobot.policies.diffusion_sparsh.configuration_diffusion_sparsh import (
    DiffusionSparshConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# =============================================================================
# Constants and feature-key helpers
# =============================================================================

# Custom keys for tactile data
OBS_TAC_FUSED = "observation.tac_fused"  # Synthetic 4-channel key (normal + depth)
OBS_TAC_RAW = "observation.tac_raw"  # Raw tactile image data
OBS_TAC_MARKER = "observation.tac_marker"  # Flattened marker displacement

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
TACTILE_PREFIXES = (
    *TACTILE_RAW_PREFIXES,
    *TACTILE_DEPTH_PREFIXES,
    *TACTILE_NORMAL_PREFIXES,
    *TACTILE_MARKER_PREFIXES,
)


def _is_tactile_feature_key(key: str) -> bool:
    """Return whether a dataset key belongs to a tactile stream."""
    return any(key.startswith(prefix) for prefix in TACTILE_PREFIXES)


def _strip_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    """Remove the first matching prefix and return a stable sensor suffix."""
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key.split(".")[-1]


def _sort_keys_by_suffix(keys: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """Sort feature keys by sensor suffix for deterministic multi-sensor packing."""
    return sorted(keys, key=lambda key: _strip_prefix(key, prefixes))


def _map_keys_by_suffix(keys: list[str], prefixes: tuple[str, ...], stream_name: str) -> dict[str, str]:
    """Map sensor suffix -> key and fail early on duplicate suffixes."""
    key_map: dict[str, str] = {}
    for key in keys:
        suffix = _strip_prefix(key, prefixes)
        if suffix in key_map:
            raise ValueError(
                f"Duplicate {stream_name} tactile suffix `{suffix}` for keys "
                f"`{key_map[suffix]}` and `{key}`. Please use unique sensor suffixes."
            )
        key_map[suffix] = key
    return key_map


def _pair_tactile_depth_normal_keys(depth_keys: list[str], normal_keys: list[str]) -> list[tuple[str, str]]:
    """Pair tactile depth and normal keys by sensor suffix rather than dict order.

    This avoids silent bugs such as pairing depth.tac1 with normal.tac2 when
    multiple tactile sensors are registered in a different order.
    """
    depth_by_suffix = _map_keys_by_suffix(depth_keys, TACTILE_DEPTH_PREFIXES, "depth")
    normal_by_suffix = _map_keys_by_suffix(normal_keys, TACTILE_NORMAL_PREFIXES, "normal")

    depth_suffixes = set(depth_by_suffix)
    normal_suffixes = set(normal_by_suffix)
    if depth_suffixes != normal_suffixes:
        missing_normal = sorted(depth_suffixes - normal_suffixes)
        missing_depth = sorted(normal_suffixes - depth_suffixes)
        raise ValueError(
            "Tactile depth/normal keys must be paired by sensor suffix. "
            f"Missing normal for {missing_normal}; missing depth for {missing_depth}."
        )

    return [(depth_by_suffix[suffix], normal_by_suffix[suffix]) for suffix in sorted(depth_suffixes)]


def _infer_image_channels_hw(shape: tuple[int, ...] | torch.Size) -> tuple[int, int, int]:
    """Infer (channels, height, width) from a PolicyFeature image shape.

    LeRobot image features are usually channel-first, but some tactile features
    can be channel-last when produced by custom preprocessing.
    """
    shape = tuple(shape)
    if len(shape) != 3:
        raise ValueError(f"Expected an image feature shape with 3 dimensions. Got {shape}.")

    if shape[0] in (1, 3, 4):
        return shape[0], shape[1], shape[2]
    if shape[-1] in (1, 3, 4):
        return shape[-1], shape[0], shape[1]

    # Fallback to channel-first, matching torchvision conventions.
    return shape[0], shape[1], shape[2]


def _image_to_channel_last(x: Tensor, expected_channels: tuple[int, ...], name: str) -> Tensor:
    """Convert (..., C, H, W) or (..., H, W, C) image tensors to channel-last."""
    if x.dim() < 3:
        raise ValueError(f"`{name}` must have at least 3 dimensions. Got shape {tuple(x.shape)}.")

    if x.shape[-1] in expected_channels:
        return x
    if x.shape[-3] in expected_channels:
        return x.movedim(-3, -1)

    raise ValueError(
        f"Cannot infer channel dimension for `{name}` with shape {tuple(x.shape)}. "
        f"Expected channel count in {expected_channels}."
    )


def _depth_to_channel_last_one_channel(depth: Tensor, key: str) -> Tensor:
    """Convert depth tensor to channel-last with one channel.

    Some video backends decode a single-channel depth image as 3 channels; in
    that case the channels are averaged back to one channel.
    """
    depth = _image_to_channel_last(depth, expected_channels=(1, 3), name=key)
    if depth.shape[-1] == 3:
        depth = depth.mean(dim=-1, keepdim=True)
    if depth.shape[-1] != 1:
        raise ValueError(f"Depth key `{key}` should be 1-channel after conversion. Got {tuple(depth.shape)}.")
    return depth


def _normal_to_channel_last_three_channels(normal: Tensor, key: str) -> Tensor:
    """Convert normal tensor to channel-last with three channels."""
    normal = _image_to_channel_last(normal, expected_channels=(3,), name=key)
    if normal.shape[-1] != 3:
        raise ValueError(f"Normal key `{key}` should be 3-channel after conversion. Got {tuple(normal.shape)}.")
    return normal


def _normal_depth_to_channel_first_4ch(normal: Tensor, depth: Tensor) -> Tensor:
    """Concatenate channel-last normal and depth tensors, then return channel-first."""
    tactile_4ch = torch.cat([normal, depth], dim=-1)  # (..., H, W, 4)
    return tactile_4ch.movedim(-1, -3)  # (..., 4, H, W)


def _make_mlp_projection(in_dim: int, out_dim: int) -> nn.Module:
    """Small projection used when a config adds `modality_projection_dim`."""
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))


# =============================================================================
# Policy wrapper and batch packing
# =============================================================================


class DiffusionSparshPolicy(PreTrainedPolicy):
    """Diffusion-Sparsh Policy with tactile sensor support.

    The policy wrapper is responsible for rollout queues and for converting raw
    dataset keys into the compact internal keys consumed by `DiffusionSparshModel`.
    """

    config_class = DiffusionSparshConfig
    name = "diffusion_sparsh"

    def __init__(
        self,
        config: DiffusionSparshConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        self._validate_features_for_diffusion_sparsh(config)
        self.config = config

        # queues are populated during rollout of the policy
        self._queues = None

        self.diffusion = DiffusionSparshModel(config)

        self.reset()

    @staticmethod
    def _validate_features_for_diffusion_sparsh(config: DiffusionSparshConfig) -> None:
        """Validate input features without mixing tactile image keys with RGB keys.

        `DiffusionSparshConfig.validate_features()` assumes all image features have
        the same shape. That is too strict for this policy because tactile depth,
        tactile normal, tactile raw and RGB can all be stored as image-like
        features while being processed by different branches.
        """
        depth_keys = list(config.tactile_depth_features.keys())
        normal_keys = list(config.tactile_normal_features.keys())
        if len(depth_keys) != len(normal_keys):
            raise ValueError(
                "Tactile depth/normal feature count mismatch: "
                f"got {len(depth_keys)} depth streams vs {len(normal_keys)} normal streams. "
                "Please provide paired depth+normal keys per tactile sensor."
            )
        if len(depth_keys) > 0:
            _pair_tactile_depth_normal_keys(depth_keys, normal_keys)

        rgb_image_features = config.enabled_rgb_image_features

        if config.crop_shape is not None:
            for key, image_ft in rgb_image_features.items():
                _, height, width = _infer_image_channels_hw(image_ft.shape)
                if config.crop_shape[0] > height or config.crop_shape[1] > width:
                    raise ValueError(
                        f"`crop_shape` should fit within RGB image shapes. Got {config.crop_shape} "
                        f"for `{key}` with shape {image_ft.shape}."
                    )

        if len(rgb_image_features) > 0:
            first_key, first_ft = next(iter(rgb_image_features.items()))
            first_shape = first_ft.shape
            for key, image_ft in rgb_image_features.items():
                if image_ft.shape != first_shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_key}`, but standard RGB image "
                        "features are expected to have the same shape. Tactile image-like "
                        "features are excluded from this check."
                    )

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.enabled_rgb_image_features:
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
        """Predict a chunk of actions given environment observations."""
        batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out.
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
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self._prepare_batch(batch)
        loss = self.diffusion.compute_loss(batch)
        return loss, None

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pack raw dataset keys into internal observation keys.

        The main correctness change is tactile depth/normal pairing: we pair by
        sensor suffix, not by Python dict order.
        """
        batch = dict(batch)  # shallow copy

        depth_keys = _sort_keys_by_suffix(list(self.config.tactile_depth_features.keys()), TACTILE_DEPTH_PREFIXES)
        normal_keys = _sort_keys_by_suffix(list(self.config.tactile_normal_features.keys()), TACTILE_NORMAL_PREFIXES)
        tac_raw_keys = _sort_keys_by_suffix(list(self.config.tactile_raw_features.keys()), TACTILE_RAW_PREFIXES)
        depth_normal_pairs = _pair_tactile_depth_normal_keys(depth_keys, normal_keys) if depth_keys else []

        # 1. Standard RGB image features only. Tactile image-like keys are routed
        #    to their own branches below.
        rgb_image_keys = list(self.config.enabled_rgb_image_features.keys())
        if len(rgb_image_keys) > 0:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in rgb_image_keys], dim=-4)

        # 2. Tactile raw images.
        if self.config.has_tactile_raw and len(tac_raw_keys) > 0:
            if len(tac_raw_keys) == 1:
                batch[OBS_TAC_RAW] = batch[tac_raw_keys[0]]  # (B, S, C, H, W)
            else:
                batch[OBS_TAC_RAW] = torch.stack([batch[key] for key in tac_raw_keys], dim=-4)

        # 3. Tactile fused normal+depth images. The channel order is [normal, depth].
        if self.config.has_tactile_fused and len(depth_normal_pairs) > 0:
            fused_list = []
            for depth_key, normal_key in depth_normal_pairs:
                depth = _depth_to_channel_last_one_channel(batch[depth_key], depth_key)
                normal = _normal_to_channel_last_three_channels(batch[normal_key], normal_key)
                fused_list.append(_normal_depth_to_channel_first_4ch(normal, depth))

            if len(fused_list) == 1:
                batch[OBS_TAC_FUSED] = fused_list[0]  # (B, S, 4, H, W) or (B, 4, H, W)
            else:
                batch[OBS_TAC_FUSED] = torch.stack(fused_list, dim=-4)  # (B, S, N, 4, H, W)

        # 4. Marker displacement: (B, S, 35, 2) -> (B, S, 70), or (B, 35, 2) -> (B, 70).
        if self.config.has_tactile_marker:
            marker_keys = list(self.config.tactile_marker_features.keys())
            if len(marker_keys) > 0:
                if len(marker_keys) > 1:
                    raise ValueError(
                        "Multiple tactile marker streams are not yet fused explicitly. "
                        f"Got marker keys: {marker_keys}."
                    )
                marker = batch[marker_keys[0]]
                batch[OBS_TAC_MARKER] = marker.flatten(start_dim=-2)

        return batch


# =============================================================================
# Scheduler and feature-level MoE
# =============================================================================

def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """Factory for noise scheduler instances."""
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class MultiModalConsensusMoE(nn.Module):
    """Multi-modal consensus MoE block.

    This module projects modality features into a shared latent space, applies
    expert routing per modality, then computes a weighted consensus across
    modalities for each observation step.
    """

    def __init__(
        self,
        modality_input_dims: dict[str, int],
        hidden_dim: int,
        num_experts: int,
        dropout: float,
        routing_dropout: float,
        topk: int,
        debug_inference: bool = False,
        debug_every_n_calls: int = 1,
        debug_print_full_weights: bool = True,
    ):
        super().__init__()
        self.modality_names = tuple(modality_input_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.routing_dropout = routing_dropout
        self.topk = topk
        self.debug_inference = debug_inference
        self.debug_every_n_calls = debug_every_n_calls
        self.debug_print_full_weights = debug_print_full_weights
        self._inference_call_count = 0
        self.last_inference_routing: dict[str, list[float] | list[str] | int] | None = None

        self.modality_projectors = nn.ModuleDict(
            {name: nn.Linear(in_dim, hidden_dim) for name, in_dim in modality_input_dims.items()}
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.modality_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _dropout_and_renorm(weights: Tensor, p: float) -> Tensor:
        """Apply dropout to mixture weights and renormalize along the last dim."""
        if p <= 0.0:
            return weights

        keep_mask = (torch.rand_like(weights) > p).to(weights.dtype)
        dropped = weights * keep_mask
        denom = dropped.sum(dim=-1, keepdim=True)
        fallback = 1.0 / weights.shape[-1]
        renorm = torch.where(denom > 0, dropped / denom.clamp_min(1e-12), torch.full_like(weights, fallback))
        return renorm

    def forward(self, modality_features: dict[str, Tensor]) -> Tensor:
        """Compute consensus feature.

        Args:
            modality_features: maps modality name -> tensor of shape (B, S, D_i).

        Returns:
            Consensus feature tensor of shape (B, S, hidden_dim).
        """
        expert_outputs = []
        modality_logits = []

        for name in self.modality_names:
            x = self.modality_projectors[name](modality_features[name])  # (B, S, H)

            routing_logits = self.router(x)  # (B, S, E)
            routing_weights = F.softmax(routing_logits, dim=-1)
            if self.training and self.routing_dropout > 0.0:
                routing_weights = self._dropout_and_renorm(routing_weights, self.routing_dropout)

            stacked_expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (B, S, E, H)
            mixed = (routing_weights.unsqueeze(-1) * stacked_expert_outputs).sum(dim=2)  # (B, S, H)

            expert_outputs.append(mixed)
            modality_logits.append(self.modality_score(mixed))

        modality_stack = torch.stack(expert_outputs, dim=2)  # (B, S, M, H)
        modality_logits = torch.cat(modality_logits, dim=2)  # (B, S, M)

        k = min(self.topk, modality_logits.shape[-1])
        if k < modality_logits.shape[-1]:
            topk_vals, topk_idx = torch.topk(modality_logits, k=k, dim=-1)
            masked_logits = torch.full_like(modality_logits, float("-inf"))
            masked_logits.scatter_(-1, topk_idx, topk_vals)
            modality_weights = F.softmax(masked_logits, dim=-1)
        else:
            modality_weights = F.softmax(modality_logits, dim=-1)

        if self.training and self.routing_dropout > 0.0:
            modality_weights = self._dropout_and_renorm(modality_weights, self.routing_dropout)

        if (not self.training) and self.debug_inference:
            self._inference_call_count += 1
            mean_weights = modality_weights.detach().mean(dim=(0, 1))
            report_k = min(self.topk, mean_weights.shape[0])
            topk_weights, topk_idx = torch.topk(mean_weights, k=report_k, dim=0)

            topk_modalities = [self.modality_names[i] for i in topk_idx.tolist()]
            topk_weights_list = topk_weights.cpu().tolist()
            all_modalities = list(self.modality_names)
            all_weights = mean_weights.cpu().tolist()

            self.last_inference_routing = {
                "call": self._inference_call_count,
                "topk_modalities": topk_modalities,
                "topk_weights": topk_weights_list,
                "all_modalities": all_modalities,
                "all_weights": all_weights,
            }

            if self._inference_call_count % self.debug_every_n_calls == 0:
                topk_pairs = ", ".join(
                    [f"{m}:{w:.4f}" for m, w in zip(topk_modalities, topk_weights_list, strict=True)]
                )
                if self.debug_print_full_weights:
                    all_pairs = ", ".join(
                        [f"{m}:{w:.4f}" for m, w in zip(all_modalities, all_weights, strict=True)]
                    )
                    print(
                        f"[modal_moe] call={self._inference_call_count} topk={topk_pairs} all={all_pairs}"
                    )
                else:
                    print(f"[modal_moe] call={self._inference_call_count} topk={topk_pairs}")

        consensus = (modality_weights.unsqueeze(-1) * modality_stack).sum(dim=2)  # (B, S, H)
        return self.out_norm(consensus)


class DiffusionSparshModel(nn.Module):
    """Core diffusion model with tactile sensor support.

    This class builds three conceptual parts:
    1. observation encoders;
    2. optional modality-level consensus MoE;
    3. either a single denoiser or a semantic denoiser-level MoE.
    """

    def __init__(self, config: DiffusionSparshConfig):
        super().__init__()
        self.config = config
        self.use_denoiser_moe = config.use_denoiser_moe
        self.denoiser_composition_strategy = config.denoiser_composition_strategy
        self.denoiser_num_modules = config.denoiser_num_modules
        self.denoiser_topk = config.denoiser_topk
        self.denoiser_grouping_strategy = config.denoiser_grouping_strategy
        self.denoiser_moe_debug_inference = config.denoiser_moe_debug_inference
        self.denoiser_moe_debug_every_n_calls = config.denoiser_moe_debug_every_n_calls
        self.denoiser_moe_debug_print_full_weights = config.denoiser_moe_debug_print_full_weights
        self._denoiser_inference_call_count = 0
        self.last_denoiser_moe_routing: dict[str, float | list[float] | list[str] | int | str] | None = None

        # Optional modality projection. Existing configs do not need this field;
        # if you add `modality_projection_dim=128` to the config, every modality
        # except the robot state can be normalized/projected before concatenation.
        self.modality_projection_dim: int | None = getattr(config, "modality_projection_dim", None)
        self.project_state: bool = bool(getattr(config, "project_state_condition", False))
        self.modality_projectors = nn.ModuleDict()

        # Keep tactile raw/depth/normal out of the RGB branch when they are stored
        # as video/image features.
        self._tactile_depth_keys = _sort_keys_by_suffix(
            list(self.config.tactile_depth_features.keys()), TACTILE_DEPTH_PREFIXES
        )
        self._tactile_normal_keys = _sort_keys_by_suffix(
            list(self.config.tactile_normal_features.keys()), TACTILE_NORMAL_PREFIXES
        )
        self._tactile_raw_keys = _sort_keys_by_suffix(
            list(self.config.tactile_raw_features.keys()), TACTILE_RAW_PREFIXES
        )
        self._tactile_fused_pairs = _pair_tactile_depth_normal_keys(
            self._tactile_depth_keys, self._tactile_normal_keys
        ) if len(self._tactile_depth_keys) > 0 else []

        self._rgb_image_keys = list(self.config.enabled_rgb_image_features.keys())

        # ------------------------------------------------------------------
        # Observation encoders and modality dimensions.
        # ------------------------------------------------------------------
        raw_modality_dims: dict[str, int] = {"state": self.config.robot_state_feature.shape[0]}
        self._modality_dims: dict[str, int] = {}
        global_cond_dim = 0

        def register_modality(name: str, raw_dim: int, *, allow_projection: bool = True) -> int:
            """Register a modality and optionally create a projection head."""
            should_project = (
                self.modality_projection_dim is not None
                and allow_projection
                and (name != "state" or self.project_state)
            )
            out_dim = self.modality_projection_dim if should_project else raw_dim
            if should_project:
                self.modality_projectors[name] = _make_mlp_projection(raw_dim, out_dim)
            raw_modality_dims[name] = raw_dim
            self._modality_dims[name] = out_dim
            return out_dim

        global_cond_dim += register_modality("state", self.config.robot_state_feature.shape[0], allow_projection=True)

        if len(self._rgb_image_keys) > 0:
            if self.config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [DiffusionRgbEncoder(config, image_shape=self.config.enabled_rgb_image_features[key].shape) for key in self._rgb_image_keys]
                )
                rgb_raw_dim = self.rgb_encoder[0].feature_dim * len(self._rgb_image_keys)
            else:
                first_shape = self.config.enabled_rgb_image_features[self._rgb_image_keys[0]].shape
                self.rgb_encoder = DiffusionRgbEncoder(config, image_shape=first_shape)
                rgb_raw_dim = self.rgb_encoder.feature_dim * len(self._rgb_image_keys)
            global_cond_dim += register_modality("rgb", rgb_raw_dim)

        if self.config.env_state_feature:
            global_cond_dim += register_modality("env_state", self.config.env_state_feature.shape[0])

        if self.config.has_tactile_raw and len(self._tactile_raw_keys) > 0:
            first_raw_shape = self.config.tactile_raw_features[self._tactile_raw_keys[0]].shape
            if config.tactile_raw_encoder_type == "sparsh":
                self.tactile_raw_encoder = SparshTactileRawEncoder(config, image_shape=first_raw_shape)
            elif config.tactile_raw_encoder_type == "resnet":
                self.tactile_raw_encoder = TactileRawEncoder(config, image_shape=first_raw_shape)
            else:
                raise ValueError(f"Unknown tactile_raw_encoder_type: {config.tactile_raw_encoder_type}")
            tactile_raw_dim = self.tactile_raw_encoder.feature_dim * len(self._tactile_raw_keys)
            global_cond_dim += register_modality("tactile_raw", tactile_raw_dim)

        if self.config.has_tactile_fused and len(self._tactile_fused_pairs) > 0:
            first_depth_key, _ = self._tactile_fused_pairs[0]
            first_depth_shape = self.config.tactile_depth_features[first_depth_key].shape
            self.tactile_fused_encoder = TactileFusedEncoder(config, depth_image_shape=first_depth_shape)
            tactile_fused_dim = self.tactile_fused_encoder.feature_dim * len(self._tactile_fused_pairs)
            global_cond_dim += register_modality("tactile_fused", tactile_fused_dim)

        if self.config.has_tactile_marker:
            self.tactile_marker_encoder = TactileMarkerEncoder(
                input_dim=config.tactile_marker_input_dim,
                embed_dim=config.tactile_marker_embed_dim,
            )
            global_cond_dim += register_modality("tactile_marker", config.tactile_marker_embed_dim)

        # Optional feature-level consensus MoE. It receives projected modality
        # features, so its input dims are exactly `self._modality_dims`.
        if config.use_modal_moe:
            self.modal_moe = MultiModalConsensusMoE(
                modality_input_dims=self._modality_dims,
                hidden_dim=config.moe_hidden_dim,
                num_experts=config.moe_num_experts,
                dropout=config.moe_dropout,
                routing_dropout=config.moe_routing_dropout,
                topk=config.moe_topk,
                debug_inference=config.modal_moe_debug_inference,
                debug_every_n_calls=config.modal_moe_debug_every_n_calls,
                debug_print_full_weights=config.modal_moe_debug_print_full_weights,
            )
            global_cond_dim += config.moe_hidden_dim
        else:
            self.modal_moe = None

        # ------------------------------------------------------------------
        # Denoiser backbone: single denoiser or semantic denoiser MoE.
        # ------------------------------------------------------------------
        global_cond_total_dim = global_cond_dim * config.n_obs_steps
        if config.denoiser_type == "unet":
            denoiser_ctor = lambda cond_dim: DiffusionConditionalUnet1d(config, global_cond_dim=cond_dim)
        elif config.denoiser_type == "dit":
            denoiser_ctor = lambda cond_dim: DiffusionConditionalDiT1d(config, global_cond_dim=cond_dim)
        else:
            raise ValueError(f"Unsupported denoiser type {config.denoiser_type}")

        if self.use_denoiser_moe:
            self._expert_groups = self._build_denoiser_expert_groups()
            if len(self._expert_groups) == 0:
                raise ValueError("`use_denoiser_moe=True` requires at least one non-state modality feature.")

            self._expert_group_names = list(self._expert_groups.keys())
            self._expert_cond_dims = {
                group_name: (
                    self._modality_dims["state"]
                    + sum(self._modality_dims[name] for name in modality_names)
                )
                * config.n_obs_steps
                for group_name, modality_names in self._expert_groups.items()
            }

            self.denoiser_experts = nn.ModuleList()
            self._denoiser_expert_names: list[str] = []
            self._expert_idx_to_group_idx: list[int] = []
            for group_idx, group_name in enumerate(self._expert_group_names):
                for module_idx in range(self.denoiser_num_modules):
                    self.denoiser_experts.append(denoiser_ctor(self._expert_cond_dims[group_name]))
                    if self.denoiser_num_modules == 1:
                        self._denoiser_expert_names.append(group_name)
                    else:
                        self._denoiser_expert_names.append(f"{group_name}_m{module_idx}")
                    self._expert_idx_to_group_idx.append(group_idx)

            self.weight_predictor = nn.Sequential(
                nn.LayerNorm(global_cond_total_dim),
                nn.Linear(global_cond_total_dim, global_cond_total_dim),
                nn.GELU(),
                nn.Linear(global_cond_total_dim, len(self.denoiser_experts)),
            )

            # Compatibility handle; MoE path does not call this directly.
            self.unet = self.denoiser_experts
        else:
            self.unet = denoiser_ctor(global_cond_total_dim)

        # ------------------------------------------------------------------
        # Noise scheduler.
        # ------------------------------------------------------------------
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
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def _project_modality(self, name: str, features: Tensor) -> Tensor:
        """Apply optional per-modality projection."""
        if name in self.modality_projectors:
            return self.modality_projectors[name](features)
        return features

    def _build_denoiser_expert_groups(self) -> dict[str, list[str]]:
        """Build denoiser expert groups from available non-state modalities.

        - `modality`: one group per modality.
        - `semantic`: three interpretable groups: visual, tactile, and other.
        """
        non_state_modalities = [name for name in self._modality_dims.keys() if name != "state"]
        if self.denoiser_grouping_strategy == "modality":
            return {name: [name] for name in non_state_modalities}

        if self.denoiser_grouping_strategy == "semantic":
            visual = [name for name in non_state_modalities if name == "rgb"]
            tactile = [name for name in non_state_modalities if name.startswith("tactile")]
            other = [name for name in non_state_modalities if name not in visual and name not in tactile]

            groups: dict[str, list[str]] = {}
            if len(visual) > 0:
                groups["visual"] = visual
            if len(tactile) > 0:
                groups["tactile"] = tactile
            if len(other) > 0:
                groups["other"] = other
            return groups

        raise ValueError(f"Unknown denoiser grouping strategy {self.denoiser_grouping_strategy}")

    def get_modal_moe_debug_info(self) -> dict[str, list[float] | list[str] | int] | None:
        """Return latest modal MoE routing summary captured during inference."""
        if self.modal_moe is None:
            return None
        return self.modal_moe.last_inference_routing

    def get_denoiser_moe_debug_info(self) -> dict[str, float | list[float] | list[str] | int | str] | None:
        """Return latest denoiser MoE routing summary captured during inference."""
        return self.last_denoiser_moe_routing

    def _get_denoiser_composition_strategy(self) -> str:
        """Resolve denoiser MoE strategy by phase.

        Training uses soft-gating for dense gradients. Inference uses the configured
        strategy. If you want no train/inference mismatch, set the config strategy
        to `soft_gating` as well.
        """
        if self.training:
            return "soft_gating"
        return self.denoiser_composition_strategy

    def _effective_denoiser_weights(self, weights: Tensor, strategy: str) -> Tensor:
        """Convert router logits to effective expert weights of shape (E, B)."""
        num_experts = weights.shape[0]
        if strategy == "soft_gating":
            return F.softmax(weights, dim=0)

        if strategy == "hard_routing":
            idx = torch.argmax(weights, dim=0)
            return F.one_hot(idx, num_classes=num_experts).to(weights.dtype).transpose(0, 1)

        if strategy == "topk_moe":
            k = min(self.denoiser_topk, num_experts)
            topk_vals, topk_idx = torch.topk(weights, k=k, dim=0)
            topk_weights = F.softmax(topk_vals, dim=0)
            effective = torch.zeros_like(weights)
            effective.scatter_(0, topk_idx, topk_weights)
            return effective

        raise ValueError(f"Unknown denoiser composition strategy {strategy}")

    def _maybe_log_denoiser_moe_weights(self, weights: Tensor) -> None:
        """Capture and optionally print denoiser MoE routing diagnostics."""
        if (not self.denoiser_moe_debug_inference) or self.training:
            return

        self._denoiser_inference_call_count += 1
        strategy = self._get_denoiser_composition_strategy()
        effective_weights = self._effective_denoiser_weights(weights, strategy)

        mean_weights = effective_weights.detach().mean(dim=1)
        entropy = -(effective_weights.detach().clamp_min(1e-12) * effective_weights.detach().clamp_min(1e-12).log()).sum(dim=0).mean()
        top1_idx = effective_weights.detach().argmax(dim=0)
        top1_ratio = torch.bincount(top1_idx, minlength=effective_weights.shape[0]).float() / top1_idx.numel()

        report_k = min(self.denoiser_topk, mean_weights.shape[0])
        topk_weights, topk_idx = torch.topk(mean_weights, k=report_k, dim=0)

        all_experts = list(self._denoiser_expert_names)
        all_weights = mean_weights.cpu().tolist()
        topk_experts = [all_experts[i] for i in topk_idx.tolist()]
        topk_weights_list = topk_weights.cpu().tolist()

        self.last_denoiser_moe_routing = {
            "call": self._denoiser_inference_call_count,
            "strategy": strategy,
            "topk_experts": topk_experts,
            "topk_weights": topk_weights_list,
            "all_experts": all_experts,
            "all_weights": all_weights,
            "routing_entropy": float(entropy.cpu()),
            "top1_expert_ratio": top1_ratio.cpu().tolist(),
        }

        if self._denoiser_inference_call_count % self.denoiser_moe_debug_every_n_calls == 0:
            topk_pairs = ", ".join(
                [f"{name}:{w:.4f}" for name, w in zip(topk_experts, topk_weights_list, strict=True)]
            )
            msg = (
                "[denoiser_moe] "
                f"call={self._denoiser_inference_call_count} strategy={strategy} "
                f"entropy={float(entropy.cpu()):.4f} topk={topk_pairs}"
            )
            if self.denoiser_moe_debug_print_full_weights:
                all_pairs = ", ".join([f"{name}:{w:.4f}" for name, w in zip(all_experts, all_weights, strict=True)])
                msg += f" all={all_pairs}"
            print(msg)

    def _expert_cond_for_expert(self, expert_idx: int, expert_conds: list[Tensor]) -> Tensor:
        group_idx = self._expert_idx_to_group_idx[expert_idx]
        return expert_conds[group_idx]

    def _predict_with_denoiser_moe(
        self,
        trajectory: Tensor,
        timestep: Tensor,
        expert_conds: list[Tensor],
        weights: Tensor,
    ) -> Tensor:
        """Predict denoising output with denoiser MoE composition.

        This version avoids the old per-sample Python loop for hard/top-k routing.
        It groups samples by expert, so the loop is over experts rather than over
        every batch element.
        """
        strategy = self._get_denoiser_composition_strategy()
        effective_weights = self._effective_denoiser_weights(weights, strategy)
        pred = torch.zeros_like(trajectory)

        for expert_idx, expert in enumerate(self.denoiser_experts):
            sample_mask = effective_weights[expert_idx] > 0
            if not torch.any(sample_mask):
                continue

            cond = self._expert_cond_for_expert(expert_idx, expert_conds)
            expert_pred = expert(
                trajectory[sample_mask],
                timestep[sample_mask],
                global_cond=cond[sample_mask],
            )
            pred[sample_mask] += effective_weights[expert_idx, sample_mask][:, None, None] * expert_pred

        return pred

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        expert_conds: list[Tensor] | None = None,
        weights: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample from the diffusion model."""
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
            timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
            if self.use_denoiser_moe:
                assert expert_conds is not None and weights is not None
                model_output = self._predict_with_denoiser_moe(
                    sample,
                    timestep=timestep,
                    expert_conds=expert_conds,
                    weights=weights,
                )
            else:
                model_output = self.unet(sample, timestep, global_cond=global_cond)
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(
        self,
        batch: dict[str, Tensor],
        return_modality_conds: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Encode observations and build global/expert conditions."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        state_features = self._project_modality("state", batch[OBS_STATE])
        global_cond_feats = [state_features]
        modality_features: dict[str, Tensor] = {"state": state_features}

        if len(self._rgb_image_keys) > 0:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = [
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                img_features_cat = torch.cat(img_features_list, dim=0)
                img_features = einops.rearrange(
                    img_features_cat, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=len(self._rgb_image_keys)
                )
            else:
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(
                    img_features, "(b s n) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=len(self._rgb_image_keys)
                )
            img_features = self._project_modality("rgb", img_features)
            global_cond_feats.append(img_features)
            modality_features["rgb"] = img_features

        if self.config.env_state_feature:
            env_features = self._project_modality("env_state", batch[OBS_ENV_STATE])
            global_cond_feats.append(env_features)
            modality_features["env_state"] = env_features

        if self.config.has_tactile_raw and OBS_TAC_RAW in batch:
            tac_raw = batch[OBS_TAC_RAW]
            if isinstance(self.tactile_raw_encoder, SparshTactileRawEncoder):
                # Sparsh needs temporal pairs, so the encoder receives the full
                # observation sequence: (B,S,3,H,W) or (B,S,N,3,H,W).
                tac_raw_features = self.tactile_raw_encoder(tac_raw)
            else:
                if tac_raw.dim() == 5:
                    tac_raw_features = self.tactile_raw_encoder(einops.rearrange(tac_raw, "b s c h w -> (b s) c h w"))
                    tac_raw_features = einops.rearrange(tac_raw_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)
                else:
                    num_raw = tac_raw.shape[2]
                    tac_raw_per_camera = einops.rearrange(tac_raw, "b s n c h w -> n (b s) c h w")
                    raw_features_list = [self.tactile_raw_encoder(images) for images in tac_raw_per_camera]
                    raw_features_cat = torch.cat(raw_features_list, dim=0)
                    tac_raw_features = einops.rearrange(
                        raw_features_cat, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=num_raw
                    )
            tac_raw_features = self._project_modality("tactile_raw", tac_raw_features)
            global_cond_feats.append(tac_raw_features)
            modality_features["tactile_raw"] = tac_raw_features

        if self.config.has_tactile_fused and OBS_TAC_FUSED in batch:
            tac_fused = batch[OBS_TAC_FUSED]
            if tac_fused.dim() == 5:
                tac_fused_flat = einops.rearrange(tac_fused, "b s c h w -> (b s) c h w")
                tac_fused_features = self.tactile_fused_encoder(tac_fused_flat)
                tac_fused_features = einops.rearrange(tac_fused_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)
            else:
                num_fused = tac_fused.shape[2]
                tac_fused_per_sensor = einops.rearrange(tac_fused, "b s n c h w -> n (b s) c h w")
                fused_features_list = [self.tactile_fused_encoder(images) for images in tac_fused_per_sensor]
                fused_features_cat = torch.cat(fused_features_list, dim=0)
                tac_fused_features = einops.rearrange(
                    fused_features_cat, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=num_fused
                )
            tac_fused_features = self._project_modality("tactile_fused", tac_fused_features)
            global_cond_feats.append(tac_fused_features)
            modality_features["tactile_fused"] = tac_fused_features

        if self.config.has_tactile_marker and OBS_TAC_MARKER in batch:
            tac_marker_features = self.tactile_marker_encoder(batch[OBS_TAC_MARKER])
            tac_marker_features = self._project_modality("tactile_marker", tac_marker_features)
            global_cond_feats.append(tac_marker_features)
            modality_features["tactile_marker"] = tac_marker_features

        if self.modal_moe is not None:
            consensus_features = self.modal_moe(modality_features)
            global_cond_feats.append(consensus_features)

        global_cond = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

        if not return_modality_conds:
            return global_cond
        if not self.use_denoiser_moe:
            return global_cond, []

        expert_conds = [
            torch.cat(
                [torch.cat([modality_features[name] for name in names], dim=-1), modality_features["state"]],
                dim=-1,
            ).flatten(start_dim=1)
            for names in self._expert_groups.values()
        ]
        return global_cond, expert_conds

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate actions given observations."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        if self.use_denoiser_moe:
            global_cond, expert_conds = self._prepare_global_conditioning(batch, return_modality_conds=True)
            weights = self.weight_predictor(global_cond).transpose(0, 1)
            self._maybe_log_denoiser_moe_weights(weights)
            actions = self.conditional_sample(
                batch_size,
                global_cond=global_cond,
                expert_conds=expert_conds,
                weights=weights,
                noise=noise,
            )
        else:
            global_cond = self._prepare_global_conditioning(batch)
            actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute training loss."""
        assert set(batch).issuperset({OBS_STATE, ACTION})
        if self.config.do_mask_loss_for_padding:
            assert "action_is_pad" in batch

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        if self.use_denoiser_moe:
            global_cond, expert_conds = self._prepare_global_conditioning(batch, return_modality_conds=True)
            weights = self.weight_predictor(global_cond).transpose(0, 1)
        else:
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

        if self.use_denoiser_moe:
            pred = self._predict_with_denoiser_moe(
                noisy_trajectory,
                timestep=timesteps,
                expert_conds=expert_conds,
                weights=weights,
            )
        else:
            pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


# =============================================================================
# Observation encoders
# =============================================================================
class TactileRawEncoder(nn.Module):
    """Encode 3-channel tactile raw data into a feature vector."""

    def __init__(self, config: DiffusionSparshConfig, image_shape: tuple[int, ...] | torch.Size | None = None):
        super().__init__()
        self.config = config

        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.tactile_crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
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
                raise ValueError("Cannot replace BatchNorm in a pretrained tactile raw backbone.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if image_shape is None:
            image_shape = next(iter(config.tactile_raw_features.values())).shape
        channels, height, width = _infer_image_channels_hw(image_shape)
        if channels != config.tactile_raw_backbone_in_channels:
            raise ValueError(
                f"Tactile raw encoder expected {config.tactile_raw_backbone_in_channels} channels, "
                f"but got feature shape {tuple(image_shape)}."
            )

        final_shape_h_w = config.tactile_resize_shape or config.tactile_crop_shape or (height, width)
        dummy_shape = (1, channels, *final_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        num_kp = config.tactile_spatial_softmax_num_keypoints
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=num_kp)
        self.feature_dim = num_kp * 2
        self.out = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        if self.do_resize:
            x = self.resize(x)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.out(x)


class SparshTactileRawEncoder(nn.Module):
    """Encode tactile raw RGB with the official Sparsh ViT backbone.

    This is a drop-in replacement for `TactileRawEncoder`, but it consumes the
    full observation sequence because Sparsh expects temporal tactile pairs:

        I_t + I_{t-k} -> (6, H, W)

    Expected input:
        single sensor: (B, S, 3, H, W)
        multi sensor:  (B, S, N, 3, H, W)

    Output:
        single sensor: (B, S, D)
        multi sensor:  (B, S, N*D)
    """

    def __init__(self, config: DiffusionSparshConfig, image_shape: tuple[int, ...] | torch.Size | None = None):
        super().__init__()
        self.config = config
        self.frozen = bool(config.sparsh_frozen)
        self.keep_eval_when_frozen = bool(config.sparsh_keep_eval_when_frozen)
        self.input_size = tuple(config.sparsh_input_size)
        self.temporal_stride = int(config.sparsh_temporal_stride)
        self.auto_rescale_uint8 = bool(config.sparsh_auto_rescale_uint8)
        self.pooling = str(config.sparsh_pooling)

        if image_shape is None:
            image_shape = next(iter(config.tactile_raw_features.values())).shape
        channels, _, _ = _infer_image_channels_hw(image_shape)
        if channels != 3:
            raise ValueError(
                "Sparsh tactile raw encoder expects RGB tactile images before temporal "
                f"pairing. Got feature shape {tuple(image_shape)}."
            )

        self.resize = torchvision.transforms.Resize(self.input_size)
        self.backbone, backbone_dim = self._build_and_load_sparsh_backbone(config)

        if self.frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if self.keep_eval_when_frozen:
                self.backbone.eval()

        self.feature_dim = int(config.sparsh_projection_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )

    def train(self, mode: bool = True):  # noqa: D401
        """Keep a frozen Sparsh backbone in eval mode while training the policy."""
        super().train(mode)
        if self.frozen and self.keep_eval_when_frozen:
            self.backbone.eval()
        return self

    @staticmethod
    def _maybe_add_sparsh_repo_to_path(config: DiffusionSparshConfig) -> None:
        """Optionally add a local facebookresearch/sparsh clone to sys.path."""
        repo_path = getattr(config, "sparsh_repo_path", None)
        if repo_path in (None, ""):
            return
        import os
        import sys

        repo_path = os.path.abspath(os.path.expanduser(str(repo_path)))
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    @staticmethod
    def _build_official_sparsh_vit(config: DiffusionSparshConfig) -> tuple[nn.Module, int]:
        """Instantiate the exact ViT builder from the Sparsh `tactile_ssl` package.

        This mirrors Sparsh configs such as:
          - config/experiment/dino_vit.yaml
          - config/task/t1_force_estimation.yaml

        In practice, for DINO base this builds:
            tactile_ssl.model.vit_base(
                img_size=(320, 240), in_chans=6, patch_size=16,
                pos_embed_fn="sinusoidal", num_register_tokens=1
            )
        """
        SparshTactileRawEncoder._maybe_add_sparsh_repo_to_path(config)
        try:
            import importlib
            sparsh_model = importlib.import_module("tactile_ssl.model")
        except ImportError as exc:
            raise ImportError(
                "Could not import `tactile_ssl.model`. Install the official Sparsh repo first, e.g.\n"
                "  git clone https://github.com/facebookresearch/sparsh.git ~/third_party/sparsh\n"
                "  pip install -e ~/third_party/sparsh\n"
                "or set `sparsh_repo_path` to your local Sparsh repo root."
            ) from exc

        builder_name = f"vit_{config.sparsh_model_size}"
        if not hasattr(sparsh_model, builder_name):
            raise ValueError(f"Official Sparsh model package does not expose `{builder_name}`.")
        builder = getattr(sparsh_model, builder_name)

        backbone = builder(
            img_size=tuple(config.sparsh_input_size),
            in_chans=int(config.sparsh_input_channels),
            patch_size=int(config.sparsh_patch_size),
            num_register_tokens=int(config.sparsh_num_register_tokens),
            pos_embed_fn=str(config.sparsh_pos_embed_fn),
        )
        backbone_dim = int(getattr(backbone, "embed_dim", getattr(backbone, "num_features", config.sparsh_feature_dim)))
        return backbone, backbone_dim

    @staticmethod
    def _load_checkpoint_file(path: str) -> dict[str, Tensor]:
        """Load .ckpt/.pt or .safetensors as a flat dict-like checkpoint."""
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading Sparsh safetensors requires `safetensors`. Install with: pip install safetensors"
                ) from exc
            return load_file(path)
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _select_encoder_prefix(state_dict: dict[str, Tensor], ssl_name: str) -> str | None:
        """Find the prefix used by the official Sparsh checkpoint for the encoder."""
        if "jepa" in ssl_name:
            preferred = ["target_encoder.backbone", "target_encoder"]
        elif "dino" in ssl_name:
            preferred = ["teacher_encoder.backbone", "teacher_encoder", "student_encoder.backbone", "student_encoder"]
        elif "mae" in ssl_name:
            preferred = ["encoder.backbone", "encoder"]
        else:
            preferred = ["teacher_encoder.backbone", "target_encoder.backbone", "encoder.backbone", "backbone"]

        # The HF/safetensors files may prepend "model." or "module."; matching by substring
        # handles both while still stripping exactly through the selected prefix below.
        for prefix in preferred:
            needle = prefix + "."
            if any(needle in key for key in state_dict.keys()):
                return prefix
        return None

    @staticmethod
    def _extract_official_encoder_state(checkpoint: dict[str, Tensor], ssl_name: str) -> dict[str, Tensor]:
        """Extract only the official Sparsh encoder weights from a checkpoint.

        This follows the logic in `tactile_ssl.downstream_task.SLModule.load_encoder`,
        but is robust to both .ckpt files with a top-level `model` dict and flat
        .safetensors files from Hugging Face.
        """
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")

        raw_state = checkpoint.get("model", checkpoint)
        if not isinstance(raw_state, dict):
            raise ValueError(f"Expected state dict under checkpoint['model'] or checkpoint itself. Got {type(raw_state)!r}.")

        tensor_state = {key: value for key, value in raw_state.items() if torch.is_tensor(value)}
        if len(tensor_state) == 0:
            raise ValueError(
                "No tensor weights found in Sparsh checkpoint. "
                f"Top-level keys: {list(checkpoint.keys())[:20]}"
            )

        encoder_prefix = SparshTactileRawEncoder._select_encoder_prefix(tensor_state, ssl_name)
        if encoder_prefix is None:
            # Fallback: maybe the file already contains raw backbone keys.
            cleaned = {}
            for key, value in tensor_state.items():
                new_key = key
                for prefix in ("module.", "model.", "backbone."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                cleaned[new_key] = value
            return cleaned

        cleaned = {}
        needle = encoder_prefix + "."
        for key, value in tensor_state.items():
            idx = key.find(needle)
            if idx < 0:
                continue
            new_key = key[idx + len(needle):]
            # Remove an extra backbone prefix for cases like target_encoder.backbone.*
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone."):]
            cleaned[new_key] = value
        return cleaned

    def _build_and_load_sparsh_backbone(self, config: DiffusionSparshConfig) -> tuple[nn.Module, int]:
        """Build the official Sparsh ViT and load the official HF checkpoint."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "SparshTactileRawEncoder requires huggingface_hub. Install with: pip install huggingface_hub"
            ) from exc

        backbone, backbone_dim = self._build_official_sparsh_vit(config)

        checkpoint_path = hf_hub_download(
            repo_id=config.sparsh_model_name,
            filename=config.sparsh_checkpoint_filename,
        )
        checkpoint = self._load_checkpoint_file(checkpoint_path)
        encoder_state = self._extract_official_encoder_state(checkpoint, config.sparsh_ssl_name)

        model_state = backbone.state_dict()
        compatible_state = {
            key: value
            for key, value in encoder_state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        skipped_shape = [
            key
            for key, value in encoder_state.items()
            if key in model_state and tuple(model_state[key].shape) != tuple(value.shape)
        ]
        skipped_missing = [key for key in encoder_state.keys() if key not in model_state]

        missing, unexpected = backbone.load_state_dict(compatible_state, strict=False)
        loaded_ratio = len(compatible_state) / max(1, len(model_state))
        message = (
            "[Sparsh] official builder loaded "
            f"{len(compatible_state)}/{len(model_state)} tensors from "
            f"{config.sparsh_model_name}/{config.sparsh_checkpoint_filename}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_skipped={len(skipped_shape)}, key_skipped={len(skipped_missing)}"
        )
        if loaded_ratio < 0.5:
            warnings.warn(
                message
                + ". Loaded ratio is low; check `sparsh_model_size`, `sparsh_num_register_tokens`, "
                "`sparsh_input_size`, and checkpoint filename.",
                RuntimeWarning,
            )
        else:
            print(message)

        return backbone, backbone_dim

    def _maybe_rescale(self, x: Tensor) -> Tensor:
        """Convert uint8-like tensors to [0,1], without touching normalized tensors."""
        if not self.auto_rescale_uint8:
            return x
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        if not torch.is_floating_point(x):
            return x.float()
        with torch.no_grad():
            max_abs = x.detach().abs().amax()
        if max_abs > 10:
            return x / 255.0
        return x

    def _make_temporal_pairs(self, x: Tensor) -> Tensor:
        """Build 6-channel tactile pairs for each observation step.

        Args:
            x: (B, S, 3, H, W)

        Returns:
            (B, S, 6, H, W), where each step i uses x_i and x_{max(0, i-k)}.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected tactile sequence (B,S,3,H,W), got {tuple(x.shape)}.")
        batch_size, n_steps, channels, _, _ = x.shape
        if channels != 3:
            raise ValueError(f"Sparsh expects RGB tactile frames with 3 channels. Got C={channels}.")

        pairs = []
        for step in range(n_steps):
            previous_step = max(0, step - self.temporal_stride)
            current = x[:, step]
            previous = x[:, previous_step]
            pairs.append(torch.cat([current, previous], dim=1))
        return torch.stack(pairs, dim=1)

    def _pool_backbone_output(self, features: Tensor | dict | tuple | list) -> Tensor:
        """Convert official Sparsh patch-token output to a vector feature."""
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                features = features["x_norm_patchtokens"]
            elif "last_hidden_state" in features:
                features = features["last_hidden_state"]
            elif "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            else:
                raise RuntimeError(f"Unknown Sparsh output dict keys: {list(features.keys())}")
        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.dim() == 3:
            if self.pooling == "mean_patch":
                return features.mean(dim=1)
            if self.pooling == "first_patch":
                return features[:, 0]
            raise ValueError(f"Unsupported Sparsh pooling mode: {self.pooling}")
        if features.dim() == 2:
            return features
        raise RuntimeError(f"Expected Sparsh features with shape (B,N,D) or (B,D), got {tuple(features.shape)}.")

    def _forward_backbone(self, x_pair: Tensor) -> Tensor:
        """Forward 6-channel pairs through Sparsh and return vector features."""
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(x_pair)
        else:
            features = self.backbone(x_pair)
        return self._pool_backbone_output(features)

    def forward(self, x: Tensor) -> Tensor:
        multi_sensor = x.dim() == 6
        if multi_sensor:
            batch_size, n_steps, n_sensors, channels, height, width = x.shape
            x = einops.rearrange(x, "b s n c h w -> (b n) s c h w")
        else:
            batch_size, n_steps, channels, height, width = x.shape
            n_sensors = 1

        x = self._maybe_rescale(x)
        x_pairs = self._make_temporal_pairs(x)  # (B*N, S, 6, H, W)
        x_pairs = einops.rearrange(x_pairs, "bn s c h w -> (bn s) c h w")
        x_pairs = self.resize(x_pairs)

        features = self._forward_backbone(x_pairs)
        features = self.projector(features)
        features = einops.rearrange(features, "(bn s) d -> bn s d", s=n_steps)

        if multi_sensor:
            features = einops.rearrange(features, "(b n) s d -> b s (n d)", b=batch_size, n=n_sensors)
        return features


class TactileFusedEncoder(nn.Module):
    """Encode 4-channel tactile normal+depth data into a feature vector."""

    def __init__(self, config: DiffusionSparshConfig, depth_image_shape: tuple[int, ...] | torch.Size | None = None):
        super().__init__()
        self.config = config

        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.tactile_crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
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
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)

        backbone_model.conv1 = new_conv
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.tactile_use_group_norm:
            if config.tactile_fused_pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained tactile fused backbone.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if depth_image_shape is None and config.tactile_depth_features:
            depth_image_shape = next(iter(config.tactile_depth_features.values())).shape
        if depth_image_shape is None:
            _, height, width = 1, 480, 640
        else:
            _, height, width = _infer_image_channels_hw(depth_image_shape)

        final_shape_h_w = config.tactile_resize_shape or config.tactile_crop_shape or (height, width)
        dummy_shape = (1, config.tactile_fused_backbone_in_channels, *final_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        num_kp = config.tactile_spatial_softmax_num_keypoints
        self.spatial_pool = SpatialSoftmax(feature_map_shape, num_kp=num_kp)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = num_kp * 2 + feature_map_shape[0]
        self.out = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        if self.do_resize:
            x = self.resize(x)

        x = self.backbone(x)
        keypoint_coords = torch.flatten(self.spatial_pool(x), start_dim=1)
        scale_features = torch.flatten(self.global_pool(x), start_dim=1)
        return self.out(torch.cat([keypoint_coords, scale_features], dim=-1))


class TactileMarkerEncoder(nn.Module):
    """Encodes flattened tactile marker displacement into a feature vector.

    Processes (N_markers × 2) coordinates similar to state encoding.
    """

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
        """
        Args:
            x: (B, ..., input_dim) flattened marker displacement.
        Returns:
            (B, ..., embed_dim) marker feature.
        """
        return self.encoder(x)


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

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
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


class DiffusionRgbEncoder(nn.Module):
    """Encode an RGB image into a 1D feature vector."""

    def __init__(self, config: DiffusionSparshConfig, image_shape: tuple[int, ...] | torch.Size | None = None):
        super().__init__()

        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
        else:
            self.do_crop = False

        if config.resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.do_resize = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained RGB backbone.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if image_shape is None:
            rgb_shapes = [ft.shape for ft in config.enabled_rgb_image_features.values()]
            if len(rgb_shapes) == 0:
                raise ValueError("DiffusionRgbEncoder requires at least one enabled non-tactile RGB image feature.")
            image_shape = rgb_shapes[0]

        channels, height, width = _infer_image_channels_hw(image_shape)
        if channels != 3:
            raise ValueError(f"RGB encoder expected 3 channels, but got feature shape {tuple(image_shape)}.")

        final_shape_h_w = config.resize_shape or config.crop_shape or (height, width)
        dummy_shape = (1, channels, *final_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        if self.do_resize:
            x = self.resize(x)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.out(x)


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """Replace submodules matching a predicate with new modules."""
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
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


# =============================================================================
# Denoisers: timestep embedding, UNet, and DiT
# =============================================================================

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

    def __init__(self, config: DiffusionSparshConfig, global_cond_dim: int):
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


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionDiTBlock1d(nn.Module):
    """A lightweight DiT block with AdaLN-Zero modulation and gated residuals."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # AdaLN-Zero: each block initially behaves close to identity, which tends
        # to stabilize DiT denoiser training on small robot datasets.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiffusionDiTFinalLayer1d(nn.Module):
    """DiT output layer with AdaLN-Zero modulation."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiffusionConditionalDiT1d(nn.Module):
    """DiT-style 1D denoiser for action diffusion.

    Conditioning path:
        timestep_emb + global_cond -> concat -> cond_proj -> cond

    The resulting `cond` modulates every DiT block through AdaLN-Zero.
    """

    def __init__(self, config: DiffusionSparshConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        d_model = config.dit_d_model

        self.input_proj = nn.Linear(config.action_feature.shape[0], d_model)
        self.cond_proj = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d_model), nn.Mish())
        self.pos_emb = nn.Parameter(torch.zeros(1, config.horizon, d_model))

        self.blocks = nn.ModuleList(
            [
                DiffusionDiTBlock1d(
                    d_model=d_model,
                    nhead=config.dit_nhead,
                    dim_feedforward=config.dit_dim_feedforward,
                    dropout=config.dit_dropout,
                )
                for _ in range(config.dit_num_layers)
            ]
        )
        self.final_layer = DiffusionDiTFinalLayer1d(d_model, config.action_feature.shape[0])

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([timesteps_embed, global_cond], axis=-1) if global_cond is not None else timesteps_embed
        cond = self.cond_proj(global_feature)

        h = self.input_proj(x)
        h = h + self.pos_emb[:, : h.shape[1]]

        for block in self.blocks:
            h = block(h, cond)

        return self.final_layer(h, cond)


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
