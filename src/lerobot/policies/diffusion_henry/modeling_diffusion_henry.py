#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
# Modified for Diffusion-Henry tactile adaptation.
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
"""Diffusion-Henry Policy: Diffusion Policy with tactile sensor support.

Extends the original Diffusion Policy to handle:
- Standard images (global, inhand, tac_raw) via shared ResNet backbone
- Tactile depth + normal (4-channel) via independent ResNet backbone
- Tactile marker displacement fused with robot state via project-then-fuse strategy
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

from lerobot.policies.diffusion_henry.configuration_diffusion_henry import DiffusionHenryConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Custom keys for tactile data
OBS_TAC_FUSED = "observation.tac_fused"  # Synthetic 4-channel key (depth + normal)
OBS_TAC_RAW = "observation.tac_raw"  # Raw tactile image data
OBS_TAC_MARKER = "observation.tac_marker"  # Flattened marker displacement


class DiffusionHenryPolicy(PreTrainedPolicy):
    """Diffusion-Henry Policy with tactile sensor support.

    Extends Diffusion Policy to handle tactile depth+normal images and marker displacement data.
    """

    config_class = DiffusionHenryConfig
    name = "diffusion_henry"

    def __init__(
        self,
        config: DiffusionHenryConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy
        self._queues = None

        self.diffusion = DiffusionHenryModel(config)

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
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        # Prepare the batch (stack images, synthesize tactile data)
        batch = self._prepare_batch(batch)

        # NOTE: It's important that this happens after preparing the batch.
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
        """Prepare the batch: collect images, synthesize tactile 4-channel, flatten marker."""
        batch = dict(batch)  # shallow copy

        depth_keys = list(self.config.tactile_depth_features.keys())
        normal_keys = list(self.config.tactile_normal_features.keys())
        tac_raw_keys = list(self.config.tactile_raw_features.keys())

        # 1. Collect standard image features while excluding tactile raw/depth/normal features.
        # This keeps tac_raw/tac_depth/tac_normal 
        if self.config.image_features:
            tactile_keys = set(depth_keys + normal_keys + tac_raw_keys)
            rgb_image_keys = [key for key in self.config.image_features if key not in tactile_keys]
            if len(rgb_image_keys) > 0:
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in rgb_image_keys], dim=-4)
        
        # 2. Collect tactile raw features (if present) without mixing them with RGB image features.
        if self.config.has_tactile_raw:
            if len(tac_raw_keys) > 0:
                # Support single or multiple tactile raw cameras
                tac_raw_images = torch.stack([batch[key] for key in tac_raw_keys], dim=-4)
                batch[OBS_TAC_RAW] = tac_raw_images  # (b, s, n_raw, h, w, c) or (b, s, h, w, c) if n_raw=1

        # 3. Synthesize tac_depth + tac_normal → 4-channel tactile images (supports multiple sensor pairs)
        if self.config.has_tactile_fused:
            if len(depth_keys) > 0 and len(normal_keys) > 0:
                # Support single or multiple depth+normal pairs
                fused_list = []
                for depth_idx in range(len(depth_keys)):
                    depth = batch[depth_keys[depth_idx]]
                    normal = batch[normal_keys[depth_idx]]  # (B, H, W, 3) or (B, n_obs_steps, H, W, 3)

                    # Video-encoded depth may decode as 3 channels. Reduce it to 1 channel via channel mean.
                    # Supports both channel-first (..., C, H, W) and channel-last (..., H, W, C).
                    if depth.dim() >= 3:
                        if depth.shape[-3] in (1, 3):
                            depth = depth.mean(dim=-3, keepdim=True)
                        elif depth.shape[-1] in (1, 3):
                            depth = depth.mean(dim=-1, keepdim=True)

                    # Ensure normal is channel-last for concatenation below.
                    if normal.dim() >= 3 and normal.shape[-3] in (1, 3):
                        normal_for_concat = normal.movedim(-3, -1)
                    else:
                        normal_for_concat = normal

                    # Ensure depth is channel-last for concatenation below.
                    if depth.dim() >= 3 and depth.shape[-3] == 1:
                        depth_for_concat = depth.movedim(-3, -1)
                    else:
                        depth_for_concat = depth

                    # Concatenate depth and normal to 4-channel
                    tac_4ch = torch.cat([normal_for_concat, depth_for_concat], dim=-1)  # (..., H, W, 4)

                    # Permute to channel-first format
                    # Handle both (B, H, W, 4) and (B, n_obs_steps, H, W, 4)
                    if tac_4ch.dim() == 4:
                        tac_4ch = tac_4ch.permute(0, 3, 1, 2)  # (B, 4, H, W)
                    else:
                        tac_4ch = tac_4ch.permute(0, 1, 4, 2, 3)  # (B, n_obs_steps, 4, H, W)

                    fused_list.append(tac_4ch)

                # Stack multiple fused pairs if present
                if len(fused_list) > 1:
                    batch[OBS_TAC_FUSED] = torch.stack(fused_list, dim=-4)  # (B, n_obs_steps, n_fused, 4, H, W)
                else:
                    batch[OBS_TAC_FUSED] = fused_list[0]  # (B, n_obs_steps, 4, H, W)

        # 4. Flatten tac_marker_displacement: (B, 35, 2) → (B, 70)
        if self.config.has_tactile_marker:
            marker_keys = list(self.config.tactile_marker_features.keys())
            if len(marker_keys) > 0:
                marker_key = marker_keys[0]
                marker = batch[marker_key]  # (B, 35, 2) or (B, n_obs_steps, 35, 2)
                # Flatten the marker coordinates
                batch[OBS_TAC_MARKER] = marker.flatten(start_dim=-2)  # (B, 70) or (B, n_obs_steps, 70)

        return batch


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
    ):
        super().__init__()
        self.modality_names = tuple(modality_input_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.routing_dropout = routing_dropout
        self.topk = topk
        self.debug_inference = debug_inference
        self.debug_every_n_calls = debug_every_n_calls
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
                print(f"[modal_moe] call={self._inference_call_count} topk={topk_pairs}")

        consensus = (modality_weights.unsqueeze(-1) * modality_stack).sum(dim=2)  # (B, S, H)
        return self.out_norm(consensus)


class DiffusionHenryModel(nn.Module):
    """Core Diffusion model with tactile sensor support."""

    def __init__(self, config: DiffusionHenryConfig):
        super().__init__()
        self.config = config
        self.use_denoiser_moe = config.use_denoiser_moe
        self.denoiser_composition_strategy = config.denoiser_composition_strategy
        self.denoiser_num_modules = config.denoiser_num_modules
        self.denoiser_topk = config.denoiser_topk

        # Keep tactile raw/depth/normal out of the RGB branch when they are video/image features.
        self._tactile_depth_keys = list(self.config.tactile_depth_features.keys())
        self._tactile_normal_keys = list(self.config.tactile_normal_features.keys())
        self._tactile_raw_keys = list(self.config.tactile_raw_features.keys())

        tactile_keys = set(self._tactile_depth_keys + self._tactile_normal_keys + self._tactile_raw_keys)

        self._rgb_image_keys = [key for key in self.config.image_features if key not in tactile_keys]

        # Build observation encoders
        # Start with robot state dimension
        global_cond_dim = self.config.robot_state_feature.shape[0]
        self._modality_dims: dict[str, int] = {"state": self.config.robot_state_feature.shape[0]}

        # === Standard image encoder (shared for visual images and tac_raw) ===
        if len(self._rgb_image_keys) > 0:
            num_images = len(self._rgb_image_keys)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
                rgb_total_dim = encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
                rgb_total_dim = self.rgb_encoder.feature_dim * num_images
            self._modality_dims["rgb"] = rgb_total_dim

        # === Environment state (if present) ===
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
            self._modality_dims["env_state"] = self.config.env_state_feature.shape[0]

        # === Environment
        if self.config.has_tactile_raw:
            self.tactile_raw_encoder = TactileRawEncoder(config)
            num_tactile_raw = len(self._tactile_raw_keys)
            tactile_raw_total_dim = self.tactile_raw_encoder.feature_dim * num_tactile_raw
            global_cond_dim += tactile_raw_total_dim
            self._modality_dims["tactile_raw"] = tactile_raw_total_dim

        # === Tactile depth/normal encoder (for depth + normal 4-channel data) ===
        if self.config.has_tactile_fused:
            self.tactile_fused_encoder = TactileFusedEncoder(config)
            num_tactile_fused = min(len(self._tactile_depth_keys), len(self._tactile_normal_keys))
            tactile_fused_total_dim = self.tactile_fused_encoder.feature_dim * num_tactile_fused
            global_cond_dim += tactile_fused_total_dim
            self._modality_dims["tactile_fused"] = tactile_fused_total_dim

        # === Tactile marker encoder (for marker displacement) ===
        if self.config.has_tactile_marker:
            self.tactile_marker_encoder = TactileMarkerEncoder(
                input_dim=config.tactile_marker_input_dim,
                embed_dim=config.tactile_marker_embed_dim,
            )
            global_cond_dim += config.tactile_marker_embed_dim
            self._modality_dims["tactile_marker"] = config.tactile_marker_embed_dim

        # === Optional MoE consensus over modalities ===
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
            )
            global_cond_dim += config.moe_hidden_dim
        else:
            self.modal_moe = None

        # === Denoiser backbone (UNet / DiT) ===
        global_cond_total_dim = global_cond_dim * config.n_obs_steps
        if config.denoiser_type == "unet":
            denoiser_ctor = lambda cond_dim: DiffusionConditionalUnet1d(config, global_cond_dim=cond_dim)
        elif config.denoiser_type == "dit":
            denoiser_ctor = lambda cond_dim: DiffusionConditionalDiT1d(config, global_cond_dim=cond_dim)
        else:
            raise ValueError(f"Unsupported denoiser type {config.denoiser_type}")

        if self.use_denoiser_moe:
            self._expert_modalities = [name for name in self._modality_dims.keys() if name != "state"]
            if len(self._expert_modalities) == 0:
                raise ValueError(
                    "`use_denoiser_moe=True` requires at least one non-state modality feature."
                )

            # Expert condition uses [state + modality] per observation step.
            self._expert_cond_dims = {
                name: (self._modality_dims["state"] + self._modality_dims[name]) * config.n_obs_steps
                for name in self._expert_modalities
            }

            self.denoiser_experts = nn.ModuleList()
            for name in self._expert_modalities:
                self.denoiser_experts.extend(
                    [denoiser_ctor(self._expert_cond_dims[name]) for _ in range(self.denoiser_num_modules)]
                )

            self.weight_predictor = nn.Sequential(
                nn.Linear(global_cond_total_dim, global_cond_total_dim),
                nn.ReLU(),
                nn.Linear(global_cond_total_dim, len(self.denoiser_experts)),
            )

            # Compatibility handle; MoE path does not call this directly.
            self.unet = self.denoiser_experts
        else:
            denoiser = denoiser_ctor(global_cond_total_dim)

            # Keep `unet` attribute name for compatibility with existing training/inference code paths.
            self.unet = denoiser

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

    def get_modal_moe_debug_info(self) -> dict[str, list[float] | list[str] | int] | None:
        """Return latest modal MoE routing summary captured during inference."""
        if self.modal_moe is None:
            return None
        return self.modal_moe.last_inference_routing

    def _expert_cond_for_expert(
        self,
        expert_idx: int,
        modality_conds: list[Tensor],
    ) -> Tensor:
        modality_idx = expert_idx // self.denoiser_num_modules
        return modality_conds[modality_idx]

    def _predict_with_denoiser_moe(
        self,
        trajectory: Tensor,
        timestep: Tensor,
        modality_conds: list[Tensor],
        weights: Tensor,
    ) -> Tensor:
        """Predict denoising output with expert composition.

        Args:
            trajectory: (B, T, Da)
            timestep: (B,) long tensor
            modality_conds: list of per-modality conditions, each (B, Dm)
            weights: (num_experts, B)
        """
        batch_size = trajectory.shape[0]

        if self.denoiser_composition_strategy == "soft_gating":
            norm_weights = F.softmax(weights, dim=0)
            pred = sum(
                norm_weights[i][:, None, None]
                * expert(trajectory, timestep, global_cond=self._expert_cond_for_expert(i, modality_conds))
                for i, expert in enumerate(self.denoiser_experts)
            )
            return pred

        if self.denoiser_composition_strategy == "hard_routing":
            idx = torch.argmax(weights, dim=0)  # (B,)
            pred = torch.stack(
                [
                    self.denoiser_experts[i](
                        trajectory[b : b + 1],
                        timestep[b : b + 1],
                        global_cond=self._expert_cond_for_expert(i, modality_conds)[b : b + 1],
                    )
                    for b, i in enumerate(idx)
                ]
            ).squeeze(1)
            return pred

        if self.denoiser_composition_strategy == "topk_moe":
            k = min(self.denoiser_topk, len(self.denoiser_experts))
            topk_vals, topk_idx = torch.topk(weights, k=k, dim=0)  # (k, B)
            norm_weights = F.softmax(topk_vals, dim=0)  # (k, B)
            pred = torch.zeros_like(trajectory)

            for j in range(k):
                idx = topk_idx[j]  # (B,)
                w = norm_weights[j]  # (B,)
                for b in range(batch_size):
                    i = idx[b].item()
                    cond_i = self._expert_cond_for_expert(i, modality_conds)
                    pred[b] += (
                        w[b]
                        * self.denoiser_experts[i](
                            trajectory[b : b + 1],
                            timestep[b : b + 1],
                            global_cond=cond_i[b : b + 1],
                        ).squeeze(0)
                    )
            return pred

        raise ValueError(f"Unknown denoiser composition strategy {self.denoiser_composition_strategy}")

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        modality_conds: list[Tensor] | None = None,
        weights: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample from the diffusion model."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
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
                assert modality_conds is not None and weights is not None
                model_output = self._predict_with_denoiser_moe(
                    sample,
                    timestep=timestep,
                    modality_conds=modality_conds,
                    weights=weights,
                )
            else:
                model_output = self.unet(
                    sample,
                    timestep,
                    global_cond=global_cond,
                )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(
        self,
        batch: dict[str, Tensor],
        return_modality_conds: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Encode all features and concatenate them for global conditioning.

        When `return_modality_conds=True`, also returns per-modality expert
        conditions in the same modality order as `self._expert_modalities`.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        modality_features: dict[str, Tensor] = {"state": batch[OBS_STATE]}

        # === Extract standard image features ===
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
            modality_features["rgb"] = img_features

        # === Extract environment state features ===
        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])
            modality_features["env_state"] = batch[OBS_ENV_STATE]

        # === Extract tactile raw features ===
        if self.config.has_tactile_raw:
            tac_raw = batch[OBS_TAC_RAW]  # (B, n_obs_steps, 3, H, W) or (B, n_obs_steps, n_raw, 3, H, W)
            if tac_raw.dim() == 5:
                # Single tactile raw camera
                tac_raw_features = self.tactile_raw_encoder(einops.rearrange(tac_raw, "b s c h w -> (b s) c h w"))
                tac_raw_features = einops.rearrange(tac_raw_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)
            else:
                # Multiple tactile raw cameras (B, n_obs_steps, n_raw, 3, H, W)
                num_raw = tac_raw.shape[2]
                tac_raw_per_camera = einops.rearrange(tac_raw, "b s n c h w -> n (b s) c h w")
                raw_features_list = [
                    self.tactile_raw_encoder(images)
                    for images in tac_raw_per_camera
                ]
                raw_features_stacked = torch.cat(raw_features_list, dim=0)
                tac_raw_features = einops.rearrange(
                    raw_features_stacked, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=num_raw
                )
            global_cond_feats.append(tac_raw_features)
            modality_features["tactile_raw"] = tac_raw_features

        # === Extract tactile fused features (depth + normal) ===
        if self.config.has_tactile_fused:
            tac_fused = batch[OBS_TAC_FUSED]  # (B, n_obs_steps, 4, H, W) or (B, n_obs_steps, n_fused, 4, H, W)
            if tac_fused.dim() == 5:
                # Single fused pair (depth + normal)
                tac_fused_flat = einops.rearrange(tac_fused, "b s c h w -> (b s) c h w")
                tac_fused_features = self.tactile_fused_encoder(tac_fused_flat)
                tac_fused_features = einops.rearrange(tac_fused_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)
            else:
                # Multiple fused pairs (B, n_obs_steps, n_fused, 4, H, W)
                num_fused = tac_fused.shape[2]
                tac_fused_per_sensor = einops.rearrange(tac_fused, "b s n c h w -> n (b s) c h w")
                fused_features_list = [
                    self.tactile_fused_encoder(images)
                    for images in tac_fused_per_sensor
                ]
                fused_features_stacked = torch.cat(fused_features_list, dim=0)
                tac_fused_features = einops.rearrange(
                    fused_features_stacked, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps, n=num_fused
                )
            global_cond_feats.append(tac_fused_features)
            modality_features["tactile_fused"] = tac_fused_features

        # === Extract tactile marker features ===
        if self.config.has_tactile_marker:
            tac_marker = batch[OBS_TAC_MARKER]  # (B, n_obs_steps, 70)
            tac_marker_features = self.tactile_marker_encoder(tac_marker)  # (B, n_obs_steps, embed_dim)
            global_cond_feats.append(tac_marker_features)
            modality_features["tactile_marker"] = tac_marker_features

        if self.modal_moe is not None:
            consensus_features = self.modal_moe(modality_features)
            global_cond_feats.append(consensus_features)

        # Concatenate features then flatten to (B, global_cond_dim).
        global_cond = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

        if not return_modality_conds:
            return global_cond

        if not self.use_denoiser_moe:
            return global_cond, []

        state_feat = modality_features["state"]
        modality_conds = [
            torch.cat([modality_features[name], state_feat], dim=-1).flatten(start_dim=1)
            for name in self._expert_modalities
        ]
        return global_cond, modality_conds

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate actions given observations."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        if self.use_denoiser_moe:
            global_cond, modality_conds = self._prepare_global_conditioning(batch, return_modality_conds=True)
            weights = self.weight_predictor(global_cond).transpose(0, 1)
            actions = self.conditional_sample(
                batch_size,
                global_cond=global_cond,
                modality_conds=modality_conds,
                weights=weights,
                noise=noise,
            )
        else:
            global_cond = self._prepare_global_conditioning(batch)
            actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract n_action_steps worth of actions
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute training loss."""
        # Input validation
        assert set(batch).issuperset({OBS_STATE, ACTION})
        if self.config.do_mask_loss_for_padding:
            assert "action_is_pad" in batch

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Prepare global conditioning
        if self.use_denoiser_moe:
            global_cond, modality_conds = self._prepare_global_conditioning(batch, return_modality_conds=True)
            weights = self.weight_predictor(global_cond).transpose(0, 1)
        else:
            global_cond = self._prepare_global_conditioning(batch)

        # Forward diffusion
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run denoising network
        if self.use_denoiser_moe:
            pred = self._predict_with_denoiser_moe(
                noisy_trajectory,
                timestep=timesteps,
                modality_conds=modality_conds,
                weights=weights,
            )
        else:
            pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute loss
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss for padded actions
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

class TactileRawEncoder(nn.Module):
    """Encodes 3-channel tactile raw data into a feature vector.
    
    Uses a ResNet backbone modified to accept 3-channel input, with pretrained
    weights for the RGB channels.
    """

    def __init__(self, config:DiffusionHenryConfig):
        super().__init__()
        self.config = config

        # Set up optional preprocessing (crop)
        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.tactile_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # optional preprocessing (resize)
        if config.tactile_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.tactile_resize_shape)
        else:
            self.do_resize = False

        backbone_model = getattr(torchvision.models, config.tactile_raw_backbone)(
            weights=config.tactile_raw_pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.use_group_norm:
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
        # Determine final input shape to backbone (resize has priority over crop)
        if config.tactile_resize_shape is not None:
            final_shape_h_w = config.tactile_resize_shape
        elif config.tactile_crop_shape is not None:
            final_shape_h_w = config.tactile_crop_shape
        else:
            final_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *final_shape_h_w)
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
        if self.do_resize:
            x = self.resize(x)  

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x
    

class TactileFusedEncoder(nn.Module):
    """Encodes 4-channel tactile data (depth + normal) into a feature vector.

    Uses a ResNet backbone modified to accept 4-channel input, with pretrained
    weights for the RGB channels and mean-initialized weights for the depth channel.
    """

    def __init__(self, config: DiffusionHenryConfig):
        super().__init__()
        self.config = config

        # Set up optional preprocessing (crop)
        if config.tactile_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.tactile_crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.tactile_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # optional preprocessing (resize)
        if config.tactile_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.tactile_resize_shape)
        else:
            self.do_resize = False

        # Set up backbone with 4-channel input
        backbone_model = getattr(torchvision.models, config.tactile_fused_backbone)(
            weights=config.tactile_fused_pretrained_backbone_weights
        )

        # Modify first conv layer to accept 4 channels
        original_conv = backbone_model.conv1
        new_conv = nn.Conv2d(
            config.tactile_fused_backbone_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize weights: copy RGB weights, init depth channel with mean
        with torch.no_grad():
            if config.tactile_fused_pretrained_backbone_weights is not None:
                # Copy RGB weights (first 3 channels)
                new_conv.weight[:, :3] = original_conv.weight[:, :3]
                # Initialize additional channels with mean of RGB weights
                for i in range(3, config.tactile_fused_backbone_in_channels):
                    new_conv.weight[:, i:i+1] = original_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

        backbone_model.conv1 = new_conv

        # Remove final FC layers, keep feature extraction part
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        # Optionally replace BatchNorm with GroupNorm
        if config.tactile_use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers
        # Get tactile image shape from config
        if config.has_tactile_fused:
            depth_features = config.tactile_depth_features
            if len(depth_features) > 0:
                first_depth_ft = next(iter(depth_features.values()))
                # Shape is (H, W, 1) for depth
                h, w = first_depth_ft.shape[0], first_depth_ft.shape[1]
            else:
                # Default
                h, w = 480, 640
        else:
            h, w = 480, 640

        # Determine final input shape to backbone (resize has priority over crop)
        if config.tactile_resize_shape is not None:
            final_shape_h_w = config.tactile_resize_shape
        elif config.tactile_crop_shape is not None:
            final_shape_h_w = config.tactile_crop_shape
        else:
            final_shape_h_w = (h, w)
        dummy_shape = (1, config.tactile_fused_backbone_in_channels, *final_shape_h_w)
        # get final out put shape
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        # Use spatial softmax to extract keypoints
        self.spatial_pool = SpatialSoftmax(feature_map_shape, num_kp=config.tactile_spatial_softmax_num_keypoints)
        # Use global average pooling to extract tactile depth features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = config.tactile_spatial_softmax_num_keypoints * 2 + feature_map_shape[0]
        self.out = nn.Linear(config.tactile_spatial_softmax_num_keypoints * 2 + feature_map_shape[0], self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 4, H, W) tactile tensor (normal + depth channels).
        Returns:
            (B, D) tactile feature.
        """
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        if self.do_resize:
            x = self.resize(x)
        
        x = self.backbone(x)

        # apply sptial pool and avg pool
        keypoint_coords = torch.flatten(self.spatial_pool(x), start_dim=1)
        scale_features = torch.flatten(self.global_pool(x), start_dim=1)

        x = torch.cat([keypoint_coords, scale_features], dim=-1)

        x = self.relu(self.out(x))
        return x


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
    """Encodes an RGB image into a 1D feature vector."""

    def __init__(self, config: DiffusionHenryConfig):
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

        if config.resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.do_resize = False

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
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if config.image_features:
            images_shape = next(iter(config.image_features.values())).shape
            # Determine final input shape to backbone (resize has priority over crop)
            if config.resize_shape is not None:
                final_shape_h_w = config.resize_shape
            elif config.crop_shape is not None:
                final_shape_h_w = config.crop_shape
            else:
                final_shape_h_w = images_shape[1:]
            dummy_shape = (1, images_shape[0], *final_shape_h_w)
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
        if self.do_resize:
            x = self.resize(x)  

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


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

    def __init__(self, config: DiffusionHenryConfig, global_cond_dim: int):
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
    """A lightweight DiT block with AdaLN modulation and gated residuals."""

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
    """DiT output layer with AdaLN modulation."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiffusionConditionalDiT1d(nn.Module):
    """DiT-style 1D denoiser for action diffusion."""

    def __init__(self, config: DiffusionHenryConfig, global_cond_dim: int):
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
        self.cond_proj = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d_model))
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

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        timesteps_embed = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

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
