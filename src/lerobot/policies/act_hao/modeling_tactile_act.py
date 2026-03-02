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
"""Tactile Action Chunking Transformer Policy (Hao variant)

Based on ACT (https://huggingface.co/papers/2304.13705) with GelSight tactile sensor integration.
Supports three tactile modalities:
  - Depth map (1ch) + Normal map (3ch) → fused 4-channel ResNet backbone (unfrozen BN)
  - Marker displacement (35×2) → independent 1D token via MLP encoder
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_hao.configuration_tactile_act import TactileACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# Constants for tactile observation keys (must match dataset info.json keys)
OBS_TAC_DEPTH = "observation.tac1_depth"
OBS_TAC_NORMAL = "observation.tac1_normal"
OBS_TAC_MARKER = "observation.tac1_marker_displacement"


class TactileACTPolicy(PreTrainedPolicy):
    """Tactile ACT Policy with GelSight tactile sensor support.

    Processes camera images, tactile depth+normal images, marker displacement,
    and robot proprioceptive state to predict action chunks.
    """

    config_class = TactileACTConfig
    name = "tactile_act_hao"

    def __init__(
        self,
        config: TactileACTConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        """Return parameter groups with separate learning rates for backbones."""
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone")
                    and not n.startswith("model.tactile_backbone")
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.tactile_backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_tactile_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705."""
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer with tactile sensor support.

    Architecture overview:
                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder during training)
                                ┌─────────────────────────────┐
                                │                Outputs      │
                                │                   ▲         │
                                │        ┌─────►┌───────┐    │
                   ┌──────┐     │        │      │Transf.│    │
                   │      │     │        ├─────►│decoder│    │
              ┌────┴────┐ │     │        │      │       │    │
              │         │ │     │ ┌──────┴──┬──►│       │    │
              │ VAE     │ │     │ │         │   └───────┘    │
              │ encoder │ │     │ │ Transf. │                │
              │         │ │     │ │ encoder │                │
              └───▲─────┘ │     │ │         │                │
                  │       │     │ └▲──▲─▲─▲─┘                │
                  │       │     │  │  │ │ │                  │
                inputs    └─────┼──┘  │ │ tactile img emb.   │
                                │    │ marker emb.           │
                                │   state emb.               │
                                └─────────────────────────────┘
    """

    def __init__(self, config: TactileACTConfig):
        super().__init__()
        self.config = config

        # =====================================================================
        # 1) VAE Encoder (unchanged from original ACT)
        # =====================================================================
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # =====================================================================
        # 2) Camera Backbone (ResNet with FrozenBatchNorm2d, same as original ACT)
        # =====================================================================
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # =====================================================================
        # 3) Tactile Image Backbone (4-channel ResNet, unfrozen BN)
        #    Fuses depth (1ch) + normal (3ch) as a 4-channel input.
        #    Pretrained weights: first 3 channels from ImageNet, 4th channel
        #    initialized by copying the red channel weights.
        #    BatchNorm is NOT frozen → fully fine-tuned during training.
        # =====================================================================
        if self.config.use_tactile_image_features:
            tac_backbone_model = getattr(torchvision.models, config.tactile_vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_tactile_backbone_weights,
                # No norm_layer override → default nn.BatchNorm2d (fully trainable)
            )
            # Modify conv1: 3 channels → 4 channels (normal 3ch + depth 1ch)
            old_conv1 = tac_backbone_model.conv1
            new_conv1 = nn.Conv2d(
                4, old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                # Copy pretrained weights for first 3 channels (normal map)
                new_conv1.weight[:, :3] = old_conv1.weight
                # Initialize 4th channel (depth) by copying red channel weights
                new_conv1.weight[:, 3:4] = old_conv1.weight[:, 0:1]
            tac_backbone_model.conv1 = new_conv1

            # Save fc.in_features before IntermediateLayerGetter removes fc
            tac_backbone_fc_in_features = tac_backbone_model.fc.in_features
            self.tactile_backbone = IntermediateLayerGetter(
                tac_backbone_model, return_layers={"layer4": "feature_map"}
            )

        # =====================================================================
        # 4) Marker Displacement Encoder (MLP → independent 1D token)
        #    Input: (B, 35, 2) → flatten → (B, 70) → MLP → (B, dim_model)
        # =====================================================================
        if self.config.use_tactile_marker:
            self.marker_encoder = nn.Sequential(
                nn.Linear(config.tactile_marker_input_dim, config.tactile_marker_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.tactile_marker_hidden_dim, config.dim_model),
            )

        # =====================================================================
        # 5) Transformer Encoder & Decoder
        # =====================================================================
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # =====================================================================
        # 6) Encoder Input Projections
        #    Token structure: [latent, (state), (marker), (env), (*cam_pixels), (*tac_pixels)]
        # =====================================================================
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        if self.config.use_tactile_image_features:
            self.encoder_tac_feat_input_proj = nn.Conv2d(
                tac_backbone_fc_in_features, config.dim_model, kernel_size=1
            )

        # =====================================================================
        # 7) Positional Embeddings
        # =====================================================================
        # 1D tokens: latent + (state) + (marker) + (env)
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.use_tactile_marker:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # 2D positional embeddings for camera and tactile image feature maps
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        if self.config.use_tactile_image_features:
            self.encoder_tac_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # =====================================================================
        # 8) Decoder
        # =====================================================================
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Tactile ACT model.

        Args:
            batch: Dictionary with keys:
                - "observation.state": (B, state_dim)
                - "observation.images.*": (B, C, H, W) per camera (via OBS_IMAGES list)
                - "observation.tac1_depth": (B, H, W, 1) depth map
                - "observation.tac1_normal": (B, H, W, 3) normal map
                - "observation.tac1_marker_displacement": (B, 35, 2) marker positions
                - "action": (B, chunk_size, action_dim) [only during training with VAE]

        Returns:
            actions: (B, chunk_size, action_dim) predicted action sequence
            (mu, log_sigma_x2): VAE latent distribution parameters, or (None, None)
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # --- Determine batch size ---
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        elif OBS_STATE in batch:
            batch_size = batch[OBS_STATE].shape[0]
        elif OBS_TAC_DEPTH in batch:
            batch_size = batch[OBS_TAC_DEPTH].shape[0]
        else:
            raise ValueError("Cannot determine batch size: no recognized observation keys in batch.")

        # --- VAE Encoder (training only) ---
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[ACTION].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            # Determine device from any available tensor
            device = batch[OBS_STATE].device if OBS_STATE in batch else batch[OBS_TAC_DEPTH].device
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

        # --- Build Transformer Encoder Input Tokens ---
        # Token order: [latent, (state), (marker), (env), (*cam_pixels), (*tac_pixels)]
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # 1D token: Robot state
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        # 1D token: Marker displacement
        if self.config.use_tactile_marker and OBS_TAC_MARKER in batch:
            marker_flat = batch[OBS_TAC_MARKER].flatten(start_dim=1)  # (B, 70)
            marker_token = self.marker_encoder(marker_flat)  # (B, dim_model)
            encoder_in_tokens.append(marker_token)

        # 1D token: Environment state
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # 2D tokens: Camera image features
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 2D tokens: Tactile image features (depth + normal → 4ch)
        if self.config.use_tactile_image_features and OBS_TAC_DEPTH in batch and OBS_TAC_NORMAL in batch:
            # Data arrives as (B, H, W, C) float32, convert to (B, C, H, W)
            tac_normal = batch[OBS_TAC_NORMAL].permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            tac_depth = batch[OBS_TAC_DEPTH].permute(0, 3, 1, 2).contiguous()    # (B, 1, H, W)
            # Fuse: normal (3ch) + depth (1ch) → 4 channels
            tac_img = torch.cat([tac_normal, tac_depth], dim=1)  # (B, 4, H, W)

            tac_features = self.tactile_backbone(tac_img)["feature_map"]  # (B, 512, H', W')
            tac_pos_embed = self.encoder_tac_feat_pos_embed(tac_features).to(dtype=tac_features.dtype)
            tac_features = self.encoder_tac_feat_input_proj(tac_features)  # (B, dim_model, H', W')
            tac_features = einops.rearrange(tac_features, "b c h w -> (h w) b c")
            tac_pos_embed = einops.rearrange(tac_pos_embed, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(tac_features))
            encoder_in_pos_embed.extend(list(tac_pos_embed))

        # --- Stack and run through Transformer ---
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # (S, B, D) → (B, S, D)
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


# =========================================================================
# Transformer Components (unchanged from original ACT)
# =========================================================================

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: TactileACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: TactileACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: TactileACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: TactileACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


# =========================================================================
# Helper Functions
# =========================================================================

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
