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
"""Tactile PointNet Diffusion Policy.

Architecture:
    1. PointNet encoder processes hand point cloud (observation.hand_pc) per frame
    2. Multi-modal Transformer encoder fuses:
       - state (observation.state)
       - state_velocity (observation.state_velocity)
       - tactile raw data (observation.tactile + observation.fsr concatenated)
       - PointNet-encoded hand_pc features
       Each (modality, timestep) pair is a token with learned modality-type and temporal embeddings.
    3. Transformer output conditions a 1D conditional UNet for diffusion-based action denoising.

No image input is used.
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.tactile_pointnet_diffusion.configuration_tactile_pointnet_diffusion import (
    TactilePointnetDiffusionConfig,
)
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION


# =============================================================================
# PointNet Encoder
# =============================================================================


class PointNetEncoder(nn.Module):
    """Simple PointNet encoder (Qi et al. 2017) for point cloud data.

    Processes per-point features through shared MLPs, applies global max pooling,
    then projects to a fixed-size output feature vector.

    Input:  (B, N, C)  where N=num_points, C=input_dim (e.g., 7 for xyz+force)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
        output_dim: int = 256,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # Shared MLP — implemented as Conv1d over points for parallel processing.
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Conv1d(prev_dim, h_dim, 1))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        self.shared_mlp = nn.Sequential(*layers)

        # Output FC
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) point cloud with N points and C features per point.
        Returns:
            (B, output_dim) global point cloud feature.
        """
        # (B, N, C) → (B, C, N) for Conv1d
        x = x.transpose(1, 2)
        x = self.shared_mlp(x)  # (B, hidden_dims[-1], N)
        # Global max pooling over points
        x = x.max(dim=-1)[0]  # (B, hidden_dims[-1])
        x = self.fc(x)  # (B, output_dim)
        return x


class SparsePointCloudEncoder(nn.Module):
    """Encode sparse point cloud features into a low-dimensional vector.

    Input:  (B, N, C)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        output_dim: int = 32,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Flatten sparse point cloud points/features into one vector per sample.
        x = x.flatten(start_dim=1)
        return self.encoder(x)


# =============================================================================
# Multi-Modal Transformer Encoder
# =============================================================================


class MultiModalTransformerEncoder(nn.Module):
    """Transformer encoder for cross-modal attention over heterogeneous observation tokens.

    Each (modality, timestep) pair is projected to d_model dimensions, augmented with
    learned modality-type embeddings and temporal position embeddings, then processed
    by a standard Transformer encoder.

    Aggregation modes:
        - "mean":    mean-pool all output tokens  → (B, d_model)
        - "cls":     prepend learnable [CLS] token, return its output → (B, d_model)
        - "flatten": flatten all output tokens → (B, num_tokens * d_model)
    """

    def __init__(
        self,
        config: TactilePointnetDiffusionConfig,
        modality_dims: dict[str, int],
    ):
        """
        Args:
            config: Policy configuration.
            modality_dims: Mapping from modality name to its input feature dimension.
                           E.g., {"state": 22, "state_velocity": 22, "tactile": 44, "hand_pc": 256}
        """
        super().__init__()
        d_model = config.transformer_d_model
        self.d_model = d_model
        self.aggregation = config.transformer_aggregation

        # Ordered list of modality names (order matters for modality-type embeddings)
        self.modality_names = list(modality_dims.keys())

        # Per-modality linear projections to d_model
        self.modality_projections = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in modality_dims.items()}
        )

        # Learned modality-type embeddings
        self.modality_type_embeddings = nn.Embedding(len(modality_dims), d_model)

        # Learned temporal position embeddings
        self.temporal_embeddings = nn.Embedding(config.n_obs_steps, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_num_layers
        )

        # Optional [CLS] token
        if self.aggregation == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Layer norm on output
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension (depends on aggregation mode)."""
        if self.aggregation in ("mean", "cls"):
            return self.d_model
        else:
            raise ValueError(
                "output_dim is not statically known for 'flatten' aggregation; "
                "it depends on the number of tokens at runtime."
            )

    def forward(self, modality_data: dict[str, Tensor]) -> Tensor:
        """
        Args:
            modality_data: dict mapping modality name → (B, n_obs_steps, feature_dim) tensor.
                           Keys must be a subset of self.modality_names.
        Returns:
            Aggregated feature tensor. Shape depends on aggregation mode.
        """
        tokens_list: list[Tensor] = []
        batch_size = None

        for idx, name in enumerate(self.modality_names):
            if name not in modality_data:
                continue
            data = modality_data[name]  # (B, T, dim)
            B, T, _ = data.shape
            batch_size = B

            # Project to d_model
            projected = self.modality_projections[name](data)  # (B, T, d_model)

            # Add modality-type embedding (broadcast over batch and time)
            mod_type_ids = torch.full(
                (B, T), idx, device=data.device, dtype=torch.long
            )
            projected = projected + self.modality_type_embeddings(mod_type_ids)

            # Add temporal position embedding
            temporal_ids = torch.arange(T, device=data.device).unsqueeze(0).expand(B, -1)
            projected = projected + self.temporal_embeddings(temporal_ids)

            tokens_list.append(projected)

        # Concatenate all tokens: (B, total_tokens, d_model)
        all_tokens = torch.cat(tokens_list, dim=1)

        # Prepend [CLS] token if needed
        if self.aggregation == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)
            all_tokens = torch.cat([cls, all_tokens], dim=1)

        # Transformer forward
        output = self.transformer(all_tokens)  # (B, total_tokens[+1], d_model)

        # Aggregate
        if self.aggregation == "cls":
            result = output[:, 0]  # (B, d_model)
        elif self.aggregation == "mean":
            result = output.mean(dim=1)  # (B, d_model)
        elif self.aggregation == "flatten":
            result = output.flatten(start_dim=1)  # (B, total_tokens * d_model)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.output_norm(result)


# =============================================================================
# Policy
# =============================================================================


class TactilePointnetDiffusionPolicy(PreTrainedPolicy):
    """Diffusion policy with PointNet + Transformer encoder for tactile/state observations.

    No image input is used. The action is generated by a conditional UNet diffusion model
    whose global conditioning comes from a Transformer encoder that fuses:
    - observation.state
    - observation.state_velocity
    - concatenated tactile data (observation.tactile + observation.fsr)
    - PointNet-encoded hand point cloud (observation.hand_pc)
    """

    config_class = TactilePointnetDiffusionConfig
    name = "tactile_pointnet_diffusion"

    def __init__(self, config: TactilePointnetDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        self.diffusion = TactilePointnetDiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        # Queues for state-like features
        for key in [
            self.config.state_feature_key,
            self.config.state_velocity_feature_key,
        ]:
            if key in self.config.input_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

        if (
            self.config.include_sparse_pc_in_state
            and self.config.sparse_pointcloud_feature_key in self.config.input_features
        ):
            self._queues[self.config.sparse_pointcloud_feature_key] = deque(
                maxlen=self.config.n_obs_steps
            )

        # Queues for tactile features
        for key in self.config.tactile_feature_keys:
            if key in self.config.input_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

        # Queue for point cloud
        if self.config.pointcloud_feature_key in self.config.input_features:
            self._queues[self.config.pointcloud_feature_key] = deque(
                maxlen=self.config.n_obs_steps
            )

        # Optional cube_pos
        if self.config.include_cube_pos and self.config.cube_pos_key in self.config.input_features:
            self._queues[self.config.cube_pos_key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss."""
        loss = self.diffusion.compute_loss(batch)
        return loss, None


# =============================================================================
# Diffusion Model (PointNet + Transformer + UNet)
# =============================================================================


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class TactilePointnetDiffusionModel(nn.Module):
    """Diffusion model combining PointNet, Transformer encoder, and conditional UNet.

    Data flow:
        hand_pc → PointNet → pc_feature
        {state, state_velocity, tactile, pc_feature, [cube_pos]} → Transformer → global_cond
        global_cond + timestep → UNet → denoised action trajectory
    """

    def __init__(self, config: TactilePointnetDiffusionConfig):
        super().__init__()
        self.config = config

        # --- PointNet encoder for hand_pc ---
        self.pointnet = PointNetEncoder(
            input_dim=config.pointnet_input_dim,
            hidden_dims=config.pointnet_hidden_dims,
            output_dim=config.pointnet_output_dim,
            use_batch_norm=config.pointnet_use_batch_norm,
        )

        self.sparse_pc_encoder = None
        if (
            config.include_sparse_pc_in_state
            and config.sparse_pointcloud_feature_key in config.input_features
        ):
            sparse_shape = config.input_features[config.sparse_pointcloud_feature_key].shape
            self.sparse_pc_encoder = SparsePointCloudEncoder(
                input_dim=_prod(sparse_shape),
                hidden_dims=config.sparse_pc_hidden_dims,
                output_dim=config.sparse_pc_output_dim,
            )

        # --- Compute per-modality dimensions for Transformer ---
        modality_dims: dict[str, int] = {}

        # State
        if config.state_feature_key in config.input_features:
            state_shape = config.input_features[config.state_feature_key].shape
            state_dim = _prod(state_shape)
            if (
                config.include_sparse_pc_in_state
                and config.sparse_pointcloud_feature_key in config.input_features
            ):
                state_dim += config.sparse_pc_output_dim
            modality_dims["state"] = state_dim

        # State velocity
        if config.state_velocity_feature_key in config.input_features:
            sv_shape = config.input_features[config.state_velocity_feature_key].shape
            modality_dims["state_velocity"] = _prod(sv_shape)

        # Tactile (concatenated tactile + fsr)
        tactile_dim = 0
        for key in config.tactile_feature_keys:
            if key in config.input_features:
                tactile_dim += _prod(config.input_features[key].shape)
        if tactile_dim > 0:
            modality_dims["tactile"] = tactile_dim

        # PointNet output
        modality_dims["hand_pc"] = config.pointnet_output_dim

        # Optional cube_pos
        if config.include_cube_pos and config.cube_pos_key in config.input_features:
            cube_shape = config.input_features[config.cube_pos_key].shape
            modality_dims["cube_pos"] = _prod(cube_shape)

        self.modality_dims = modality_dims

        # --- Transformer encoder ---
        self.transformer_encoder = MultiModalTransformerEncoder(config, modality_dims)

        # --- Compute global_cond_dim for UNet ---
        if config.transformer_aggregation in ("mean", "cls"):
            global_cond_dim = config.transformer_d_model
        elif config.transformer_aggregation == "flatten":
            # Number of tokens = num_modalities * n_obs_steps [+ 1 for CLS, but cls is separate]
            num_tokens = len(modality_dims) * config.n_obs_steps
            global_cond_dim = num_tokens * config.transformer_d_model
        else:
            raise ValueError(f"Unknown aggregation: {config.transformer_aggregation}")

        # --- UNet ---
        self.unet = ConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        # --- Noise scheduler ---
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

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the global conditioning vector from all observation modalities.

        Steps:
            1. Process hand_pc through PointNet for each observation step.
            2. Concatenate tactile feature keys into a single tactile vector.
            3. Assemble modality_data dict for the Transformer.
            4. Run Transformer encoder → conditioning vector.
        """
        config = self.config
        modality_data: dict[str, Tensor] = {}

        # Determine B and T from any available feature
        B, T = None, None
        for key in [config.state_feature_key, config.state_velocity_feature_key]:
            if key in batch:
                B, T = batch[key].shape[0], batch[key].shape[1]
                break
        if B is None:
            for key in config.tactile_feature_keys:
                if key in batch:
                    B, T = batch[key].shape[0], batch[key].shape[1]
                    break
        if B is None and config.pointcloud_feature_key in batch:
            B, T = batch[config.pointcloud_feature_key].shape[0], batch[config.pointcloud_feature_key].shape[1]
        if B is None and config.sparse_pointcloud_feature_key in batch:
            B, T = batch[config.sparse_pointcloud_feature_key].shape[0], batch[config.sparse_pointcloud_feature_key].shape[1]

        # State → (B, T, state_dim)
        if config.state_feature_key in batch:
            state_data = batch[config.state_feature_key]

            if (
                config.include_sparse_pc_in_state
                and self.sparse_pc_encoder is not None
                and config.sparse_pointcloud_feature_key in batch
            ):
                sparse_pc = batch[config.sparse_pointcloud_feature_key]  # (B, T, N, C)
                B_sp, T_sp, N_sp, C_sp = sparse_pc.shape
                sparse_pc_flat = sparse_pc.reshape(B_sp * T_sp, N_sp, C_sp)
                sparse_features = self.sparse_pc_encoder(sparse_pc_flat)
                sparse_features = sparse_features.reshape(B_sp, T_sp, -1)
                state_data = torch.cat([state_data, sparse_features], dim=-1)

            modality_data["state"] = state_data

        # State velocity → (B, T, sv_dim)
        if config.state_velocity_feature_key in batch:
            modality_data["state_velocity"] = batch[config.state_velocity_feature_key]

        # Tactile: concatenate tactile + fsr → (B, T, tactile_dim)
        tactile_parts = []
        for key in config.tactile_feature_keys:
            if key in batch:
                tactile_parts.append(batch[key])
        if tactile_parts:
            modality_data["tactile"] = torch.cat(tactile_parts, dim=-1)

        # Hand PC: run PointNet per timestep
        # batch[pc_key] has shape (B, T, N, C)
        if config.pointcloud_feature_key in batch:
            pc = batch[config.pointcloud_feature_key]  # (B, T, N, C)
            B_pc, T_pc, N, C = pc.shape
            # Reshape for PointNet: (B*T, N, C)
            pc_flat = pc.reshape(B_pc * T_pc, N, C)
            pc_features = self.pointnet(pc_flat)  # (B*T, pointnet_output_dim)
            pc_features = pc_features.reshape(B_pc, T_pc, -1)  # (B, T, pointnet_output_dim)
            modality_data["hand_pc"] = pc_features

        # Optional cube_pos
        if config.include_cube_pos and config.cube_pos_key in batch:
            modality_data["cube_pos"] = batch[config.cube_pos_key]

        # Transformer encoder → (B, global_cond_dim)
        global_cond = self.transformer_encoder(modality_data)
        return global_cond

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
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample

        return sample

    def generate_actions(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Generate action predictions.

        Expected batch keys:
            Required: observation.state, observation.state_velocity, 
                      observation.tactile, observation.fsr, observation.hand_pc
            Optional: cube_pos (if include_cube_pos=True)
        All with shape (B, n_obs_steps, ...).
        """
        # Determine batch size & n_obs_steps
        batch_size, n_obs_steps = None, None
        for key in [self.config.state_feature_key, self.config.state_velocity_feature_key]:
            if key in batch:
                batch_size = batch[key].shape[0]
                n_obs_steps = batch[key].shape[1]
                break
        if batch_size is None:
            for key in self.config.tactile_feature_keys:
                if key in batch:
                    batch_size = batch[key].shape[0]
                    n_obs_steps = batch[key].shape[1]
                    break
        if batch_size is None and self.config.pointcloud_feature_key in batch:
            batch_size = batch[self.config.pointcloud_feature_key].shape[0]
            n_obs_steps = batch[self.config.pointcloud_feature_key].shape[1]
        if batch_size is None and self.config.sparse_pointcloud_feature_key in batch:
            batch_size = batch[self.config.sparse_pointcloud_feature_key].shape[0]
            n_obs_steps = batch[self.config.sparse_pointcloud_feature_key].shape[1]

        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute training loss.

        Expected batch:
            Same observation keys as generate_actions, plus:
            - "action": (B, horizon, action_dim)
            - "action_is_pad": (B, horizon)
        """
        assert ACTION in batch
        assert "action_is_pad" in batch

        # Determine n_obs_steps from available features
        n_obs_steps = None
        for key in [self.config.state_feature_key, self.config.state_velocity_feature_key]:
            if key in batch:
                n_obs_steps = batch[key].shape[1]
                break
        if n_obs_steps is None:
            for key in self.config.tactile_feature_keys:
                if key in batch:
                    n_obs_steps = batch[key].shape[1]
                    break
        if n_obs_steps is None and self.config.pointcloud_feature_key in batch:
            n_obs_steps = batch[self.config.pointcloud_feature_key].shape[1]
        if n_obs_steps is None and self.config.sparse_pointcloud_feature_key in batch:
            n_obs_steps = batch[self.config.sparse_pointcloud_feature_key].shape[1]

        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

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
# Helper: product of shape
# =============================================================================


def _prod(shape: tuple[int, ...] | list[int]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for s in shape:
        result *= s
    return result


# =============================================================================
# UNet and building blocks (shared with diffusion_baseline)
# =============================================================================


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

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
    """Conv1d → GroupNorm → Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1D convolutional block with FiLM modulation for conditioning."""

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
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
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


class ConditionalUnet1d(nn.Module):
    """1D convolutional UNet with FiLM modulation for conditioning.

    This is the same UNet architecture as in diffusion_baseline, used for
    diffusion-based action denoising.
    """

    def __init__(self, config: TactilePointnetDiffusionConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # Timestep encoder
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # FiLM conditioning dimension
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # UNet encoder
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

        # Mid modules
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

        # UNet decoder
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
        """
        Args:
            x: (B, T, input_dim)
            timestep: (B,) diffusion timestep
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, input_dim)
        """
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
