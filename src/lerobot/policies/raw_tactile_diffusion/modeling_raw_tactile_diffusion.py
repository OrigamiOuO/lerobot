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
"""Raw Tactile Diffusion Policy.

Architecture:
    1. Multi-modal state features (observation.state, observation.state_velocity,
       observation.tactile, observation.fsr) are concatenated per timestep
    2. The sequence of concatenated states (B, n_obs_steps, composite_state_dim)
       is flattened and passed through an MLP encoder to produce a global
       conditioning vector
    3. A 1D Conditional UNet denoises the action trajectory conditioned on
       the encoded state embedding and diffusion timestep

No image input, no point cloud processing, no Transformer cross-attention.
This is a lightweight policy designed for tactile-rich manipulation tasks.
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
from lerobot.policies.raw_tactile_diffusion.configuration_raw_tactile_diffusion import (
    RawTactileDiffusionConfig,
)
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION


# =============================================================================
# State Encoder MLP
# =============================================================================


class StateMLPEncoder(nn.Module):
    """Simple MLP encoder for concatenated multi-modal state features.

    Takes the flattened sequence of composite state vectors and encodes it
    into a fixed-size conditioning embedding for the UNet.

    Input:  (B, composite_state_dim * n_obs_steps)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        output_dim: int = 256,
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
        """
        Args:
            x: (B, input_dim) flattened composite state.
        Returns:
            (B, output_dim) encoded state embedding.
        """
        return self.encoder(x)


# =============================================================================
# Policy
# =============================================================================


class RawTactileDiffusionPolicy(PreTrainedPolicy):
    """Diffusion policy using raw tactile and state observations (no images, no point cloud).

    Multi-modal state features are concatenated and encoded by an MLP, which produces
    a global conditioning vector for the 1D Conditional UNet diffusion denoiser.
    """

    config_class = RawTactileDiffusionConfig
    name = "raw_tactile_diffusion"

    def __init__(self, config: RawTactileDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Compute composite state dimension
        self.composite_state_dim = config.compute_composite_state_dim()

        # Queues for rollout
        self._queues = None

        self.diffusion = RawTactileDiffusionModel(config, composite_state_dim=self.composite_state_dim)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        # Add queues for all configured state features
        for key in self.config.state_feature_keys:
            if key in self.config.input_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

        # Optional cube_pos
        if self.config.include_cube_pos and self.config.cube_pos_key in self.config.input_features:
            self._queues[self.config.cube_pos_key] = deque(maxlen=self.config.n_obs_steps)

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
# Diffusion Model
# =============================================================================


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    """Factory for noise scheduler instances."""
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class RawTactileDiffusionModel(nn.Module):
    """Diffusion model with MLP-based state encoding and conditional UNet.

    Data flow:
        {state, state_velocity, tactile, fsr, [cube_pos]} → concatenate → composite_state
        composite_state → flatten temporal → MLP encoder → global_cond
        global_cond + timestep → UNet → denoised action trajectory
    """

    def __init__(self, config: RawTactileDiffusionConfig, composite_state_dim: int):
        super().__init__()
        self.config = config
        self.composite_state_dim = composite_state_dim

        # MLP state encoder
        # Input: (B, composite_state_dim * n_obs_steps) — flattened temporal + feature dims
        encoder_input_dim = composite_state_dim * config.n_obs_steps
        self.state_encoder = StateMLPEncoder(
            input_dim=encoder_input_dim,
            hidden_dims=config.state_encoder_hidden_dims,
            output_dim=config.state_embed_dim,
        )
        global_cond_dim = config.state_embed_dim

        # Conditional UNet
        self.unet = ConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        # Noise scheduler
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

    def _concatenate_state_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Concatenate multiple state features in the configured order.

        Args:
            batch: Dictionary containing state features.

        Returns:
            Concatenated state tensor of shape (B, n_obs_steps, composite_state_dim).
        """
        state_parts = []
        for key in self.config.state_feature_keys:
            if key in batch:
                state_parts.append(batch[key])

        if self.config.include_cube_pos and self.config.cube_pos_key in batch:
            state_parts.append(batch[self.config.cube_pos_key])

        if not state_parts:
            raise ValueError("No state features found in batch!")

        return torch.cat(state_parts, dim=-1)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode the composite state into a global conditioning vector.

        Args:
            batch: Dictionary containing state features with shape (B, n_obs_steps, ...).

        Returns:
            (B, state_embed_dim) global conditioning vector.
        """
        # Concatenate state features → (B, n_obs_steps, composite_state_dim)
        composite_state = self._concatenate_state_features(batch)

        # Flatten temporal dimension → (B, composite_state_dim * n_obs_steps)
        flat_state = composite_state.flatten(start_dim=1)

        # Encode → (B, state_embed_dim)
        global_cond = self.state_encoder(flat_state)
        return global_cond

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Run the denoising diffusion process to sample an action trajectory."""
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

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate action predictions.

        Expected batch keys (subset of):
            - observation.state: (B, n_obs_steps, 22)
            - observation.state_velocity: (B, n_obs_steps, 22)
            - observation.tactile: (B, n_obs_steps, 32)
            - observation.fsr: (B, n_obs_steps, 12)
            - cube_pos: (B, n_obs_steps, 7) [optional]

        Returns:
            (B, n_action_steps, action_dim) action trajectory.
        """
        batch_size, n_obs_steps = self._get_batch_dims(batch)
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

        _, n_obs_steps = self._get_batch_dims(batch)
        horizon = batch[ACTION].shape[1]

        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode state into global conditioning
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

    def _get_batch_dims(self, batch: dict[str, Tensor]) -> tuple[int, int]:
        """Extract batch_size and n_obs_steps from available features."""
        batch_size, n_obs_steps = None, None

        for key in self.config.state_feature_keys:
            if key in batch:
                batch_size = batch[key].shape[0]
                n_obs_steps = batch[key].shape[1]
                break

        if batch_size is None and self.config.include_cube_pos and self.config.cube_pos_key in batch:
            batch_size = batch[self.config.cube_pos_key].shape[0]
            n_obs_steps = batch[self.config.cube_pos_key].shape[1]

        if batch_size is None:
            raise ValueError("Could not determine batch dimensions from batch keys.")

        return batch_size, n_obs_steps


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

    def __init__(self, config: RawTactileDiffusionConfig, global_cond_dim: int):
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
            x: (B, T, input_dim) input to the UNet.
            timestep: (B,) diffusion timestep.
            global_cond: (B, global_cond_dim) conditioning vector.
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Encoder
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x
