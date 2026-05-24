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
"""
Raw Tactile Diffusion Policy.

A lightweight diffusion policy that uses only raw tactile and state observations
(no images, no point cloud). Multi-modal state features are concatenated and
encoded by an MLP to condition a 1D Conditional UNet for action denoising.
"""

from lerobot.policies.raw_tactile_diffusion.configuration_raw_tactile_diffusion import (
    RawTactileDiffusionConfig,
)
from lerobot.policies.raw_tactile_diffusion.modeling_raw_tactile_diffusion import (
    RawTactileDiffusionPolicy,
)
from lerobot.policies.raw_tactile_diffusion.processor_raw_tactile_diffusion import (
    make_raw_tactile_diffusion_pre_post_processors,
)

__all__ = [
    "RawTactileDiffusionConfig",
    "RawTactileDiffusionPolicy",
    "make_raw_tactile_diffusion_pre_post_processors",
]
