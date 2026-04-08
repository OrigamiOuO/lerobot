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
"""Tactile PointNet Diffusion Policy.

Architecture: PointNet + Transformer Encoder + Conditional UNet Diffusion.
No image input. Fuses state, tactile, and point cloud modalities via Transformer
cross-modal attention.
"""

from lerobot.policies.tactile_pointnet_diffusion.configuration_tactile_pointnet_diffusion import (
    TactilePointnetDiffusionConfig,
)
from lerobot.policies.tactile_pointnet_diffusion.modeling_tactile_pointnet_diffusion import (
    TactilePointnetDiffusionPolicy,
)
from lerobot.policies.tactile_pointnet_diffusion.processor_tactile_pointnet_diffusion import (
    make_tactile_pointnet_diffusion_pre_post_processors,
)

__all__ = [
    "TactilePointnetDiffusionConfig",
    "TactilePointnetDiffusionPolicy",
    "make_tactile_pointnet_diffusion_pre_post_processors",
]
