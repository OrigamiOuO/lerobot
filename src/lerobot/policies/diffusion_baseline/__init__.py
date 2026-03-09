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
Diffusion Baseline Policy with Multi-Modal State Concatenation.

This policy combines the diffusion-based action prediction approach with 
multi-modal state feature concatenation (similar to act_baseline).
"""

from lerobot.policies.diffusion_baseline.configuration_diffusion_baseline import DiffusionBaselineConfig
from lerobot.policies.diffusion_baseline.modeling_diffusion_baseline import DiffusionBaselinePolicy
from lerobot.policies.diffusion_baseline.processor_diffusion_baseline import make_diffusion_baseline_pre_post_processors

__all__ = [
    "DiffusionBaselineConfig",
    "DiffusionBaselinePolicy",
    "make_diffusion_baseline_pre_post_processors",
]
