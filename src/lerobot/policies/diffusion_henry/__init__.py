# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion-Henry: Diffusion Policy with tactile sensor support."""

from lerobot.policies.diffusion_henry.configuration_diffusion_henry import DiffusionHenryConfig
from lerobot.policies.diffusion_henry.modeling_diffusion_henry import DiffusionHenryPolicy

__all__ = ["DiffusionHenryConfig", "DiffusionHenryPolicy"]
