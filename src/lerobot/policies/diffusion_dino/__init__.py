# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion-Dino: DINOv2-based Diffusion Policy with tactile sensor support."""

from lerobot.policies.diffusion_dino.configuration_diffusion_dino import DiffusionDinoConfig
from lerobot.policies.diffusion_dino.modeling_diffusion_dino import DiffusionDinoPolicy

__all__ = ["DiffusionDinoConfig", "DiffusionDinoPolicy"]
