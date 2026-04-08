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
"""Processor pipeline for Diffusion Baseline Policy with multi-modal state support."""

from typing import Any

import torch

from lerobot.policies.diffusion_baseline.configuration_diffusion_baseline import DiffusionBaselineConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_ENV_STATE
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class KeepObservationKeysProcessorStep(ObservationProcessorStep):
    """Drop observation entries that are not consumed by the policy."""

    def __init__(self, keep_keys: list[str]):
        self.keep_keys = set(keep_keys)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in observation.items() if k in self.keep_keys}

    def transform_features(self, features: dict) -> dict:
        transformed = dict(features)
        if "observation" in transformed:
            transformed_observation = dict(transformed["observation"])
            transformed_observation = {
                k: v for k, v in transformed_observation.items() if k in self.keep_keys
            }
            transformed["observation"] = transformed_observation
        return transformed


def make_diffusion_baseline_pre_post_processors(
    config: DiffusionBaselineConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion baseline policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion baseline policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    required_observation_keys = [
        key for key in config.state_feature_keys if key in config.input_features
    ]
    required_observation_keys.extend(list(config.image_features.keys()))
    if config.env_state_feature is not None:
        required_observation_keys.append(OBS_ENV_STATE)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        KeepObservationKeysProcessorStep(keep_keys=required_observation_keys),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
