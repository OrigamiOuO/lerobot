"""Processor for DiffusionBaselineV3 — delegates to the baseline processor."""

from lerobot.policies.diffusion_baseline.processor_diffusion_baseline import (
    make_diffusion_baseline_pre_post_processors,
)


def make_diffusion_baseline_v3_pre_post_processors(
    config, dataset_stats=None
):
    """DiffusionBaselineV3 uses the same processor as DiffusionBaseline."""
    return make_diffusion_baseline_pre_post_processors(config, dataset_stats=dataset_stats)
