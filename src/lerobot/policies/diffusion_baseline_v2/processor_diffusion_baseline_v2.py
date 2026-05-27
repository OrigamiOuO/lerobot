"""Processor for DiffusionBaselineV2 — delegates to the baseline processor."""

from lerobot.policies.diffusion_baseline.processor_diffusion_baseline import (
    make_diffusion_baseline_pre_post_processors,
)


def make_diffusion_baseline_v2_pre_post_processors(
    config, dataset_stats=None
):
    """DiffusionBaselineV2 uses the same processor as DiffusionBaseline."""
    return make_diffusion_baseline_pre_post_processors(config, dataset_stats=dataset_stats)
