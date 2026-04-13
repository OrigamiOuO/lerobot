#!/usr/bin/env python
"""
Test script for PretrainDiffusionPolicy.

Given a dataset path and a checkpoint path, randomly samples data points from the
dataset and runs inference through the pretrain_diffusion policy, comparing predicted
actions against ground-truth actions.

Two test modes:
  1. Batch inference: take a full dataset item (with delta_timestamps giving
     [n_obs_steps, ...] observations and [horizon, ...] actions), add batch dim,
     and call generate_actions / compute_loss directly.
  2. Sequential select_action: feed single frames one by one through select_action
     (mimicking the eval loop).

Usage:
    python tests/pretrain_dp_test.py \
        --repo-id grasp_multi \
        --dataset-root datasets/luo_proj/Blind_Grasping_LeRobot_Multimodal \
        --ckpt-path checkpoints/luo_proj/pretrain_dp_v2/checkpoints/020000/pretrained_model \
        --num-samples 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import torch
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
# Import config class FIRST to trigger @register_subclass decorator
from lerobot.policies.pretrain_diffusion.configuration_pretrain_diffusion import PretrainDiffusionConfig
from lerobot.policies.pretrain_diffusion.modeling_pretrain_diffusion import PretrainDiffusionPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION

# ── observation keys used by pretrain_diffusion ──────────────────────────────
OBS_STATE = "observation.state"
OBS_STATE_VELOCITY = "observation.state_velocity"
OBS_TACTILE = "observation.tactile"
OBS_FSR = "observation.fsr"
OBS_SPARSE_PC = "observation.sparse_pc"

OBS_KEYS = [OBS_STATE, OBS_STATE_VELOCITY, OBS_TACTILE, OBS_FSR, OBS_SPARSE_PC]


def parse_args():
    parser = argparse.ArgumentParser(description="Test pretrain_diffusion policy on dataset samples")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id (e.g. grasp_multi)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Local root path for the dataset")
    parser.add_argument(
        "--ckpt-path", type=str, required=True,
        help="Path to the pretrained_model directory containing config.json and model.safetensors",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_policy(ckpt_path: str, ds_meta: LeRobotDatasetMetadata, device: str):
    """Load the pretrain_diffusion policy from a checkpoint."""
    # Load config from checkpoint (base class handles 'type' dispatch)
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    assert isinstance(config, PretrainDiffusionConfig), (
        f"Expected PretrainDiffusionConfig, got {type(config)}"
    )
    config.device = device

    # Set up input/output features from dataset metadata
    features = dataset_to_policy_features(ds_meta.features)
    config.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {key: ft for key, ft in features.items() if key not in config.output_features}

    # Load model weights
    policy = PretrainDiffusionPolicy.from_pretrained(
        ckpt_path, config=config, dataset_stats=ds_meta.stats, dataset_meta=ds_meta,
    )
    policy.to(device)
    policy.eval()
    return policy, config


def build_normalized_batch(
    item: dict[str, torch.Tensor],
    config: PretrainDiffusionConfig,
    ds_meta: LeRobotDatasetMetadata,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Build a normalized batch from a dataset item for direct model inference.

    Takes the raw dataset item (with delta_timestamps giving [T, ...] observations),
    applies MIN_MAX normalization using dataset stats, and adds a batch dimension.
    Returns a batch dict with shapes [1, T, ...].
    """
    stats = ds_meta.stats
    batch = {}

    for key in list(OBS_KEYS) + [ACTION]:
        if key not in item:
            continue
        val = item[key].float()

        # Apply MIN_MAX normalization: x_norm = (x - min) / (max - min) * 2 - 1
        if key in stats and "min" in stats[key] and "max" in stats[key]:
            s_min = torch.as_tensor(stats[key]["min"], dtype=torch.float32)
            s_max = torch.as_tensor(stats[key]["max"], dtype=torch.float32)
            val = (val - s_min) / (s_max - s_min + 1e-8) * 2 - 1

        batch[key] = val.unsqueeze(0).to(device)  # [1, T, ...]

    # action_is_pad
    if "action_is_pad" in item:
        batch["action_is_pad"] = item["action_is_pad"].unsqueeze(0).to(device)
    else:
        batch["action_is_pad"] = torch.zeros(
            1, config.horizon, dtype=torch.bool, device=device
        )

    return batch


def unnormalize_action(action: torch.Tensor, ds_meta: LeRobotDatasetMetadata) -> torch.Tensor:
    """Reverse MIN_MAX normalization for actions."""
    stats = ds_meta.stats
    if ACTION in stats and "min" in stats[ACTION] and "max" in stats[ACTION]:
        s_min = torch.as_tensor(stats[ACTION]["min"], dtype=torch.float32).to(action.device)
        s_max = torch.as_tensor(stats[ACTION]["max"], dtype=torch.float32).to(action.device)
        return (action + 1) / 2 * (s_max - s_min + 1e-8) + s_min
    return action


def test_batch_inference(
    policy: PretrainDiffusionPolicy,
    dataset: LeRobotDataset,
    sample_idx: int,
    config: PretrainDiffusionConfig,
    ds_meta: LeRobotDatasetMetadata,
    device: str,
):
    """
    Test direct batch-level inference: build a full batch from the dataset item,
    run generate_actions on it, and compare with ground truth.
    """
    item = dataset[sample_idx]
    batch = build_normalized_batch(item, config, ds_meta, device)

    # Log shapes for the first sample
    for k, v in batch.items():
        if hasattr(v, "shape"):
            logging.debug(f"  batch[{k}]: {v.shape}")

    # ── generate_actions (inference) ──
    policy.eval()
    with torch.inference_mode():
        actions_norm = policy.diffusion.generate_actions(batch)
    # actions_norm: [1, n_action_steps, action_dim] (normalized)
    actions_unnorm = unnormalize_action(actions_norm, ds_meta)

    # Ground-truth: raw (unnormalized) from dataset
    gt_actions_raw = item[ACTION]  # [horizon, action_dim]
    n_obs = config.n_obs_steps
    n_act = config.n_action_steps
    gt_chunk = gt_actions_raw[n_obs - 1 : n_obs - 1 + n_act]  # [n_action_steps, action_dim]

    return actions_unnorm.squeeze(0).cpu(), gt_chunk.cpu()


def test_select_action(
    policy: PretrainDiffusionPolicy,
    dataset: LeRobotDataset,
    sample_idx: int,
    config: PretrainDiffusionConfig,
    ds_meta: LeRobotDatasetMetadata,
    device: str,
):
    """
    Test select_action by feeding single frames one-by-one (eval-loop style).

    The dataset item with delta_timestamps gives [n_obs_steps, ...] observations.
    We feed each timestep as if it came from an environment step.
    """
    policy.eval()
    policy.reset()

    item = dataset[sample_idx]
    stats = ds_meta.stats

    n_obs = config.n_obs_steps
    predicted_action = None

    for t in range(n_obs):
        single_obs = {}
        for key in OBS_KEYS:
            if key not in item:
                continue
            val = item[key][t].float()  # single frame

            # Normalize
            if key in stats and "min" in stats[key] and "max" in stats[key]:
                s_min = torch.as_tensor(stats[key]["min"], dtype=torch.float32)
                s_max = torch.as_tensor(stats[key]["max"], dtype=torch.float32)
                val = (val - s_min) / (s_max - s_min + 1e-8) * 2 - 1

            # Add batch dim and move to device
            single_obs[key] = val.unsqueeze(0).to(device)

        with torch.inference_mode():
            predicted_action = policy.select_action(single_obs)

    # predicted_action: [1, action_dim] (normalized)
    pred_unnorm = unnormalize_action(predicted_action, ds_meta)

    # GT: first action after current observation
    gt_action = item[ACTION][n_obs - 1].cpu()  # single action

    return pred_unnorm.squeeze(0).cpu(), gt_action


def compute_metrics(predicted: torch.Tensor, ground_truth: torch.Tensor):
    """Compute L1, L2, and max absolute error between predicted and GT actions."""
    pred = predicted.detach().float()
    gt = ground_truth.detach().float()
    l1 = torch.abs(pred - gt).mean().item()
    l2 = torch.sqrt(((pred - gt) ** 2).mean()).item()
    max_err = torch.abs(pred - gt).max().item()
    return {"l1": l1, "l2": l2, "max_abs": max_err}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # ── 1. Load dataset metadata ─────────────────────────────────────────────
    logging.info(f"Loading dataset metadata: repo_id={args.repo_id}, root={args.dataset_root}")
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    logging.info(f"Dataset: {ds_meta.total_episodes} episodes, {ds_meta.total_frames} frames, fps={ds_meta.fps}")
    logging.info(f"Features: {list(ds_meta.features.keys())}")

    # Verify required features exist
    for key in OBS_KEYS:
        if key not in ds_meta.features:
            logging.error(f"Required feature '{key}' not found. Available: {list(ds_meta.features.keys())}")
            sys.exit(1)

    # ── 2. Load policy ───────────────────────────────────────────────────────
    logging.info(f"Loading policy from: {args.ckpt_path}")
    policy, config = load_policy(args.ckpt_path, ds_meta, device)
    logging.info(f"Config: n_obs_steps={config.n_obs_steps}, horizon={config.horizon}, "
                 f"n_action_steps={config.n_action_steps}")

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Model params: {num_params:,} total, {num_trainable:,} trainable")

    # ── 3. Build dataset with delta_timestamps ───────────────────────────────
    logging.info("Loading dataset with delta_timestamps ...")
    delta_timestamps = resolve_delta_timestamps(config, ds_meta)
    for k, v in delta_timestamps.items():
        logging.info(f"  {k}: {v}")

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    logging.info(f"Dataset: {len(dataset)} frames available")

    # ── 4. Sample random indices (avoid episode boundaries) ──────────────────
    num_samples = min(args.num_samples, len(dataset))
    safe_indices = []
    for ep_idx in range(ds_meta.total_episodes):
        ep_data = ds_meta.episodes[ep_idx]
        ep_start = ep_data["dataset_from_index"]
        ep_end = ep_data["dataset_to_index"]
        ep_len = ep_end - ep_start
        margin = config.n_obs_steps + config.horizon
        if ep_len > 2 * margin:
            safe_indices.extend(range(ep_start + margin, ep_end - margin))

    if not safe_indices:
        logging.warning("No safe indices found, using all dataset indices")
        safe_indices = list(range(len(dataset)))

    sample_indices = random.sample(safe_indices, min(num_samples, len(safe_indices)))
    logging.info(f"Testing on {len(sample_indices)} samples: {sample_indices}")

    # ── 5. Test: batch-level generate_actions ────────────────────────────────
    logging.info("=" * 60)
    logging.info("TEST 1: Batch-level generate_actions")
    logging.info("=" * 60)
    batch_metrics = []
    for i, idx in enumerate(sample_indices):
        pred, gt = test_batch_inference(policy, dataset, idx, config, ds_meta, device)
        m = compute_metrics(pred, gt)
        batch_metrics.append(m)
        logging.info(
            f"  [{i+1}/{len(sample_indices)}] idx={idx} | "
            f"L1={m['l1']:.6f}  L2={m['l2']:.6f}  MaxAbs={m['max_abs']:.6f}"
        )
        if i == 0:
            logging.info(f"    Pred shape: {pred.shape}")
            logging.info(f"    Pred[0]: {pred[0].numpy().round(4)}")
            logging.info(f"    GT[0]:   {gt[0].numpy().round(4)}")

    avg = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]}
    logging.info(f"  >> Avg L1={avg['l1']:.6f}  L2={avg['l2']:.6f}  MaxAbs={avg['max_abs']:.6f}")

    # ── 6. Test: sequential select_action (eval-loop style) ──────────────────
    logging.info("=" * 60)
    logging.info("TEST 2: Sequential select_action (eval-loop style)")
    logging.info("=" * 60)
    sa_metrics = []
    for i, idx in enumerate(sample_indices):
        pred, gt = test_select_action(policy, dataset, idx, config, ds_meta, device)
        m = compute_metrics(pred, gt)
        sa_metrics.append(m)
        logging.info(
            f"  [{i+1}/{len(sample_indices)}] idx={idx} | "
            f"L1={m['l1']:.6f}  L2={m['l2']:.6f}  MaxAbs={m['max_abs']:.6f}"
        )
        if i == 0:
            logging.info(f"    Pred: {pred.numpy().round(4)}")
            logging.info(f"    GT:   {gt.numpy().round(4)}")

    avg = {k: np.mean([m[k] for m in sa_metrics]) for k in sa_metrics[0]}
    logging.info(f"  >> Avg L1={avg['l1']:.6f}  L2={avg['l2']:.6f}  MaxAbs={avg['max_abs']:.6f}")

    # ── 7. Test: forward pass (training loss) ────────────────────────────────
    logging.info("=" * 60)
    logging.info("TEST 3: Forward pass (compute_loss)")
    logging.info("=" * 60)
    policy.train()
    test_idx = sample_indices[0]
    batch = build_normalized_batch(dataset[test_idx], config, ds_meta, device)
    with torch.inference_mode():
        loss, _ = policy.forward(batch)
    logging.info(f"  Training loss on idx={test_idx}: {loss.item():.6f}")

    logging.info("=" * 60)
    logging.info("All tests completed successfully!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
