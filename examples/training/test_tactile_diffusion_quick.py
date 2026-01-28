#!/usr/bin/env python

# Copyright 2024 Custom Policy Implementation
# Quick test script for Tactile Diffusion Policy

"""Quick test to verify Tactile Diffusion Policy can load data and perform forward pass."""

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.tactile_diffusion.configuration_tactile_diffusion import TactileDiffusionConfig
from lerobot.policies.tactile_diffusion.modeling_tactile_diffusion import TactileDiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    print("=" * 70)
    print("Tactile Diffusion Policy - Quick Test")
    print("=" * 70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/7] Device: {device}")

    # Dataset path (local dataset)
    # Use both repo_id and root for local datasets
    repo_id = "tactile_dp_test_data/xarm_leap_tactile_lift_blind"
    dataset_root = "datasets/tactile_dp_test_data/xarm_leap_tactile_lift_blind"
    print(f"[2/7] Loading dataset from: {dataset_root}")
    
    # Load metadata
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"      - Episodes: {dataset_metadata.total_episodes}")
    print(f"      - Frames: {dataset_metadata.total_frames}")
    print(f"      - Input features: {list(input_features.keys())}")
    
    # Verify tactile features
    assert "observation.tactile_fsr" in input_features
    assert "observation.tactile_taxel" in input_features
    print(f"      ✓ Tactile features found")

    # Create configuration
    print(f"[3/7] Creating policy configuration...")
    cfg = TactileDiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        use_tactile_features=True,
        tactile_encoder_hidden_dim=64,
    )
    print(f"      ✓ Configuration created")

    # Create policy
    print(f"[4/7] Initializing policy...")
    policy = TactileDiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"      ✓ Policy initialized ({total_params:,} parameters)")

    # Create processors
    print(f"[5/7] Creating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, 
        dataset_stats=dataset_metadata.stats
    )
    print(f"      ✓ Processors created")

    # Load dataset
    print(f"[6/7] Loading dataset...")
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.tactile_fsr": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.tactile_taxel": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"      ✓ Dataset loaded ({len(dataset)} samples)")

    # Test forward pass
    print(f"[7/7] Testing forward pass...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    
    batch = next(iter(dataloader))
    print(f"      - Batch keys: {list(batch.keys())}")
    print(f"      - observation.state shape: {batch['observation.state'].shape}")
    print(f"      - observation.tactile_fsr shape: {batch['observation.tactile_fsr'].shape}")
    print(f"      - observation.tactile_taxel shape: {batch['observation.tactile_taxel'].shape}")
    print(f"      - action shape: {batch['action'].shape}")
    
    # Preprocess
    batch = preprocessor(batch)
    
    # Forward pass
    with torch.no_grad():
        loss, _ = policy.forward(batch)
    
    print(f"      ✓ Forward pass successful")
    print(f"      - Loss: {loss.item():.6f}")
    
    # Test action prediction via select_action (which calls predict_action_chunk internally)
    print(f"\n[Bonus] Testing action prediction...")
    
    policy.reset()  # Reset queues for inference
    
    # Create a single observation batch for inference (only tensor keys)
    single_batch = {k: v[0:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    with torch.no_grad():
        action = policy.select_action(single_batch)
    
    print(f"      ✓ Action prediction successful")
    print(f"      - Predicted action shape: {action.shape}")
    print(f"      - Expected shape: (action_dim={cfg.action_feature.shape[0]},)")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! Tactile Diffusion Policy is working correctly.")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run full training: python examples/training/train_tactile_diffusion.py")
    print("  - Monitor training progress and adjust hyperparameters as needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
