#!/usr/bin/env python

# Copyright 2024 Custom Policy Implementation
# Training script for Tactile Diffusion Policy

"""This script demonstrates how to train Tactile Diffusion Policy on tactile sensor dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.tactile_diffusion.configuration_tactile_diffusion import TactileDiffusionConfig
from lerobot.policies.tactile_diffusion.modeling_tactile_diffusion import TactileDiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/tactile_diffusion_test")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Number of offline training steps
    # Adjust as you prefer. Start with a small number for testing.
    training_steps = 1000
    log_freq = 100

    # Dataset path (local dataset)
    # Use both repo_id and root for local datasets
    repo_id = "tactile_dp_test_data/xarm_leap_tactile_lift_blind"
    dataset_root = "datasets/tactile_dp_test_data/xarm_leap_tactile_lift_blind"
    
    # Load dataset metadata to get feature information
    print(f"Loading dataset from: {dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    
    # Extract features from metadata
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print("\n=== Dataset Information ===")
    print(f"Total episodes: {dataset_metadata.total_episodes}")
    print(f"Total frames: {dataset_metadata.total_frames}")
    print(f"FPS: {dataset_metadata.fps}")
    print(f"\nInput features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    
    # Verify tactile features are present
    assert "observation.tactile_fsr" in input_features, "Missing tactile_fsr feature"
    assert "observation.tactile_taxel" in input_features, "Missing tactile_taxel feature"
    print("\n✓ Tactile features detected")

    # Create policy configuration with tactile support
    # Using the same temporal parameters as standard DiffusionPolicy
    cfg = TactileDiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        # Temporal parameters
        n_obs_steps=2,           # Look at 2 observation steps
        horizon=16,              # Generate 16 action steps
        n_action_steps=8,        # Execute 8 actions per policy call
        drop_n_last_frames=7,    # horizon - n_action_steps - n_obs_steps + 1
        # Tactile-specific parameters
        use_tactile_features=True,
        tactile_encoder_hidden_dim=64,
        # Training parameters (adjust as needed)
        optimizer_lr=1e-4,
    )
    
    print("\n=== Policy Configuration ===")
    print(f"n_obs_steps: {cfg.n_obs_steps}")
    print(f"horizon: {cfg.horizon}")
    print(f"n_action_steps: {cfg.n_action_steps}")
    print(f"use_tactile_features: {cfg.use_tactile_features}")
    print(f"tactile_encoder_hidden_dim: {cfg.tactile_encoder_hidden_dim}")

    # Instantiate the policy
    print("\n=== Initializing Policy ===")
    policy = TactileDiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create preprocessor and postprocessor
    print("\n=== Creating Data Processors ===")
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, 
        dataset_stats=dataset_metadata.stats
    )
    print("✓ Preprocessor and postprocessor created")

    # Configure delta_timestamps for the dataset
    # This tells the dataset which frames to load relative to the current frame
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.tactile_fsr": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.tactile_taxel": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    
    print("\n=== Delta Timestamps ===")
    print(f"observation.state: {delta_timestamps['observation.state']}")
    print(f"observation.tactile_fsr: {delta_timestamps['observation.tactile_fsr']}")
    print(f"observation.tactile_taxel: {delta_timestamps['observation.tactile_taxel']}")
    print(f"action: {delta_timestamps['action'][:5]}... (showing first 5 of {len(delta_timestamps['action'])})")

    # Load the dataset
    print("\n=== Loading Dataset ===")
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create optimizer and dataloader
    print("\n=== Setting up Training ===")
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=32,  # Adjust based on your GPU memory
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    print(f"Batch size: 32")
    print(f"Number of batches per epoch: {len(dataloader)}")

    # Run training loop
    print("\n=== Starting Training ===")
    print(f"Training for {training_steps} steps...")
    print("-" * 60)
    
    step = 0
    done = False
    epoch = 0
    
    while not done:
        epoch += 1
        for batch in dataloader:
            # Preprocess the batch
            batch = preprocessor(batch)
            
            # Forward pass
            loss, _ = policy.forward(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if step % log_freq == 0:
                print(f"Epoch {epoch:3d} | Step {step:5d} | Loss: {loss.item():.6f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    print("-" * 60)
    print(f"\n=== Training Complete ===")
    print(f"Final loss: {loss.item():.6f}")

    # Save the policy checkpoint
    print("\n=== Saving Checkpoint ===")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"✓ Checkpoint saved to: {output_directory}")
    
    print("\n=== Training Summary ===")
    print(f"Device: {device}")
    print(f"Total steps: {step}")
    print(f"Total epochs: {epoch}")
    print(f"Dataset: {dataset_root}")
    print(f"Output directory: {output_directory}")
    print("\n✓ Training script completed successfully!")


if __name__ == "__main__":
    main()