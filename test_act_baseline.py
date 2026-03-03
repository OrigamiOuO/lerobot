#!/usr/bin/env python
"""
Test script to verify ACT Baseline policy implementation.

This script tests:
1. Configuration loading and feature dimension computation
2. Policy initialization
3. Forward pass with dummy data
"""

import sys
import torch

# Add src to path
sys.path.insert(0, '/home/user/Code/lerobot/src')

from lerobot.policies.act_baseline import ACTBaselineConfig, ACTBaselinePolicy
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_STATE, ACTION


def test_configuration():
    """Test configuration creation and feature dimension computation."""
    print("\n" + "="*60)
    print("TEST 1: Configuration and Feature Dimensions")
    print("="*60)
    
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(22,)),
        "observation.state_velocity": PolicyFeature(type=FeatureType.STATE, shape=(22,)),
        "observation.tactile": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        "observation.fsr": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    
    config = ACTBaselineConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )
    
    composite_dim = config.compute_composite_state_dim()
    
    print(f"✓ Configuration created successfully")
    print(f"\nFeature breakdown:")
    print(f"  - observation.state:           22 dims")
    print(f"  - observation.state_velocity:  22 dims")
    print(f"  - observation.tactile:         32 dims")
    print(f"  - observation.fsr:             12 dims")
    print(f"  {'─'*35}")
    print(f"  Total composite state:        {composite_dim} dims")
    
    assert composite_dim == 88, f"Expected 88, got {composite_dim}"
    print(f"\n✓ Composite dimension calculation verified")
    
    return config


def test_policy_initialization(config):
    """Test policy model initialization."""
    print("\n" + "="*60)
    print("TEST 2: Policy Model Initialization")
    print("="*60)
    
    policy = ACTBaselinePolicy(config)
    
    print(f"✓ ACTBaselinePolicy initialized successfully")
    print(f"\nModel architecture:")
    print(f"  - Policy class: {type(policy).__name__}")
    print(f"  - Model class: {type(policy.model).__name__}")
    print(f"  - Composite state dim: {policy.composite_state_dim}")
    print(f"  - Chunk size: {config.chunk_size}")
    print(f"  - Action dim: {config.action_feature.shape[0]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    return policy


def test_forward_pass(policy, config):
    """Test forward pass with dummy data."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass with Dummy Data")
    print("="*60)
    
    batch_size = 2
    chunk_size = config.chunk_size
    
    # Create dummy batch
    # Note: state features should be (B, feature_dim), not (B, T, feature_dim)
    # The chunk_size dimension is only for actions
    batch = {
        OBS_STATE: torch.randn(batch_size, 22),
        "observation.state_velocity": torch.randn(batch_size, 22),
        "observation.tactile": torch.randn(batch_size, 32),
        "observation.fsr": torch.randn(batch_size, 12),
        ACTION: torch.randn(batch_size, chunk_size, 8),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }
    
    print(f"Input batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key:35s}: {tuple(value.shape)}")
    
    # Forward pass
    with torch.no_grad():
        policy.eval()
        actions, (mu, log_sigma) = policy.model(batch)
    
    print(f"\n✓ Forward pass successful")
    print(f"\nOutput shapes:")
    print(f"  - actions:                     {tuple(actions.shape)}")
    if mu is not None:
        print(f"  - VAE mu:                      {tuple(mu.shape)}")
        print(f"  - VAE log_sigma_x2:           {tuple(log_sigma.shape)}")
    
    # Verify output dimensions
    assert actions.shape == (batch_size, chunk_size, 8), f"Unexpected action shape: {actions.shape}"
    print(f"\n✓ Output dimensions verified")
    
    return actions


def test_loss_computation(policy, config):
    """Test loss computation in training mode."""
    print("\n" + "="*60)
    print("TEST 4: Loss Computation (Training Mode)")
    print("="*60)
    
    batch_size = 2
    chunk_size = config.chunk_size
    
    # Create dummy batch
    # Note: state features should be (B, feature_dim)
    batch = {
        OBS_STATE: torch.randn(batch_size, 22),
        "observation.state_velocity": torch.randn(batch_size, 22),
        "observation.tactile": torch.randn(batch_size, 32),
        "observation.fsr": torch.randn(batch_size, 12),
        ACTION: torch.randn(batch_size, chunk_size, 8),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }
    
    # Compute loss
    policy.train()
    loss, loss_dict = policy(batch)
    
    print(f"✓ Loss computation successful")
    print(f"\nLoss breakdown:")
    for name, value in loss_dict.items():
        print(f"  - {name:30s}: {value:.6f}")
    print(f"  - Total loss: {loss:.6f}")
    
    assert loss.item() > 0, "Loss should be positive"
    print(f"\n✓ Loss values verified")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ACT Baseline Implementation Tests")
    print("="*60)
    
    # Test 1: Configuration
    config = test_configuration()
    
    # Test 2: Policy initialization
    policy = test_policy_initialization(config)
    
    # Test 3: Forward pass
    test_forward_pass(policy, config)
    
    # Test 4: Loss computation
    test_loss_computation(policy, config)
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
