# ACT Baseline: Multi-Modal State Feature Concatenation Policy

## Overview

The **ACT Baseline** policy is a variant of the standard [ACT (Action Chunking Transformer)](https://huggingface.co/papers/2304.13705) policy designed to handle datasets with **multiple 1D state observation modalities**. Instead of relying on visual/image-based tactile sensors, it simply concatenates arbitrary state features (joint positions, tactile sensors, force-sensitive resistors, etc.) into a composite state vector.

## Design Motivation

Some robotics datasets may have multiple 1D sensor modalities that need to be combined:

- **observation.state**: 22D - Joint positions
- **observation.state_velocity**: 22D - Joint velocities  
- **observation.tactile**: 32D - Taxel sensor readings
- **observation.fsr**: 12D - Force-sensitive resistor pressures

**Total composite state**: 22 + 22 + 32 + 12 = 88D

Standard ACT expects a single `robot_state_feature` input. ACT Baseline solves this by:

1. **Automatic dimension inference**: The configuration computes total state dimension by summing all configured state features
2. **Flexible feature ordering**: Users can specify which state features to concatenate via `state_feature_keys`
3. **Optional features**: Gracefully handles missing features at inference time
4. **Simple concatenation**: No complex feature encoding—just stack features along the feature dimension

## Architecture Modifications

### Configuration (`ACTBaselineConfig`)

```python
@dataclass
class ACTBaselineConfig(PreTrainedConfig):
    # New field: list of features to concatenate
    state_feature_keys: list[str] = field(
        default_factory=lambda: [
            OBS_STATE,
            "observation.state_velocity", 
            "observation.tactile",
            "observation.fsr"
        ]
    )
    
    # New method: compute total state dimension
    def compute_composite_state_dim(self) -> int:
        """Return sum of shapes of all features in state_feature_keys."""
```

### Modeling (`ACTBaselinePolicy` and `ACT`)

**Key changes in `ACT` class**:

1. **Constructor adjustment**:
   ```python
   composite_state_dim = config.compute_composite_state_dim()
   self.encoder_robot_state_input_proj = nn.Linear(
       composite_state_dim,  # Now: sum of all state features
       config.dim_model
   )
   ```

2. **Feature concatenation in forward()**:
   ```python
   def _concatenate_state_features(self, batch: dict) -> Tensor:
       """Stack all state features along feature dimension."""
       state_parts = [batch[key] for key in config.state_feature_keys if key in batch]
       return torch.cat(state_parts, dim=-1)  # Shape: (B, composite_state_dim)
   ```

3. **Flexible feature handling**:
   - Only concatenates features that exist in the batch
   - Gracefully skips missing optional features
   - Works with partial feature sets at inference time

### Processing (`make_act_baseline_pre_post_processors`)

The processor is identical to standard ACT:
- Normalizes all input/output features via `NormalizerProcessorStep`
- Batching and device placement
- Post-processing unnormalization

## Usage Example

### Training with Blind Grasping Dataset

```bash
lerobot-train \
  --dataset.repo_id=luo_proj/Blind_Grasping_LeRoBot_v1 \
  --dataset.root=./datasets/luo_proj/Blind_Grasping_LeRoBot_v1 \
  --policy.type=act_baseline \
  --policy.repo_id=act_baseline_test \
  --policy.push_to_hub=false \
  --output_dir=./checkpoints/act_baseline_test \
  --batch_size=4 \
  --num_workers=0 \
  --steps=100
```

### Customizing State Features

Create a JSON policy config:

```json
{
  "state_feature_keys": [
    "observation.state",
    "observation.state_velocity",
    "observation.tactile"
  ]
}
```

Or override via CLI:
```bash
lerobot-train ... --policy.state_feature_keys='["observation.state", "observation.tactile"]'
```

## Comparison with Other Variants

| Aspect | Standard ACT | ACT Hao | ACT Baseline |
|--------|-------------|---------|--------------|
| **Input State** | Single `robot_state_feature` | Dual: robot_state + image-based tactile (depth/normal) | Multiple 1D features concatenated |
| **Tactile Processing** | None | Separate ResNet18 backbone for depth+normal images + MLP for marker displacement | None—tactile as raw 1D vector |
| **Use Case** | Standard manipulation with joint/gripper state | GelSight sensors (visual tactile) | Blind grasping with multiple 1D sensors |
| **Composite State Dim** | ~22D | ~22D + ~256D image features | Dynamic (22+22+32+12 = 88D) |
| **Number of Backbones** | 0 (if no images) | 2 (camera + tactile) | 0 (just linear projections) |

## File Structure

```
src/lerobot/policies/act_baseline/
├── __init__.py                           # Package exports
├── configuration_act_baseline.py          # Config with composite state support
├── modeling_act_baseline.py               # ACTBaselinePolicy + ACT implementation
├── processor_act_baseline.py              # Pre/post-processor pipeline
└── README.md                             # This file
```

## Implementation Notes

1. **VAE Support**: The VAE encoder also uses concatenated state during training (`vae_encoder_robot_state_input_proj`)

2. **Positional Embeddings**: The transformer correctly accounts for the composite state token in its positional encoding calculations

3. **Optional Features**: Missing features are skipped gracefully:
   ```python
   state_parts = [batch[key] for key in keys if key in batch]
   ```

4. **Feature Ordering**: Order matters! Features are concatenated in `state_feature_keys` order. Ensure consistent ordering across training, evaluation, and deployment.

5. **Normalization**: Each individual feature is normalized independently via the dataset statistics before concatenation.

## Performance Considerations

- **Computation**: Simpler than ACT Hao (no CNN backbones), just linear projections
- **Memory**: Proportional to composite state dimension
- **Training Speed**: Similar to standard ACT, often faster than ACT Hao due to fewer CNN layers

## Debugging Tips

1. **Check composite state dimension**:
   ```python
   config = ACTBaselineConfig.from_pretrained("checkpoints/act_baseline_test")
   print(config.compute_composite_state_dim())
   ```

2. **Verify feature concatenation order**:
   ```python
   print(config.state_feature_keys)
   ```

3. **Inspect normalized features**:
   Add logging in `_concatenate_state_features()` to inspect feature values before concatenation

4. **Missing feature errors at inference**:
   Either include all features in the observation or update `state_feature_keys` to match available features

## Future Extensions

- Support for feature weighting (apply different scales to different features)
- Feature selection via learned gates
- Attention-based feature fusion instead of concatenation
- Support for hierarchical state (position + velocity pairs)

## References

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705)
- LeRoBot framework: https://github.com/huggingface/lerobot
