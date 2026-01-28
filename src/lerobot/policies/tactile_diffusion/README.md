# My Custom Policy

A custom policy implementation based on Diffusion Policy with added support for tactile sensor observations.

## Overview

This policy extends the standard Diffusion Policy to incorporate tactile sensor data (`observation.tactile`) as an additional input modality alongside state, vision, and environment state.

## Key Features

- **Tactile Sensor Support**: Processes `observation.tactile` through a dedicated encoder
- **Flexible Configuration**: Enable/disable tactile features via `use_tactile_features` flag
- **Compatible with Diffusion Framework**: Reuses the same UNet architecture and noise schedulers
- **Configurable Encoder**: Adjust tactile feature encoding via `tactile_encoder_hidden_dim`

## Architecture

```
Observations:
├── observation.state (required)
├── observation.tactile (optional, NEW)
├── observation.images (optional)
└── observation.environment_state (optional)
        ↓
[Tactile Encoder] (NEW)
        ↓
[Concatenate all features]
        ↓
[Diffusion UNet]
        ↓
Action predictions
```

## Usage

### Training

```bash
lerobot-train \
    --policy.type my_custom_policy \
    --policy.use_tactile_features=true \
    --policy.tactile_encoder_hidden_dim=64 \
    --dataset.repo_id=YourDataset \
    --dataset.root=./datasets/your_data \
    --steps=200000
```

### Configuration

Key parameters in `MyCustomPolicyConfig`:

- `use_tactile_features` (bool): Enable/disable tactile feature processing (default: True)
- `tactile_encoder_hidden_dim` (int): Hidden dimension for tactile encoder (default: 64)
- All other parameters inherited from `DiffusionConfig`

### Dataset Requirements

Your dataset should include:
- `observation.state`: Robot joint states (required)
- `observation.tactile`: Tactile sensor readings (optional, used if `use_tactile_features=True`)
- `action`: Target actions (required)

Example `info.json` structure:
```json
{
    "features": {
        "observation.state": {"dtype": "float32", "shape": [22]},
        "observation.tactile": {"dtype": "float32", "shape": [44]},
        "action": {"dtype": "float32", "shape": [22]}
    }
}
```

## Implementation Details

### Files

- `configuration_my_custom_policy.py`: Config class with tactile-specific parameters
- `modeling_my_custom_policy.py`: Policy and model implementation with tactile encoder
- `processor_my_custom_policy.py`: Data preprocessing/postprocessing pipelines
- `__init__.py`: Package exports

### Tactile Feature Processing

The tactile encoder is a simple 2-layer MLP:
```python
nn.Sequential(
    nn.Linear(tactile_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)
```

You can modify this in `MyCustomDiffusionModel.__init__()` to use more sophisticated architectures.

### Extending the Policy

To customize further:

1. **Modify tactile encoder**: Edit `MyCustomDiffusionModel.__init__()`
2. **Add new modalities**: Follow the same pattern in `_prepare_global_conditioning()`
3. **Change validation logic**: Implement `validate_features()` in the config class

## TODO

- [ ] Implement custom validation logic in `validate_features()`
- [ ] Add support for separate tactile encoders per sensor type (FSR vs. taxel)
- [ ] Experiment with attention mechanisms for tactile-vision fusion
- [ ] Add pre-trained weights support

## Differences from Diffusion Policy

| Feature | Diffusion Policy | My Custom Policy |
|---------|------------------|------------------|
| Tactile support | ❌ | ✅ |
| Configurable tactile encoder | ❌ | ✅ |
| Other features | Same | Same |

## Citation

Based on Diffusion Policy:
```bibtex
@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```
