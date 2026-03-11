# Diffusion-Hao: Tactile-Augmented Diffusion Policy

## 概述

`diffusion_hao` 是基于原始 Diffusion Policy 的触觉增强版本，专门设计用于处理包含触觉传感器数据的机器人学习任务。

## 触觉数据支持

### 1. 触觉原始图像 (`observation.images.tac_raw.xx`)

**处理方式**：与普通视觉图像 (`observation.images.global`, `observation.images.inhand`) 共享同一个 ResNet 编码器权重。

**原理**：
- 触觉原始图像本质上也是 RGB 图像（H×W×3）
- 使用 `DiffusionRgbEncoder` 进行编码
- 所有视觉图像特征最终拼接在一起作为 global conditioning 的一部分

**配置**：只需在数据集的 `observation.images.tac_raw.xx` 键中指定，系统会自动使用共享编码器。

### 2. 触觉深度+法线数据 (`observation.tac_depth.xx` + `observation.tac_normal.xx`)

**处理方式**：
- `tac_depth`: (H, W, 1) - 触觉深度图
- `tac_normal`: (H, W, 3) - 触觉法线图
- **合并为 4 通道输入**：将深度和法线拼接为 (H, W, 4) 的张量
- 使用**独立的 ResNet 编码器**（权重初始化自 ImageNet 预训练模型，但修改第一层 conv 以接受 4 通道输入）
- 训练过程中**全参数更新**，让模型自主学习触觉特征表示

**架构**：
```
tac_depth (H, W, 1) + tac_normal (H, W, 3)
          ↓
    concat → (H, W, 4)
          ↓
    permute → (4, H, W)
          ↓
  TactileVisionEncoder (独立 ResNet18)
          ↓
    SpatialSoftmax → (K, 2)
          ↓
    flatten + Linear → feature_dim
```

### 3. 触觉标记点位移 (`observation.tac_marker_displacement.xx`)

**处理方式**：
- 输入形状: (35, 2) - 35 个标记点的 (x, y) 位移
- **类似 state 的处理**：先 flatten 为 (70,)，然后通过 MLP 编码到与 state 相近的维度
- 编码后与 `observation.state` **拼接**，共同作为 conditioning 的一部分

**架构**：
```
tac_marker_displacement (35, 2)
          ↓
    flatten → (70,)
          ↓
    TactileMarkerEncoder (Linear + ReLU + Linear)
          ↓
    tactile_marker_embed_dim (与 state_dim 相近)
          ↓
    concat with state → global_cond
```

## 代码结构

```
diffusion_hao/
├── __init__.py
├── README.md
├── configuration_diffusion_hao.py    # 配置类
└── modeling_diffusion_hao.py         # 模型实现
```

## 配置参数

```python
@dataclass
class DiffusionHaoConfig(PreTrainedConfig):
    # 继承自 DiffusionConfig 的参数...
    
    # === 触觉相关新增参数 ===
    
    # 触觉4通道编码器配置
    tactile_vision_backbone: str = "resnet18"
    tactile_backbone_in_channels: int = 4  # depth(1) + normal(3)
    tactile_pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    tactile_spatial_softmax_num_keypoints: int = 32
    
    # 触觉标记点编码器配置
    tactile_marker_input_dim: int = 70  # 35 markers × 2 coordinates
    tactile_marker_embed_dim: int = 64  # 输出维度（与 state_dim 相近）
```

## 使用方法

### 训练

```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion_hao \
    --dataset.repo_id=your_tactile_dataset \
    --output_dir=checkpoints/diffusion_hao
```

### 配置文件示例

```yaml
policy:
  type: diffusion_hao
  n_obs_steps: 2
  horizon: 16
  n_action_steps: 8
  
  # 视觉编码器
  vision_backbone: resnet18
  use_separate_rgb_encoder_per_camera: false
  
  # 触觉4通道编码器
  tactile_vision_backbone: resnet18
  tactile_pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  tactile_spatial_softmax_num_keypoints: 32
  
  # 触觉标记点编码器
  tactile_marker_embed_dim: 64
```

## Global Conditioning 维度计算

最终的 `global_cond_dim` 计算如下：

```
global_cond_dim = (
    state_dim                                    # observation.state
    + image_feature_dim × num_cameras            # observation.images.*
    + tactile_raw_feature_dim × num_tac_raw      # observation.images.tac_raw.* (共享编码器)
    + tactile_vision_feature_dim                 # tac_depth + tac_normal (独立编码器)
    + tactile_marker_embed_dim                   # tac_marker_displacement
    + env_state_dim (if exists)                  # observation.environment_state
) × n_obs_steps
```

## 实现细节

### TactileVisionEncoder

为了让 ResNet 接受 4 通道输入，我们需要修改第一层卷积：

```python
# 加载预训练 ResNet
backbone_model = torchvision.models.resnet18(weights=pretrained_weights)

# 修改第一层 conv 以接受 4 通道
original_conv = backbone_model.conv1
new_conv = nn.Conv2d(
    4, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# 复制 RGB 权重，新通道初始化为均值
with torch.no_grad():
    new_conv.weight[:, :3] = original_conv.weight
    new_conv.weight[:, 3:4] = original_conv.weight.mean(dim=1, keepdim=True)

backbone_model.conv1 = new_conv
```

### TactileMarkerEncoder

```python
class TactileMarkerEncoder(nn.Module):
    def __init__(self, input_dim=70, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
    
    def forward(self, x):
        # x: (B, 35, 2) or (B, n_obs_steps, 35, 2)
        x = x.flatten(start_dim=-2)  # (B, 70) or (B, n_obs_steps, 70)
        return self.encoder(x)
```

## 关于权重共享的说明

| 数据类型 | 编码器 | 权重共享 |
|---------|--------|---------|
| `observation.images.*` | DiffusionRgbEncoder | 共享 |
| `observation.images.tac_raw.*` | DiffusionRgbEncoder | **与视觉共享** |
| `observation.tac_depth.*` + `tac_normal.*` | TactileVisionEncoder | **独立** |
| `observation.tac_marker_displacement.*` | TactileMarkerEncoder | 独立 |
| `observation.state` | 直接使用 | - |

## 依赖

- torch >= 2.0
- torchvision >= 0.15
- diffusers >= 0.24
- einops

## 参考

- [Diffusion Policy](https://huggingface.co/papers/2303.04137)
- [Deep Spatial Autoencoders for Visuomotor Learning](https://huggingface.co/papers/1509.06113)
