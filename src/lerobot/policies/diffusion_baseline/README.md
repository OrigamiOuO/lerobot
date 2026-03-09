# Diffusion Baseline Policy

## 概述

`diffusion_baseline` 是一个基于扩散模型（Diffusion Policy）的策略变体，支持多模态状态特征拼接。它结合了：

1. **Diffusion Policy** 的核心扩散模型架构（基于 [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://huggingface.co/papers/2303.04137)）
2. **ACT Baseline** 的多模态状态拼接方式（支持 observation.state, observation.tactile, observation.fsr 等多种状态特征）

## 主要特性

### 多模态状态拼接

与标准 Diffusion Policy 只支持单一 `observation.state` 不同，`diffusion_baseline` 支持自动拼接多种状态观测模态：

```python
state_feature_keys: list[str] = [
    "observation.state",           # 关节位置
    "observation.state_velocity",  # 关节速度
    "observation.tactile",         # 触觉传感器
    "observation.fsr",             # 力敏电阻
]
```

所有配置的状态特征会按顺序拼接成一个复合状态向量，用于 UNet 的全局条件调制。

### 扩散模型架构

- **Conditional UNet1D**: 1D 卷积 UNet 架构，用于动作序列的去噪
- **FiLM 调制**: 使用 Feature-wise Linear Modulation 进行时间步和状态条件的注入
- **DDPM/DDIM 调度器**: 支持 DDPM 和 DDIM 两种噪声调度策略
- **SpatialSoftmax**: 用于图像特征提取的空间 softmax 关键点检测

## 文件结构

```
diffusion_baseline/
├── __init__.py                          # 模块导出
├── configuration_diffusion_baseline.py  # 配置类
├── modeling_diffusion_baseline.py       # 模型实现
└── README.md                           # 本文档
```

## 与标准 Diffusion Policy 的对比

| 特性 | Diffusion Policy | Diffusion Baseline |
|------|------------------|-------------------|
| 状态输入 | 单一 `observation.state` | 多模态状态拼接 |
| 状态维度 | 固定 | 自动计算复合维度 |
| 配置字段 | `robot_state_feature` | `state_feature_keys` |
| 其他特性 | 相同 | 相同 |

## 使用方法

### 1. 配置文件示例

```python
from lerobot.policies.diffusion_baseline.configuration_diffusion_baseline import DiffusionBaselineConfig

config = DiffusionBaselineConfig(
    # 基本参数
    n_obs_steps=2,
    horizon=16,
    n_action_steps=8,
    
    # 多模态状态特征键
    state_feature_keys=[
        "observation.state",
        "observation.tactile",
        "observation.fsr",
    ],
    
    # UNet 架构
    down_dims=(512, 1024, 2048),
    diffusion_step_embed_dim=128,
    
    # 噪声调度
    noise_scheduler_type="DDPM",
    num_train_timesteps=100,
    prediction_type="epsilon",
)
```

### 2. 训练配置 (YAML)

```yaml
policy:
  type: diffusion_baseline
  
  # 观测步数和动作范围
  n_obs_steps: 2
  horizon: 16
  n_action_steps: 8
  
  # 多模态状态配置
  state_feature_keys:
    - observation.state
    - observation.state_velocity
    - observation.tactile
    - observation.fsr
  
  # 视觉骨干网络
  vision_backbone: resnet18
  crop_shape: [84, 84]
  crop_is_random: true
  spatial_softmax_num_keypoints: 32
  
  # UNet 配置
  down_dims: [512, 1024, 2048]
  kernel_size: 5
  n_groups: 8
  diffusion_step_embed_dim: 128
  use_film_scale_modulation: true
  
  # 扩散过程
  noise_scheduler_type: DDPM
  num_train_timesteps: 100
  beta_schedule: squaredcos_cap_v2
  prediction_type: epsilon
  
  # 训练参数
  optimizer_lr: 1e-4
  optimizer_weight_decay: 1e-6
```

### 3. 数据集要求

数据集需要包含配置中指定的所有状态特征键。例如：

```python
# 数据集特征示例
features = {
    "observation.state": (22,),           # 关节位置
    "observation.state_velocity": (22,),  # 关节速度
    "observation.tactile": (32,),         # 触觉传感器数据
    "observation.fsr": (12,),             # 力敏电阻数据
    "observation.image": (3, 480, 640),   # RGB 图像
    "action": (14,),                      # 动作空间
}
# 总状态维度: 22 + 22 + 32 + 12 = 88
```

## 核心组件说明

### DiffusionBaselinePolicy

主策略类，负责：
- 管理观测队列（多模态状态、图像）
- 调用扩散模型生成动作
- 计算训练损失

### DiffusionBaselineModel

扩散模型核心，包含：
- `_concatenate_state_features()`: 拼接多模态状态特征
- `_prepare_global_conditioning()`: 准备全局条件（状态 + 图像特征）
- `conditional_sample()`: 扩散采样过程
- `compute_loss()`: 计算扩散损失

### DiffusionBaselineConditionalUnet1d

条件 UNet 网络：
- 编码器-解码器架构带跳跃连接
- FiLM 条件注入
- 正弦位置编码

### DiffusionBaselineRgbEncoder

图像编码器：
- ResNet 骨干网络
- SpatialSoftmax 池化
- 可选随机/中心裁剪

## 配置参数详解

### 输入输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_obs_steps` | 2 | 观测历史步数 |
| `horizon` | 16 | 扩散模型预测的总时间范围 |
| `n_action_steps` | 8 | 实际使用的动作步数 |

### 多模态状态参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_feature_keys` | `["observation.state", ...]` | 要拼接的状态特征键列表 |

### 视觉参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vision_backbone` | `"resnet18"` | 视觉骨干网络 |
| `crop_shape` | `(84, 84)` | 图像裁剪尺寸 |
| `crop_is_random` | `True` | 训练时是否随机裁剪 |
| `spatial_softmax_num_keypoints` | 32 | SpatialSoftmax 关键点数量 |
| `use_separate_rgb_encoder_per_camera` | `False` | 是否为每个相机使用独立编码器 |

### UNet 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `down_dims` | `(512, 1024, 2048)` | 下采样各阶段特征维度 |
| `kernel_size` | 5 | 卷积核大小 |
| `n_groups` | 8 | GroupNorm 分组数 |
| `diffusion_step_embed_dim` | 128 | 时间步嵌入维度 |
| `use_film_scale_modulation` | `True` | 是否使用 FiLM 缩放调制 |

### 扩散参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `noise_scheduler_type` | `"DDPM"` | 噪声调度器类型 (DDPM/DDIM) |
| `num_train_timesteps` | 100 | 训练时间步数 |
| `num_inference_steps` | `None` | 推理时间步数（默认同训练） |
| `beta_schedule` | `"squaredcos_cap_v2"` | Beta 调度策略 |
| `beta_start` | 0.0001 | Beta 起始值 |
| `beta_end` | 0.02 | Beta 结束值 |
| `prediction_type` | `"epsilon"` | 预测类型 (epsilon/sample) |
| `clip_sample` | `True` | 是否裁剪采样结果 |
| `clip_sample_range` | 1.0 | 裁剪范围 |

## 注意事项

1. **状态特征键顺序**: `state_feature_keys` 中的顺序决定了特征拼接的顺序，请确保训练和推理时使用相同的顺序。

2. **缺失特征处理**: 如果某个配置的状态特征在数据集中不存在，会被自动跳过，不会报错。

3. **Horizon 约束**: `horizon` 必须是 `2^len(down_dims)` 的倍数（因为 UNet 的下采样结构）。

4. **图像形状一致性**: 所有输入图像必须具有相同的形状。

5. **归一化**: 建议对状态特征使用 `MIN_MAX` 归一化，对图像使用 `MEAN_STD` 归一化。

## 与 ACT Baseline 的区别

| 方面 | ACT Baseline | Diffusion Baseline |
|------|-------------|-------------------|
| 核心架构 | Transformer Encoder-Decoder | UNet + Diffusion |
| 动作生成 | 直接回归 | 迭代去噪 |
| VAE | 可选 | 无 |
| 推理速度 | 快 | 较慢（需要多步去噪） |
| 多模态性 | 较好 | 更好（适合复杂分布） |

## 参考资料

1. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://huggingface.co/papers/2303.04137)
2. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705) (ACT)
3. [FiLM: Visual Reasoning with a General Conditioning Layer](https://huggingface.co/papers/1709.07871)
4. [Deep Spatial Autoencoders for Visuomotor Learning](https://huggingface.co/papers/1509.06113) (SpatialSoftmax)
