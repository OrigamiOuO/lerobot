# Tactile Blind Diffusion Policy

## 📝 概述

这是一个**去除视觉特征**的 Diffusion Policy 实现，专门用于基于**状态 + 触觉传感器**的机器人学习任务。

与标准的 DiffusionPolicy 相比，此策略完全移除了视觉编码器（ResNet），仅使用：
- **Robot State** (`observation.state`)
- **Tactile FSR** (`observation.tactile_fsr`) - 12 维力传感器
- **Tactile Taxel** (`observation.tactile_taxel`) - 32 维触觉阵列

---

## 🔧 主要改动记录

### 1. **`configuration_tactile_diffusion.py`**

#### 删除的参数：
- `vision_backbone` (原: `"resnet18"`)
- `crop_shape` (原: `(84, 84)`)
- `crop_is_random` (原: `True`)
- `pretrained_backbone_weights` (原: `None`)
- `use_group_norm` (原: `True`)
- `spatial_softmax_num_keypoints` (原: `32`)
- `use_separate_rgb_encoder_per_camera` (原: `False`)

#### 保留的参数：
```python
# Tactile features
use_tactile_features: bool = True
tactile_encoder_hidden_dim: int = 64

# Time parameters
n_obs_steps: int = 2          # 修改：原为 5
horizon: int = 16
n_action_steps: int = 8
drop_n_last_frames: int = 7   # 修改：原为 4

# U-Net parameters
down_dims: tuple[int, ...] = (512, 1024, 2048)
kernel_size: int = 5
n_groups: int = 8
diffusion_step_embed_dim: int = 128
use_film_scale_modulation: bool = True
```

#### 修改的方法：
- **`__post_init__()`**: 删除了 ResNet 检查逻辑
- **`validate_features()`**: 删除了图像相关的验证，只保留触觉特征验证

---

### 2. **`modeling_tactile_diffusion.py`**

#### 删除的导入：
```python
# 删除
import torchvision
import numpy as np
from lerobot.policies.utils import get_output_shape
from lerobot.utils.constants import OBS_IMAGES
```

#### 删除的类：
- `SpatialSoftmax` - 图像特征提取用的空间软最大值
- `DiffusionRgbEncoder` - ResNet 图像编码器
- `_replace_submodules` - 用于替换 BatchNorm 的辅助函数

#### 修改的方法：

##### `TactileDiffusionPolicy.reset()`
```python
# 删除：
if self.config.image_features:
    self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

# 保留：
self._queues = {
    OBS_STATE: deque(maxlen=self.config.n_obs_steps),
    ACTION: deque(maxlen=self.config.n_action_steps),
}
if self.config.env_state_feature:
    self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
if self.config.use_tactile_features:
    self._queues[OBS_TACTILE1] = deque(maxlen=self.config.n_obs_steps)
    self._queues[OBS_TACTILE2] = deque(maxlen=self.config.n_obs_steps)
```

##### `TactileDiffusionPolicy.select_action()`
```python
# 删除：
if self.config.image_features:
    batch = dict(batch)
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
```

##### `TactileDiffusionPolicy.forward()`
```python
# 删除：
if self.config.image_features:
    batch = dict(batch)
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
```

##### `TactileDiffusionModel.__init__()`
```python
# 删除：
if self.config.image_features:
    num_images = len(self.config.image_features)
    if self.config.use_separate_rgb_encoder_per_camera:
        encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
        self.rgb_encoder = nn.ModuleList(encoders)
        global_cond_dim += encoders[0].feature_dim * num_images
    else:
        self.rgb_encoder = DiffusionRgbEncoder(config)
        global_cond_dim += self.rgb_encoder.feature_dim * num_images

# 保留：
global_cond_dim = self.config.robot_state_feature.shape[0]

if self.config.use_tactile_features:
    self.tactile_encoder = nn.Sequential(
        nn.Linear(44, self.config.tactile_encoder_hidden_dim),  # 12 (fsr) + 32 (taxel) = 44
        nn.ReLU(),
        nn.Linear(self.config.tactile_encoder_hidden_dim, self.config.tactile_encoder_hidden_dim)
    )
    global_cond_dim += self.config.tactile_encoder_hidden_dim
```

##### `TactileDiffusionModel._prepare_global_conditioning()`
```python
# 删除：
if self.config.image_features:
    if self.config.use_separate_rgb_encoder_per_camera:
        images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
        img_features_list = torch.cat([...])
        img_features = einops.rearrange(img_features_list, "(n b s) ... -> b s (n ...)", ...)
    else:
        img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
        img_features = einops.rearrange(img_features, "(b s n) ... -> b s (n ...)", ...)
    global_cond_feats.append(img_features)

# 保留：
global_cond_feats = [batch[OBS_STATE]]

if self.config.use_tactile_features and OBS_TACTILE1 in batch:
    tactile_fsr = batch[OBS_TACTILE1]
    tactile_taxel = batch[OBS_TACTILE2]
    tactile_features = torch.cat([tactile_fsr, tactile_taxel], dim=-1)
    tactile_flat = einops.rearrange(tactile_features, "b s d -> (b s) d")
    tactile_encoded = self.tactile_encoder(tactile_flat)
    tactile_encoded = einops.rearrange(tactile_encoded, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)
    global_cond_feats.append(tactile_encoded)
```

---

### 3. **`processor_tactile_diffusion.py`**

**无需修改** - Processor 逻辑与视觉无关，保持原样。

---

### 4. **`factory.py`**

#### 添加的注册：
```python
elif name == "tactile_blind_diffusion":
    from lerobot.policies.tactile_blind_diffusion.modeling_tactile_diffusion import TactileDiffusionPolicy
    return TactileDiffusionPolicy
```

位置：在 `tactile_diffusion` 之后添加。

---

## 🎯 使用方法

### 训练命令

```bash
lerobot-train \
    --dataset.repo_id=xarm_leap_tactile_lift_blind \
    --dataset.root=./datasets/tactile_dp_test_data/xarm_leap_tactile_lift_blind \
    --policy.type=tactile_blind_diffusion \
    --output_dir=./checkpoints/tactile_blind_full \
    --batch_size=128 \
    --num_workers=10 \
    --policy.use_amp=true \
    --steps=200000 \
    --policy.push_to_hub=false \
    --wandb.enable=true
```

### 推理/评估

```python
from lerobot.policies.pretrained import PreTrainedPolicy

policy = PreTrainedPolicy.from_pretrained("./checkpoints/tactile_blind_full")

# 输入只需要 state 和 tactile
obs = {
    "observation.state": state_tensor,          # (22,)
    "observation.tactile_fsr": fsr_tensor,      # (12,)
    "observation.tactile_taxel": taxel_tensor,  # (32,)
}

action = policy.select_action(obs)
```

---

## 📊 网络结构对比

### 原始 Tactile Diffusion Policy
```
Input:
  - observation.image (多相机)
  - observation.state
  - observation.tactile_fsr
  - observation.tactile_taxel

Encoder:
  - ResNet18 + SpatialSoftmax → visual_features (64)
  - MLP → tactile_features (64)
  - state (直接使用)

Global Conditioning: [visual_features, state, tactile_features]
                                ↓
                        U-Net Diffusion Model
```

### Tactile Blind Diffusion Policy
```
Input:
  - observation.state
  - observation.tactile_fsr
  - observation.tactile_taxel

Encoder:
  - MLP → tactile_features (64)
  - state (直接使用)

Global Conditioning: [state, tactile_features]
                                ↓
                        U-Net Diffusion Model
```

---

## ⚠️ 注意事项

1. **数据集要求**：
   - 必须包含 `observation.state`
   - 必须包含 `observation.tactile_fsr` 和 `observation.tactile_taxel`
   - **不需要**图像数据

2. **维度匹配**：
   - `observation.state`: 22 维 (根据你的机器人)
   - `observation.tactile_fsr`: 12 维
   - `observation.tactile_taxel`: 32 维

3. **预训练模型不兼容**：
   - 由于删除了视觉编码器，**无法加载**原 `tactile_diffusion` 的 checkpoint
   - 需要从头训练

4. **配置文件**：
   - Config 注册名：`"tactile_blind_diffusion"`
   - 继承自 `PreTrainedConfig.register_subclass("tactile_diffusion")`
   - 实际使用时需指定 `--policy.type=tactile_blind_diffusion`

---

## 📈 预期性能

对于**无视觉任务**（如盲操作、触觉感知为主的任务），此策略应该：
- ✅ 训练更快（无需处理图像）
- ✅ 内存占用更小
- ✅ 推理速度更快
- ✅ 专注于触觉和状态信息

---

## 🔗 相关文件

- Configuration: `configuration_tactile_diffusion.py`
- Modeling: `modeling_tactile_diffusion.py`
- Processor: `processor_tactile_diffusion.py`
- Factory Registration: `lerobot/policies/factory.py` (line ~138)

---

## 📚 参考

- 原始 Diffusion Policy: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)
- LeRobot Framework: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- 触觉传感器集成: 基于 LEAP Hand 的 FSR + Taxel 传感器

---

**Created:** 2026-01-29  
**Version:** 1.0 - Blind Policy (No Vision)
