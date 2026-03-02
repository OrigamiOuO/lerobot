# Tactile ACT (Hao Variant)

基于原版 ACT (Action Chunking Transformer) 改进的触觉感知策略，支持 GelSight 触觉传感器的三种模态数据。

## 支持的输入模态

| 模态 | 数据格式 | 处理方式 |
|------|---------|---------|
| 摄像头图像 (RGB) | `observation.images.*`: (B, 3, H, W) video | ResNet18 backbone (FrozenBatchNorm2d) |
| 触觉深度图 (depth) | `observation.tac1_depth`: (B, H, W, 1) float32 | 与 normal 拼接为 4 通道输入 |
| 触觉法线图 (normal) | `observation.tac1_normal`: (B, H, W, 3) float32 | 与 depth 拼接为 4 通道输入 |
| 触觉标记位移 (marker) | `observation.tac1_marker_displacement`: (B, 35, 2) float32 | MLP 编码为独立 1D token |
| 机器人状态 | `observation.state`: (B, 6) float32 | 线性投影为 1D token |
| 动作 | `action`: (B, chunk_size, 6) float32 | 输出目标 |

## 相对于原版 ACT 的改动清单

### 1. `configuration_tactile_act.py`

| 改动项 | 说明 |
|--------|------|
| 注册名改为 `tactile_act_hao` | 避免与 `tactile_act` 冲突 |
| `n_obs_steps` 改为 1 | 单帧输入，避免触觉图像多帧的显存开销 |
| 新增 `use_tactile_image_features` | 控制是否启用 depth+normal 触觉图像 |
| 新增 `tactile_vision_backbone` | 触觉图像的 ResNet 骨干网络类型（默认 resnet18） |
| 新增 `pretrained_tactile_backbone_weights` | 触觉骨干网络的预训练权重 |
| 新增 `use_tactile_marker` | 控制是否启用 marker displacement |
| 新增 `tactile_marker_input_dim` | marker 展平后的维度（35×2=70） |
| 新增 `tactile_marker_hidden_dim` | marker MLP 编码器的隐藏层维度 |
| 新增 `optimizer_lr_tactile_backbone` | 触觉骨干网络的独立学习率 |
| 删除旧的 `use_tactile_features` / `tactile_encoder_hidden_dim` | 旧版 FSR/Taxel 相关配置已移除 |
| 修改 `validate_features()` | 允许仅触觉输入（无需摄像头或环境状态） |
| 修改 `observation_delta_indices` | n_obs_steps=1 时返回 None（无时间维度） |

### 2. `modeling_tactile_act.py`

| 改动项 | 位置 | 说明 |
|--------|------|------|
| 修正导入路径 | 文件顶部 | 从 `lerobot.policies.act_hao.configuration_tactile_act` 导入 |
| 替换触觉常量 | 文件顶部 | `OBS_TAC_DEPTH`, `OBS_TAC_NORMAL`, `OBS_TAC_MARKER` 替代旧的 FSR/Taxel 常量 |
| 策略名改为 `tactile_act_hao` | `TactileACTPolicy` | 与配置注册名一致 |
| 新增触觉backbone参数组 | `get_optim_params()` | 三个参数组：通用参数、摄像头backbone、触觉backbone，各自独立学习率 |
| 简化 `reset()` | `TactileACTPolicy` | 移除旧的多帧 observation queue（n_obs_steps=1 不需要） |
| 简化 `predict_action_chunk()` | `TactileACTPolicy` | 无需多帧历史管理 |
| **新增触觉backbone** | `ACT.__init__` | 4通道ResNet18（normal 3ch + depth 1ch），不使用 FrozenBatchNorm2d，预训练权重第4通道从红色通道复制初始化 |
| **新增 marker_encoder** | `ACT.__init__` | `Linear(70, hidden) → ReLU → Linear(hidden, dim_model)`，作为独立 1D token |
| **新增触觉图像投影层** | `ACT.__init__` | `Conv2d(512, dim_model, 1)` 用于触觉 feature map |
| **新增触觉位置编码** | `ACT.__init__` | `ACTSinusoidalPositionEmbedding2d` 用于触觉 feature map |
| 更新 `n_1d_tokens` | `ACT.__init__` | 加入 marker token 的计数 |
| 移除 `_prepare_state_and_tactile()` | `ACT` | 旧版多帧state+FSR/Taxel融合方法已删除 |
| 更新 `encoder_robot_state_input_proj` | `ACT.__init__` | 维度改回纯 state_dim（不再拼接触觉数据） |
| **新增触觉图像前向处理** | `ACT.forward()` | depth+normal → permute → cat → tactile_backbone → 投影 → rearrange → 加入 encoder tokens |
| **新增 marker 前向处理** | `ACT.forward()` | flatten → marker_encoder → 加入 encoder 1D tokens |

### 3. `processor_tactile_act.py`

| 改动项 | 说明 |
|--------|------|
| 修正导入路径 | 改为从 `lerobot.policies.act_hao.configuration_tactile_act` 导入 |

## 关键设计决策

### 触觉图像融合（方案A）
- **Depth (1ch) + Normal (3ch) → 4通道拼接**，共用一个 ResNet18 backbone
- 预训练权重：前3通道使用 ImageNet 权重，第4通道（depth）从红色通道权重复制初始化
- 拼接顺序：`[normal(3ch), depth(1ch)]`（normal 对齐 ImageNet RGB 通道位置）

### Marker Displacement（方案B）
- **(35, 2) → flatten → 70维 → MLP → dim_model**，作为独立 1D token
- 与 latent、state、env_state 一起使用 learnable 位置编码

### 触觉 Backbone 参数策略
- **完全不冻结**：使用默认 `nn.BatchNorm2d`（不使用 `FrozenBatchNorm2d`）
- 所有参数参与梯度更新，通过 `optimizer_lr_tactile_backbone` 可设置独立学习率
- 初始权重来自 ImageNet 预训练，训练过程中会被更新
- Checkpoint 中保存的是更新后的参数

### Token 结构
Transformer Encoder 的输入 token 顺序：
```
[latent, (state), (marker), (env_state), (*camera_pixels), (*tactile_pixels)]
```

## 数据集要求

数据集 `info.json` 中需包含以下 features：
```json
{
    "observation.state": {"dtype": "float32", "shape": [6]},
    "observation.images.global": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.tac1_depth": {"dtype": "float32", "shape": [480, 640, 1]},
    "observation.tac1_normal": {"dtype": "float32", "shape": [480, 640, 3]},
    "observation.tac1_marker_displacement": {"dtype": "float32", "shape": [35, 2]},
    "action": {"dtype": "float32", "shape": [6]}
}
```
