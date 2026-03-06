# ACT-Hao 策略适配指南

## 1. 概述

本文档描述如何基于默认的 ACT 策略，编写一个名为 `act_hao` 的自定义策略，使其适配包含以下新增特征的数据集：

| 数据集键名 | shape | 类型 | 处理方式 |
|---|---|---|---|
| `observation.images.global` | (480,640,3) | video | 标准图像处理（已有） |
| `observation.images.inhand` | (480,640,3) | video | 标准图像处理（已有） |
| `observation.images.tac_raw.tac1` | (480,640,3) | video | **按标准图像处理**（新增） |
| `observation.tac_depth.tac1` | (480,640,1) | float32 | **合成4通道 → 独立ResNet**（新增） |
| `observation.tac_normal.tac1` | (480,640,3) | float32 | **合成4通道 → 独立ResNet**（新增） |
| `observation.tac_marker_displacement.tac1` | (35,2) | float32 | **flatten → 与state拼接**（新增） |
| `observation.state` | (6,) | float32 | 标准state处理（已有） |
| `action` | (6,) | float32 | 标准action输出（已有） |

---

## 2. 需要创建/修改的文件

整个适配涉及 **4 个文件**的创建和 **2 个已有文件**的修改：

```
src/lerobot/policies/act_hao/
├── __init__.py                      # 新建
├── configuration_act_hao.py         # 新建 - 配置类
├── modeling_act_hao.py              # 新建 - 模型核心
├── processor_act_hao.py             # 新建 - 前后处理管线
└── ADAPTATION_GUIDE.md              # 本文档

src/lerobot/policies/factory.py      # 修改 - 注册新策略
src/lerobot/datasets/utils.py        # 可能需要修改 - 特征分类
```

---

## 3. 架构设计

### 3.1 数据流全图

```
输入数据
├── observation.images.global        ──┐
├── observation.images.inhand        ──┼──→ [共享 backbone (ResNet18, pretrained, FrozenBN)]
├── observation.images.tac_raw.tac1  ──┘        → encoder_img_feat_input_proj (Conv2d 1x1)
│                                                → flatten + 2D位置编码 → tokens
│
├── observation.tac_depth.tac1 (H,W,1)  ─┐
│                                         ├─ cat dim=-1 → (B,H,W,4) → permute → (B,4,H,W)
├── observation.tac_normal.tac1 (H,W,3) ─┘
│       → [独立 tactile_backbone (ResNet18, 首层conv1改4通道, 无pretrained, 无FrozenBN)]
│       → encoder_tac_feat_input_proj (Conv2d 1x1)
│       → flatten + 2D位置编码 → tokens
│
├── observation.state (6,)                 ─┐
│                                           ├─ 各自投影到dim_model/2 → concat → 融合投影
├── observation.tac_marker_displacement     │   → (B, dim_model) → 1个token
│     .tac1 (35,2) → flatten → (70,)      ─┘
│
└── latent (从VAE或全零)                    → 1个token

所有tokens拼接 → Transformer Encoder → Transformer Decoder → action_head → action
```

> **state token 融合**：
> - state(6) → Linear(6, dim_model/2)
> - marker(70) → Linear(70, dim_model/2)  
> - concat → (dim_model/2 + dim_model/2) = dim_model → Linear(dim_model, dim_model) → 最终 token

### 3.2 关键设计决策

1. **`observation.images.tac_raw.tac1`**：直接视为普通图像，与 `global`、`inhand` 共享同一个 ResNet backbone。因此在 `dataset_to_policy_features` 中它已经被分类为 `FeatureType.VISUAL`（因为它的 dtype 是 "video"），无需额外处理。

2. **`tac_depth` + `tac_normal` 合成 4 通道**：
   - 在第三维 concat：`(B, 480, 640, 1)` + `(B, 480, 640, 3)` → `(B, 480, 640, 4)` → permute → `(B, 4, 480, 640)`
   - 使用**独立的** ResNet18 作为 backbone（可通过 `tactile_vision_backbone` 配置），首层 `conv1` 从 3 通道改为 4 通道
   - **不使用 pretrained 权重**（4 通道无法使用 ImageNet 预训练）
   - **不使用 FrozenBatchNorm2d**（全部参与训练，使用普通 BatchNorm2d）
   - 输出 feature map 经过独立的 `encoder_tac_feat_input_proj`（Conv2d 1x1）投影到 `dim_model`
   - 然后和普通图像一样展平为 token 序列 + 2D 正弦位置编码

3. **`tac_marker_displacement`**：
   - Flatten 为 70 维向量：`(B, 35, 2)` → `(B, 70)`
   - **各自独立投影**：
     - state(6) → Linear(6, dim_model/2) → `(B, dim_model/2)`
     - marker(70) → Linear(70, dim_model/2) → `(B, dim_model/2)`
   - **融合**：concat([state_proj, marker_proj], dim=-1) → `(B, dim_model)`
   - **最终投影**：Linear(dim_model, dim_model) → `(B, dim_model)` 作为 1 个 token
   - 优点：维度平衡（各占 50%），特征解耦，数值稳定

4. **VAE Encoder**：使用相同的融合策略，输入序列形式保持原始 ACT 的形式（因为 marker 不作为独立 token）
   - `[cls, robot_state_with_marker, *action_sequence]`（state_with_marker 是融合后的结果）
   - 不需要改变位置编码长度

---

## 4. 逐文件实现指南

### 4.1 `configuration_act_hao.py`

基于 `configuration_act.py`，添加触觉相关配置项：

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_hao")
@dataclass
class ACTHaoConfig(PreTrainedConfig):
    # ========== 与原始 ACT 相同的配置 ==========
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,  # 新增：触觉数据归一化
        }
    )

    # Vision backbone (用于普通图像和 tac_raw)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # ========== 新增：触觉视觉 backbone 配置 ==========
    # 用于 tac_depth + tac_normal 合成的 4 通道数据
    tactile_vision_backbone: str = "resnet18"
    tactile_backbone_in_channels: int = 4  # depth(1) + normal(3)
    # 不使用预训练权重（4通道无法匹配ImageNet 3通道预训练）
    tactile_pretrained_backbone_weights: str | None = None
    tactile_replace_final_stride_with_dilation: int = False

    # Transformer layers
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training
    dropout: float = 0.1
    kl_weight: float = 10.0
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        # 输入校验（与原始ACT相同）
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")
        if not self.tactile_vision_backbone.startswith("resnet"):
            raise ValueError(f"`tactile_vision_backbone` must be a ResNet variant. Got {self.tactile_vision_backbone}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})"
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"n_obs_steps must be 1, got {self.n_obs_steps}")

    # --- 新增：触觉特征属性方法 ---
    @property
    def tactile_depth_features(self) -> dict:
        """返回所有 tac_depth 类型的特征"""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items()
                if "tac_depth" in k}

    @property
    def tactile_normal_features(self) -> dict:
        """返回所有 tac_normal 类型的特征"""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items()
                if "tac_normal" in k}

    @property
    def tactile_marker_features(self) -> dict:
        """返回所有 tac_marker_displacement 类型的特征"""
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items()
                if "tac_marker" in k}

    @property
    def has_tactile_vision(self) -> bool:
        """是否有需要视觉backbone处理的触觉数据"""
        return bool(self.tactile_depth_features) and bool(self.tactile_normal_features)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.has_tactile_vision:
            raise ValueError(
                "You must provide at least one of: image features, environment state, "
                "or tactile vision features."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
```

### 4.2 `modeling_act_hao.py`

这是核心文件。以下标注**所有需要相对于原始 `modeling_act.py` 做修改的位置**：

#### 4.2.1 新增常量

```python
# 在文件顶部定义新的常量键名
OBS_TAC_DEPTH = "observation.tac_depth.tac1"
OBS_TAC_NORMAL = "observation.tac_normal.tac1"
OBS_TAC_MARKER = "observation.tac_marker_displacement.tac1"
OBS_TAC_VISION = "observation.tac_vision"  # 合成后的4通道数据
```

#### 4.2.2 `ACTHaoPolicy` 类的 `forward` 和 `predict_action_chunk`

需要在将 batch 传入 `self.model()` 之前，完成以下预处理：

```python
def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """准备 batch，处理图像列表和触觉数据合成"""
    batch = dict(batch)  # shallow copy

    # 1. 收集所有标准图像特征（包括 tac_raw，因为它被归类为 VISUAL）
    if self.config.image_features:
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    # 2. 合成 tac_depth + tac_normal → 4通道触觉图
    if self.config.has_tactile_vision:
        # tac_depth: (B, 1, H, W) 已经是 channel-first（经过 dataset_to_policy_features 转换）
        # tac_normal: (B, 3, H, W) 同上
        # 但注意：这些触觉数据是 float32 且 FeatureType=TACTILE，
        # shape 没有经过 channel-first 转换！需要自行处理。
        depth = batch[OBS_TAC_DEPTH]    # (B, 480, 640, 1)
        normal = batch[OBS_TAC_NORMAL]  # (B, 480, 640, 3)
        tac_4ch = torch.cat([depth, normal], dim=-1)  # (B, H, W, 4)
        tac_4ch = tac_4ch.permute(0, 3, 1, 2)         # (B, 4, H, W)
        batch[OBS_TAC_VISION] = tac_4ch

    # 3. tac_marker_displacement: flatten
    if self.config.tactile_marker_features:
        marker = batch[OBS_TAC_MARKER]  # (B, 35, 2)
        batch[OBS_TAC_MARKER] = marker.flatten(start_dim=1)  # (B, 70)

    return batch
```

> **重要说明**：`observation.tac_depth.tac1` 和 `observation.tac_normal.tac1` 在 `dataset_to_policy_features` 中会因 `_is_tactile_feature` 检测被归类为 `FeatureType.TACTILE`。由于它们不是 image/video dtype，shape 不会自动做 channel-first 转换。因此需要在模型中手动 permute。

#### 4.2.3 `ACTHao` (nn.Module) 的 `__init__` — 新增模块

在 `ACT.__init__` 的基础上，新增以下模块：

```python
# ======== 触觉视觉 backbone（处理 depth+normal 4通道数据） ========
if self.config.has_tactile_vision:
    # 创建独立的 ResNet，不冻结 BN，不使用预训练权重
    tactile_backbone_model = getattr(torchvision.models, config.tactile_vision_backbone)(
        replace_stride_with_dilation=[
            False, False, config.tactile_replace_final_stride_with_dilation
        ],
        weights=config.tactile_pretrained_backbone_weights,
        # 注意：不使用 FrozenBatchNorm2d，让 BN 层正常训练
    )
    # 修改第一层 conv1 以接受 4 通道输入
    old_conv1 = tactile_backbone_model.conv1
    tactile_backbone_model.conv1 = nn.Conv2d(
        config.tactile_backbone_in_channels,  # 4
        old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=old_conv1.bias is not None,
    )
    self.tactile_backbone = IntermediateLayerGetter(
        tactile_backbone_model, return_layers={"layer4": "feature_map"}
    )
    # 触觉 feature map → dim_model 的 1x1 卷积投影
    self.encoder_tac_feat_input_proj = nn.Conv2d(
        tactile_backbone_model.fc.in_features, config.dim_model, kernel_size=1
    )
    # 触觉 feature map 的 2D 位置编码（复用 ACTSinusoidalPositionEmbedding2d）
    self.encoder_tac_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

# ======== robot state 和 marker 的独立投影 ========
# 策略：先各自投影到中间维度（dim_model/2），再 concat 后融合投影
# 这样可以保证 state 和 marker 的信息平衡（各占 50%），避免高维特征压制低维特征
if self.config.robot_state_feature:
    state_dim = self.config.robot_state_feature.shape[0]  # 6
    # state 单独投影到 dim_model/2
    self.encoder_state_proj = nn.Linear(state_dim, config.dim_model // 2)

if self.config.tactile_marker_features:
    marker_dim = 70  # (35, 2) flatten
    # marker 单独投影到 dim_model/2
    self.encoder_marker_proj = nn.Linear(marker_dim, config.dim_model // 2)
    # state 和 marker concat 后的融合投影
    self.encoder_state_marker_fusion = nn.Linear(config.dim_model, config.dim_model)
```

#### 4.2.4 VAE Encoder 的 state 和 marker 融合投影

```python
# VAE encoder 也使用相同的策略：state 和 marker 各自投影到 dim_model/2，再融合
if self.config.use_vae:
    if self.config.robot_state_feature:
        state_dim = self.config.robot_state_feature.shape[0]  # 6
        self.vae_encoder_state_proj = nn.Linear(state_dim, config.dim_model // 2)
    
    if self.config.tactile_marker_features:
        marker_dim = 70
        self.vae_encoder_marker_proj = nn.Linear(marker_dim, config.dim_model // 2)
        self.vae_encoder_state_marker_fusion = nn.Linear(config.dim_model, config.dim_model)

# VAE encoder 的位置编码长度保持不变（marker 不作为独立 token）：
num_input_token_encoder = 1 + config.chunk_size  # cls + actions
if self.config.robot_state_feature:
    num_input_token_encoder += 1
```

#### 4.2.5 `ACTHao.forward` — 修改 Encoder 输入

```python
# ---- Encoder 输入构建 ----
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

# Robot state + marker displacement 合并后作为单个 token
if self.config.robot_state_feature:
    state = batch[OBS_STATE]  # (B, 6)
    state_proj = self.encoder_state_proj(state)  # (B, dim_model/2)
    
    if self.config.tactile_marker_features:
        marker = batch[OBS_TAC_MARKER]  # (B, 70)
        marker_proj = self.encoder_marker_proj(marker)  # (B, dim_model/2)
        # Concat：(B, dim_model/2) + (B, dim_model/2) = (B, dim_model)
        merged = torch.cat([state_proj, marker_proj], dim=-1)
        # 融合投影：(B, dim_model) → (B, dim_model)
        state_token = self.encoder_state_marker_fusion(merged)
    else:
        state_token = state_proj
    
    encoder_in_tokens.append(state_token)

# 标准图像特征（包括 global, inhand, tac_raw.tac1）
if self.config.image_features:
    for img in batch[OBS_IMAGES]:
        cam_features = self.backbone(img)["feature_map"]
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_features = self.encoder_img_feat_input_proj(cam_features)
        cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
        cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
        encoder_in_tokens.extend(list(cam_features))
        encoder_in_pos_embed.extend(list(cam_pos_embed))

# 触觉视觉特征（tac_depth + tac_normal 合成的 4 通道）
if self.config.has_tactile_vision:
    tac_vision = batch[OBS_TAC_VISION]  # (B, 4, H, W)
    tac_features = self.tactile_backbone(tac_vision)["feature_map"]
    tac_pos_embed = self.encoder_tac_feat_pos_embed(tac_features).to(dtype=tac_features.dtype)
    tac_features = self.encoder_tac_feat_input_proj(tac_features)
    tac_features = einops.rearrange(tac_features, "b c h w -> (h w) b c")
    tac_pos_embed = einops.rearrange(tac_pos_embed, "b c h w -> (h w) b c")
    encoder_in_tokens.extend(list(tac_features))
    encoder_in_pos_embed.extend(list(tac_pos_embed))
```

#### 4.2.6 VAE Encoder 输入修改

```python
# VAE encoder 的输入形式保持原始 ACT 的结构：[cls, robot_state_with_marker, *action_sequence]
# 使用相同的融合策略：state 和 marker 各自投影到 dim_model/2，再融合

if self.config.use_vae and self.training:
    cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)

    if self.config.robot_state_feature:
        state = batch[OBS_STATE]  # (B, 6)
        state_proj = self.vae_encoder_state_proj(state)  # (B, dim_model/2)
        
        if self.config.tactile_marker_features:
            marker = batch[OBS_TAC_MARKER]  # (B, 70)
            marker_proj = self.vae_encoder_marker_proj(marker)  # (B, dim_model/2)
            # Concat 后融合
            merged = torch.cat([state_proj, marker_proj], dim=-1)  # (B, dim_model)
            robot_state_embed = self.vae_encoder_state_marker_fusion(merged).unsqueeze(1)
        else:
            robot_state_embed = state_proj.unsqueeze(1)

    action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

    # 构建输入序列
    if self.config.robot_state_feature:
        vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)
    else:
        vae_encoder_input = torch.cat([cls_embed, action_embed], axis=1)
```

**位置编码和 key_padding_mask 无需改变**，因为 marker 不作为独立 token，只是 state 的扩展。

#### 4.2.7 `get_optim_params` — 独立学习率

触觉 backbone 完全参与训练，但你可能希望为其设置独立的学习率:

```python
def get_optim_params(self) -> dict:
    return [
        {
            "params": [
                p for n, p in self.named_parameters()
                if not n.startswith("model.backbone")
                and not n.startswith("model.tactile_backbone")
                and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in self.named_parameters()
                if n.startswith("model.backbone") and p.requires_grad
            ],
            "lr": self.config.optimizer_lr_backbone,
        },
        {
            # 触觉 backbone 使用主学习率（全部训练，不冻结）
            "params": [
                p for n, p in self.named_parameters()
                if n.startswith("model.tactile_backbone") and p.requires_grad
            ],
            "lr": self.config.optimizer_lr,  # 或者新增一个 optimizer_lr_tactile_backbone
        },
    ]
```

### 4.3 `processor_act_hao.py`

基本与 `processor_act.py` 相同。需要确保 normalization 能覆盖到 `TACTILE` 类型的特征。
关键点：在 `normalization_mapping` 中加入 `"TACTILE"` 键（已在 configuration 中添加）。

```python
# 与 processor_act.py 基本相同，复制即可
# 唯一确认点：NormalizerProcessorStep 能根据 normalization_mapping 正确处理 TACTILE 类型
```

### 4.4 `factory.py` 修改

在 `get_policy_class` 中添加：

```python
elif name == "act_hao":
    from lerobot.policies.act_hao.modeling_act_hao import ACTHaoPolicy
    return ACTHaoPolicy
```

在 `make_pre_post_processors` 中添加：

```python
elif cfg.type == "act_hao":
    from lerobot.policies.act_hao.processor_act_hao import make_act_hao_pre_post_processors
    processors = make_act_hao_pre_post_processors(config=cfg, dataset_stats=ds_stats)
```

---

## 5. 特征分类关键点

你的数据集中各特征在 `dataset_to_policy_features()` 中的自动分类结果：

| 键 | dtype | _is_tactile_feature? | 最终 FeatureType | shape 处理 |
|---|---|---|---|---|
| `observation.images.global` | video | No | **VISUAL** | 自动 HWC→CHW |
| `observation.images.inhand` | video | No | **VISUAL** | 自动 HWC→CHW |
| `observation.images.tac_raw.tac1` | video | Yes, 但 dtype=video 先判断 | **VISUAL** | 自动 HWC→CHW |
| `observation.tac_depth.tac1` | float32 | Yes | **TACTILE** | 保持原始 (480,640,1) |
| `observation.tac_normal.tac1` | float32 | Yes | **TACTILE** | 保持原始 (480,640,3) |
| `observation.tac_marker_displacement.tac1` | float32 | Yes | **TACTILE** | 保持原始 (35,2) |
| `observation.state` | float32 | No | **STATE** | 保持原始 (6,) |
| `action` | float32 | No | **ACTION** | 保持原始 (6,) |

> **注意**：`observation.images.tac_raw.tac1` 因为 `dtype == "video"`，在 `dataset_to_policy_features` 中会首先匹配 video/image 条件，被归类为 `FeatureType.VISUAL`。这正好符合你的需求。

> **注意**：`tac_depth` 和 `tac_normal` 的 shape 是 `(480, 640, 1)` 和 `(480, 640, 3)`，dtype 是 float32 而非 video。它们不会经过 channel-first 转换。你需要在模型内部手动处理维度排列。

---

## 6. Token 结构总结

### Transformer Encoder 输入

```
[latent(1)] [state_token(1)] [global_img(H'*W')] [inhand_img(H'*W')] [tac_raw_img(H'*W')] [tac_depth_normal(H''*W'')]
```

其中 `H'*W'` 是 ResNet18 对 640×480 图像下采样后的 feature map 大小（约 15×20=300 tokens per camera）。

**state_token 的生成过程**：
- state(6) → Linear(6, dim_model/2) → `(B, dim_model/2)`
- marker(70) → Linear(70, dim_model/2) → `(B, dim_model/2)`
- concat([state_proj, marker_proj]) → `(B, dim_model)`
- Linear(dim_model, dim_model) → `(B, dim_model)` ✓ 最终 token

### VAE Encoder 输入

```
[cls(1)] [state_token(1)] [action_sequence(chunk_size)]
```

**state_token 的生成过程**（相同融合策略）：
- state(6) → Linear(6, dim_model/2) → `(B, dim_model/2)`
- marker(70) → Linear(70, dim_model/2) → `(B, dim_model/2)`
- concat([state_proj, marker_proj]) → `(B, dim_model)`
- Linear(dim_model, dim_model) → `(B, dim_model)` ✓ 最终 token

---

## 7. 训练命令示例

```bash
lerobot-train \
    --dataset.repo_id=act_hao_task1 \
    --dataset.root=hao_datasets/grasp_cup \
    --policy.type=act_hao \
    --output_dir=./checkpoints/hao_proj/task1 \
    --policy.repo_id=train_task1 \
    --batch_size=8 \
    --num_workers=8 \
    --steps=100000 \
    --wandb.enable=true \
    --policy.push_to_hub=false
```

> **注意 batch_size**：由于每帧有 3 个普通图像 + 1 个 480×640 的 4 通道触觉图经过 ResNet，显存占用会显著增大。建议从 batch_size=8 开始尝试，根据 GPU 显存调整。

---

## 8. 实现清单

按以下顺序实现：

- [x] 创建 `src/lerobot/policies/act_hao/__init__.py`
- [x] 创建 `configuration_act_hao.py`（参考 §4.1）
- [x] 创建 `modeling_act_hao.py`（参考 §4.2，这是最复杂的部分）
- [x] 创建 `processor_act_hao.py`（参考 §4.3，基本复制 ACT 的 processor）
- [x] 修改 `factory.py` 添加 `act_hao` 注册（参考 §4.4）
- [x] 修改 `policies/__init__.py` 导出 `ACTHaoConfig`
- [x] 验证 `dataset_to_policy_features` 对各键的分类是否正确
- [x] 小 batch 冒烟测试（batch_size=2, steps=10, 通过）
- [ ] 完整训练

---

## 9. 潜在风险与注意事项

1. **显存压力**：480×640 分辨率的图像通过 ResNet18 会生成较大的 feature map（约 15×20=300 tokens）。4 个图像源 = ~1200 tokens。加上触觉图再加 ~300 tokens。Transformer 的自注意力对 token 数量是 O(n²)，可能会很慢。考虑降低输入分辨率或增大 `replace_final_stride_with_dilation`。

2. **tac_depth / tac_normal 归一化**：作为 TACTILE 类型的特征，需确保 `normalization_mapping` 中包含 `"TACTILE"` 键，且训练时能正确计算这些特征的统计量。

3. **tac_raw 与视觉图像共享 backbone**：共享 backbone 意味着触觉原始图像和普通相机图像使用相同的特征提取器。如果触觉图像域差异很大，可能需要考虑是否应使用独立 backbone。当前设计是共享的。

4. **4 通道 ResNet 无法使用预训练权重**：这意味着触觉 backbone 需要从头训练，可能需要更多数据或更长训练时间。

5. **✓ State 和 Marker 融合的优势**（§4.2.3 和 §4.2.4）：
   - **维度平衡**：state(6维) 和 marker(70维) 各投影到 dim_model/2，确保两者贡献相等
   - **特征解耦**：独立的投影层让模型学习各自的表示空间
   - **数值稳定**：避免 70 维特征压制 6 维特征导致的梯度失衡
   - **两个编码器一致**：Transformer Encoder 和 VAE Encoder 使用相同的融合策略，保证 state 信息的对称处理

---

## 10. 实现记录

### 实现日期：2026-03-05

### 创建的文件

| 文件 | 说明 |
|---|---|
| `__init__.py` | 包初始化，导出 `ACTHaoConfig` 和 `ACTHaoPolicy` |
| `configuration_act_hao.py` | 配置类 `ACTHaoConfig`，使用 `@PreTrainedConfig.register_subclass("act_hao")` 注册 |
| `modeling_act_hao.py` | 核心模型，包含 `ACTHaoPolicy`、`ACTHao`、`ACTEncoder`、`ACTDecoder` 等 |
| `processor_act_hao.py` | 前后处理管线 `make_act_hao_pre_post_processors`，与原始 ACT 一致，新增 TACTILE 归一化支持 |
| `ADAPTATION_GUIDE.md` | 本设计与实现文档 |

### 修改的文件

| 文件 | 修改内容 |
|---|---|
| `policies/__init__.py` | 添加 `from .act_hao.configuration_act_hao import ACTHaoConfig as ACTHaoConfig` 导出 |
| `policies/factory.py` | 添加 `ACTHaoConfig` 导入、`get_policy_class` 注册、`make_policy_config` 注册、`make_pre_post_processors` 注册 |

### 关键实现细节

1. **配置类 (`ACTHaoConfig`)**：
   - 继承 `PreTrainedConfig`，通过 `@register_subclass("act_hao")` 注册
   - 新增 `tactile_vision_backbone`、`tactile_backbone_in_channels`、`tactile_pretrained_backbone_weights` 等触觉配置
   - 新增 `normalization_mapping` 中的 `"TACTILE"` 键
   - 新增属性方法：`tactile_depth_features`、`tactile_normal_features`、`tactile_marker_features`、`has_tactile_vision`

2. **模型类 (`ACTHao`)**：
   - **Image Backbone**：共享 ResNet18（pretrained + FrozenBN），处理 global、inhand、tac_raw 三个 VISUAL 图像
   - **Tactile Backbone**：独立 ResNet18（无 pretrained、普通 BN），首层 conv1 改为 4 通道输入
   - **State + Marker 融合**：project-then-fuse 策略
     - `encoder_state_proj`: Linear(6, dim_model/2)
     - `encoder_marker_proj`: Linear(70, dim_model/2)
     - `encoder_state_marker_fusion`: Linear(dim_model, dim_model)
     - VAE Encoder 使用独立但相同策略的投影层
   - **数据准备** (`_prepare_batch`):
     - 收集 VISUAL 图像列表到 `OBS_IMAGES`
     - 合成 depth(1ch) + normal(3ch) → 4ch，手动 permute 到 BCHW
     - Flatten marker: (B,35,2) → (B,70)

3. **Optimizer 分组**：
   - 主参数：默认学习率
   - Image backbone：`optimizer_lr_backbone`
   - Tactile backbone：主学习率（从零训练）

### 冒烟测试结果

```
测试命令:
lerobot-train \
    --dataset.repo_id=act_hao_task1 \
    --dataset.root=hao_datasets/grasp_cup \
    --policy.type=act_hao \
    --output_dir=./checkpoints/hao_proj/task1_test \
    --batch_size=2 --num_workers=4 --steps=10 \
    --wandb.enable=false --policy.push_to_hub=false

结果: 10 步训练成功完成
模型参数量: 63,597,574 (64M)
环境: conda activate lerobot_hao
```
