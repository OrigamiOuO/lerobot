# PI05 模型深度分析：Attention Mask 与 KV Cache

## 第一部分：Attention Mask 详解

### 1. 架构理解

PaliGemma 是一个 **Transformer Decoder 架构**，但这里有个特殊设计：它同时处理两个序列（前缀和后缀），通过 `PaliGemmaWithExpertModel` 中两个模型共享 attention：

- **PaliGemma**: 处理 prefix（观察）
- **Gemma Expert**: 处理 suffix（动作）

```
前缀 (Prefix)         后缀 (Suffix)
┌──────────────────┐  ┌──────────────┐
│ 图像特征          │  │ 动作特征      │
│ 语言 tokens       │  │ 时间步 embedding│
└──────────────────┘  └──────────────┘
```

#### 源代码位置

**[modeling_pi05.py:435-449]** - PaliGemmaWithExpertModel.forward 方法

```python
def forward(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: list[torch.FloatTensor] | None = None,
    use_cache: bool | None = None,
    adarms_cond: list[torch.Tensor] | None = None,
):
    if adarms_cond is None:
        adarms_cond = [None, None]
    if inputs_embeds[1] is None:
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
        )
```

### 2. 两层 Mask 系统

#### 📊 一维 Mask（逻辑控制）

**源代码位置：[modeling_pi05.py:633-680]** - embed_prefix 和 embed_suffix 方法

**embed_prefix 中的设置**：
```python
def embed_prefix(
    self, images, img_masks, tokens, masks
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed images with SigLIP and language tokens with embedding layer."""
    embs = []
    pad_masks = []
    att_masks = []

    # Process images
    for img, img_mask in zip(images, img_masks, strict=True):
        # ...
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs  # 图像: 0 (都可以互相看)

    # Process language tokens
    # ...
    embs.append(lang_emb)
    pad_masks.append(masks)
    
    num_lang_embs = lang_emb.shape[1]
    att_masks += [0] * num_lang_embs  # 语言: 0 (都可以互相看)
```

**embed_suffix 中的设置**：
```python
def embed_suffix(self, noisy_actions, timestep):
    """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
    embs = []
    pad_masks = []
    att_masks = []
    
    # ... 时间步和动作embedding ...
    
    # Set attention masks so that image, language and state inputs do not attend to action tokens
    att_masks += [1] + ([0] * (self.config.chunk_size - 1))
    # 动作: [1, 0, 0, ...] (causal - 只能看历史)
```

**含义**：
- `0` = "attentional token"（可以看之前所有 0 tokens）
- `1` = "causal token"（开始新的 causal 序列，只能看历史）

#### 📍 二维 Mask（硬件执行）

**源代码位置：[modeling_pi05.py:99-128]** - make_att_2d_masks 函数

```python
def make_att_2d_masks(pad_masks, att_masks):  
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
```

import torch
import torch.nn as nn

class TransformerDenoiser(nn.Module):
    def __init__(self, action_dim, obs_dim, n_layer=8, n_head=8, n_emb=256):
        super().__init__()
        # 1. 动作轨迹的线性投影 (Input: Batch, T, action_dim)
        self.action_proj = nn.Linear(action_dim, n_emb)

        # 2. 环境观测的线性投影 (Input: Batch, obs_dim)
        self.obs_proj = nn.Linear(obs_dim, n_emb)

        # 3. 扩散步数 (Time Step) 的 Embedding
        self.time_proj = nn.Embedding(1000, n_emb) 

        # 4. Transformer 层 (通常使用 Encoder 结构作为去噪器)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 5. 输出层：还原回动作维度
        self.output_head = nn.Linear(n_emb, action_dim)

    def forward(self, sample, timestep, obs):
        """
        sample: 带有噪声的动作序列 [B, T, action_dim]
        timestep: 当前扩散步数 [B]
        obs: 环境观测特征 [B, obs_dim]
        """
        # 准备输入 Token
        a_tokens = self.action_proj(sample) # [B, T, n_emb]

        # 将 obs 和 time 转化为与 action 维度一致的 Token
        o_token = self.obs_proj(obs).unsqueeze(1)    # [B, 1, n_emb]
        t_token = self.time_proj(timestep).unsqueeze(1) # [B, 1, n_emb]

        # 拼接所有 Token: [Obs, Time, Action_1, ..., Action_T]
        # 这里体现了 cat 操作的用法
        input_tokens = torch.cat([o_token, t_token, a_tokens], dim=1) 

        # 通过 Transformer 处理全局关联
        output = self.transformer(input_tokens)

        # 只取 Action 部分对应的输出进行去噪预测
        # 这里体现了切片操作
        action_output = output[:, 2:, :] 

        return self.output_head(action_output)


**生成逻辑示例**：

假设：3个图像 tokens + 5个语言 tokens + 4个动作 tokens

```
    img    lang    action
    001    01234   t+0123
img [××××××××××××|····]
    [××××××××××××|····]
    [××××××××××××|····]
lng [××××××××××××|····]
    [××××××××××××|····]
    [××××××××××××|····]
    [××××××××××××|····]
    [××××××××××××|····]
act [——————————×|××××]  ← 动作只能看 prefix (和历史动作)
    [——————————×|××××]
    [——————————×|××××]
    [——————————×|××××]

× = 可以 attend
— = masked (不能看)
```

**cum_sum 计算细节**：
```
att_masks = [0,0,0, 0,0,0,0,0, 1,0,0,0]
cumsum    = [0,0,0, 0,0,0,0,0, 1,1,1,1]

# att_2d_masks[i,j] = True 当且仅当 cumsum[j] <= cumsum[i]
# 对于动作部分 (cumsum=1,1,1,1)，只能看前8个 (cumsum=0) 和自己之前的
```

### 3. 训练 vs 推理的两大差异

#### **训练过程** (`forward` 方法)

**源代码位置：[modeling_pi05.py:720-767]** - forward 方法

```python
def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
    """Do a full training forward pass and compute the loss."""
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

    # 组合 mask
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

    def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],  # 【同时处理】
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    suffix_out = self._apply_checkpoint(
        forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
    )
```

特点：
- ✅ 完整序列: [prefix + suffix]
- ✅ 使用完整 2D mask
- ✅ suffix 的每个 token 都同时看到整个 prefix
- ✅ 一次 forward 传播，计算完整 MSE 损失

#### **推理过程** (`sample_actions` → `denoise_step`)

**源代码位置：[modeling_pi05.py:778-855]** - sample_actions 方法

```python
@torch.no_grad()
def sample_actions(
    self,
    images,
    img_masks,
    tokens,
    masks,
    noise=None,
    num_steps=None,
    **kwargs: Unpack[ActionSelectKwargs],
) -> Tensor:
    """Do a full inference forward and compute the action."""
    if num_steps is None:
        num_steps = self.config.num_inference_steps

    bsize = tokens.shape[0]
    device = tokens.device

    # ...
    
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],  # 【只处理 prefix】
        use_cache=True,
    )

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        v_t = denoise_step_partial_call(x_t)
        x_t = x_t + dt * v_t

    return x_t
```

特点：
- ✅ 两阶段处理：先处理 prefix，再逐步处理 suffix
- ✅ 利用 KV cache 复用 prefix 的特征
- ✅ 多次 forward（100+ 步），每步推理成本低

**源代码位置：[modeling_pi05.py:859-894]** - denoise_step 方法

```python
def denoise_step(
    self,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """Apply one denoising step of the noise `x_t` at a given timestep."""
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 【关键】使用缓存
        inputs_embeds=[None, suffix_embs],  # 【只处理新 suffix】
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)
```

### 4. 对比总结

| 方面 | 训练 | 推理 |
|------|------|------|
| **序列组合** | [Prefix: 8 tokens] + [Suffix: 4 tokens] **同时** | Prefix 先处理一次，Suffix 逐步增长 |
| **Attention 方向** | 双向：Suffix → Prefix + Suffix 前向 | 单向：Suffix → Prefix (缓存) + Suffix 前向 |
| **Mask 含义** | 完整的 [12×12] 2D mask | 动态的多个小 masks，利用 KV cache |
| **效率** | 一次 forward，计算完整 MSE 损失 | 多次 forward (100+ 步)，每步推理成本低 |

---

## 第二部分：KV Cache 详解

### 1. 显式 KV Cache 设置

虽然底层实现来自 Hugging Face transformers 库，但在 PI05 代码中有**明确的显式设置**。

#### 📍 第一步：计算并缓存 Prefix 的 KV

**源代码位置：[modeling_pi05.py:814-822]**

```python
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,              # ← 初始为空
    inputs_embeds=[prefix_embs, None], # ← 只处理 prefix
    use_cache=True,                    # ← 【关键】启用缓存
)
```

**发生了什么**：
- `use_cache=True` 信号 transformers 库保存每一层的 **Key 和 Value 向量**
- 返回的 `past_key_values` 是一个元组列表

**结构**：
```python
past_key_values = [
    (layer_0_key, layer_0_value),      # 第0层的 K,V [B, 1, 100, 256]
    (layer_1_key, layer_1_value),      # 第1层的 K,V
    ...
    (layer_17_key, layer_17_value),    # 第17层的 K,V (Gemma有18层)
]
```

#### 📍 第二步：推理循环中复用缓存

**源代码位置：[modeling_pi05.py:825-850]**

```python
for step in range(num_steps):
    time = 1.0 + step * dt
    time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

    def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
        return self.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,  # 【复用缓存】
            x_t=input_x_t,
            timestep=current_timestep,
        )

    # ...
    v_t = denoise_step_partial_call(x_t)
    x_t = x_t + dt * v_t
```

#### 📍 第三步：在 denoise_step 中使用缓存

**源代码位置：[modeling_pi05.py:883-894]**

```python
outputs_embeds, _ = self.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,        # ← 【关键】传入缓存
    inputs_embeds=[None, suffix_embs],      # ← 【关键】只处理新的 suffix
    use_cache=False,                        # ← 不再更新缓存
    adarms_cond=[None, adarms_cond],
)

suffix_out = outputs_embeds[1]
suffix_out = suffix_out[:, -self.config.chunk_size :]
suffix_out = suffix_out.to(dtype=torch.float32)
return self.action_out_proj(suffix_out)
```

### 2. KV Cache 在 PaliGemmaWithExpertModel 中的具体处理

**源代码位置：[modeling_pi05.py:448-463]**

```python
def forward(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: list[torch.FloatTensor] | None = None,
    use_cache: bool | None = None,
    adarms_cond: list[torch.Tensor] | None = None,
):
    if adarms_cond is None:
        adarms_cond = [None, None]
    if inputs_embeds[1] is None:
        # 只处理 prefix，保存 KV 缓存
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,  # None 在第一次
            use_cache=use_cache,              # True
            adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
        )
        prefix_past_key_values = prefix_output.past_key_values  # 【提取缓存】
        prefix_output = prefix_output.last_hidden_state
        suffix_output = None
        
    elif inputs_embeds[0] is None:
        # 只处理 suffix，复用缓存
        suffix_output = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds[1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,  # 【复用缓存】
            use_cache=use_cache,
            adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
        )
        suffix_output = suffix_output.last_hidden_state
        prefix_output = None
        prefix_past_key_values = None
```

### 3. 两个流程的差异

#### **训练时** (`forward` 方法)

```python
def forward(self, images, img_masks, tokens, masks, actions, ...):
    # ...
    inputs_embeds=[prefix_embs, suffix_embs]  # 都是 non-None
    
    # PaliGemmaWithExpertModel.forward 中的处理
    # → inputs_embeds[1] 不是 None，进入 "两者都处理" 的分支
    # → 不使用 KV cache（同时处理 prefix 和 suffix）
    # → 详见 [modeling_pi05.py:468-538]
```

#### **推理时** (`sample_actions` → `denoise_step`)

```python
# Step 1: 只处理 prefix（[modeling_pi05.py:814-822]）
inputs_embeds=[prefix_embs, None]    # ← suffix=None
use_cache=True                       # ← 保存 K,V
→ 进入 inputs_embeds[1] is None 分支
→ 计算 prefix 并提取 past_key_values

# Step 2-101: 重复使用缓存（[modeling_pi05.py:825-894]）
inputs_embeds=[None, suffix_embs]    # ← prefix=None
past_key_values=xxx                  # ← 复用 prefix 的缓存
use_cache=False                      # ← 不更新缓存
→ 进入 inputs_embeds[0] is None 分支
→ 只计算 suffix，不保存 K,V
```

### 4. 底层实现说明

KV cache 的**底层合并逻辑**在 Hugging Face transformers 库的 Gemma 模型中实现。在 PI05 代码中虽然没有显式的拼接代码，但通过 `use_cache` 和 `past_key_values` 参数确保了缓存的正确使用。

**transformers/models/gemma/modeling_gemma.py 中的逻辑（仅为说明）**：

```python
class GemmaAttention(nn.Module):
    def forward(self, hidden_states, ..., past_key_value=None, use_cache=False):
        # Q, K, V projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 【关键】如果有缓存，concatenate 旧的 K,V
        if past_key_value is not None:
            # past_key_value 形状: (batch, heads, seq_len_old, head_dim)
            # key_states 形状: (batch, heads, 4, head_dim)  ← 新 suffix
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            # 结果: (batch, heads, seq_len_old+4, head_dim)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)
        
        # Attention 计算
        attn_output = scaled_dot_product_attention(
            query_states,           # (B, H, 4, head_dim)
            key_states,             # (B, H, 100+4, head_dim)
            value_states,           # (B, H, 100+4, head_dim)
            ...
        )
        
        # 保存新的 K,V 供下一步使用
        if use_cache:
            cache = (key_states, value_states)
        else:
            cache = None
            
        return attn_output, cache
```

### 5. 内存节省效果

假设 Prefix 序列长度 = 100，Suffix 每步长度 = 4，推理步数 = 100

**Attention 计算对比**：

| 方案 | 计算量 | Attention 复杂度 |
|------|-------|-----------------|
| **无缓存** | 100×100 + 100×(100+4)² ≈ 1.08M | O(100 × 104²) |
| **有缓存** | 100×100 + 100×(4×100) ≈ 0.05M | O(100×100 + 100×4×100) |
| **加速比** | ~20倍 | ~100倍 |

**流程可视化**：

```
Step 0 (prefill - 只发生一次):
┌─ Prefix [B, 100, D]
├─> Q,K,V 计算 [B, H, 100, head_dim]
├─> Attention [B, H, 100, 100]
└─> K,V 保存到 past_key_values
    ↓
    past_key_values = {
        layer_0: (K[B,H,100,head_dim], V[B,H,100,head_dim]),
        layer_1: (...),
        ...
        layer_17: (...)
    }

Step 1 (decode - 重复 100 次):
┌─ Suffix [B, 4, D]
├─> Q 计算 [B, H, 4, head_dim]
├─> K,V 从 past_key_values 读取 [B, H, 100, head_dim]
├─> cat: [K, K_new] → [B, H, 104, head_dim]（仅在 transformers 内部）
├─> Attention(Q: [B,H,4,D], K: [B,H,100,D], V: [B,H,100,D])
│   → Matmul: (4 × head_dim) @ (head_dim × 100) = 4×100 operations
│   → vs 无缓存: (104 × head_dim) @ (head_dim × 104) = 104×104 operations
└─> 输出 [B, H, 4, head_dim]

说明：每步只需计算新 token 对所有历史 token 的 attention，
而不是重新计算整个序列的 attention
```

### 6. 重要细节

#### **为什么在 denoise_step 中 use_cache=False？**

```python
# denoise_step 中的设置
outputs_embeds, _ = self.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,  # 【输入：使用缓存】
    inputs_embeds=[None, suffix_embs],
    use_cache=False,                  # 【不保存新缓存】
)
```

**原因**：

- `use_cache=True`：计算新的 K,V 并保存回缓存（用于下一步）
- `use_cache=False`：只读取缓存，不更新它（节省内存）

在推理循环中，prefix 的 KV 缓存在所有 100 步中都是**不变的**，只有 query 在变化（因为 suffix 在变化）。因此：

```python
for step in range(num_steps):
    # 每次迭代：
    # - past_key_values 保持不变（只有 prefix 的 K,V）
    # - suffix_embs 改变（新的表示）
    # - 计算新 suffix 对旧 prefix 的 attention
    denoise_step(..., past_key_values=past_key_values, use_cache=False)
```

---

## 总结

### Attention Mask 的核心机制

1. **一维 mask** 编码了 Prefix-LM 式注意力规则
2. **二维 mask** 通过累积求和从一维 mask 生成
3. **训练**：完整序列同时处理，确保模型学习正确的因果关系
4. **推理**：利用 KV 缓存加速多步去噪过程

### KV Cache 的核心机制

1. **计算阶段**（Prefill）：处理整个 prefix，保存所有层的 K,V
2. **生成阶段**（Decode）：只计算新 suffix 对缓存 prefix 的 attention
3. **效率提升**：从 $O(n^2)$ 降至 $O(n)$ 的增量计算
4. **内存优化**：不再更新缓存（use_cache=False），节省显存

这个设计使 PI05 能够在保证训练稳定性的同时，实现高效的推理性能。
