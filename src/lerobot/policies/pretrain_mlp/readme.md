# Pretrain MLP Policy

最简单的 ablation baseline：用一个大 MLP 替代 Transformer Decoder / Diffusion UNet 作为动作生成头。

## 与 pretrain_act / pretrain_diffusion 的差异

- **观测编码管线完全相同**：state_mlp + pretrained sparse_pc encoder
- **动作生成器**：一个多层的 MLP（默认 4 层，每层 2048 dim），直接从全局条件向量回归整个 action chunk
- 没有时序建模（self-attention），没有迭代去噪，就是一个简单的前馈网络

## 训练指令（参考）

```bash
lerobot-train \
    --dataset.preload_to_memory=true \
    --dataset.repo_id=grasp_multi_10 \
    --dataset.root=datasets/luo_proj/Blind_Grasping_LeRobot_No_HandPC \
    --policy.type=pretrain_mlp \
    --output_dir=./checkpoints/luo_proj/mlp_pretrain_10obj \
    --policy.repo_id=pretrain_mlp_10obj \
    --batch_size=512 \
    --num_workers=12 \
    --policy.use_amp=true \
    --steps=60000 \
    --log_freq=1000 \
    --wandb.enable=true \
    --job_name=mlp_pretrain_10obj_test \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=2000 \
    --save_freq=10000
```

## MLP 结构

默认:

```
Input(global_cond_dim=896)
  → Linear(896, 2048) → ReLU
  → Linear(2048, 2048) → ReLU
  → Linear(2048, 2048) → ReLU
  → Linear(2048, 2048) → ReLU
  → Linear(2048, horizon * action_dim)
  → Reshape(B, horizon, action_dim)
```

可通过 `--policy.mlp_hidden_dims='[1024,1024,1024]'` 调整隐藏层维度。
