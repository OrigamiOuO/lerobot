训练指令（参考）：
lerobot-train \
    --dataset.preload_to_memory=true \
    --dataset.repo_id=grasp_multi_10 \
    --dataset.root=datasets/luo_proj/Blind_Grasping_LeRobot_No_HandPC \
    --policy.type=pretrain_act \
    --output_dir=./checkpoints/luo_proj/act_pretrain_10obj \
    --policy.repo_id=pretrain_act_10obj \
    --batch_size=512 \
    --num_workers=12 \
    --policy.use_amp=true \
    --steps=60000 \
    --log_freq=1000 \
    --wandb.enable=true \
    --job_name=act_pretrain_10obj_test \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=2000 \
    --save_freq=10000

与 pretrain_diffusion 的差异：
- 动作生成器：用 Transformer Decoder (ACT-style) 直接回归 action chunk，而不是 Diffusion UNet 迭代去噪
- 共享相同的观测编码管线（state_mlp + pretrained sparse_pc encoder）
- 适合做 ablation study，对比不同 action generation head 的效果
