#!/usr/bin/env python

# 评估 Tactile Diffusion Policy Checkpoint
# 脚本位置: examples/training/evaluate_tactile_ckpt.py

import argparse
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Tactile Diffusion Policy checkpoint")
    parser.add_argument(
        "--pretrained_path", 
        type=str, 
        required=True, 
        help="Path to the pretrained model directory (containing config.json and model.safetensors)"
    )
    parser.add_argument(
        "--dataset_repo_id", 
        type=str, 
        default="tactile_dp_test_data/xarm_leap_tactile_lift_blind",
        help="Dataset repo_id"
    )
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default=None, 
        help="Path to local dataset root. If None, will use random data."
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate (if None, evaluate all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_dummy_data", action="store_true", help="Force use of dummy random data")
    return parser.parse_args()

def get_dummy_batch(cfg, batch_size=4, device="cpu"):
    """生成符合策略输入要求的随机数据"""
    batch = {}
    
    # 构建 observation
    # 注意：policy 训练时通常期望输入 shape 为 (B, n_obs_steps, feature_dim)
    for name, feature in cfg.input_features.items():
        # feature.shape 是 (dim,)
        dim = feature.shape[0]
        # 随机生成 (B, n_obs_steps, dim)
        batch[name] = torch.randn(batch_size, cfg.n_obs_steps, dim, device=device)
        
    # 构建 ground truth action
    # 训练时的 dataset 通常提供 (B, horizon, action_dim) 的 action 序列
    action_dim = cfg.output_features["action"].shape[0]
    batch["action"] = torch.randn(batch_size, cfg.horizon, action_dim, device=device)
    
    return batch

def evaluate_batch(policy, batch, device):
    """
    运行单次推理并计算与 Ground Truth 的对比指标
    
    Args:
        policy: 加载好的策略模型
        batch: 数据字典，包含 observation.* 和 action
        device: 设备
        
    Returns:
        metrics: 字典，包含 MSE loss 等指标
    """
    policy.eval()
    
    # 准备输入
    # 过滤出 observation 键，并确保在正确设备上
    obs_batch = {k: v.to(device) for k, v in batch.items() if k.startswith("observation.")}
    
    # Ground Truth Action
    # dataset 返回的 batch["action"] 通常是 (B, horizon, action_dim)
    # 或者是 (B, action_dim) 取决于 dataset 配置。
    # 标准 LeRobot 训练中通常是有 horizon 维度的。
    gt_act_seq = batch["action"].to(device)
    
    # 获取 Ground Truth 的第一步动作，用于对比 select_action 的输出
    # 如果 gt_act_seq 是 3D (B, T, D)
    if gt_act_seq.ndim == 3:
        gt_next_action = gt_act_seq[:, 0, :]
    else:
        # 假设已经是 (B, D)
        gt_next_action = gt_act_seq

    # 运行策略
    # reset() 很重要，因为 DiffusionPolicy 通常是有状态的 (action queue)
    # 对于开环测试 (Open-loop Evaluation)，我们希望基于当前的 observation history 预测
    policy.reset()
    
    with torch.no_grad():
        # select_action 返回下一步要执行的动作 (B, action_dim)
        pred_action = policy.select_action(obs_batch)
    
    # 计算指标
    # MSE Loss
    mse = torch.nn.functional.mse_loss(pred_action, gt_next_action)
    # L1 Loss / MAE
    mae = torch.nn.functional.l1_loss(pred_action, gt_next_action)

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "pred_sample": pred_action[0].cpu().numpy(),
        "gt_sample": gt_next_action[0].cpu().numpy()
    }

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading policy from {args.pretrained_path}...")
    # make_policy 会自动读取 config.json 并加载权重
    policy = make_policy(pretrained_model_name_or_path=args.pretrained_path)
    policy.to(device)
    policy.eval()
    print("Policy loaded successfully.")
    
    dataloader = None
    
    if args.use_dummy_data or not args.dataset_root:
        print("Using Dummy/Random Data for evaluation...")
        # 只是为了演示循环，造几个 dummy batch
        dummy_batches = [get_dummy_batch(policy.config, args.batch_size, device) for _ in range(5)]
        dataloader = dummy_batches
    else:
        print(f"Loading Dataset from {args.dataset_root}...")
        
        # 构造 delta_timestamps 来正确读取时间窗口
        # 通常从 config 中获取这些参数
        cfg = policy.config
        
        # 如果 config 中没有 delta_indices，使用默认值或报错
        obs_delta = getattr(cfg, "observation_delta_indices", None)
        act_delta = getattr(cfg, "action_delta_indices", None)
        
        if obs_delta is None or act_delta is None:
            print("Warning: Config missing delta_indices, assuming defaults (obs: [0], act: [0])")
            # 这对于 diffusion policy 通常是不对的，但作为 fallback
            obs_dict = {k: [0.0] for k in cfg.input_features}
            act_dict = {"action": [0.0]}
            delta_timestamps = {**obs_dict, **act_dict}
        else:
            # 需要 fps 来转换 index 到 timestamp？
            # 实际上 LeRobotDatasetMetadata 会提供 fps
            metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
            fps = metadata.fps
            
            delta_timestamps = {}
            for k in cfg.input_features:
                if k.startswith("observation."):
                    delta_timestamps[k] = [i / fps for i in obs_delta]
            
            delta_timestamps["action"] = [i / fps for i in act_delta]
            
        dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            delta_timestamps=delta_timestamps
        )
        print(f"Dataset size: {len(dataset)}")
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

    print("Starting evaluation...")
    metrics_list = []
    
    # limit episodes if needed
    for i, batch in tqdm(enumerate(dataloader)):
        if args.num_episodes and i >= args.num_episodes:
            break
            
        metrics = evaluate_batch(policy, batch, device)
        metrics_list.append(metrics)
        
        if i == 0:
            print(f"\n[Preview Batch 0]")
            print(f"MSE: {metrics['mse']:.6f}")
            print(f"Pred Action (dim 0-5): {metrics['pred_sample'][:5]}")
            print(f"GT Action   (dim 0-5): {metrics['gt_sample'][:5]}")

    # 汇总指标
    avg_mse = np.mean([m["mse"] for m in metrics_list])
    avg_mae = np.mean([m["mae"] for m in metrics_list])
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Samples Evaluated: {len(metrics_list) * args.batch_size}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print("="*50)

if __name__ == "__main__":
    init_logging()
    main()
