#!/usr/bin/env python
"""
测试脚本：检验 Diffusion Policy 在训练数据上的过拟合程度

这个脚本用于：
1. 加载已训练的 Diffusion checkpoint
2. 在训练集数据上进行推理
3. 计算预测动作和真实动作之间的误差
4. 评估模型是否能够较好地拟合训练数据
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference


def compute_action_metrics(predicted_actions: np.ndarray, ground_truth_actions: np.ndarray) -> dict:
    """
    计算预测动作和真实动作之间的误差指标

    Args:
        predicted_actions: (N, action_dim) 预测的动作
        ground_truth_actions: (N, action_dim) 真实的动作

    Returns:
        dict: 包含各种误差指标的字典
    """
    # 计算各维度的误差
    errors = predicted_actions - ground_truth_actions
    abs_errors = np.abs(errors)

    metrics = {
        "mse": np.mean((errors**2)),  # 均方误差
        "mae": np.mean(abs_errors),  # 平均绝对误差
        "max_error": np.max(abs_errors),  # 最大误差
        "min_error": np.min(abs_errors),  # 最小误差
        "rmse": np.sqrt(np.mean((errors**2))),  # 均方根误差
    }

    # 计算每个动作维度的误差
    per_dim_mae = np.mean(abs_errors, axis=0)
    per_dim_mse = np.mean((errors**2), axis=0)

    return metrics, per_dim_mae, per_dim_mse


def test_diffusion_overfit(
    checkpoint_path: str,
    dataset_repo_id: str,
    dataset_root: str,
    num_episodes: int = 10,
    device: str = "cuda",
):
    """
    在训练数据上测试模型的过拟合程度

    Args:
        checkpoint_path: 模型 checkpoint 的路径
        dataset_repo_id: 数据集的 repo_id
        dataset_root: 数据集的根目录
        num_episodes: 要测试的 episode 数
        device: 计算设备 (cuda/cpu)
    """
    print("=" * 80)
    print("Diffusion Policy 训练数据过拟合测试")
    print("=" * 80)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    print(f"\n【1】加载模型")
    print(f"  Checkpoint: {checkpoint_path}")

    try:
        # 使用 DiffusionPolicy 直接加载
        policy = DiffusionPolicy.from_pretrained(checkpoint_path)
        policy = policy.to(device)
        policy.eval()
        print(f"✓ 模型加载成功")
        print(f"  策略类型: {policy.config.__class__.__name__}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n【2】加载数据集")
    print(f"  Dataset: {dataset_repo_id}")
    print(f"  Root: {dataset_root}")

    try:
        dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
        print(f"✓ 数据集加载成功")
        print(f"  总 Episodes: {dataset.num_episodes}")
        print(f"  总 Frames: {dataset.num_frames}")
        print(f"  FPS: {dataset.fps}")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n【3】在数据集上进行推理")
    print(f"  测试 Episodes: {min(num_episodes, dataset.num_episodes)}")
    print(f"  GPU优化: 启用批处理和预加载")

    # 获取要测试的 episode 索引
    episodes_to_test = list(range(min(num_episodes, dataset.num_episodes)))
    all_predicted_actions = []
    all_ground_truth_actions = []
    episode_metrics = []

    # 预加载所有 episode 的数据到内存
    print(f"  正在预加载 {len(episodes_to_test)} 个 episodes...")
    preloaded_episodes = {}
    total_frames = 0
    
    for ep_idx in episodes_to_test:
        ep_start = dataset.meta.episodes["dataset_from_index"][ep_idx]
        ep_end = dataset.meta.episodes["dataset_to_index"][ep_idx]
        
        frames_data = []
        for frame_idx in range(ep_start, ep_end):
            try:
                frame = dataset[frame_idx]
                frames_data.append(frame)
                total_frames += 1
            except Exception as e:
                print(f"    ⚠️  预加载 Episode {ep_idx} Frame {frame_idx} 失败: {e}")
                continue
        
        preloaded_episodes[ep_idx] = frames_data
    
    print(f"  ✓ 预加载完成: {total_frames} 帧")
    
    # 批量推理：减少每帧调用开销（对 CPU 很重要），并移除调试打印
    inference_batch_size = 32
    with torch.no_grad():
        for ep_idx in tqdm(episodes_to_test, desc="处理 Episodes"):
            # 重置策略的内部状态（清空队列）
            policy.reset()

            frames_data = preloaded_episodes[ep_idx]
            if not frames_data:
                continue

            episode_predicted_actions = []
            episode_ground_truth_actions = []

            batch_frames = []
            batch_actions = []

            def flush_batch():
                """内部函数：把累计的 batch_frames 组织成 batch 并推理。"""
                nonlocal batch_frames, batch_actions
                if not batch_frames:
                    return
                # 构造 batch dict：对每个 observation key 进行拼接
                batch: dict = {}
                obs_keys = list(batch_frames[0].keys())
                for k in obs_keys:
                    # 每 batch_frames[i][k] 是 (1, ...), 使用 cat 得到 (B, ...)
                    batch[k] = torch.cat([bf[k] for bf in batch_frames], dim=0)

                # 推理（支持 batch）
                try:
                    preds = policy.select_action(batch)
                except Exception as e:
                    print(f"  ⚠️  Episode {ep_idx} 批量推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 清空并返回以避免重复调用同一批数据
                    batch_frames = []
                    batch_actions = []
                    return

                # preds 可能是 (B, action_dim) 或 (action_dim,)（B==1），统一为 numpy array
                preds = preds.detach().cpu().numpy()
                if preds.ndim == 1:
                    preds = preds.reshape(1, -1)

                # 添加到结果
                episode_predicted_actions.extend(preds)
                episode_ground_truth_actions.extend(batch_actions)

                batch_frames = []
                batch_actions = []

            for frame in frames_data:
                # 收集 observation 和 ground truth action
                obs = {k: v.unsqueeze(0).to(device) for k, v in frame.items() if k.startswith("observation")}
                gt_action = frame["action"].numpy()

                batch_frames.append(obs)
                batch_actions.append(gt_action)

                if len(batch_frames) >= inference_batch_size:
                    flush_batch()

            # flush 剩余的
            flush_batch()

            if len(episode_predicted_actions) > 0:
                episode_predicted_actions = np.array(episode_predicted_actions)
                episode_ground_truth_actions = np.array(episode_ground_truth_actions)

                metrics, per_dim_mae, per_dim_mse = compute_action_metrics(
                    episode_predicted_actions, episode_ground_truth_actions
                )

                episode_metrics.append(
                    {
                        "episode_idx": ep_idx,
                        "num_frames": len(episode_predicted_actions),
                        "metrics": metrics,
                        "per_dim_mae": per_dim_mae,
                        "per_dim_mse": per_dim_mse,
                    }
                )

                all_predicted_actions.append(episode_predicted_actions)
                all_ground_truth_actions.append(episode_ground_truth_actions)

    print(f"\n【4】汇总结果")

    if not episode_metrics:
        print("✗ 没有成功推理任何数据")
        return

    # 合并所有 episodes 的结果
    all_predicted_actions = np.concatenate(all_predicted_actions, axis=0)
    all_ground_truth_actions = np.concatenate(all_ground_truth_actions, axis=0)

    overall_metrics, overall_per_dim_mae, overall_per_dim_mse = compute_action_metrics(
        all_predicted_actions, all_ground_truth_actions
    )

    print(f"\n✓ 总体性能指标 (所有 {len(episode_metrics)} 个 Episodes)")
    print(f"  MSE:         {overall_metrics['mse']:.6f}")
    print(f"  MAE:         {overall_metrics['mae']:.6f}")
    print(f"  RMSE:        {overall_metrics['rmse']:.6f}")
    print(f"  Max Error:   {overall_metrics['max_error']:.6f}")
    print(f"  Min Error:   {overall_metrics['min_error']:.6f}")
    print(f"  总帧数:       {len(all_predicted_actions)}")

    print(f"\n【5】各 Episode 的性能")
    print(f"{'Episode':<10} {'Frames':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 56)

    for ep_metric in episode_metrics:
        ep_idx = ep_metric["episode_idx"]
        num_frames = ep_metric["num_frames"]
        metrics = ep_metric["metrics"]
        print(
            f"{ep_idx:<10} {num_frames:<10} {metrics['mse']:<12.6f} "
            f"{metrics['mae']:<12.6f} {metrics['rmse']:<12.6f}"
        )

    print(f"\n【6】各动作维度的性能")
    action_dim = len(overall_per_dim_mae)
    print(f"  动作维度: {action_dim}")
    print(f"\n{'维度':<8} {'MAE':<12} {'MSE':<12} {'Max Error':<12}")
    print("-" * 44)

    for dim in range(min(action_dim, 10)):  # 最多显示 10 维
        mae = overall_per_dim_mae[dim]
        mse = overall_per_dim_mse[dim]
        max_err = np.max(np.abs(all_predicted_actions[:, dim] - all_ground_truth_actions[:, dim]))
        print(f"{dim:<8} {mae:<12.6f} {mse:<12.6f} {max_err:<12.6f}")

    if action_dim > 10:
        print(f"  ... (还有 {action_dim - 10} 维)")

    print(f"\n【7】结论")
    if overall_metrics["mae"] < 0.1:
        print(f"  ✓ 模型在训练数据上表现很好（MAE < 0.1）")
        print(f"    这表明模型成功过拟合了训练数据，能够准确预测训练集中的动作")
    elif overall_metrics["mae"] < 0.2:
        print(f"  ◐ 模型在训练数据上表现中等（0.1 <= MAE < 0.2）")
    else:
        print(f"  ✗ 模型在训练数据上表现较差（MAE >= 0.2）")
        print(f"    可能需要调整训练参数或检查数据质量")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 Diffusion Policy 在训练数据上的过拟合程度")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/blind_dp1/checkpoints/last",
        help="Checkpoint 路径",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="Opendrawer",
        help="数据集 repo_id",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets/test_data/xarm_leap_tactile_lift_no_image",
        help="数据集根目录",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="要测试的 episode 数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)",
    )

    args = parser.parse_args()

    test_diffusion_overfit(
        checkpoint_path=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        num_episodes=args.num_episodes,
        device=args.device,
    )
