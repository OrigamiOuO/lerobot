#!/usr/bin/env python
"""
在数据集上逐帧评估 checkpoint 的推理质量。

工作流程
--------
1. 加载本地 LeRobot 数据集（指定目录），选择某条 / 所有 episode。
2. 加载训练好的 policy（指定 checkpoint 路径，内含 pretrained_model/）。
3. 根据 policy 配置自动解析 delta_timestamps，构建 pre/post-processor 管道。
4. 对该 episode 中的 **每一帧** 执行推理：
   - 用数据集中的观测 (observation.*) 作为输入
   - 调用 policy.select_action 得到预测动作
   - 与数据集中的 ground-truth action 逐维对比
5. 输出逐帧对比表 + 整体 / 逐维度 / 逐关节的 L1 / L2 / MSE 统计。

用法
----
    # 评估单条 episode
    python scripts/eval_ckpt_on_dataset.py \
        --ckpt_path checkpoints/050000/pretrained_model \
        --dataset_dir datasets/test_new_tac \
        --episode_index 0 \
        --device cuda

    # 评估所有 episode
    python scripts/eval_ckpt_on_dataset.py \
        --ckpt_path checkpoints/050000 \
        --dataset_dir datasets/test_new_tac \
        --episode_index -1

    # 将结果保存为 CSV
    python scripts/eval_ckpt_on_dataset.py \
        --ckpt_path checkpoints/050000/pretrained_model \
        --dataset_dir datasets/test_new_tac \
        --episode_index 0 \
        --save_csv outputs/eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# ── 确保 lerobot/src 在 sys.path 中 ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent  # 项目根目录
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# 注册第三方 / 自定义策略插件
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import get_safe_torch_device


# ═══════════════════════════════════════════════════════════════════
#  辅助工具
# ═══════════════════════════════════════════════════════════════════


def _add_batch_dim(item: dict) -> dict:
    """给 dataset __getitem__ 返回的单帧字典加上 batch=1 维度。"""
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            out[k] = [v]
        else:
            out[k] = v
    return out


def _to_device(item: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}


def _strip_action_keys(item: dict) -> dict:
    """移除 batch 中与 action 相关的键，仅保留 observation。"""
    return {k: v for k, v in item.items() if not k.startswith("action")}


# ═══════════════════════════════════════════════════════════════════
#  单 Episode 评估
# ═══════════════════════════════════════════════════════════════════


def evaluate_episode(
    dataset: LeRobotDataset,
    policy: torch.nn.Module,
    preprocessor,
    postprocessor,
    device: torch.device,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """对一个 episode 的所有帧逐帧推理并收集 GT / Pred。

    Returns
    -------
    gt_arr :  (n_frames, action_dim)
    pred_arr: (n_frames, action_dim)
    """
    n_frames = len(dataset)
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    # 每条 episode 开始时必须 reset（清空 action chunk 缓存等）
    policy.reset()
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()

    if verbose:
        print(f"\n{'帧':>5}  {'GT action (前6维)':^50}  {'Pred action (前6维)':^50}  {'AvgL1':>8}")
        print("─" * 125)

    for frame_idx in range(n_frames):
        item = dataset[frame_idx]

        # ---- Ground-truth action ----
        gt_action: torch.Tensor = item[ACTION]  # (action_dim,) 或 (n_steps, action_dim)
        if gt_action.ndim > 1:
            # delta_timestamps 模式下，取最后一步（当前时刻的 action）
            gt_action = gt_action[-1]
        gt_np = gt_action.cpu().float().numpy()

        # ---- 调试：首帧打印数据形状 ----
        if verbose and frame_idx == 0:
            print(f"[DEBUG] 第 0 帧数据形状：")
            for key, val in item.items():
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: {val.shape} {val.dtype}")

        # ---- 构造只有 observation 的 batch ----
        obs_item = _strip_action_keys(item)
        obs_batch = _add_batch_dim(obs_item)
        obs_batch = _to_device(obs_batch, device)

        with torch.inference_mode():
            # preprocessor: normalize + device
            obs_processed = preprocessor(obs_batch)

            # policy 推理
            # 注意：对于 diffusion_hao 的离线评估，dataset 已经提供了时间维度
            # (B, n_obs_steps, ...)。如果调用 select_action，会再次走队列堆叠，
            # 导致输入多出一层时间维，引发后续 resize 维度错误。
            # 因此这里直接走 _prepare_batch + diffusion.generate_actions。
            if getattr(policy, "name", None) == "diffusion_hao" and hasattr(policy, "diffusion"):
                prepared_batch = policy._prepare_batch(dict(obs_processed))
                pred_action_chunk = policy.diffusion.generate_actions(prepared_batch)
                pred_action = pred_action_chunk[:, 0, :]  # 取当前步
            else:
                # 其他策略保持常规路径
                pred_action = policy.select_action(obs_processed)

            # postprocessor: unnormalize → cpu
            pred_action = postprocessor(pred_action)

        pred_np = pred_action.squeeze(0).cpu().float().numpy()
        if pred_np.ndim > 1:
            pred_np = pred_np[0]  # action chunk → 取第一步

        all_gt.append(gt_np)
        all_pred.append(pred_np)

        if verbose:
            avg_l1 = float(np.abs(pred_np - gt_np).mean())
            show_dims = min(6, len(gt_np))
            gt_str = " ".join(f"{v:8.4f}" for v in gt_np[:show_dims])
            pred_str = " ".join(f"{v:8.4f}" for v in pred_np[:show_dims])
            print(f"{frame_idx:>5}  {gt_str:<50}  {pred_str:<50}  {avg_l1:>8.5f}")

    gt_arr = np.stack(all_gt, axis=0)
    pred_arr = np.stack(all_pred, axis=0)
    return gt_arr, pred_arr


# ═══════════════════════════════════════════════════════════════════
#  统计输出
# ═══════════════════════════════════════════════════════════════════


def print_summary(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    action_names: list[str] | None = None,
    episode_label: str = "",
):
    """打印 L1 / L2 / MSE 的整体与逐维统计。"""
    err_l1 = np.abs(pred_arr - gt_arr)
    err_l2 = (pred_arr - gt_arr) ** 2

    n_frames, action_dim = gt_arr.shape
    dim_names = action_names if action_names else [f"dim_{i}" for i in range(action_dim)]

    header = f"汇总统计 {episode_label}"
    print(f"\n{'═' * 100}")
    print(f"  {header}")
    print(f"{'═' * 100}")
    print(f"  帧数             : {n_frames}")
    print(f"  动作维度         : {action_dim}")
    print()
    print(f"  整体 L1  均值    : {err_l1.mean():.6f}")
    print(f"  整体 L1  中位数  : {np.median(err_l1):.6f}")
    print(f"  整体 L1  最大值  : {err_l1.max():.6f}")
    print(f"  整体 L1  最小值  : {err_l1.min():.6f}")
    print(f"  整体 L1  标准差  : {err_l1.std():.6f}")
    print()
    mse = err_l2.mean()
    rmse = np.sqrt(mse)
    print(f"  整体 MSE         : {mse:.6f}")
    print(f"  整体 RMSE        : {rmse:.6f}")

    # 逐维度
    print(f"\n  {'关节名':^34} {'L1均值':>10} {'L1中位':>10} {'L1最大':>10} {'MSE':>10} {'RMSE':>10}")
    print(f"  {'─' * 90}")
    for d in range(action_dim):
        d_l1 = err_l1[:, d]
        d_l2 = err_l2[:, d]
        print(
            f"  {dim_names[d]:<34} "
            f"{d_l1.mean():>10.6f} "
            f"{np.median(d_l1):>10.6f} "
            f"{d_l1.max():>10.6f} "
            f"{d_l2.mean():>10.6f} "
            f"{np.sqrt(d_l2.mean()):>10.6f}"
        )
    print(f"{'═' * 100}")


def save_csv(
    filepath: str | Path,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    action_names: list[str] | None,
    episode_index: int,
):
    """将逐帧结果保存为 CSV。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_frames, action_dim = gt_arr.shape
    dim_names = action_names if action_names else [f"dim_{i}" for i in range(action_dim)]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode", "frame"]
        for name in dim_names:
            header += [f"gt_{name}", f"pred_{name}", f"err_l1_{name}"]
        header += ["avg_l1", "avg_l2"]
        writer.writerow(header)

        for i in range(n_frames):
            row: list = [episode_index, i]
            for d in range(action_dim):
                row += [
                    f"{gt_arr[i, d]:.6f}",
                    f"{pred_arr[i, d]:.6f}",
                    f"{abs(pred_arr[i, d] - gt_arr[i, d]):.6f}",
                ]
            l1 = float(np.abs(pred_arr[i] - gt_arr[i]).mean())
            l2 = float(np.sqrt(((pred_arr[i] - gt_arr[i]) ** 2).mean()))
            row += [f"{l1:.6f}", f"{l2:.6f}"]
            writer.writerow(row)

    print(f"\n[INFO] 结果已保存到 {filepath}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在数据集上逐帧评估 policy checkpoint 的推理质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help=(
            "checkpoint 目录。可以直接指向 pretrained_model/ 目录，"
            "也可以指向其父目录（脚本会自动查找内部的 pretrained_model/）。"
        ),
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="LeRobot 数据集根目录（含 meta/ data/ videos/ 子目录）。",
    )
    p.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="要评估的 episode 索引（0 起始）。设为 -1 则评估所有 episode。",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，如 cuda / cpu（默认自动检测）。",
    )
    p.add_argument(
        "--no_delta_timestamps",
        action="store_true",
        help="禁用 delta_timestamps（仅使用当前帧，便于调试）。",
    )
    p.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="将逐帧结果保存为 CSV 文件的路径（可选）。",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="精简输出，不打印逐帧详情。",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── 路径解析 ──────────────────────────────────────────────────
    workspace = _ROOT

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = workspace / ckpt_path

    # 允许用户直接指向 pretrained_model/ 或其父目录
    if (ckpt_path / "pretrained_model").is_dir():
        pretrained_path = ckpt_path / "pretrained_model"
    elif (ckpt_path / "config.json").is_file():
        pretrained_path = ckpt_path
    else:
        raise FileNotFoundError(
            f"找不到有效的 checkpoint 目录。请确保 {ckpt_path} 下存在 "
            f"pretrained_model/ 子目录或直接包含 config.json。"
        )

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = workspace / dataset_dir

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    print(f"[INFO] Checkpoint  : {pretrained_path}")
    print(f"[INFO] Dataset     : {dataset_dir}")
    print(f"[INFO] Device      : {args.device}")

    # ── 加载数据集 metadata ────────────────────────────────────────
    repo_id = dataset_dir.name
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_dir))
    total_episodes = ds_meta.total_episodes
    print(f"[INFO] 数据集共 {total_episodes} 条 episode，{ds_meta.total_frames} 帧，fps={ds_meta.fps}")

    # ── 解析要评估的 episode 列表 ─────────────────────────────────
    if args.episode_index == -1:
        episode_indices = list(range(total_episodes))
    else:
        if args.episode_index >= total_episodes:
            raise ValueError(
                f"episode_index={args.episode_index} 超出范围，"
                f"数据集共 {total_episodes} 条 episode（0~{total_episodes - 1}）"
            )
        episode_indices = [args.episode_index]

    print(f"[INFO] 将评估 {len(episode_indices)} 条 episode: {episode_indices}")

    # ── 加载策略配置 ──────────────────────────────────────────────
    print(f"[INFO] 从 {pretrained_path} 加载策略配置...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy_cfg.pretrained_path = str(pretrained_path)

    device = get_safe_torch_device(args.device, log=True)
    policy_cfg.device = str(device)

    # ── 解析 delta_timestamps ─────────────────────────────────────
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    if args.no_delta_timestamps:
        print("[INFO] --no_delta_timestamps 已启用，不使用 delta_timestamps")
        delta_timestamps = None
    else:
        print(f"[INFO] delta_timestamps 键数: {len(delta_timestamps) if delta_timestamps else 0}")
        if delta_timestamps:
            for key, ts in delta_timestamps.items():
                print(f"       {key}: {len(ts)} 步, range=[{ts[0]:.4f}, {ts[-1]:.4f}]")

    # ── 加载策略 ──────────────────────────────────────────────────
    print("[INFO] 加载策略权重...")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    # ── 加载 pre/post-processor ───────────────────────────────────
    print("[INFO] 加载 pre/post processor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── 获取 action 维度名称 ──────────────────────────────────────
    action_names: list[str] | None = None
    if ACTION in ds_meta.features:
        feat = ds_meta.features[ACTION]
        if "names" in feat and feat["names"] is not None:
            action_names = feat["names"]

    # ── 逐 episode 评估 ───────────────────────────────────────────
    all_episode_gt: list[np.ndarray] = []
    all_episode_pred: list[np.ndarray] = []

    for ep_idx in episode_indices:
        print(f"\n{'▓' * 100}")
        print(f"  Episode {ep_idx}")
        print(f"{'▓' * 100}")

        # 加载该 episode
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(dataset_dir),
            episodes=[ep_idx],
            delta_timestamps=delta_timestamps,
        )
        n_frames = len(dataset)
        print(f"[INFO] Episode {ep_idx} 共 {n_frames} 帧")

        gt_arr, pred_arr = evaluate_episode(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            verbose=not args.quiet,
        )

        print_summary(gt_arr, pred_arr, action_names, episode_label=f"(Episode {ep_idx})")

        if args.save_csv:
            csv_path = Path(args.save_csv)
            if len(episode_indices) > 1:
                # 多 episode 模式下，每个 episode 单独保存
                csv_path = csv_path.parent / f"{csv_path.stem}_ep{ep_idx}{csv_path.suffix}"
            save_csv(csv_path, gt_arr, pred_arr, action_names, ep_idx)

        all_episode_gt.append(gt_arr)
        all_episode_pred.append(pred_arr)

    # ── 跨 episode 总汇总 ─────────────────────────────────────────
    if len(episode_indices) > 1:
        combined_gt = np.concatenate(all_episode_gt, axis=0)
        combined_pred = np.concatenate(all_episode_pred, axis=0)
        print_summary(
            combined_gt,
            combined_pred,
            action_names,
            episode_label=f"(所有 {len(episode_indices)} 条 episode 合计)",
        )

    print("\n[DONE] 评估完成。")


if __name__ == "__main__":
    main()
