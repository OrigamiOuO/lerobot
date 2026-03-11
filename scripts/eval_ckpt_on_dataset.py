#!/usr/bin/env python
"""
用数据集中的数据测试 checkpoint 推理输出与标准答案的差异。

对于指定 episode 中的每一帧，依次将数据集观测值输入策略并得到预测动作，
再与数据集中的标准动作逐帧对比，输出每步的差异及汇总统计。

默认用法：
    python scripts/eval_ckpt_on_dataset.py

显式指定参数：
    python scripts/eval_ckpt_on_dataset.py \
        --ckpt_path checkpoints/hao_proj/task1_test_diffusion/checkpoints/004000 \
        --dataset_dir hao_datasets/grasp_cup_test \
        --episode_index 0 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ---- 把 lerobot 的 src 加入路径，方便直接 python 运行 ---- #
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# 注册所有策略子类
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import get_safe_torch_device


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _add_batch_dim(item: dict) -> dict:
    """给 dataset 返回的单帧字典加上 batch=1 维度。"""
    batched = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            batched[k] = [v]
        else:
            batched[k] = v
    return batched


def _to_device(item: dict, device: torch.device) -> dict:
    """将字典中的 Tensor 移到指定设备。"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用数据集数据逐帧测试 checkpoint 输出")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/hao_proj/task1_test_diffusion/checkpoints/004000",
        help="checkpoint 根目录（内含 pretrained_model 子目录）",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="hao_datasets/grasp_cup_test",
        help="LeRobot 数据集根目录（含 meta/data/videos）",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="数据集中的第几条 episode（0 起始）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备，如 cuda / cpu",
    )
    parser.add_argument(
        "--no_delta_timestamps",
        action="store_true",
        help="禁用 delta_timestamps，仅使用当前帧（便于调试）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  路径解析
    # ------------------------------------------------------------------ #
    workspace = _ROOT
    ckpt_root = workspace / args.ckpt_path
    pretrained_path = ckpt_root / "pretrained_model"
    dataset_root = workspace / args.dataset_dir

    if not pretrained_path.exists():
        raise FileNotFoundError(f"pretrained_model 目录不存在：{pretrained_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")

    # ------------------------------------------------------------------ #
    #  加载数据集 metadata 并校验 episode_index
    # ------------------------------------------------------------------ #
    # 本地 dataset：root 直接指向数据集根目录，repo_id 只作为名称标识
    repo_id = dataset_root.name
    print(f"[INFO] 加载数据集 metadata：{dataset_root}")
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

    total_episodes = ds_meta.total_episodes
    if args.episode_index >= total_episodes:
        raise ValueError(
            f"episode_index={args.episode_index} 超出数据集范围，"
            f"该数据集共有 {total_episodes} 条 episode（0~{total_episodes - 1}）"
        )
    print(f"[INFO] 数据集共 {total_episodes} 条 episode，选择第 {args.episode_index} 条")

    # ------------------------------------------------------------------ #
    #  加载策略 config 并解析 delta_timestamps
    # ------------------------------------------------------------------ #
    print(f"[INFO] 从 {pretrained_path} 加载策略配置")
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy_cfg.pretrained_path = str(pretrained_path)

    device = get_safe_torch_device(args.device, log=True)
    policy_cfg.device = str(device)

    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    if args.no_delta_timestamps:
        print(f"[INFO] --no_delta_timestamps 已启用，禁用 delta_timestamps")
        delta_timestamps = None
    print(f"[INFO] delta_timestamps = {delta_timestamps}")

    # ------------------------------------------------------------------ #
    #  加载数据集（只加载目标 episode）
    # ------------------------------------------------------------------ #
    print(f"[INFO] 加载 episode {args.episode_index} 的帧数据")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        episodes=[args.episode_index],
        delta_timestamps=delta_timestamps,
    )
    n_frames = len(dataset)
    print(f"[INFO] episode {args.episode_index} 共 {n_frames} 帧")

    # ------------------------------------------------------------------ #
    #  加载策略和预/后处理器
    # ------------------------------------------------------------------ #
    print(f"[INFO] 加载策略权重")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    print(f"[INFO] 加载 pre/post 处理器")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path),
        # 让 device_processor 把数据移到正确的推理设备
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 从 env config 创建环境预/后处理器（虽然在这里只用 dataset，但仍需要创建以支持某些策略）
    # 对于数据集模式，我们用一个最小的 env config
    try:
        from lerobot.envs.configs import PushtEnv
        env_cfg = PushtEnv()  # 使用最小的 env config
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    except Exception as e:
        print(f"[WARN] 无法创建环境处理器（{e}），将使用空处理器")
        # 如果无法创建，使用空处理器（Identity）
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        env_preprocessor = PolicyProcessorPipeline(steps=[])
        env_postprocessor = PolicyProcessorPipeline(steps=[])

    # 获取动作维度名称（用于逐维度统计）
    # 从数据集 metadata 中获取，而不是从 policy config
    action_names: list[str] | None = None
    if ACTION in ds_meta.features and "names" in ds_meta.features[ACTION]:
        action_names = ds_meta.features[ACTION]["names"]

    # ------------------------------------------------------------------ #
    #  逐帧推理并对比
    # ------------------------------------------------------------------ #
    policy.reset()

    # per-frame accumulator: (n_frames, action_dim)
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    col_w = 8 * 7  # 每组数值列的显示宽度，供参考
    print("\n" + "=" * 100)
    print(f"{'帧':>5}  {'GT action':^42}  {'Pred action':^42}  {'avg L1':>8}")
    print("=" * 100)

    for frame_idx in range(n_frames):
        item = dataset[frame_idx]

        # 取标准动作 ground truth
        gt_action: torch.Tensor = item[ACTION]   # (action_dim,) 或 (n_obs_steps, action_dim)
        if gt_action.ndim > 1:
            # delta_timestamps 模式下，取最后一步（最新时刻）
            gt_action = gt_action[-1]
        gt_np = gt_action.cpu().float().numpy()  # (action_dim,)

        # 构造 batch=1 观测（去掉 action 键，仅传观测）
        obs_item = {k: v for k, v in item.items() if k != ACTION}
        
        # 调试：打印第一帧的数据形状
        if frame_idx == 0:
            print(f"[DEBUG] 第 0 帧数据形状：")
            for key, val in obs_item.items():
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: {val.shape} {val.dtype}")
        
        obs_batch = _add_batch_dim(obs_item)
        # 数据集数据已经是 float tensor，device_processor 会自动移设备，
        # 这里也提前移好以防 preprocessor 中无 device_processor
        obs_batch = _to_device(obs_batch, device)

        with torch.no_grad():
            obs_processed = env_preprocessor(obs_batch)
            obs_processed = preprocessor(obs_processed)
            # select_action 返回 PolicyAction (= torch.Tensor)，shape (batch, action_dim) 或
            # (batch, n_action_steps, action_dim)（action-chunking 策略会缓存并逐步吐出）
            pred_action: torch.Tensor = policy.select_action(obs_processed)
            
            # 通过 postprocessor
            action_transition = {ACTION: pred_action}
            action_transition = env_postprocessor(action_transition)
            pred_action = postprocessor(action_transition[ACTION])

        # 取 batch 0；若是 action chunk 则取第一步
        pred_np = pred_action[0].cpu().float().numpy()
        if pred_np.ndim > 1:
            pred_np = pred_np[0]

        all_gt.append(gt_np)
        all_pred.append(pred_np)

        avg_l1 = float(np.abs(pred_np - gt_np).mean())
        gt_str   = " ".join(f"{v:7.4f}" for v in gt_np)
        pred_str = " ".join(f"{v:7.4f}" for v in pred_np)
        print(f"{frame_idx:>5}  {gt_str:^42}  {pred_str:^42}  {avg_l1:>8.5f}")

    # ------------------------------------------------------------------ #
    #  汇总统计
    # ------------------------------------------------------------------ #
    gt_arr   = np.stack(all_gt,   axis=0)   # (n_frames, action_dim)
    pred_arr = np.stack(all_pred, axis=0)   # (n_frames, action_dim)
    err_arr  = np.abs(pred_arr - gt_arr)    # (n_frames, action_dim)

    print("\n" + "=" * 100)
    print("整体汇总统计（所有帧、所有维度的 L1 误差）：")
    print(f"  帧数      : {n_frames}")
    print(f"  均值      : {err_arr.mean():.6f}")
    print(f"  中位数    : {float(np.median(err_arr)):.6f}")
    print(f"  最大值    : {err_arr.max():.6f}")
    print(f"  最小值    : {err_arr.min():.6f}")
    print(f"  标准差    : {err_arr.std():.6f}")

    # 逐维度统计
    print("\n逐维度 L1 误差（均值）：")
    dim_names = action_names if action_names else [f"dim_{i}" for i in range(err_arr.shape[1])]
    for dim_i, name in enumerate(dim_names):
        d_err = err_arr[:, dim_i]
        print(
            f"  {name:<30}: mean={d_err.mean():.6f}  "
            f"median={float(np.median(d_err)):.6f}  "
            f"max={d_err.max():.6f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
