"""
合并 black_board_v1, black_board_v2, black_board_v3 三个数据集为 black_board_merged
用于训练 diffusion_hao 策略
"""

import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)

DATASETS_ROOT = Path(__file__).parent.parent / "datasets"
OUTPUT_REPO_ID = "black_board_merged"
OUTPUT_DIR = DATASETS_ROOT / OUTPUT_REPO_ID

SOURCE_DATASETS = [
    "black_board_v1",
    "black_board_v2",
    "black_board_v3",
]

if __name__ == "__main__":
    logging.info(f"正在加载 {len(SOURCE_DATASETS)} 个数据集...")

    datasets = []
    for name in SOURCE_DATASETS:
        root = DATASETS_ROOT / name
        logging.info(f"  加载: {root}")
        ds = LeRobotDataset(name, root=root)
        logging.info(f"    episodes: {ds.meta.total_episodes}, frames: {ds.meta.total_frames}")
        datasets.append(ds)

    if OUTPUT_DIR.exists():
        import shutil
        logging.warning(f"输出目录已存在，正在删除: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    logging.info(f"开始合并 -> {OUTPUT_DIR}")
    merged = merge_datasets(
        datasets,
        output_repo_id=OUTPUT_REPO_ID,
        output_dir=OUTPUT_DIR,
    )

    logging.info("合并完成！")
    logging.info(f"  输出路径: {OUTPUT_DIR}")
    logging.info(f"  总 episodes: {merged.meta.total_episodes}")
    logging.info(f"  总 frames:   {merged.meta.total_frames}")
