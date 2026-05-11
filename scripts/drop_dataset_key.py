"""
从 LeRobot 数据集中删除指定 key，生成新数据集。

用法：
    python scripts/drop_dataset_key.py \
        --src datasets/luo_proj/Blind_Grasping_LeRobot_Multimodal_10 \
        --dst datasets/luo_proj/Blind_Grasping_LeRobot_No_HandPC \
        --key observation.hand_pc \
        [--workers 8]
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


def process_parquet(args):
    src_file, dst_file, key_to_drop = args
    table = pq.read_table(src_file)
    if key_to_drop in table.schema.names:
        table = table.drop([key_to_drop])
    os.makedirs(dst_file.parent, exist_ok=True)
    pq.write_table(table, dst_file)
    return str(dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="源数据集路径")
    parser.add_argument("--dst", type=Path, required=True, help="新数据集输出路径")
    parser.add_argument("--key", type=str, required=True, help="要删除的 key 名，如 observation.hand_pc")
    parser.add_argument("--workers", type=int, default=8, help="并行工作进程数")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    key_to_drop: str = args.key

    if not src.exists():
        raise FileNotFoundError(f"源数据集不存在: {src}")
    if dst.exists():
        raise FileExistsError(f"目标路径已存在，请先删除或换一个路径: {dst}")

    # ---------- 1. 复制 meta 目录 ----------
    print("复制 meta 目录...")
    dst_meta = dst / "meta"
    shutil.copytree(src / "meta", dst_meta)

    # ---------- 2. 修改 info.json ----------
    info_path = dst_meta / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if key_to_drop in info.get("features", {}):
        del info["features"][key_to_drop]
        print(f"已从 info.json features 中删除 '{key_to_drop}'")
    else:
        print(f"警告：info.json 中不存在 key '{key_to_drop}'，跳过")

    # data_files_size_in_mb 在处理后无法精确预知，保留原值（用户可后续手动更新）
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # ---------- 3. 修改 stats.json ----------
    stats_path = dst_meta / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        if key_to_drop in stats:
            del stats[key_to_drop]
            print(f"已从 stats.json 中删除 '{key_to_drop}'")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    # ---------- 4. 并行处理 parquet 文件 ----------
    src_data = src / "data"
    parquet_files = sorted(src_data.rglob("*.parquet"))
    print(f"共找到 {len(parquet_files)} 个 parquet 文件，开始并行处理（workers={args.workers}）...")

    tasks = []
    for src_file in parquet_files:
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        tasks.append((src_file, dst_file, key_to_drop))

    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_parquet, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()  # 异常会在此处抛出
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  进度: {done}/{total}")

    print(f"\n完成！新数据集已保存到: {dst}")


if __name__ == "__main__":
    main()
