"""
修复 grasp_chip_v1 数据集：视频中混入了 reset 环境期间的帧。

问题：clear_episode_buffer 只清理 image_keys 的 PNG，不清理 video_keys 的 PNG。
取消/重录 episode 时，残留 PNG 被编进视频，导致某些 episode 的视频区间比实际多出 reset 帧。

受影响 episode: 1(+256帧), 3(+31帧), 5(+257帧), 6(+48帧), 7(+272帧)，共 864 帧。
视频总帧: 6032, Parquet 总帧: 5168, 差值: 864。

修复策略：
1. 从 .bak (6032帧原始视频) 中，对每个 episode 按 from_timestamp 提取 length 帧
2. 拼接为新视频 (5168帧)，跳过 reset 帧
3. 更新 episode 元数据的 from_timestamp / to_timestamp
"""
import subprocess
import shutil
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATASET_DIR = Path("/home/user/Code/xuhao_lerobot/lerobot/datasets/grasp_chip_v1")
FPS = 30
FFMPEG = "/home/user/miniconda3/envs/lerobot_hao/bin/ffmpeg"
FFPROBE = "/home/user/miniconda3/envs/lerobot_hao/bin/ffprobe"

VIDEO_KEYS = [
    "observation.images.global",
    "observation.images.inhand",
    "observation.images.tac_raw.tac1",
    "observation.images.tac_depth.tac1",
    "observation.images.tac_normal.tac1",
]


def get_video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return int(info["streams"][0]["nb_frames"])


def get_video_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def extract_segment(src: Path, dst: Path, start_s: float, num_frames: int):
    """从 src 视频中提取 num_frames 帧 (从 start_s 开始)，重编码到 dst。"""
    subprocess.run(
        [FFMPEG, "-y",
         "-ss", f"{start_s:.6f}",
         "-i", str(src),
         "-vframes", str(num_frames),
         "-c:v", "libsvtav1",
         "-pix_fmt", "yuv420p",
         str(dst)],
        capture_output=True, text=True, check=True,
    )


def concat_segments(segment_paths: list, output: Path):
    """用 ffmpeg concat demuxer 拼接多个视频片段 (无重编码)。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
        concat_list = f.name

    try:
        subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", str(output)],
            capture_output=True, text=True, check=True,
        )
    finally:
        Path(concat_list).unlink()


def main():
    ep_path = DATASET_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_df = pd.read_parquet(ep_path)
    total_parquet_frames = ep_df["length"].sum()

    print(f"数据集: {DATASET_DIR}")
    print(f"Episode 数: {len(ep_df)}")
    print(f"Parquet 总帧数: {total_parquet_frames}")
    print()

    # Step 0: 确认 .bak 来源
    sample_bak = DATASET_DIR / "videos" / VIDEO_KEYS[0] / "chunk-000" / "file-000.mp4.bak"
    if not sample_bak.exists():
        # 如果没有 .bak，用当前文件作为源
        print("未找到 .bak 备份，将使用当前 mp4 作为源")
        src_ext = ".mp4"
    else:
        bak_frames = get_video_frame_count(sample_bak)
        print(f"备份视频帧数: {bak_frames}")
        src_ext = ".mp4.bak"

    # 用 episode_index 列作为索引方便查找
    ep_df = ep_df.set_index("episode_index")

    # Step 1: 识别受影响的 episode
    print("\n--- 分析 episode ---")
    episodes_info = []
    for ep in ep_df.index:
        row = ep_df.loc[ep]
        length = int(row["length"])
        v_from = row["videos/observation.images.global/from_timestamp"]
        v_to = row["videos/observation.images.global/to_timestamp"]
        video_dur = v_to - v_from
        expected_dur = length / FPS
        extra = round((video_dur - expected_dur) * FPS)
        episodes_info.append({
            "ep": ep, "length": length,
            "old_from": v_from, "old_to": v_to,
            "extra_frames": extra,
        })
        if extra > 0:
            print(f"  Episode {ep:2d}: length={length}, video_dur={video_dur:.3f}s, "
                  f"expected={expected_dur:.3f}s, extra={extra} frames ←★")
        else:
            print(f"  Episode {ep:2d}: length={length}, OK")

    total_extra = sum(e["extra_frames"] for e in episodes_info)
    print(f"\n总多余帧: {total_extra}")

    if total_extra == 0:
        print("无需修复，退出。")
        return

    # Step 2: 对每个 video key，提取各 episode 的有效帧并拼接
    print("\n--- 开始修复视频 ---")
    for vkey in VIDEO_KEYS:
        video_dir = DATASET_DIR / "videos" / vkey / "chunk-000"
        src_path = video_dir / ("file-000" + src_ext)
        dst_path = video_dir / "file-000.mp4"

        if not src_path.exists():
            print(f"[跳过] 源文件不存在: {src_path}")
            continue

        print(f"\n处理 {vkey}:")
        src_frames = get_video_frame_count(src_path)
        print(f"  源帧数: {src_frames}")

        # 对该 video key 读取每个 episode 的 from_timestamp
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_paths = []
            for info in episodes_info:
                ep = info["ep"]
                length = info["length"]
                extra = info["extra_frames"]

                # 有效帧在 episode 时间段的末尾
                # 对正常 episode: from_timestamp 即可
                # 对问题 episode: 需要跳过前面的 reset 帧
                old_from = ep_df.loc[ep, f"videos/{vkey}/from_timestamp"]
                old_to = ep_df.loc[ep, f"videos/{vkey}/to_timestamp"]
                valid_start = old_to - length / FPS  # = old_from + extra/FPS
                seg_path = str(Path(tmpdir) / f"ep_{ep:03d}.mp4")

                print(f"  Episode {ep:2d}: 提取 {length} 帧 (从 {valid_start:.3f}s, extra={extra})...", end="", flush=True)
                extract_segment(src_path, seg_path, valid_start, length)

                actual = get_video_frame_count(Path(seg_path))
                if actual != length:
                    print(f" ⚠ 帧数={actual}, 期望={length}")
                    # 容差: av1 编码可能差 1 帧
                    if abs(actual - length) > 1:
                        print(f"  ⚠ 差距过大，中止！")
                        sys.exit(1)
                else:
                    print(f" ✓")
                segment_paths.append(seg_path)

            # 拼接所有片段
            tmp_output = str(Path(tmpdir) / "merged.mp4")
            print(f"  拼接 {len(segment_paths)} 个片段...")
            concat_segments(segment_paths, Path(tmp_output))

            merged_frames = get_video_frame_count(Path(tmp_output))
            print(f"  拼接结果: {merged_frames} 帧 (期望 {total_parquet_frames})")

            if abs(merged_frames - total_parquet_frames) > len(episodes_info):
                print(f"  ⚠ 帧数差距过大，不替换！")
                sys.exit(1)

            # 替换目标文件 (不动 .bak)
            shutil.copy2(tmp_output, str(dst_path))
            final_frames = get_video_frame_count(dst_path)
            print(f"  写入完成: {final_frames} 帧 ✓")

    # Step 3: 更新 episode 元数据的时间戳
    print("\n--- 更新 episode 元数据 ---")
    for vkey in VIDEO_KEYS:
        cumulative_ts = 0.0
        for ep in ep_df.index:
            length = int(ep_df.loc[ep, "length"])
            ep_dur = length / FPS
            ep_df.at[ep, f"videos/{vkey}/from_timestamp"] = cumulative_ts
            ep_df.at[ep, f"videos/{vkey}/to_timestamp"] = cumulative_ts + ep_dur
            cumulative_ts += ep_dur

    # 保存更新后的元数据 (恢复 episode_index 为普通列)
    ep_df.reset_index().to_parquet(ep_path, index=False)
    print(f"已更新: {ep_path}")

    # Step 4: 验证
    print("\n--- 最终验证 ---")
    ep_df_check = pd.read_parquet(ep_path)
    all_ok = True

    for vkey in VIDEO_KEYS:
        video_path = DATASET_DIR / "videos" / vkey / "chunk-000" / "file-000.mp4"
        frames = get_video_frame_count(video_path)
        ok = "✓" if abs(frames - total_parquet_frames) <= len(ep_df) else "✗"
        if ok == "✗":
            all_ok = False
        print(f"  {vkey}: {frames} 帧 {ok}")

    # 验证时间戳连续性
    for _, row in ep_df_check.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        v_from = row["videos/observation.images.global/from_timestamp"]
        v_to = row["videos/observation.images.global/to_timestamp"]
        expected_dur = length / FPS
        actual_dur = v_to - v_from
        if abs(actual_dur - expected_dur) > 0.001:
            print(f"  ⚠ Episode {ep}: 时间戳异常 dur={actual_dur:.3f} expected={expected_dur:.3f}")
            all_ok = False

    if all_ok:
        print(f"\n修复完成！视频帧数与 parquet 一致 ({total_parquet_frames} 帧)。")
        print("备份文件(.mp4.bak)已保留，确认无误后可手动删除。")
    else:
        print("\n⚠ 部分检查未通过，请检查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
