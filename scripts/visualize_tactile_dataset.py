"""
触觉数据集可视化工具

从已录制的 lerobot 数据集中读取触觉数据（depth、normal、marker_displacement），
用 OpenCV 窗口逐帧回放可视化，类似 6_3_test_mlp_v2.py 的显示风格。

用法:
    python scripts/visualize_tactile_dataset.py --dataset ./datasets/task2
    python scripts/visualize_tactile_dataset.py --dataset ./datasets/task2 --fps 10
    python scripts/visualize_tactile_dataset.py --dataset ./datasets/task2 --start 50 --end 200

快捷键:
    Space  - 暂停/继续
    ← →   - 暂停时逐帧前进/后退
    q/Esc  - 退出
    s      - 保存当前帧截图
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
import pyarrow.parquet as pq
import pyarrow as pa


# ──────────────────── 着色函数（与 6_3_test_mlp_v2.py 一致）────────────────────

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """深度图着色：按压越深颜色越亮"""
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    h, w = depth.shape
    depth_range = depth.max() - depth.min()
    if depth_range < 1e-6:
        return np.full((h, w, 3), [128, 0, 68], dtype=np.uint8)
    depth_normalized = (depth.max() - depth) / depth_range
    depth_normalized = np.clip(depth_normalized, 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)


def colorize_normals(normals: np.ndarray) -> np.ndarray:
    """法向量着色"""
    if normals.shape[-1] != 3:
        h, w = normals.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    n = np.clip((normals + 1) / 2.0, 0, 1)
    bgr = np.stack([n[:, :, 2], n[:, :, 1], n[:, :, 0]], axis=-1)
    return (bgr * 255).astype(np.uint8)


def visualize_marker_displacement(disp: np.ndarray, canvas_size=(480, 640)) -> np.ndarray:
    """可视化 marker 位移"""
    h, w = canvas_size
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if disp is None or disp.size == 0:
        return vis
    if disp.ndim != 2 or disp.shape[1] != 2:
        return vis

    n_markers = len(disp)
    total_d = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)
    avg_d, max_d = np.mean(total_d), np.max(total_d)
    moving = int(np.sum(total_d > 0.5))

    cols = max(1, int(np.ceil(np.sqrt(n_markers * w / h))))
    rows_count = max(1, int(np.ceil(n_markers / cols)))
    sx, sy = w // (cols + 1), (h - 40) // (rows_count + 1)
    scale = 8.0

    for i in range(n_markers):
        r, c = divmod(i, cols)
        cx, cy = (c + 1) * sx, (r + 1) * sy + 35
        if cy >= h or cx >= w:
            continue
        dx, dy = disp[i]
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 0.5:
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        else:
            ex, ey = int(cx + dx * scale), int(cy + dy * scale)
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    cv2.putText(vis, f"Markers:{n_markers}  Avg:{avg_d:.2f}px  Max:{max_d:.2f}px  Moving:{moving}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


# ──────────────────── 数据集加载（使用 pyarrow 高效读取）────────────────────

class TactileDataset:
    """高效加载触觉数据集"""

    def __init__(self, dataset_path: str):
        parquet_files = sorted(glob.glob(
            os.path.join(dataset_path, "data", "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"在 {dataset_path}/data/ 中没有找到 parquet 文件")

        tables = [pq.read_table(f) for f in parquet_files]
        self.table = pa.concat_tables(tables)
        self.num_frames = self.table.num_rows
        self.columns = self.table.column_names

        self.depth_col = self._find_col('depth')
        self.normal_col = self._find_col('normal')
        self.marker_col = self._find_col('displacement', 'marker')

    def _find_col(self, *keywords):
        for col in self.columns:
            cl = col.lower()
            if all(kw in cl for kw in keywords):
                return col
        for col in self.columns:
            cl = col.lower()
            if any(kw in cl for kw in keywords):
                return col
        return None

    def get_frame(self, idx: int, col: str) -> np.ndarray:
        val = self.table.column(col)[idx].as_py()
        return np.array(val, dtype=np.float32)

    def get_timestamp(self, idx: int) -> float:
        if 'timestamp' in self.columns:
            return float(self.table.column('timestamp')[idx].as_py())
        return idx / 30.0


# ──────────────────── 主函数 ────────────────────

def main():
    parser = argparse.ArgumentParser(description="触觉数据集可视化工具")
    parser.add_argument("--dataset", type=str, default="./datasets/task2", help="数据集路径")
    parser.add_argument("--fps", type=int, default=30, help="回放帧率")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=-1, help="结束帧 (-1=全部)")
    parser.add_argument("--save-dir", type=str, default=None, help="截图保存目录")
    parser.add_argument("--blur", type=int, default=5, help="高斯模糊核大小 (奇数, 0=关闭)")
    args = parser.parse_args()

    # 保证核大小为奇数
    if args.blur > 0 and args.blur % 2 == 0:
        args.blur += 1

    ds = TactileDataset(args.dataset)
    print(f"✅ 加载了 {ds.num_frames} 帧数据")
    print(f"   列: {ds.columns}")
    print(f"   深度: {ds.depth_col or '未找到'}")
    print(f"   法向量: {ds.normal_col or '未找到'}")
    print(f"   标记位移: {ds.marker_col or '未找到'}")

    if not any([ds.depth_col, ds.normal_col, ds.marker_col]):
        print("❌ 数据集中没有触觉数据列!")
        sys.exit(1)

    start = args.start
    end = args.end if args.end > 0 else ds.num_frames
    end = min(end, ds.num_frames)

    # 快速采样检查
    print("\n数据有效性检查（采样 5 帧）:")
    sample_indices = np.linspace(start, end - 1, min(5, end - start), dtype=int)
    for label, col in [("深度", ds.depth_col), ("法向量", ds.normal_col), ("标记位移", ds.marker_col)]:
        if col is None:
            continue
        ok = 0
        for si in sample_indices:
            arr = ds.get_frame(si, col)
            if np.any(arr != 0):
                ok += 1
        status = "✅" if ok > 0 else "⚠️"
        s = ds.get_frame(sample_indices[len(sample_indices)//2], col)
        print(f"  {status} {label}: {ok}/{len(sample_indices)} 帧有效, shape={s.shape}, range=[{s.min():.3f}, {s.max():.3f}]")

    print(f"\n▶ 回放: 帧 {start}~{end-1} (共 {end-start} 帧), FPS={args.fps}")
    print("  快捷键: Space=暂停  ←/→或a/d=逐帧  q/Esc=退出  s=截图")
    print("=" * 60)

    cv2.namedWindow("Tactile Data", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tactile Data", 1920, 480)

    paused = False
    idx = start
    save_dir = args.save_dir

    while idx < end:
        panels = []

        if ds.depth_col:
            depth = ds.get_frame(idx, ds.depth_col)
            if args.blur > 0:
                d2 = depth.squeeze(-1) if depth.ndim == 3 else depth
                d2 = cv2.GaussianBlur(d2, (args.blur, args.blur), 0)
                depth = d2[:, :, None] if depth.ndim == 3 else d2
            depth_vis = colorize_depth(depth)
            cv2.putText(depth_vis, f"Depth (blur={args.blur})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(depth_vis)

        if ds.normal_col:
            normal = ds.get_frame(idx, ds.normal_col)
            if args.blur > 0:
                normal = cv2.GaussianBlur(normal, (args.blur, args.blur), 0)
            normal_vis = colorize_normals(normal)
            cv2.putText(normal_vis, f"Normal (blur={args.blur})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(normal_vis)

        if ds.marker_col:
            marker = ds.get_frame(idx, ds.marker_col)
            h = panels[0].shape[0] if panels else 480
            w = panels[0].shape[1] if panels else 640
            marker_vis = visualize_marker_displacement(marker, canvas_size=(h, w))
            panels.append(marker_vis)

        if panels:
            th, tw = panels[0].shape[:2]
            resized = [cv2.resize(p, (tw, th)) if p.shape[:2] != (th, tw) else p for p in panels]
            combined = np.hstack(resized)

            ts = ds.get_timestamp(idx)
            info = f"Frame {idx}/{end-1}  t={ts:.3f}s"
            if paused:
                info += "  [PAUSED]"
            cv2.putText(combined, info, (10, combined.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Tactile Data", combined)

        wait_ms = max(1, int(1000 / args.fps)) if not paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'⏸ 暂停' if paused else '▶ 继续'} @ 帧 {idx}")
        elif key in (83, ord('d')):
            if paused:
                idx = min(idx + 1, end - 1)
            continue
        elif key in (81, ord('a')):
            if paused:
                idx = max(idx - 1, start)
            continue
        elif key == ord('s'):
            if save_dir is None:
                save_dir = os.path.join(args.dataset, "screenshots")
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(fname, combined)
            print(f"  📸 截图: {fname}")

        if not paused:
            idx += 1

    cv2.destroyAllWindows()
    print("\n可视化结束。")


if __name__ == "__main__":
    main()
