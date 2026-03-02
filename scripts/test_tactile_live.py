#!/usr/bin/env python
"""
触觉传感器实时测试脚本

实时显示触觉传感器的：
  - 原始图像 (透视变换后)
  - 深度图 (Viridis 色彩映射)
  - 法向量图 (RGB 着色)
  - Marker 位移箭头

所有面板横向拼接在一个窗口中，轻量无 3D 依赖。

用法:
    python scripts/test_tactile_live.py
    python scripts/test_tactile_live.py --device /dev/video4
    python scripts/test_tactile_live.py --num-markers 35

快捷键:
    r - 重置背景帧和 marker 追踪器
    m - 重置 marker 初始位置
    s - 保存当前帧数据 (depth + normal .npy)
    q - 退出
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# 确保可以导入 lerobot 模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_script_dir, "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import MLPProcessor
from lerobot.cameras.tactile_cam.gelsight_marker_tracker import GelSightMarkerTracker
from lerobot.cameras.configs import ColorMode, Cv2Rotation


# ──────────────────────────────────────────────────────────────
# 可视化辅助
# ──────────────────────────────────────────────────────────────

def draw_marker_arrows(frame: np.ndarray, tracker: GelSightMarkerTracker, scale: float = 6.0, threshold: float = 6) -> np.ndarray:
    """在帧上绘制 marker 位移箭头，小于 threshold 的位移视为静止"""
    vis = frame.copy()
    if tracker.flowcenter is None or len(tracker.flowcenter) == 0:
        return vis
    if tracker.markerU is None or tracker.markerV is None:
        return vis

    centers = np.around(tracker.flowcenter[:, 0:2]).astype(np.int16)
    displacements = tracker.get_marker_displacements()
    if displacements is None:
        return vis

    total_d = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    avg_d, max_d = np.mean(total_d), np.max(total_d)
    moving = int(np.sum(total_d > threshold))

    for i in range(min(tracker.MarkerCount, len(centers))):
        if i >= len(tracker.markerU):
            break
        cx, cy = int(centers[i, 0]), int(centers[i, 1])
        dx, dy = tracker.markerU[i], tracker.markerV[i]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        if mag < threshold:
            cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)
        else:
            ex, ey = int(cx + dx * scale), int(cy + dy * scale)
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    cv2.putText(vis, f"Markers:{tracker.MarkerCount}  Avg:{avg_d:.2f}  Max:{max_d:.2f}  Moving:{moving}  Thr:{threshold}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="触觉传感器实时测试")
    parser.add_argument("--device", type=str, default="/dev/video4", help="相机设备路径")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25, help="相机帧率")
    parser.add_argument("--exposure", type=int, default=600)
    parser.add_argument("--wb", type=int, default=4000, help="白平衡色温")
    parser.add_argument("--num-markers", type=int, default=35, help="marker 数量")
    parser.add_argument("--threshold", type=float, default=6, help="marker 位移阈值 (像素), 低于此值视为噪声归零")
    parser.add_argument("--save-dir", type=str, default=None, help="保存目录 (默认 outputs/tactile_test)")
    args = parser.parse_args()

    # ── 相机配置 ──
    camera_config = TactileCameraConfig(
        index_or_path=args.device,
        fps=args.fps,
        width=args.width,
        height=args.height,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        exposure=args.exposure,
        auto_exposure=False,
        wb_temperature=args.wb,
        auto_wb=False,
    )

    # ── 路径 ──
    tactile_dir = os.path.abspath(os.path.join(_src_dir, "lerobot", "cameras", "tactile_cam"))
    model_path = os.path.join(tactile_dir, "load", "nnmodel_v2.pth")
    calib_file = os.path.join(tactile_dir, "calibration_data", "homography_matrix.npz")

    # 加载 ppmm
    mm_file = os.path.join(tactile_dir, "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_file):
        ppmm = 1.0 / float(np.load(mm_file)["mm_per_pixel"])
        print(f"[INFO] ppmm = {ppmm:.2f} pixel/mm")
    else:
        ppmm = 7.6
        print(f"[WARN] 使用默认 ppmm = {ppmm:.2f}")

    save_dir = args.save_dir or os.path.join(_script_dir, "..", "outputs", "tactile_test")
    os.makedirs(save_dir, exist_ok=True)

    # ── 初始化 ──
    camera = TactileCamera(camera_config)
    processor = MLPProcessor(
        model_path=model_path if os.path.exists(model_path) else None,
        pad=20,
        calib_file=calib_file if os.path.exists(calib_file) else None,
        ppmm=ppmm,
    )
    tracker = GelSightMarkerTracker()
    tracker.IsDisplay = False
    marker_initialized = False

    cv2.namedWindow("Tactile Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tactile Live", 1920, 480)

    camera.connect()
    print("[INFO] 相机已连接")
    print("\n=== 触觉传感器实时测试 ===")
    print("  r - 重置背景 + marker    m - 重置 marker 位置")
    print("  s - 保存数据              q - 退出")
    print("=" * 50)

    fps_counter = 0
    fps_time = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            try:
                frame = camera.async_read(timeout_ms=200)
            except TimeoutError:
                continue
            except RuntimeError as e:
                print(f"[WARN] 帧读取错误: {e}")
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ── MLP 处理 ──
            depth_colored, normal_colored, raw_depth, raw_normals = processor.process_frame(
                frame_bgr, apply_warp=True
            )
            warped = processor.warp_perspective(frame_bgr)

            # ── Marker 追踪 ──
            if not processor.con_flag:
                if not marker_initialized:
                    bg = processor.bg_image
                    if bg is not None:
                        tracker.reinit(bg)
                        marker_initialized = True
                        print(f"[INFO] Marker 追踪器初始化完成，检测到 {tracker.MarkerCount} 个标记点")
                else:
                    tracker.update_markerMotion(warped)

            # ── 拼接面板 ──
            panels = []

            # 1) 原始图像
            raw_panel = warped.copy()
            cv2.putText(raw_panel, "Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panels.append(raw_panel)

            # 2) 深度图
            depth_panel = depth_colored.copy() if depth_colored is not None else np.zeros_like(warped)
            cv2.putText(depth_panel, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(depth_panel)

            # 3) 法向量图
            normal_panel = normal_colored.copy() if normal_colored is not None else np.zeros_like(warped)
            cv2.putText(normal_panel, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(normal_panel)

            # 4) Marker 位移
            if marker_initialized and not processor.con_flag:
                marker_panel = draw_marker_arrows(warped, tracker, threshold=args.threshold)
            else:
                marker_panel = np.zeros_like(warped)
                if processor.con_flag:
                    cv2.putText(marker_panel, "Collecting background...", (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(marker_panel, "Marker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(marker_panel)

            # 统一尺寸拼接
            th, tw = panels[0].shape[:2]
            resized = [cv2.resize(p, (tw, th)) if p.shape[:2] != (th, tw) else p for p in panels]
            combined = np.hstack(resized)

            # FPS 统计
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                display_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now
            cv2.putText(combined, f"FPS: {display_fps:.1f}", (combined.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Tactile Live", combined)

            # ── 按键处理 ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                processor.reset()
                marker_initialized = False
                print("[INFO] 背景帧已重置，正在重新采集...")
            elif key == ord("m"):
                if marker_initialized:
                    tracker.iniMarkerPos()
                    print("[INFO] Marker 位置已重置为当前位置")
            elif key == ord("s") and raw_depth is not None:
                ts = int(time.time() * 1000)
                np.save(os.path.join(save_dir, f"depth_{ts}.npy"), raw_depth)
                if raw_normals is not None:
                    np.save(os.path.join(save_dir, f"normal_{ts}.npy"), raw_normals)
                cv2.imwrite(os.path.join(save_dir, f"raw_{ts}.png"), warped)
                print(f"[INFO] 已保存: depth_{ts}.npy, normal_{ts}.npy, raw_{ts}.png")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
