"""
MLP方法测试脚本（梯度角度版本）

使用训练好的MLP模型实时处理触觉传感器图像，
计算深度图和法向量图。

与 gs_sdk 的 Reconstructor 逻辑一致：
1. 加载背景图，计算背景梯度角度
2. 计算当前帧梯度角度，减去背景得到差分梯度
3. 使用泊松方程重建深度

使用方法:
1. 先运行 6_1_collect_mlp_dataset_angle.py 收集数据
2. 运行 6_2_train_mlp_angle.py 训练模型
3. 运行本脚本测试效果
"""

import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvista as pv
from scipy import fftpack
import math

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.tactile_cam.visualization import TactileVisualizer
from lerobot.cameras.tactile_cam.gelsight_marker_tracker import GelSightMarkerTracker
from lerobot.cameras.tactile_cam.model import BGRXYMLPNet
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def image2bgrxys(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ys = np.linspace(0, 1, h, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, 1, w, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255.0, xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


def poisson_dct_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    使用DCT的泊松求解器 
    
    Args:
        gx: X方向梯度 (H, W)
        gy: Y方向梯度 (H, W)
        
    Returns:
        深度图 (H, W)
    """
    # 计算拉普拉斯算子
    gxx = 1 * (
        gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])]
        - gx[:, ([0] + list(range(gx.shape[1] - 1)))]
    )
    gyy = 1 * (
        gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :]
        - gy[([0] + list(range(gx.shape[0] - 1))), :]
    )
    f = gxx + gyy

    # 边界条件
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    # 边界修正
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    # 角落修正
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    # DCT变换
    tt = fftpack.dct(f, norm="ortho")
    fcos = fftpack.dct(tt.T, norm="ortho").T

    # 频域求解
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    ).astype(np.float32)
    denom[denom == 0] = 1  # 避免除零

    # 逆DCT变换
    f = -fcos / denom
    tt = fftpack.idct(f, norm="ortho")
    img_tt = fftpack.idct(tt.T, norm="ortho").T
    img_tt = img_tt - img_tt.mean()

    return img_tt


class Reconstructor:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化重建器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.device = device
        self.bg_image = None
        self.bg_G = None
        
        # 加载模型
        if not os.path.isfile(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        self.gxy_net = BGRXYMLPNet()
        self.gxy_net.load_state_dict(torch.load(model_path, map_location=device))
        self.gxy_net.eval()
        self.gxy_net.to(device)
        
        print(f"[INFO] 模型已加载: {model_path}")
        print(f"[INFO] 使用设备: {device}")
    
    def load_bg(self, bg_image: np.ndarray):
        """
        加载背景图像并计算背景梯度
        
        Args:
            bg_image: 背景图像 (BGR)
        """
        self.bg_image = bg_image.copy()
        
        # 计算背景梯度
        bgrxys = image2bgrxys(bg_image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().numpy()
            # 将梯度角度转换为梯度值
            self.bg_G = np.tan(gxyangles.reshape(bg_image.shape[0], bg_image.shape[1], 2))
        
        print(f"[INFO] 背景图像已加载，尺寸: {bg_image.shape[:2]}")
    
    def get_surface_info(self, image: np.ndarray, ppmm: float = 7.6,
                         color_dist_threshold: float = 15,
                         height_threshold: float = 0.2) -> tuple:
        """
        获取表面信息：梯度、深度、接触掩膜
        
        Args:
            image: 当前帧 (BGR)
            ppmm: 像素每毫米
            color_dist_threshold: 颜色距离阈值
            height_threshold: 高度阈值 (mm)
            
        Returns:
            G: 梯度 (H, W, 2)
            H: 深度图 (H, W)
            C: 接触掩膜 (H, W)
        """
        if self.bg_image is None:
            raise ValueError("请先调用 load_bg() 加载背景图像")
        
        # 计算当前帧梯度
        bgrxys = image2bgrxys(image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().numpy()
            G = np.tan(gxyangles.reshape(image.shape[0], image.shape[1], 2))
            # 减去背景梯度
            G = G - self.bg_G
        
        # 计算深度图
        H = poisson_dct_neumann(G[:, :, 0], G[:, :, 1]).astype(np.float32)

        
        # 计算接触掩膜
        diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
        color_mask = np.linalg.norm(diff_image, axis=-1) > color_dist_threshold
        color_mask = cv2.dilate(color_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
        color_mask = cv2.erode(color_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
        
        # 高度掩膜
        cutoff = np.percentile(H, 85) - height_threshold / ppmm
        height_mask = H < cutoff
        
        # 组合掩膜
        C = np.logical_and(color_mask, height_mask)
        
        return G, H, C


class MLPProcessor(BaseProcessor):
    
    def __init__(self, model_path: str = None, pad: int = 20,
                 calib_file: str = None, device: str = None, ppmm: float = 7.6):
        """
        初始化处理器
        
        Args:
            model_path: MLP模型文件路径
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            device: 计算设备
            ppmm: 像素每毫米
        """
        # 如果没有指定标定文件，使用默认路径
        if calib_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(current_dir, "calibration_data", "homography_matrix.npz")
        
        super().__init__(pad=pad, calib_file=calib_file)
        
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.ppmm = ppmm
        
        # 加载模型
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "load", "mlp_angle_model.pth")
        
        try:
            self.reconstructor = Reconstructor(model_path, self.device)
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            self.reconstructor = None
        
        # 参考帧采集
        self.ref_frames_buffer = []
        self.ref_avg_count = 10
        self.con_flag = True
    
    def reset(self):
        """重置处理器"""
        self.ref_frames_buffer = []
        self.con_flag = True
        if self.reconstructor:
            self.reconstructor.bg_image = None
            self.reconstructor.bg_G = None
        print("[INFO] 处理器已重置，将采集多帧平均作为背景")
    
    def _colorize_gradient(self, G: np.ndarray) -> np.ndarray:
        """
        梯度可视化 (与gs_sdk test_model一致)
        
        红色: gx
        蓝色: gy
        """
        red = G[:, :, 0] * 255 / 3.0 + 127
        red = np.clip(red, 0, 255)
        blue = G[:, :, 1] * 255 / 3.0 + 127
        blue = np.clip(blue, 0, 255)
        grad_image = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
        return grad_image
    
    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """深度图着色 - 按压越深颜色越亮"""
        h, w = depth.shape
        
        depth_range = depth.max() - depth.min()
        if depth_range < 0.01:
            return np.full((h, w, 3), [128, 0, 68], dtype=np.uint8)
        
        # 反转深度：按压越深（值越小/越负）-> 显示越亮
        depth_normalized = (depth.max() - depth) / depth_range
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    
    def _colorize_normals(self, G: np.ndarray) -> np.ndarray:
        """
        从梯度计算法向量并着色
        
        法向量 n = (-gx, -gy, 1) / |n|
        """
        gx = G[:, :, 0]
        gy = G[:, :, 1]
        gz = np.ones_like(gx)
        
        # 归一化
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        magnitude[magnitude == 0] = 1
        
        nx = -gx / magnitude
        ny = -gy / magnitude
        nz = gz / magnitude
        
        # 映射到颜色 [-1, 1] -> [0, 255]
        normals = np.stack([nx, ny, nz], axis=-1)
        normals_normalized = (normals + 1) / 2
        
        # BGR格式
        normals_bgr = np.stack([
            normals_normalized[:, :, 2],  # B <- nz
            normals_normalized[:, :, 1],  # G <- ny
            normals_normalized[:, :, 0],  # R <- nx
        ], axis=-1)
        
        return (normals_bgr * 255).astype(np.uint8)
    
    def process_frame(self, frame: np.ndarray, apply_warp: bool = False):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像
            apply_warp: 是否应用透视变换
            
        Returns:
            depth_colored, normal_colored, grad_colored, raw_depth, G, C
        """
        if apply_warp:
            frame = self.warp_perspective(frame)
        
        h, w = frame.shape[:2]
        
        if self.reconstructor is None:
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            return empty, empty, empty, np.zeros((h, w)), None, None
        
        # 采集背景帧
        if self.con_flag:
            self.ref_frames_buffer.append(frame.astype(np.float32))
            
            if len(self.ref_frames_buffer) < self.ref_avg_count:
                progress = len(self.ref_frames_buffer)
                progress_img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(progress_img, f"Collecting bg: {progress}/{self.ref_avg_count}",
                           (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return progress_img, progress_img, progress_img, np.zeros((h, w)), None, None
            else:
                bg_image = np.mean(self.ref_frames_buffer, axis=0).astype(np.uint8)
                self.reconstructor.load_bg(bg_image)
                self.con_flag = False
                self.ref_frames_buffer = []
                return (np.zeros((h, w, 3), dtype=np.uint8),) * 3 + (np.zeros((h, w)), None, None)
        
        # 获取表面信息
        G, H, C = self.reconstructor.get_surface_info(frame, self.ppmm)
        
        # 可视化
        depth_colored = self._colorize_depth(H)
        normal_colored = self._colorize_normals(G)
        grad_colored = self._colorize_gradient(G)
        
        return depth_colored, normal_colored, grad_colored, H, G, C


def main():
    """主函数"""
    
    # 相机配置
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video2",
        fps=25,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        exposure=600,
        auto_exposure=False,
        wb_temperature=4000,
        auto_wb=False,
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0,
    )
    
    # 数据保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "data", "tactile_data_mlp_angle")
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载 ppmm
    mm_per_pixel_file = os.path.join(current_dir, "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_per_pixel_file):
        data = np.load(mm_per_pixel_file)
        mm_per_pixel = float(data['mm_per_pixel'])
        PPMM = 1.0 / mm_per_pixel
        print(f"[INFO] ppmm: {PPMM:.2f} pixel/mm")
    else:
        PPMM = 7.6
        print(f"[WARNING] 使用默认 ppmm: {PPMM:.2f}")
    
    # 初始化组件
    camera = TactileCamera(camera_config)
    processor = MLPProcessor(ppmm=PPMM)
    visualizer = TactileVisualizer(
        windows=['original', 'depth', 'normal', 'gradient', 'marker'],
        window_size=(640, 480)
    )
    
    # 初始化 marker 追踪器
    marker_tracker = GelSightMarkerTracker()
    marker_tracker_initialized = False
    
    # 连续采集数据缓冲
    depth_buffer = []       # 存储深度数据
    normal_buffer = []      # 存储法向量数据
    marker_buffer = []      # 存储 marker 位移数据
    
    # 按键状态（用于在PyVista和OpenCV之间共享）
    key_pressed = {'key': -1}
    
    def on_key_press(key):
        """PyVista 键盘回调"""
        key_pressed['key'] = ord(key) if isinstance(key, str) and len(key) == 1 else -1
    
    # 初始化3D可视化
    plotter = pv.Plotter(window_size=(800, 600), title="3D Depth")
    plotter.set_background('white')
    plotter.add_axes()
    plotter.add_key_event('r', lambda: on_key_press('r'))
    plotter.add_key_event('m', lambda: on_key_press('m'))
    plotter.add_key_event('c', lambda: on_key_press('c'))
    plotter.add_key_event('q', lambda: on_key_press('q'))
    plotter.show(interactive_update=True, auto_close=False)
    
    if processor.reconstructor is None:
        print("[ERROR] 模型未加载，无法继续")
        return
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        print("\n=== 触觉传感器测试 (MLP Angle) ===")
        print("操作说明（请确保 OpenCV 窗口有焦点）:")
        print("  r - 重置背景帧和marker追踪器")
        print("  m - 重置marker初始位置")
        print("  c - 开始/停止连续采集（自动保存）")
        print("  q - 退出")
        print("=" * 50)
        
        is_collecting = False  # 是否正在连续采集
        
        while True:
            try:
                frame = camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                warped_frame = processor.warp_perspective(frame_bgr)
                
                depth_colored, normal_colored, grad_colored, raw_depth, G, C = \
                    processor.process_frame(warped_frame, apply_warp=False)
                
                visualizer.show('original', warped_frame)
                visualizer.show('depth', depth_colored)
                visualizer.show('normal', normal_colored)
                visualizer.show('gradient', grad_colored)
                
                # === Marker 追踪与可视化 ===
                if not processor.con_flag:  # 背景已采集完成
                    if not marker_tracker_initialized:
                        # 初始化 marker 追踪器，使用背景帧
                        bg_frame = processor.reconstructor.bg_image
                        marker_tracker.reinit(bg_frame)
                        marker_tracker_initialized = True
                        print(f"[INFO] Marker追踪器已初始化, 检测到 {marker_tracker.MarkerCount} 个marker点")
                    else:
                        # 更新 marker 运动
                        marker_tracker.update_markerMotion(warped_frame)
                        
                        # 获取位移数据
                        displacements = marker_tracker.get_marker_displacements()
                        
                        # 可视化 marker 位移
                        marker_vis = warped_frame.copy()
                        if marker_tracker.flowcenter is not None and len(marker_tracker.flowcenter) > 0:
                            for i in range(marker_tracker.MarkerCount):
                                if i < len(marker_tracker.markerU):
                                    cx = int(marker_tracker.flowcenter[i, 0])
                                    cy = int(marker_tracker.flowcenter[i, 1])
                                    dx = marker_tracker.markerU[i]
                                    dy = marker_tracker.markerV[i]
                                    
                                    # 画原始位置（绿色圆点）
                                    cv2.circle(marker_vis, (cx, cy), 3, (0, 255, 0), -1)
                                    
                                    # 画位移箭头（黄色）
                                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                                        scale = 5  # 放大显示位移
                                        end_x = int(cx + dx * scale)
                                        end_y = int(cy + dy * scale)
                                        cv2.arrowedLine(marker_vis, (cx, cy), (end_x, end_y), 
                                                       (0, 255, 255), 2, tipLength=0.3)
                        
                        visualizer.show('marker', marker_vis)
                        
                        # 连续采集数据
                        if is_collecting:
                            # 采集深度
                            if raw_depth is not None:
                                depth_buffer.append(raw_depth.copy())
                            # 采集法向量（从梯度计算）
                            if G is not None:
                                # 计算法向量 n = (-gx, -gy, 1) / |n|
                                gx, gy = G[:, :, 0], G[:, :, 1]
                                gz = np.ones_like(gx)
                                magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
                                magnitude[magnitude == 0] = 1
                                normals = np.stack([-gx/magnitude, -gy/magnitude, gz/magnitude], axis=-1)
                                normal_buffer.append(normals.astype(np.float32))
                            # 采集marker位移
                            if displacements is not None:
                                marker_buffer.append({
                                    'displacements': displacements.copy(),
                                    'positions': marker_tracker.flowcenter.copy() if marker_tracker.flowcenter is not None else None
                                })
                else:
                    # 背景采集中，显示空白
                    visualizer.show('marker', np.zeros_like(warped_frame))
                
                # === 3D 深度可视化 ===
                if raw_depth is not None and not processor.con_flag:
                    h, w = raw_depth.shape
                    step = 4
                    depth_ds = raw_depth[::step, ::step]
                    hh, ww = depth_ds.shape
                    
                    depth_range = depth_ds.max() - depth_ds.min()
                    if depth_range > 0.01:
                        # 反转深度：按压越深 -> 3D中显示越高
                        depth_norm = (depth_ds.max() - depth_ds) / depth_range
                    else:
                        depth_norm = np.zeros_like(depth_ds)
                    
                    x = np.arange(ww)
                    y = np.arange(hh)
                    x, y = np.meshgrid(x, y)
                    
                    grid = pv.StructuredGrid(x, y, depth_norm * 10.0)
                    grid["depth"] = depth_norm.flatten(order="F")
                    
                    plotter.clear()
                    plotter.add_mesh(grid, scalars="depth", cmap='viridis', show_scalar_bar=True)
                    plotter.update()
                
                # 检查按键（同时支持 OpenCV 窗口和 PyVista 窗口）
                key = visualizer.wait_key(1)
                # 如果 PyVista 窗口有按键，优先使用
                if key_pressed['key'] != -1:
                    key = key_pressed['key']
                    key_pressed['key'] = -1  # 重置
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    processor.reset()
                    marker_tracker_initialized = False
                    depth_buffer = []
                    normal_buffer = []
                    marker_buffer = []
                    print("[INFO] 处理器已重置，Marker追踪器将重新初始化")
                elif key == ord('m') and marker_tracker_initialized:
                    marker_tracker.iniMarkerPos()
                    print("[INFO] Marker初始位置已重置")
                elif key == ord('c'):
                    is_collecting = not is_collecting
                    if is_collecting:
                        # 开始新的采集，清空缓冲
                        depth_buffer = []
                        normal_buffer = []
                        marker_buffer = []
                        print("[INFO] 开始采集数据...")
                    else:
                        # 停止采集，自动保存
                        frame_count = len(depth_buffer)
                        if frame_count > 0:
                            timestamp = int(cv2.getTickCount())
                            
                            # 保存深度数据
                            depth_file = os.path.join(save_dir, f"depth_{timestamp}.npy")
                            np.save(depth_file, np.array(depth_buffer))
                            print(f"[INFO] 深度数据已保存: {depth_file} ({frame_count} 帧)")
                            
                            # 保存法向量数据
                            if len(normal_buffer) > 0:
                                normal_file = os.path.join(save_dir, f"normal_{timestamp}.npy")
                                np.save(normal_file, np.array(normal_buffer))
                                print(f"[INFO] 法向量数据已保存: {normal_file} ({len(normal_buffer)} 帧)")
                            
                            # 保存marker数据
                            if len(marker_buffer) > 0:
                                marker_file = os.path.join(save_dir, f"marker_{timestamp}.npz")
                                np.savez(marker_file, 
                                        data=marker_buffer,
                                        marker_count=marker_tracker.MarkerCount if marker_tracker_initialized else 0)
                                print(f"[INFO] Marker数据已保存: {marker_file} ({len(marker_buffer)} 帧)")
                            
                            # 清空缓冲
                            depth_buffer = []
                            normal_buffer = []
                            marker_buffer = []
                        else:
                            print("[INFO] 停止采集，无数据保存")
                
            except TimeoutError:
                continue
            except RuntimeError as e:
                print(f"[WARNING] 帧读取错误: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    finally:
        plotter.close()
        visualizer.cleanup()
        camera.disconnect()
        if marker_tracker_initialized:
            marker_tracker.cleanup()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
