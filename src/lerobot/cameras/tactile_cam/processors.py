"""
触觉传感器图像处理器模块

提供两种方法计算法向量和深度：
1. LookupTableProcessor: 基于查找表的方法（适用于校准过的传感器）
2. GradientProcessor: 基于梯度的方法（通用方法）
"""

import numpy as np
import cv2
import os
import math
import scipy.fftpack

from .fast_poisson import fast_poisson


class BaseProcessor:
    """处理器基类，提供公共功能"""
    
    def __init__(self, pad: int = 20, calib_file: str = None):
        """
        初始化处理器
        
        Args:
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
        """
        self.pad = pad
        self.con_flag = True  # 是否是第一帧
        
        # 透视变换相关
        self.homography_matrix = None
        self.output_size = None
        
        if calib_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(current_dir, "calibration_data", "homography_matrix_320x240.npz")
        self.calib_file = calib_file
        
        self._load_homography()
    
    def _load_homography(self):
        """加载透视变换矩阵"""
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵，输出尺寸: {self.output_size}')
        except FileNotFoundError:
            print(f'[WARNING] 透视变换矩阵文件不存在: {self.calib_file}')
        except Exception as e:
            print(f'[WARNING] 加载透视变换矩阵失败: {e}')
    
    def warp_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        应用透视变换
        
        Args:
            image: 输入BGR图像
            
        Returns:
            变换后的图像
        """
        if self.homography_matrix is None:
            return image
        return cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size if self.output_size else (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST
        )
    
    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """裁剪图像边缘"""
        if self.pad > 0:
            return img[self.pad:-self.pad, self.pad:-self.pad]
        return img
    
    def reset(self):
        """重置处理器状态，下一帧将作为新的参考帧"""
        self.con_flag = True
        print("[INFO] 处理器已重置，下一帧将作为参考帧")
    
    def process_frame(self, frame: np.ndarray):
        """
        处理单帧图像（子类需实现）
        
        Args:
            frame: BGR格式图像
            
        Returns:
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
        """
        raise NotImplementedError


class LookupTableProcessor(BaseProcessor):
    """
    基于查找表的 GelSight 图像处理器
    
    使用预先校准的查找表将颜色差异映射到梯度值，
    然后通过泊松方程重建深度。
    """
    
    def __init__(self, table_path: str = None, pad: int = 20, calib_file: str = None, 
                 has_marker: bool = False):
        """
        初始化处理器
        
        Args:
            table_path: 查找表文件路径
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            has_marker: 是否有标记点，False则不进行marker检测
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        # 是否有marker
        self.has_marker = has_marker
        
        # 查找表参数
        self.zeropoint = [-38, -30, -66]
        self.lookscale = [80, 62, 132]
        self.bin_num = 90
        
        # 加载查找表
        if table_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            table_path = os.path.join(current_dir, "load", "table_smooth.npy")
        self.table = np.load(table_path)
        
        # 状态变量
        self.ref_blur = None
        self.blur_inverse = None
        self.red_mask = None
        self.dmask = None
        self.kernel = self._make_kernel(9, 'circle')
        self.kernel2 = self._make_kernel(9, 'circle')
        self.reset_shape = True
    
    def _make_kernel(self, n: int, k_type: str) -> np.ndarray:
        """创建形态学核"""
        if k_type == 'circle':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    
    def _defect_mask(self, img: np.ndarray) -> np.ndarray:
        """创建缺陷区域掩膜"""
        pad = 20
        im_mask = np.ones(img.shape)
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad * 2 + 20] = 0
        im_mask[:, -pad:] = 0
        return im_mask.astype(int)
    
    def _marker_detection(self, raw_image: np.ndarray) -> np.ndarray:
        """检测标记点区域"""
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        ref_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (25, 25), 0)
        
        diff = ref_blur - raw_image_blur
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.
        
        mask = ((diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) & (diff[:, :, 1] > 120))
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel2, iterations=1)
        return mask
    
    def _find_dots(self, binary_image: np.ndarray):
        """查找标记点"""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 5
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(binary_image.astype(np.uint8))
    
    def _make_mask(self, img: np.ndarray, keypoints) -> np.ndarray:
        """根据关键点创建掩膜"""
        img_mask = np.zeros_like(img[:, :, 0])
        for kp in keypoints:
            cv2.ellipse(img_mask, (int(kp.pt[0]), int(kp.pt[1])),
                       (9, 6), 0, 0, 360, (1), -1)
        return img_mask
    
    def _matching_v2(self, test_img: np.ndarray, ref_blur: np.ndarray, 
                     blur_inverse: np.ndarray) -> np.ndarray:
        """使用查找表将颜色差异映射到梯度"""
        diff_temp1 = test_img - ref_blur
        diff_temp2 = diff_temp1 * blur_inverse
        
        # 分通道归一化
        diff_temp3 = np.zeros_like(diff_temp2, dtype=np.float32)
        for ch in range(3):
            diff_temp3[..., ch] = (diff_temp2[..., ch] - self.zeropoint[ch]) / self.lookscale[ch]
        
        diff_temp3 = np.clip(diff_temp3, 0, 0.999)
        diff = (diff_temp3 * self.bin_num).astype(int)
        diff = np.clip(diff, 0, self.bin_num - 1)
        
        # 查表获取梯度
        grad_img = self.table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
        return grad_img
    
    def process_frame(self, frame: np.ndarray):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像（已透视变换）
            
        Returns:
            depth_colored: 深度图可视化 (H, W, 3)
            normal_colored: 法向量可视化 (H, W, 3)
            raw_depth: 原始深度数据 (H, W) 或 None
            raw_normals: 原始法向量数据 (H, W, 3) 或 None
        """
        raw_image = self._crop_image(frame)
        h, w = raw_image.shape[:2]
        
        if self.con_flag:
            # 第一帧作为参考
            ref_image = raw_image.copy()
            
            if self.has_marker:
                marker = self._marker_detection(ref_image)
                keypoints = self._find_dots((1 - marker) * 255)
                
                if self.reset_shape:
                    marker_mask = self._make_mask(ref_image, keypoints)
                    ref_image = cv2.inpaint(ref_image, marker_mask, 3, cv2.INPAINT_TELEA)
            
            if self.reset_shape:
                self.red_mask = (ref_image[:, :, 2] > 12).astype(np.uint8)
                self.dmask = self._defect_mask(ref_image[:, :, 0])
                self.ref_blur = cv2.GaussianBlur(ref_image.astype(np.float32), (5, 5), 0)
                self.blur_inverse = 1 + ((np.mean(self.ref_blur) / (self.ref_blur + 1)) - 1) * 2
                self.reset_shape = False
            
            self.con_flag = False
            
            return (
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                None, None
            )
        
        # 处理后续帧
        raw_image = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        
        # 根据是否有marker决定是否检测marker
        if self.has_marker:
            marker_mask = self._marker_detection(raw_image)
        else:
            marker_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 计算梯度
        grad_img = self._matching_v2(raw_image, self.ref_blur, self.blur_inverse)
        grad_x = grad_img[:, :, 0] * (1 - marker_mask)
        grad_y = grad_img[:, :, 1] * (1 - marker_mask)
        
        # 平滑梯度
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), sigmaX=0) * (1 - marker_mask)
        grad_y = cv2.GaussianBlur(grad_y, (5, 5), sigmaX=0) * (1 - marker_mask)
        
        # 计算法向量
        denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
        normal_x = -grad_x / denom
        normal_y = -grad_y / denom
        normal_z = 1.0 / denom
        raw_normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        
        # 法向量可视化（强力增强对比度）
        # 放大 x, y 分量以增强可视化效果
        normal_x_enhanced = np.clip(raw_normals[:, :, 0] * 8.0, -1, 1)
        normal_y_enhanced = np.clip(raw_normals[:, :, 1] * 8.0, -1, 1)
        normals_enhanced = np.stack([normal_x_enhanced, normal_y_enhanced, raw_normals[:, :, 2]], axis=-1)
        N_disp = 0.5 * (normals_enhanced + 1.0)
        N_disp = np.clip(N_disp, 0, 1)
        normal_colored = (N_disp * 255).astype(np.uint8)
        normal_colored = cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR)
        
        # 计算深度
        raw_depth = fast_poisson(grad_x, grad_y)
        depth_min = np.nanmin(raw_depth[raw_depth != 0]) if np.any(raw_depth != 0) else 0
        raw_depth = raw_depth - depth_min
        raw_depth[raw_depth < 0] = 0
        
        # 深度可视化
        depth_denoised = cv2.bilateralFilter(raw_depth.astype(np.float32), d=9, 
                                             sigmaColor=75, sigmaSpace=75)
        depth_normalized = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        return depth_colored, normal_colored, raw_depth, raw_normals
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.reset_shape = True
        self.ref_blur = None
        self.blur_inverse = None


def _recon_poisson_dst(img, frame0, x_ratio=0.5, y_ratio=0.5, bias=1.0):
    """
    使用DST求解泊松方程重建深度
    
    Args:
        img: 当前帧
        frame0: 参考帧
        x_ratio: x方向比例系数
        y_ratio: y方向比例系数
        bias: 偏置系数
        
    Returns:
        result: 重建的深度图
        dx_display: 梯度X
        dy_display: 梯度Y
    """
    img = np.int32(img)
    frame0 = np.int32(frame0)
    diff = (img - frame0) * bias

    dx1 = diff[:, :, 1] * x_ratio / 255.0
    dy1 = (diff[:, :, 2] * y_ratio - diff[:, :, 0] * (1 - y_ratio)) / 255.0
    
    dx1 = np.clip(dx1, -0.99, 0.99)
    dy1 = np.clip(dy1, -0.99, 0.99)
    
    dx_display = dx1 / np.sqrt(1 - dx1 ** 2)
    dy_display = dy1 / np.sqrt(1 - dy1 ** 2)
    
    dx = dx_display / 32
    dy = dy_display / 32

    gxx = dx[:-1, 1:] - dx[:-1, :-1]
    gyy = dy[1:, :-1] - dy[:-1, :-1]

    f = np.zeros(dx.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    x, y = np.meshgrid(range(1, f.shape[1]+1), range(1, f.shape[0]+1))
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin / denom

    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    result = np.zeros(f.shape)
    result[1:-1, 1:-1] = img_tt[1:-1, 1:-1]

    return result, dx_display, dy_display


class GradientProcessor(BaseProcessor):
    """
    基于梯度的 GelSight 图像处理器
    
    使用图像颜色差异直接计算梯度，不依赖查找表。
    适用于未校准或通用的传感器。
    """
    
    def __init__(self, pad: int = 20, sensor_id: str = "right", calib_file: str = None):
        """
        初始化处理器
        
        Args:
            pad: 边缘裁剪像素数
            sensor_id: 传感器ID，"left" 或 "right"
            calib_file: 透视变换矩阵文件路径
        """
        super().__init__(pad=pad, calib_file=calib_file)
        self.sensor_id = sensor_id
        
        # 状态变量
        self.ref_frame = None
        self.ref_blur = None
        self.frame_count = 0
    
    def img2grad(self, frame0: np.ndarray, frame: np.ndarray, 
                 bias: float = 4.0) -> tuple:
        """
        从图像差异计算梯度（线性方法）
        
        Args:
            frame0: 参考帧
            frame: 当前帧
            bias: 偏置系数
            
        Returns:
            dx, dy: x和y方向的梯度
        """
        diff = (frame.astype(np.float32) - frame0.astype(np.float32)) * bias
        dx = diff[:, :, 1] / 255.0  # Green通道
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0  # B - R
        return dx, dy
    
    def img2depth(self, frame0: np.ndarray, frame: np.ndarray, 
                  bias: float = 1.0, x_ratio: float = 0.5, 
                  y_ratio: float = 0.5) -> tuple:
        """
        从图像计算深度
        
        Args:
            frame0: 参考帧
            frame: 当前帧
            bias: 偏置系数
            x_ratio: x方向比例系数
            y_ratio: y方向比例系数
            
        Returns:
            depth: 深度图
            dx, dy: 梯度
        """
        return _recon_poisson_dst(frame, frame0, x_ratio, y_ratio, bias)
    
    def process_frame(self, frame: np.ndarray, apply_warp: bool = True):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像
            apply_warp: 是否应用透视变换
            
        Returns:
            depth_colored: 深度图可视化 (H, W, 3)
            normal_colored: 法向量可视化 (H, W, 3)
            raw_depth: 原始深度数据 (H, W) 或 None
            raw_normals: 原始法向量数据 (H, W, 3) 或 None
            grad_x, grad_y: 梯度数据或 None
            diff_img: 差分图（用于调试）或 None
        """
        if apply_warp:
            warped = self.warp_perspective(frame)
        else:
            warped = frame
        raw_image = self._crop_image(warped)
        h, w = raw_image.shape[:2]
        
        if self.con_flag:
            # 第一帧作为参考
            self.ref_frame = raw_image.copy()
            self.ref_blur = cv2.GaussianBlur(self.ref_frame.astype(np.float32), (13, 13), 0)
            self.con_flag = False
            
            return (
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                None, None, None, None, None
            )
        
        # 当前帧处理
        frame_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        
        # 差分图（用于显示）
        diff = frame_blur - self.ref_blur
        diff_display = np.clip(diff * 2 + 127, 0, 255).astype(np.uint8)
        
        # 计算深度和梯度
        raw_depth, grad_x, grad_y = self.img2depth(self.ref_blur, frame_blur)
        
        # 打印统计信息（可选）
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            grad_x_valid = grad_x[np.abs(grad_x) > 0.01]
            if len(grad_x_valid) > 0:
                print(f"[梯度法] grad_x: [{np.min(grad_x):.4f}, {np.max(grad_x):.4f}], "
                      f"grad_y: [{np.min(grad_y):.4f}, {np.max(grad_y):.4f}]")
        
        # 计算法向量
        denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
        normal_x = -grad_x / denom
        normal_y = -grad_y / denom
        normal_z = 1.0 / denom
        raw_normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        
        # 法向量可视化
        N_disp = 0.5 * (raw_normals + 1.0)
        N_disp = np.clip(N_disp, 0, 1)
        normal_colored = (N_disp * 255).astype(np.uint8)
        normal_colored = cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR)
        
        # 深度可视化
        depth_normalized = raw_depth - np.min(raw_depth)
        if np.max(depth_normalized) > 0:
            depth_normalized = depth_normalized / np.max(depth_normalized) * 255
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        return depth_colored, normal_colored, raw_depth, raw_normals, grad_x, grad_y, diff_display
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.ref_frame = None
        self.ref_blur = None
        self.frame_count = 0


# ============================================================================
# MLPProcessor: 基于神经网络的方法 (与gs_sdk/6_3_test_mlp_v2.py一致)
# ============================================================================

class BGRXYMLPNet:
    """与gs_sdk一致的MLP网络结构 (纯torch模块)"""
    pass  # 在_load_model中动态创建


def _image2bgrxys(image: np.ndarray) -> np.ndarray:
    """将BGR图像转换为BGRXY特征 (与gs_sdk一致)"""
    h, w = image.shape[:2]
    ys = np.linspace(0, 1, h, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, 1, w, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255.0, xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


def _poisson_dct_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    使用DCT的泊松求解器 (与gs_sdk一致)
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
    tt = scipy.fftpack.dct(f, norm="ortho")
    fcos = scipy.fftpack.dct(tt.T, norm="ortho").T

    # 频域求解
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    ).astype(np.float32)
    denom[denom == 0] = 1  # 避免除零

    # 逆DCT变换
    f = -fcos / denom
    tt = scipy.fftpack.idct(f, norm="ortho")
    img_tt = scipy.fftpack.idct(tt.T, norm="ortho").T
    img_tt = img_tt - img_tt.mean()

    return img_tt


class MLPProcessor(BaseProcessor):
    """
    基于MLP神经网络的触觉图像处理器 (与gs_sdk/6_3_test_mlp_v2.py一致)
    
    流程:
    1. 加载背景图，计算背景梯度
    2. 计算当前帧梯度，减去背景梯度得到差分梯度
    3. 使用泊松方程重建深度
    """
    
    def __init__(self, model_path: str = None, pad: int = 20, 
                 calib_file: str = None, device: str = None, ppmm: float = 7.6):
        """
        初始化处理器
        
        Args:
            model_path: MLP模型文件路径 (默认: load/nnmodel_v2.pth)
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            ppmm: 像素每毫米
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        self.ppmm = ppmm
        
        # 延迟导入torch
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            self.torch = torch
            self.nn = nn
            self.F = F
        except ImportError:
            print("[ERROR] 需要安装 PyTorch: pip install torch")
            self.torch = None
            self.model = None
            return
        
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 模型路径 - 默认使用 nnmodel_v2.pth (与6_3_test_mlp_v2.py一致)
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "load", "nnmodel_v2.pth")
        
        self.model_path = model_path
        self.model = None
        self._load_model()
        
        # 背景帧和背景梯度
        self.bg_image = None
        self.bg_G = None
        
        # 多帧平均采集背景
        self.ref_frames_buffer = []
        self.ref_avg_count = 10
    
    def _load_model(self):
        """加载MLP模型 (BGRXYMLPNet架构)"""
        if self.torch is None:
            return
        
        try:
            # 动态创建BGRXYMLPNet
            class BGRXYMLPNet(self.nn.Module):
                def __init__(inner_self):
                    super().__init__()
                    input_size = 5
                    inner_self.fc1 = self.nn.Linear(input_size, 128)
                    inner_self.fc2 = self.nn.Linear(128, 32)
                    inner_self.fc3 = self.nn.Linear(32, 32)
                    inner_self.fc4 = self.nn.Linear(32, 2)

                def forward(inner_self, x):
                    x = self.F.relu(inner_self.fc1(x))
                    x = self.F.relu(inner_self.fc2(x))
                    x = self.F.relu(inner_self.fc3(x))
                    x = inner_self.fc4(x)
                    return x
            
            self.model = BGRXYMLPNet()
            state_dict = self.torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            print(f"[INFO] MLP模型已加载: {self.model_path}")
            print(f"[INFO] 设备: {self.device}")
            
        except FileNotFoundError:
            print(f"[WARNING] 模型文件不存在: {self.model_path}")
            self.model = None
        except Exception as e:
            print(f"[WARNING] 加载MLP模型失败: {e}")
            self.model = None
    
    def _compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """使用MLP计算图像的梯度"""
        if self.model is None or self.torch is None:
            return np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
        
        bgrxys = _image2bgrxys(image).reshape(-1, 5)
        features = self.torch.from_numpy(bgrxys).float().to(self.device)
        
        with self.torch.no_grad():
            gxyangles = self.model(features)
            gxyangles = gxyangles.cpu().numpy()
            # 将梯度角度转换为梯度值
            G = np.tan(gxyangles.reshape(image.shape[0], image.shape[1], 2))
        
        return G
    
    def load_bg(self, bg_image: np.ndarray):
        """加载背景图像并计算背景梯度"""
        self.bg_image = bg_image.copy()
        self.bg_G = self._compute_gradient(bg_image)
        print(f"[INFO] 背景图像已加载，尺寸: {bg_image.shape[:2]}")
    
    def _compute_normals(self, G: np.ndarray) -> np.ndarray:
        """从梯度计算法向量"""
        gx = G[:, :, 0]
        gy = G[:, :, 1]
        gz = np.ones_like(gx)
        
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        magnitude[magnitude == 0] = 1
        
        nx = -gx / magnitude
        ny = -gy / magnitude
        nz = gz / magnitude
        
        return np.stack([nx, ny, nz], axis=-1)
    
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
    
    def _colorize_normals(self, normals: np.ndarray) -> np.ndarray:
        """法向量着色"""
        # 映射到颜色 [-1, 1] -> [0, 255]
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
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
        """
        if apply_warp:
            frame = self.warp_perspective(frame)
        
        h, w = frame.shape[:2]
        
        if self.model is None:
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    np.zeros((h, w, 3), dtype=np.float32))
        
        # 采集背景帧（多帧平均）
        if self.con_flag:
            self.ref_frames_buffer.append(frame.astype(np.float32))
            
            if len(self.ref_frames_buffer) < self.ref_avg_count:
                return (np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w), dtype=np.float32),
                        np.zeros((h, w, 3), dtype=np.float32))
            else:
                # 计算平均背景帧
                bg_image = np.mean(self.ref_frames_buffer, axis=0).astype(np.uint8)
                self.load_bg(bg_image)
                self.con_flag = False
                self.ref_frames_buffer = []
                return (np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w), dtype=np.float32),
                        np.zeros((h, w, 3), dtype=np.float32))
        
        # 计算当前帧梯度
        G = self._compute_gradient(frame)
        
        # 减去背景梯度
        G = G - self.bg_G
        
        # 泊松重建深度
        depth = _poisson_dct_neumann(G[:, :, 0], G[:, :, 1]).astype(np.float32)
        
        # 计算法向量
        normals = self._compute_normals(G)
        
        # 可视化
        depth_colored = self._colorize_depth(depth)
        normal_colored = self._colorize_normals(normals)
        
        return depth_colored, normal_colored, depth, normals
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.bg_image = None
        self.bg_G = None
        self.ref_frames_buffer = []
        print("[INFO] 处理器已重置，将采集多帧平均作为背景")
