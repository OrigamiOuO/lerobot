"""
重新生成MLP训练数据集

读取已采集的样本图像和参考图，
使用颜色差分值重新生成npy数据集

使用方法:
    python 6_1_regenerate_dataset.py
"""

import cv2
import numpy as np
import os
import sys
from scipy import signal

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def compute_gradient_from_sphere(center: tuple, radius: int, img_shape: tuple, 
                                  mask: np.ndarray, ball_radius_mm: float = 4.0,
                                  mm_per_pixel: float = 0.1316) -> tuple:
    """
    根据球面几何计算真实梯度
    """
    h, w = img_shape[:2]
    ball_radius_pixel = ball_radius_mm / mm_per_pixel
    
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x, y)
    
    xv = x - center[0]
    yv = y - center[1]
    
    r_sq = xv**2 + yv**2
    R_sq = ball_radius_pixel**2
    
    valid = (r_sq < R_sq) & (mask > 0)
    
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    
    denom = np.sqrt(R_sq - r_sq[valid])
    gx[valid] = -xv[valid] / denom
    gy[valid] = -yv[valid] / denom
    
    return gx, gy


def main():
    """主函数：重新生成数据集"""
    
    # 数据目录
    data_dir = os.path.join(_current_dir, "data", "mlp_calibration")
    
    # 检查参考图
    ref_path = os.path.join(data_dir, "ref.jpg")
    if not os.path.exists(ref_path):
        print(f"[ERROR] 参考图不存在: {ref_path}")
        return
    
    ref_image = cv2.imread(ref_path)
    print(f"[INFO] 加载参考图: {ref_image.shape}")
    
    # 参数设置
    mm_per_pixel_file = os.path.join(_current_dir, "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_per_pixel_file):
        data = np.load(mm_per_pixel_file)
        MM_PER_PIXEL = float(data['mm_per_pixel'])
        print(f"[INFO] 加载 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    else:
        MM_PER_PIXEL = 0.1316
        print(f"[WARNING] 使用默认 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    
    BALL_RADIUS_MM = 4.0
    
    # 查找所有样本
    sample_idx = 1
    all_data = []
    
    while True:
        sample_img_path = os.path.join(data_dir, f"sample_{sample_idx}.jpg")
        sample_txt_path = os.path.join(data_dir, f"sample_{sample_idx}.txt")
        
        if not os.path.exists(sample_img_path):
            break
        
        if not os.path.exists(sample_txt_path):
            print(f"[WARNING] 样本 {sample_idx} 缺少位置信息，跳过")
            sample_idx += 1
            continue
        
        # 读取图像
        image = cv2.imread(sample_img_path)
        
        # 读取圆心和半径
        with open(sample_txt_path, 'r') as f:
            parts = f.read().strip().split()
            if len(parts) >= 3:
                center = (int(parts[0]), int(parts[1]))
                radius = int(parts[2])
            else:
                print(f"[WARNING] 样本 {sample_idx} 位置信息格式错误，跳过")
                sample_idx += 1
                continue
        
        h, w = image.shape[:2]
        
        # 计算颜色差分图像（当前帧 - 参考帧）
        diff_image = image.astype(np.float32) - ref_image.astype(np.float32)
        
        # 创建接触区域掩膜
        contact_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(contact_mask, center, radius, 255, -1)
        valid_mask = (contact_mask > 0).astype(np.float32)
        
        # 计算真实梯度
        gx, gy = compute_gradient_from_sphere(
            center, radius, image.shape, valid_mask,
            ball_radius_mm=BALL_RADIUS_MM, mm_per_pixel=MM_PER_PIXEL
        )
        
        # 提取有效像素的数据
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            print(f"[WARNING] 样本 {sample_idx} 无有效数据，跳过")
            sample_idx += 1
            continue
        
        # 颜色差分值（归一化到 [-1, 1]）
        diff_values = diff_image[y_coords, x_coords] / 255.0
        
        # 梯度值
        gx_values = gx[y_coords, x_coords]
        gy_values = gy[y_coords, x_coords]
        
        # 过滤掉NaN值
        valid_idx = ~(np.isnan(gx_values) | np.isnan(gy_values))
        
        # 组合数据 [dB, dG, dR, X, Y, gx, gy]
        data = np.column_stack([
            diff_values[valid_idx, 0],  # dB (差分)
            diff_values[valid_idx, 1],  # dG (差分)
            diff_values[valid_idx, 2],  # dR (差分)
            x_coords[valid_idx],        # X
            y_coords[valid_idx],        # Y
            gx_values[valid_idx],       # gx
            gy_values[valid_idx]        # gy
        ])
        
        all_data.extend(data.tolist())
        print(f"[INFO] 样本 {sample_idx}: {len(data)} 个数据点, 圆心={center}, 半径={radius}")
        
        sample_idx += 1
    
    if len(all_data) == 0:
        print("[ERROR] 没有找到任何样本数据")
        return
    
    # 保存新数据集
    dataset = np.array(all_data)
    
    # 备份旧数据集
    old_dataset_path = os.path.join(data_dir, "mlp_dataset.npy")
    if os.path.exists(old_dataset_path):
        backup_path = os.path.join(data_dir, "mlp_dataset_old_rgb.npy")
        os.rename(old_dataset_path, backup_path)
        print(f"[INFO] 旧数据集已备份到: {backup_path}")
    
    # 保存新数据集
    dataset_path = os.path.join(data_dir, "mlp_dataset.npy")
    np.save(dataset_path, dataset)
    
    print(f"\n{'='*60}")
    print(f"[INFO] 新数据集已保存: {dataset_path}")
    print(f"[INFO] 总样本数: {sample_idx - 1}")
    print(f"[INFO] 总数据点数: {len(dataset)}")
    print(f"[INFO] 数据形状: {dataset.shape}")
    print(f"[INFO] 数据格式: [dB, dG, dR, X, Y, gx, gy]")
    print(f"{'='*60}")
    
    # 显示数据统计
    print("\n数据统计:")
    columns = ['dB', 'dG', 'dR', 'X', 'Y', 'gx', 'gy']
    for i, col in enumerate(columns):
        data_col = dataset[:, i]
        print(f"  {col}: [{data_col.min():.4f}, {data_col.max():.4f}], mean={data_col.mean():.4f}")


if __name__ == "__main__":
    main()
