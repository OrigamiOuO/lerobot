"""
mm/px 标定工具
使用10mm直径的圆柱按压传感器，检测透视变换后图像中每个像素对应的毫米数
"""

import numpy as np
import cv2
import os
from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class MMPerPixelCalibrator:
    """测量透视变换后图像的 mm/px 比例"""
    
    def __init__(self, camera_config, cylinder_diameter_mm=10.0):
        """
        初始化标定器
        
        Args:
            camera_config: 相机配置
            cylinder_diameter_mm: 圆柱直径（毫米），默认10mm
        """
        self.camera_config = camera_config
        self.cylinder_diameter_mm = cylinder_diameter_mm
        self.camera = None
        self.homography_matrix = None
        self.output_size = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.calib_file = os.path.join(current_dir, "calibration_data", "homography_matrix.npz")
        self.result_file = os.path.join(current_dir, "calibration_data", "mm_per_pixel.npz")
        
        # 检测结果
        self.detected_circles = []
        self.mm_per_pixel = None
        
    def initialize_camera(self):
        """初始化相机"""
        print("[INFO] 初始化相机...")
        self.camera = TactileCamera(self.camera_config)
        self.camera.connect()
        print("[INFO] 相机连接成功")
        
    def load_homography(self):
        """加载透视变换矩阵"""
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵')
            print(f'[INFO] 输出尺寸：{self.output_size}')
            return True
        except Exception as e:
            print(f'[ERROR] 加载透视变换矩阵失败: {e}')
            return False
    
    def apply_perspective(self, image):
        """应用透视变换"""
        if self.homography_matrix is None:
            return image
        return cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size,
            flags=cv2.INTER_NEAREST
        )
    
    def detect_circle_canny(self, image, ref_image=None):
        """
        使用 Canny 边缘检测 + 霍夫圆检测
        
        Args:
            image: 当前帧（透视变换后）
            ref_image: 参考图像（无按压时），用于差分检测
            
        Returns:
            circles: 检测到的圆 [(x, y, radius), ...]
            edges: Canny 边缘图像（用于显示）
        """
        if ref_image is not None:
            # 使用差分图像
            diff = cv2.absdiff(image, ref_image)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        
        # Canny 边缘检测 - 降低阈值使其更敏感
        edges = cv2.Canny(blurred, threshold1=10, threshold2=50)
        
        # 膨胀边缘使其更连续
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 霍夫圆检测 - 降低 param2 使其更敏感
        circles = cv2.HoughCircles(
            edges_dilated,
            cv2.HOUGH_GRADIENT,
            dp=1.2,         # 稍微增加，减少计算量
            minDist=30,     # 减小最小距离
            param1=50,      
            param2=15,      # 降低阈值，更容易检测到圆
            minRadius=5,    # 减小最小半径
            maxRadius=200   # 增大最大半径
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return circles, edges_dilated
        return None, edges_dilated
    
    def detect_circle_contour(self, image, ref_image=None):
        """
        使用轮廓检测方法（备选方案）
        
        Args:
            image: 当前帧
            ref_image: 参考图像
            
        Returns:
            circles: 检测到的圆 [(x, y, radius), ...]
            thresh: 二值化图像
        """
        if ref_image is not None:
            diff = cv2.absdiff(image, ref_image)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        
        # 二值化 - 使用 Otsu 自动阈值
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作：闭运算填充空洞
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 过滤太小的轮廓
                continue
            
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            # 计算圆形度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # 圆形度接近1表示是圆形
                if circularity > 0.5:  # 降低阈值
                    circles.append([int(x), int(y), int(radius)])
        
        if len(circles) > 0:
            return np.array(circles), thresh
        return None, thresh

    def detect_circle(self, image, ref_image=None):
        """
        检测图像中的圆形按压区域（尝试多种方法）
        
        Args:
            image: 当前帧（透视变换后）
            ref_image: 参考图像（无按压时），用于差分检测
            
        Returns:
            circles: 检测到的圆 [(x, y, radius), ...]
        """
        # 首先尝试 Canny + 霍夫圆
        circles, _ = self.detect_circle_canny(image, ref_image)
        if circles is not None:
            return circles
        
        # 如果失败，尝试轮廓检测
        circles, _ = self.detect_circle_contour(image, ref_image)
        return circles
    
    def calculate_mm_per_pixel(self, detected_radius_px):
        """
        计算 mm/px 比例
        
        Args:
            detected_radius_px: 检测到的圆形半径（像素）
            
        Returns:
            mm_per_pixel: 每像素对应的毫米数
        """
        # 圆柱直径 = 2 * 半径
        detected_diameter_px = 2 * detected_radius_px
        mm_per_pixel = self.cylinder_diameter_mm / detected_diameter_px
        return mm_per_pixel
    
    def run_calibration(self):
        """运行标定流程"""
        if self.camera is None:
            raise RuntimeError("相机未初始化")
        
        if not self.load_homography():
            raise RuntimeError("透视变换矩阵加载失败")
        
        print("\n=== mm/px 标定工具 ===")
        print(f"圆柱直径: {self.cylinder_diameter_mm} mm")
        print("\n操作说明:")
        print("  1. 按 'c' 采集当前图像")
        print("  2. 采集后使用 IJKL 移动圆心 (I上 K下 J左 L右)")
        print("  3. 按 'm' 扩大半径, 'n' 缩小半径")
        print("  4. 按 空格 保存标定结果")
        print("  5. 按 'q' 退出")
        print("========================\n")
        
        captured_image = None
        current_result = None
        
        # 手动调整变量
        manual_center = None  # (x, y)
        manual_radius = 50    # 默认半径
        move_step = 1         # 移动步长
        
        cv2.namedWindow('mm/px Calibration')
        
        while True:
            try:
                # 如果没有采集图像，显示实时画面
                if captured_image is None:
                    frame = self.camera.async_read(timeout_ms=200)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    warped = self.apply_perspective(frame_bgr)
                    display = warped.copy()
                    cv2.putText(display, "[LIVE] Press 'c' to capture", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # 显示采集的静态图像
                    display = captured_image.copy()
                    
                    # 绘制圆形
                    if manual_center is not None:
                        x, y = manual_center
                        cv2.circle(display, (x, y), manual_radius, (0, 255, 0), 2)
                        cv2.circle(display, (x, y), 3, (0, 0, 255), -1)
                        
                        # 绘制十字线辅助定位
                        cv2.line(display, (x - 20, y), (x + 20, y), (255, 0, 0), 1)
                        cv2.line(display, (x, y - 20), (x, y + 20), (255, 0, 0), 1)
                        
                        mm_px = self.calculate_mm_per_pixel(manual_radius)
                        current_result = (x, y, manual_radius, mm_px)
                        self.mm_per_pixel = mm_px
                        
                        # 显示信息
                        info_text = f"D={2*manual_radius}px, {self.cylinder_diameter_mm}mm"
                        cv2.putText(display, info_text, (x - 60, y - manual_radius - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        mm_px_text = f"mm/px = {mm_px:.4f}"
                        cv2.putText(display, mm_px_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        px_per_mm_text = f"px/mm = {1/mm_px:.2f}"
                        cv2.putText(display, px_per_mm_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 显示操作提示
                    cv2.putText(display, "[CAPTURED] IJKL:move  M/N:radius  SPACE:save", 
                               (10, display.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    cv2.putText(display, f"Center: {manual_center}  Radius: {manual_radius}px", 
                               (10, display.shape[0] - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                cv2.imshow('mm/px Calibration', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    # 采集当前图像
                    frame = self.camera.async_read(timeout_ms=200)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    captured_image = self.apply_perspective(frame_bgr)
                    # 初始化圆心为图像中心
                    h, w = captured_image.shape[:2]
                    manual_center = (w // 2, h // 2)
                    print(f"[INFO] 图像已采集，圆心初始化为图像中心: {manual_center}")
                    print("[INFO] 使用 WASD 移动圆心，M/N 调整半径")
                
                elif key == ord('i') and captured_image is not None and manual_center is not None:
                    # 向上移动
                    x, y = manual_center
                    manual_center = (x, max(0, y - move_step))
                    
                elif key == ord('k') and captured_image is not None and manual_center is not None:
                    # 向下移动
                    x, y = manual_center
                    h = captured_image.shape[0]
                    manual_center = (x, min(h - 1, y + move_step))
                    
                elif key == ord('j') and captured_image is not None and manual_center is not None:
                    # 向左移动
                    x, y = manual_center
                    manual_center = (max(0, x - move_step), y)
                    
                elif key == ord('l') and captured_image is not None and manual_center is not None:
                    # 向右移动
                    x, y = manual_center
                    w = captured_image.shape[1]
                    manual_center = (min(w - 1, x + move_step), y)
                
                elif key == ord('m') and captured_image is not None:
                    # 扩大半径
                    manual_radius += 1
                    if manual_center is not None:
                        mm_px = self.calculate_mm_per_pixel(manual_radius)
                        print(f"[调整] 半径: {manual_radius}px, mm/px={mm_px:.4f}")
                
                elif key == ord('n') and captured_image is not None:
                    # 缩小半径
                    if manual_radius > 5:
                        manual_radius -= 1
                        if manual_center is not None:
                            mm_px = self.calculate_mm_per_pixel(manual_radius)
                            print(f"[调整] 半径: {manual_radius}px, mm/px={mm_px:.4f}")
                
                elif key == ord(' '):  # 空格保存
                    # 保存结果
                    if self.mm_per_pixel is not None:
                        self.save_calibration()
                    else:
                        print("[WARNING] 请先采集图像并调整圆形")
                        
                elif key == ord('q'):
                    break
                    
            except TimeoutError:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        
        cv2.destroyAllWindows()
        
    def save_calibration(self):
        """保存标定结果"""
        if self.mm_per_pixel is None:
            print("[ERROR] 没有标定结果可保存")
            return
        
        np.savez(
            self.result_file,
            mm_per_pixel=self.mm_per_pixel,
            cylinder_diameter_mm=self.cylinder_diameter_mm,
            output_size=np.array(self.output_size)
        )
        print(f"\n[INFO] 标定结果已保存到: {self.result_file}")
        print(f"  mm/px = {self.mm_per_pixel:.4f}")
        print(f"  px/mm = {1/self.mm_per_pixel:.2f}")
        
    def cleanup(self):
        """清理资源"""
        if self.camera is not None:
            self.camera.disconnect()
            print("[INFO] 相机已断开")


def main():
    # 相机配置（与 1_quick_roi_calibrator.py 一致）
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video0",  # 替换为你的TactileCamera设备路径
        fps=25,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        # 曝光设置 (范围: 1-10000, 较小值=较暗)
        exposure=600,
        auto_exposure=False,
        # 白平衡设置 (色温范围: 2000-8000K)
        wb_temperature=4000,
        auto_wb=False,
        # RGB增益 (范围: 0.0 - 3.0)
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0,
    )
    
    # 创建标定器，使用10mm直径圆柱
    calibrator = MMPerPixelCalibrator(camera_config, cylinder_diameter_mm=10.0)
    
    try:
        calibrator.initialize_camera()
        calibrator.run_calibration()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()
