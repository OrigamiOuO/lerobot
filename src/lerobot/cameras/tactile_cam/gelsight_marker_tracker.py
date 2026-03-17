# gelsight_marker_tracker.py
import numpy as np
import cv2

class GelSightMarkerTracker:
    """GelSight 标记点追踪器"""
    
    def __init__(self, node=None):
        """
        初始化标记点追踪器
        
        Args:
            node: ROS2 节点对象（在非ROS2环境中设为None）
        """
        self.node = node  
        self.bridge = None
        self.contactmap = np.zeros([480, 640])

        # 标记点相关参数
        self.MarkerAvailable = True  # 标记点是否可用
        self.calMarker_on = False
        self.IsDisplay = True  # 是否显示标记点运动图像
        self.showScale = 8

        # 状态变量
        self.img = None  # 当前图像
        self.f0 = None   # 参考背景图像
        self.flowcenter = None  # 标记点中心坐标 (x, y, area)
        self.marker_last = None  # 上一帧标记点位置
        self.MarkerCount = 0  # 标记点数量
        self.markerU = None  # X方向运动 (位移)
        self.markerV = None  # Y方向运动 (位移)
        self.MarkerMask = None  # 标记点掩膜
        self.current_markers = None  # 当前帧检测到的标记点
        
        # 接触检测参数
        self.touchthresh = None
        self.touchMarkerMovThresh = 1
        self.touchMarkerNumThresh = 20
        
        # 匹配距离上限（像素），超过此距离认为 marker 丢失而非移动
        self.match_max_dist = 25.0
        
        # EMA 时间滤波系数 (0~1, 越小越平滑, 1=不滤波)
        self.ema_alpha = 0.4
        self._smoothU = None
        self._smoothV = None
        
        # 窗口名称
        self.window_name = 'Marker Motion'

    def reinit(self, frame, frame0=None):
        """
        重新初始化追踪器
        
        Args:
            frame: 当前帧
            frame0: 参考背景帧（如果为None，则使用高斯模糊当前帧）
        """
        self.img = frame  # 当前帧
        
        if frame0 is not None:
            self.f0 = frame0  # 使用提供的参考帧
        else:
            # 使用高斯模糊作为参考背景
            self.f0 = np.int16(cv2.GaussianBlur(self.img, (101, 101), 50))
        
        # 初始化接触检测参数
        self._ini_contactDetect()
        
        # 查找标记点
        self.flowcenter = self.find_markers()  # 所有标记点的中心坐标
        self.marker_last = self.flowcenter.copy()
        self.MarkerCount = len(self.flowcenter)
        self.markerU = np.zeros(self.MarkerCount)  # X方向运动
        self.markerV = np.zeros(self.MarkerCount)  # Y方向运动
        self._smoothU = np.zeros(self.MarkerCount)
        self._smoothV = np.zeros(self.MarkerCount)
        
        # 创建显示窗口
        if self.IsDisplay:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    # def _loc_markerArea(self):
    #     """定位标记点区域（适用于Bnz GelSight）"""
    #     if self.img is None or self.f0 is None:
    #         return
            
    #     I = self.img.astype(np.double) - self.f0
    #     self.MarkerMask = np.amax(I, 2) < -2


    def _loc_markerArea(self, thresh=30, mode="dark"):
        if self.img is None or self.f0 is None:
            self.MarkerMask = None
            return

        I = self.img.astype(np.float32) - self.f0.astype(np.float32)
        diff_max = np.max(I, axis=2)

        if mode == "dark":      # marker 比背景暗
            signal = -diff_max  # 翻转使 marker 区域为正
        else:                   # marker 比背景亮
            signal = diff_max

        # 自适应阈值：用 Otsu 在信号图上自动选阈值，比固定 thresh 更鲁棒
        signal_clipped = np.clip(signal, 0, 255).astype(np.uint8)
        otsu_val, mask_otsu = cv2.threshold(signal_clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 如果 Otsu 阈值太低（噪声主导），回退到固定阈值
        if otsu_val < thresh:
            if mode == "dark":
                mask = (diff_max < -thresh).astype(np.uint8)
            else:
                mask = (diff_max > thresh).astype(np.uint8)
        else:
            mask = (mask_otsu // 255).astype(np.uint8)

        # === 形态学开闭去噪 ===
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.MarkerMask = mask.astype(bool)


    def displayIm(self):
        """显示标记点运动图像"""
        if self.img is None:
            return
            
        disIm = self.img.copy()
        
        # 确保有标记点数据
        if self.flowcenter is not None and len(self.flowcenter) > 0:
            markerCenter = np.around(self.flowcenter[:, 0:2]).astype(np.int16)
            
            for i in range(min(self.MarkerCount, len(markerCenter))):
                if i < len(self.markerU) and self.markerU[i] != 0:
                    cv2.line(disIm, 
                            (markerCenter[i, 0], markerCenter[i, 1]),
                            (int(self.flowcenter[i, 0] + self.markerU[i] * self.showScale),
                             int(self.flowcenter[i, 1] + self.markerV[i] * self.showScale)),
                            (0, 255, 255), 2)
        
        cv2.imshow(self.window_name, disIm)
        cv2.waitKey(1)

    def find_markers(self):
        """查找标记点（鲁棒版本：不因数量少而整体放弃）"""
        if self.img is None:
            return np.empty([0, 3])
            
        self._loc_markerArea()
        if self.MarkerMask is None:
            return np.empty([0, 3])
        
        areaThresh1 = 20   # 最小面积（放宽以适应不同曝光）
        areaThresh2 = 600   # 最大面积（放宽一点）
        MarkerCenter = np.empty([0, 3])

        contours = cv2.findContours(self.MarkerMask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
        
        # 适配OpenCV不同版本的contours返回值
        if len(contours) == 3:
            contours = contours[1]
        else:
            contours = contours[0]
        
        # 不再因 contour 数量少而直接放弃；
        # 找到多少就用多少，匹配阶段会处理缺失
        for contour in contours:
            AreaCount = cv2.contourArea(contour)
            if areaThresh1 < AreaCount < areaThresh2:
                t = cv2.moments(contour)
                if t['m00'] != 0:  # 避免除以零
                    MarkerCenter = np.append(MarkerCenter,
                                           [[t['m10'] / t['m00'], 
                                             t['m01'] / t['m00'], 
                                             AreaCount]], axis=0)
        
        if len(MarkerCenter) == 0:
            self.MarkerAvailable = False
        else:
            self.MarkerAvailable = True
        
        self.current_markers = MarkerCenter  # 存储当前帧检测到的标记点
        return MarkerCenter

    def _cal_marker_center_motion(self, MarkerCenter):
        """计算标记点中心运动（带距离上限，防止错误匹配导致位移飙升）"""
        Nt = len(MarkerCenter)
        if Nt == 0 or self.MarkerCount == 0:
            return np.zeros([self.MarkerCount, 3])
            
        no_seq2 = np.zeros(Nt, dtype=int)
        min_dist2 = np.full(Nt, 1e9)  # 记录每个当前marker到最近上一帧marker的距离
        center_now = np.zeros([self.MarkerCount, 3])
        
        # 第一轮匹配：从当前帧到上一帧
        for i in range(Nt):
            if len(self.marker_last) > 0:
                dif = np.abs(MarkerCenter[i, 0] - self.marker_last[:, 0]) + \
                      np.abs(MarkerCenter[i, 1] - self.marker_last[:, 1])
                if len(dif) > 0:
                    cost = dif * (100 + np.abs(MarkerCenter[i, 2] - self.flowcenter[:, 2]))
                    no_seq2[i] = np.argmin(cost)
                    min_dist2[i] = np.min(dif)

        # 第二轮匹配：从上一帧到当前帧
        for i in range(self.MarkerCount):
            if len(MarkerCenter) > 0:
                dif = np.abs(MarkerCenter[:, 0] - self.marker_last[i, 0]) + \
                      np.abs(MarkerCenter[:, 1] - self.marker_last[i, 1])
                if len(dif) > 0:
                    t = dif * (100 + np.abs(MarkerCenter[:, 2] - self.flowcenter[i, 2]))
                    a = np.amin(t) / 100
                    b = np.argmin(t)
                    nearest_dist = np.min(dif)
                    
                    # 距离超过上限 → 认为该 marker 丢失，位移归零
                    if nearest_dist > self.match_max_dist:
                        self.markerU[i] = 0
                        self.markerV[i] = 0
                        center_now[i] = self.flowcenter[i]
                    elif self.flowcenter[i, 2] < a:  # 小区域忽略
                        self.markerU[i] = 0
                        self.markerV[i] = 0
                        center_now[i] = self.flowcenter[i]
                    elif i == no_seq2[b]:
                        # 计算位移：当前帧位置 - 参考位置
                        self.markerU[i] = MarkerCenter[b, 0] - self.flowcenter[i, 0]
                        self.markerV[i] = MarkerCenter[b, 1] - self.flowcenter[i, 1]
                        center_now[i] = MarkerCenter[b]
                    else:
                        self.markerU[i] = 0
                        self.markerV[i] = 0
                        center_now[i] = self.flowcenter[i]
        
        return center_now

    def update_markerMotion(self, img=None):
        """更新标记点运动（含 EMA 时间滤波）"""
        if img is not None:
            self.img = img
            
        MarkerCenter = self.find_markers()
        if len(MarkerCenter) > 0:
            self.marker_last = self._cal_marker_center_motion(MarkerCenter)
        
        # 死区过滤：小于 0.5px 的位移视为噪声
        disp = np.sqrt(self.markerU**2 + self.markerV**2)
        small_motion_mask = disp < 0.5
        self.markerU[small_motion_mask] = 0
        self.markerV[small_motion_mask] = 0
        
        # EMA 时间滤波：平滑帧间跳动
        if self._smoothU is not None and len(self._smoothU) == self.MarkerCount:
            alpha = self.ema_alpha
            self._smoothU = alpha * self.markerU + (1 - alpha) * self._smoothU
            self._smoothV = alpha * self.markerV + (1 - alpha) * self._smoothV
            self.markerU = self._smoothU.copy()
            self.markerV = self._smoothV.copy()
        else:
            self._smoothU = self.markerU.copy()
            self._smoothV = self.markerV.copy()
        
        if self.IsDisplay:
            self.displayIm()

    def get_marker_displacements(self):
        """
        获取所有标记点的位移 (x, y)
        
        Returns:
            displacements: (n, 2) 的数组，n为标记点数量
                          displacements[i, 0] = x方向位移
                          displacements[i, 1] = y方向位移
        """
        if self.markerU is None or self.markerV is None or self.MarkerCount == 0:
            return None
        
        displacements = np.zeros((self.MarkerCount, 2))
        for i in range(min(self.MarkerCount, len(self.markerU), len(self.markerV))):
            displacements[i, 0] = self.markerU[i]  # x位移
            displacements[i, 1] = self.markerV[i]  # y位移
        
        return displacements

    def iniMarkerPos(self):
        """将当前标记点位置设置为初始位置"""
        self.flowcenter = self.marker_last.copy()

    def start_display_markerIm(self):
        """开始显示标记点图像"""
        self.IsDisplay = True

    def stop_display_markerIm(self):
        """停止显示标记点图像"""
        self.IsDisplay = False

    def detect_contact(self, img=None, ColorThresh=1):
        """
        检测接触
        
        Args:
            img: 输入图像
            ColorThresh: 颜色阈值系数
            
        Returns:
            isContact: 是否接触
            countnum: 接触像素数量
        """
        if not self.calMarker_on:
            self.update_markerMotion(img)

        isContact = False

        # 基于颜色的接触检测
        diffim = np.int16(self.img) - self.f0
        self.contactmap = diffim.max(axis=2) - diffim.min(axis=2)
        countnum = np.logical_and(self.contactmap > 10, diffim.max(axis=2) > 0).sum()

        if countnum > self.touchthresh * ColorThresh:  # 有接触
            isContact = True

        # 基于标记点运动的接触检测
        motion = np.abs(self.markerU) + np.abs(self.markerV)
        MotionNum = (motion > self.touchMarkerMovThresh * np.sqrt(ColorThresh)).sum()
        if MotionNum > self.touchMarkerNumThresh:
            isContact = True

        return isContact, countnum

    def _ini_contactDetect(self):
        """初始化接触检测参数"""
        if self.img is None or self.f0 is None:
            return
            
        diffim = np.int16(self.img) - self.f0
        maxim = diffim.max(axis=2)
        contactmap = maxim - diffim.min(axis=2)
        countnum = np.logical_and(contactmap > 10, maxim > 0).sum()

        contactmap[contactmap < 10] = 0
        contactmap[maxim <= 0] = 0
        
        # 保存初始接触图
        # cv2.imwrite('iniContact.png', contactmap)

        self.touchthresh = round((countnum + 1500) * 1.0)
        self.touchMarkerMovThresh = 1
        self.touchMarkerNumThresh = 20

    def cleanup(self):
        """清理资源"""
        cv2.destroyWindow(self.window_name)