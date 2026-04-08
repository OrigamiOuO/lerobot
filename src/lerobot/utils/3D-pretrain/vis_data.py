import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ================= 1. 加载文件 =================
file_path = "/home/luoshch/Data/3D-pretrain/Egg/episode_238.pkl"
print(f"📦 正在加载轨迹: {file_path}")

with open(file_path, "rb") as f:
    ep_data = pickle.load(f)

# ================= 2. 挑选极具价值的一帧 =================
# 我们不看第 0 帧（还没碰到），直接抽取中间帧（通常是抓取发生的时候）
frame_idx = len(ep_data) // 2
step = ep_data[frame_idx]

# 提取特征
rgb = step['rgb_img']
mask = step['seg_mask']
hand_pc = step['hand_pc']  # (1355, 7)
obj_pc = step['obj_pc']    # (1024, 3)

# ================= 3. 绘制 2D + 3D 联合视图 =================
print("🎨 正在渲染数据看板...")
fig = plt.figure(figsize=(18, 6))

# --- 面板 1: RGB 视觉 ---
ax1 = fig.add_subplot(1, 3, 1)
# MuJoCo 给的可能是 BGR 格式，为了 Matplotlib 颜色正常，转一下 RGB
# 如果你的图颜色发蓝，把这行取消注释： rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
ax1.imshow(rgb)
ax1.set_title(f"1. Raw RGB Image (Frame {frame_idx})", fontsize=14)
ax1.axis("off")

# --- 面板 2: 语义掩码 ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(mask, cmap='gray')
ax2.set_title("2. Segmentation Mask", fontsize=14)
ax2.axis("off")

# --- 面板 3: 三维几何互动 (核心！) ---
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

# 1. 画机械手的点云 [1355, 7]
# 假设前 3 维是 xyz，第 7 维(索引6)是受力大小 f_magnitude
hx, hy, hz = hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2]
h_forces = hand_pc[:, 6] if hand_pc.shape[1] == 7 else hz 
sc = ax3.scatter(hx, hy, hz, c=h_forces, cmap='hot', s=5, alpha=0.8, label="Hand PC (Hot)")

# 2. 画被抓物体的点云 [1024, 3]
ox, oy, oz = obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2]
ax3.scatter(ox, oy, oz, c='cyan', s=10, alpha=0.5, label="Object PC (Cyan)")

ax3.set_title("3. 3D Kinematics & Blind Grasping", fontsize=14)
fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Contact Force Magnitude")
ax3.legend()

# 强制坐标轴比例一致，防止点云被拉伸变形
max_range = np.array([hx.max()-hx.min(), hy.max()-hy.min(), hz.max()-hz.min()]).max() / 2.0
mid_x = (hx.max()+hx.min()) * 0.5
mid_y = (hy.max()+hy.min()) * 0.5
mid_z = (hz.max()+hz.min()) * 0.5
ax3.set_xlim(mid_x - max_range, mid_x + max_range)
ax3.set_ylim(mid_y - max_range, mid_y + max_range)
ax3.set_zlim(mid_z - max_range, mid_z + max_range)
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
save_path = "visualize_raw_data.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"🎉 渲染大功告成！图片已保存至: {save_path}")