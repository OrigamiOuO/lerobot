import zarr
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 核心路径 =================
ZARR_PATH = "/home/luoshch/Data/3D-pretrain/multimodal_pretrain.zarr"

print(f"📦 正在打开 Zarr 数据集: {ZARR_PATH}")
try:
    root = zarr.open(ZARR_PATH, mode='r')
except Exception as e:
    print(f"🚨 无法打开 Zarr 库，请确认路径是否正确: {e}")
    exit()

# ================= 2. 全局维度与统计健康检查 =================
print("\n📊 [数据集全局结构报告]")
for key in root['data'].keys():
    shape = root[f'data/{key}'].shape
    dtype = root[f'data/{key}'].dtype
    print(f"  - {key:<12}: Shape {shape}, Dtype {dtype}")

total_frames = root['data/sparse_pc'].shape[0]
if total_frames == 0:
    print("\n🚨 致命错误：Zarr 库为空 (0 帧)！请检查打包脚本的过滤条件。")
    exit()

# 💡 针对 Object PC 的专项验尸报告
test_obj = root['data/object_pc'][total_frames // 2]
obj_mean = np.mean(test_obj)
obj_std = np.std(test_obj)

print("\n🔍 [Object PC 专项质检]")
print(f"  - 均值 (Mean) : {obj_mean:.6f}")
print(f"  - 标准差 (Std): {obj_std:.6f}")

if obj_mean == 0.0 and obj_std == 0.0:
    print("  ❌ 惨烈翻车：Object PC 依然全为 0！请务必检查原始 pkl 的 Key 名以及打包脚本！")
else:
    print("  ✅ 完美通关：Object PC 拥有真实的几何分布！")

# ================= 3. 寻找一个“接触发生”的精彩瞬间 =================
print("\n🕵️ 正在茫茫数据海中寻找发生了物理干涉的帧...")
sample_idx = -1

# 随机抽样 100 次，找一个有受力的帧（展示效果最好）
for _ in range(100):
    idx = np.random.randint(0, total_frames)
    sparse_pc = root['data/sparse_pc'][idx]
    total_force = np.sum(np.abs(sparse_pc[:, 3]))
    if total_force > 0.01:
        sample_idx = idx
        break

if sample_idx == -1:
    print("⚠️ 没找到明显的接触帧，将随机展示中间帧。")
    sample_idx = total_frames // 2

print(f"🎬 锁定第 {sample_idx} 帧，准备渲染...")

# ================= 4. 提取数据 =================
# 视觉数据: [3, 128, 128] -> [128, 128, 3]
rgb = np.transpose(root['data/rgb'][sample_idx], (1, 2, 0))
# Mask 数据: [1, 128, 128] -> [128, 128]
mask = root['data/seg_mask'][sample_idx].squeeze()

# 几何数据
sparse_pc = root['data/sparse_pc'][sample_idx]  # [44, 4]
hand_pc = root['data/dense_pc'][sample_idx][:, :3]  # [1024, 7] -> 取前 3 维 xyz
obj_pc = root['data/object_pc'][sample_idx]  # [1024, 3]

# ================= 5. 开始多模态硬核绘图 =================
fig = plt.figure(figsize=(18, 6))

# --- 面板 1: 视觉 RGB ---
ax1 = fig.add_subplot(1, 3, 1)
# 确保 RGB 在 0-1 之间
ax1.imshow(np.clip(rgb, 0, 1))
ax1.set_title(f"1. RGB Image (Frame {sample_idx})", fontsize=14)
ax1.axis("off")

# --- 面板 2: 语义 Mask ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(mask, cmap='gray')
ax2.set_title("2. Segmentation Mask", fontsize=14)
ax2.axis("off")

# --- 面板 3: 3D 世界坐标系点云交汇 ---
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

# 1. 画致密的手部点云 (灰色)
ax3.scatter(hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2], c='gray', s=5, alpha=0.3, label="Dense Hand")

# 2. 画致密的物体点云 (青色)
ax3.scatter(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], c='cyan', s=10, alpha=0.8, label="Dense Object")

# 3. 画稀疏的 44 个传感器点 (用受力大小作为颜色热力图)
sx, sy, sz = sparse_pc[:, 0], sparse_pc[:, 1], sparse_pc[:, 2]
forces = sparse_pc[:, 3]
sc = ax3.scatter(sx, sy, sz, c=forces, cmap='hot', s=40, edgecolors='black', linewidth=0.5, label="Sensors (Hot=Force)")

ax3.set_title("3. World Coordinate Kinematics", fontsize=14)
fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Contact Force Magnitude")
ax3.legend(loc='upper right', fontsize=8)

# 强制坐标轴比例完全一致（极其关键，否则点云会被拉长成面条）
all_pts = np.concatenate([hand_pc, obj_pc, sparse_pc[:, :3]])
max_range = np.array([all_pts[:,0].ptp(), all_pts[:,1].ptp(), all_pts[:,2].ptp()]).max() / 2.0
mid_x = (all_pts[:,0].max() + all_pts[:,0].min()) * 0.5
mid_y = (all_pts[:,1].max() + all_pts[:,1].min()) * 0.5
mid_z = (all_pts[:,2].max() + all_pts[:,2].min()) * 0.5

ax3.set_xlim(mid_x - max_range, mid_x + max_range)
ax3.set_ylim(mid_y - max_range, mid_y + max_range)
ax3.set_zlim(mid_z - max_range, mid_z + max_range)
ax3.view_init(elev=30, azim=45)

plt.tight_layout()
save_path = "check_zarr_result.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"🎉 质检图已生成！请立刻查看: {save_path}")