import torch
import torch.nn as nn
import zarr
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ================= 1. 搬运网络架构 (保持与训练脚本完全一致) =================
class TemporalTactileEncoder(nn.Module):
    def __init__(self, in_dim=4, embed_dim=384, seq_len=5): 
        super().__init__()
        self.point_embed = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 44, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, batch_first=True, dim_feedforward=embed_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, sparse_history):
        x = self.point_embed(sparse_history)
        x = x + self.pos_embed + self.time_embed 
        x = x.flatten(1, 2)
        return self.transformer(x)

class PureGeometryDecoder(nn.Module):
    def __init__(self, embed_dim=384): 
        super().__init__()
        self.pts_per_token = 16
        self.num_hand_tokens = 1024 // self.pts_per_token  
        self.num_obj_tokens = 1024 // self.pts_per_token   
        
        self.hand_token = nn.Parameter(torch.zeros(1, self.num_hand_tokens, embed_dim))
        self.obj_token = nn.Parameter(torch.zeros(1, self.num_obj_tokens, embed_dim))
        self.pose_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.qpos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.contact_token = nn.Parameter(torch.zeros(1, 44, embed_dim))
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=6, batch_first=True, dim_feedforward=embed_dim*4), 
            num_layers=6
        )
        
        self.hand_pc_predictor = nn.Linear(embed_dim, self.pts_per_token * 3)
        self.obj_pc_predictor = nn.Linear(embed_dim, self.pts_per_token * 3)
        self.pose_predictor = nn.Linear(embed_dim, 7)
        self.qpos_predictor = nn.Linear(embed_dim, 22)
        self.contact_predictor = nn.Linear(embed_dim, 1)

    def forward(self, z_real):
        B = z_real.shape[0]
        tgt = torch.cat([
            self.hand_token.repeat(B, 1, 1), self.obj_token.repeat(B, 1, 1),
            self.pose_token.repeat(B, 1, 1), self.qpos_token.repeat(B, 1, 1),
            self.contact_token.repeat(B, 1, 1)
        ], dim=1)
        
        decoded = self.decoder(tgt=tgt, memory=z_real)
        
        pred_hand_pc = self.hand_pc_predictor(decoded[:, :self.num_hand_tokens, :]).reshape(B, 1024, 3)
        end_obj = self.num_hand_tokens + self.num_obj_tokens
        pred_obj_pc = self.obj_pc_predictor(decoded[:, self.num_hand_tokens:end_obj, :]).reshape(B, 1024, 3)
        
        pred_pose = self.pose_predictor(decoded[:, end_obj, :])     
        pred_qpos = self.qpos_predictor(decoded[:, end_obj+1, :])     
        pred_contact = torch.sigmoid(self.contact_predictor(decoded[:, end_obj+2:, :]).squeeze(-1)) # 🌟 加上 Sigmoid 变概率
        
        return pred_hand_pc, pred_obj_pc, pred_pose, pred_qpos, pred_contact

# ================= 2. 加载权重与数据 =================
def load_ddp_state_dict(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # 去除 DDP 保存时带有的 'module.' 前缀
    enc_state = OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint['encoder_state_dict'].items()])
    dec_state = OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint['decoder_state_dict'].items()])
    return enc_state, dec_state, checkpoint['epoch']

def local_to_world_pc(pc_local, pose):
    """
    极速向量化 3D 坐标系变换 (支持 GPU 和 CPU)
    pc_local: [N, 3] 或 [B, N, 3] 的局部点云
    pose: [7] 或 [B, 7] 的位姿 (假定 MuJoCo 默认顺序: x, y, z, qw, qx, qy, qz)
    """
    # 统一增加 Batch 维度方便处理
    if pc_local.dim() == 2:
        pc_local = pc_local.unsqueeze(0)
        pose = pose.unsqueeze(0)
        
    B, N, _ = pc_local.shape
    t = pose[:, :3].unsqueeze(1) # 平移向量 [B, 1, 3]
    
    # 提取四元数 (注意：检查你的数据是 qw,qx,qy,qz 还是 qx,qy,qz,qw)
    qw, qx, qy, qz = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    
    # 极速计算旋转矩阵 R [B, 3, 3]
    R = torch.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], dim=1).reshape(-1, 3, 3)
    
    # R @ P + T
    pc_world = torch.bmm(pc_local, R.transpose(1, 2)) + t
    
    # 如果本来没有 Batch 维度，就再挤压回去
    return pc_world.squeeze(0) if pc_local.shape[0] == 1 else pc_world

print("⚙️ 正在加载模型权重...")
encoder = TemporalTactileEncoder()
decoder = PureGeometryDecoder()

ckpt_path = "checkpoints/latest_geometry_ckpt.pth"
enc_state, dec_state, trained_epochs = load_ddp_state_dict(ckpt_path)
encoder.load_state_dict(enc_state)
decoder.load_state_dict(dec_state)

encoder.eval()
decoder.eval()

print("📦 正在从 Zarr 中搜索发生接触的精彩瞬间...")
zarr_root = zarr.open("/home/luoshch/Data/3D-pretrain/multimodal_pretrain.zarr", mode='r')
total_frames = zarr_root['data/sparse_pc'].shape[0]

# 🌟 核心修改：搜索逻辑
test_idx = -1
# 随机采样 500 次，找一个受力显著的帧
for _ in range(500):
    idx = np.random.randint(10, total_frames)
    # 取出该帧 44 个点的受力总和
    force_sum = np.sum(zarr_root['data/sparse_pc'][idx, :, 3])
    if force_sum > 0.2:  # 设定一个明显的受力阈值，确保是真的抓住了
        test_idx = idx
        print(f"🎯 锁定第 {test_idx} 帧，总受力: {force_sum:.4f}")
        break

if test_idx == -1:
    print("⚠️ 没找到受力大的帧，可能数据集里接触帧比例较低。使用默认第 1000 帧。")
    test_idx = 1000

# 提取 5 帧历史并做预处理
sparse_history = torch.tensor(zarr_root['data/sparse_pc'][test_idx-4:test_idx+1], dtype=torch.float32).unsqueeze(0)
sparse_force = sparse_history[..., 3].abs()
sparse_history[..., 3] = torch.log1p(sparse_force)
sparse_history = torch.clamp(torch.nan_to_num(sparse_history, nan=0.0), min=-10.0, max=10.0)

# 提取 Ground Truth
gt_hand = zarr_root['data/dense_pc'][test_idx, :, :3]
# 在 vis_ckpt.py 提取 GT 的地方：
gt_obj = torch.tensor(zarr_root['data/object_pc'][test_idx, :, :3], dtype=torch.float32)
gt_pose = torch.tensor(zarr_root['data/pose'][test_idx], dtype=torch.float32)
# gt_obj = local_to_world_pc(gt_obj_local.unsqueeze(0), gt_pose.unsqueeze(0)).squeeze(0).numpy()
gt_contact = (zarr_root['data/sparse_pc'][test_idx, :, 3] > 0.001).astype(float)

# ================= 3. 执行前向脑补 =================
print("🧠 正在进行时空脑补推理...")
with torch.no_grad():
    z_real = encoder(sparse_history)
    pred_hand, pred_obj, pred_pose, pred_qpos, pred_contact = decoder(z_real)

# 转为 numpy 以便绘图
pred_hand = pred_hand.squeeze(0).numpy()
pred_obj = pred_obj.squeeze(0).numpy()
pred_pose = pred_pose.squeeze(0).numpy()
pred_contact = pred_contact.squeeze(0).numpy()

# ================= 4. 打印特权信息对比 =================
print(f"\n✅ 推理完成！当前模型已训练 Epoch: {trained_epochs}")
print("-" * 50)
print(f"🎯 [特权信息对比]")
print(f"  👉 物体 7D Pose (GT)   : {np.round(gt_pose, 3)}")
print(f"  👉 物体 7D Pose (Pred) : {np.round(pred_pose, 3)}")
print("-" * 50)
print(f"  👉 接触判定 (GT 中接触点数)   : {int(gt_contact.sum())} / 44")
print(f"  👉 接触判定 (Pred 激活点数) : {int((pred_contact > 0.5).sum())} / 44")
print("-" * 50)

# ================= 5. 绘制 3D 几何对比图 =================
fig = plt.figure(figsize=(14, 7))

# 统一坐标轴范围，防止两张图比例失调
all_pts = np.concatenate([gt_hand, gt_obj])
max_range = np.array([all_pts[:,0].ptp(), all_pts[:,1].ptp(), all_pts[:,2].ptp()]).max() / 2.0
mid_x = (all_pts[:,0].max() + all_pts[:,0].min()) * 0.5
mid_y = (all_pts[:,1].max() + all_pts[:,1].min()) * 0.5
mid_z = (all_pts[:,2].max() + all_pts[:,2].min()) * 0.5

# --- 面板 1: Ground Truth ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(gt_hand[:,0], gt_hand[:,1], gt_hand[:,2], c='gray', s=5, alpha=0.5, label='Hand (GT)')
ax1.scatter(gt_obj[:,0], gt_obj[:,1], gt_obj[:,2], c='cyan', s=10, alpha=0.8, label='Object (GT)')
ax1.set_title("Ground Truth PC", fontsize=14)
ax1.legend()

# --- 面板 2: Prediction ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(pred_hand[:,0], pred_hand[:,1], pred_hand[:,2], c='orange', s=5, alpha=0.5, label='Hand (Pred)')
ax2.scatter(pred_obj[:,0], pred_obj[:,1], pred_obj[:,2], c='magenta', s=10, alpha=0.8, label='Object (Pred)')
ax2.set_title(f"Prediction (Epoch {trained_epochs})", fontsize=14)
ax2.legend()

for ax in [ax1, ax2]:
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("visualize_epoch_ckpt.png", dpi=300)
print("🎨 对比大图已保存至 visualize_epoch_ckpt.png！")