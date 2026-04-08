import torch
import torch.nn as nn
import torch.nn.functional as F  # 🌟 补上这行！
import zarr
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score

# ================= 1. 搬运你的 Encoder 和 Decoder 代码 =================
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


# ================= 2. 加载权重 =================
print("⚙️ 正在加载验证器...")
encoder, decoder = TemporalTactileEncoder().cuda(), PureGeometryDecoder().cuda()
checkpoint = torch.load("checkpoints/latest_geometry_ckpt.pth", map_location='cuda')
encoder.load_state_dict(OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint['encoder_state_dict'].items()]))
decoder.load_state_dict(OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint['decoder_state_dict'].items()]))
encoder.eval(), decoder.eval()

# ================= 3. 大规模验证循环 =================
zarr_root = zarr.open("/home/luoshch/Data/3D-pretrain/multimodal_pretrain.zarr", mode='r')
total_frames = zarr_root['data/sparse_pc'].shape[0]

NUM_TESTS = 10000
metrics = {"trans_err": [], "qpos_err": [], "contact_gt_ones": [], "contact_probs": []}
all_gt_contact, all_pred_contact = [], []

print(f"🚀 开始在 {NUM_TESTS} 个高受力帧上进行量化评估...")
with torch.no_grad():
    for _ in tqdm(range(NUM_TESTS)):
        # 找一个发生接触的帧
        idx = np.random.randint(10, total_frames - 1)
        while np.sum(zarr_root['data/sparse_pc'][idx, :, 3]) < 0.2:
            idx = np.random.randint(10, total_frames - 1)
            
        # 准备数据
        sparse_history = torch.tensor(zarr_root['data/sparse_pc'][idx-4:idx+1], dtype=torch.float32).unsqueeze(0).cuda()
        sparse_force = sparse_history[..., 3].abs()
        sparse_history[..., 3] = torch.log1p(sparse_force)
        sparse_history = torch.clamp(torch.nan_to_num(sparse_history, nan=0.0), min=-10.0, max=10.0)
        
        gt_pose = torch.tensor(zarr_root['data/pose'][idx], dtype=torch.float32).cuda()
        gt_qpos = torch.tensor(zarr_root['data/qpos'][idx], dtype=torch.float32).cuda()
        gt_contact = (zarr_root['data/sparse_pc'][idx, :, 3] > 0.001).astype(float)
        
        # 前向推理
        z_real = encoder(sparse_history)
        _, _, pred_pose, pred_qpos, pred_contact = decoder(z_real)
        
        # 计算位姿和平移误差
        trans_err = torch.norm(pred_pose[0, :3] - gt_pose[:3]).item()
        qpos_err = F.mse_loss(pred_qpos[0], gt_qpos).item()
        
        metrics["trans_err"].append(trans_err)
        metrics["qpos_err"].append(qpos_err)
        metrics["contact_probs"].append(pred_contact[0].cpu().numpy().max()) # 看看它最大能输出多少概率
        
        all_gt_contact.extend(gt_contact)
        # 🌟 动态阈值：把要求降低到 0.2 看看
        all_pred_contact.extend((pred_contact[0].cpu().numpy() > 0.2).astype(float))

# ================= 4. 输出震撼的量化报告 =================
avg_trans_err = np.mean(metrics["trans_err"]) * 100 # 转为厘米
precision = precision_score(all_gt_contact, all_pred_contact, zero_division=0)
recall = recall_score(all_gt_contact, all_pred_contact, zero_division=0)
f1 = f1_score(all_gt_contact, all_pred_contact, zero_division=0)
avg_max_prob = np.mean(metrics["contact_probs"])

print("\n" + "="*50)
print("🏆 [预训练模型终极性能测评报告]")
print("="*50)
print(f"🎯 空间定位 (Pose):")
print(f"   - 平均绝对平移误差 : {avg_trans_err:.2f} 厘米  <-- (如果小于 2cm, 绝对的神级表现！)")
print(f"🦾 关节自我感知 (Qpos):")
print(f"   - 关节角 MSE 误差  : {np.mean(metrics['qpos_err']):.4f}")
print(f"🤏 触觉激活 (Contact):")
print(f"   - 网络输出的最大概率均值 : {avg_max_prob:.3f}  <-- (你看，它根本不敢输出 0.9！)")
print(f"   - Precision (查准率) : {precision:.1%}")
print(f"   - Recall    (查全率) : {recall:.1%}")
print(f"   - F1-Score  (综合分) : {f1:.1%}")
print("="*50)