import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import zarr
import numpy as np
from tqdm import tqdm
import numcodecs
numcodecs.blosc.use_threads = False

# ================= 1. 核心路径与超参 =================
ZARR_PATH = "/home/luoshch/Data/3D-pretrain/multimodal_pretrain.zarr"
BATCH_SIZE_PER_GPU = 412
NUM_EPOCHS = 200
LR = 3e-4
SEQ_LEN = 5  # 🌟 历史帧长度

# ================= 2. 几何倒角距离 (Chamfer Distance) =================
def chamfer_distance_loss(pc1, pc2):
    """ pc1: [B, N, 3], pc2: [B, M, 3] """
    dist_matrix = torch.cdist(pc1, pc2, p=2.0) ** 2
    min_dist_1_to_2, _ = torch.min(dist_matrix, dim=2) # [B, N]
    min_dist_2_to_1, _ = torch.min(dist_matrix, dim=1) # [B, M]
    # 🌟 核心：返回 [B] 维度的向量，而不是一个标量
    return min_dist_1_to_2.mean(dim=1) + min_dist_2_to_1.mean(dim=1)

# ================= 3. 时序 Zarr 数据集 (滑动窗口机制) =================
class ZarrTemporalDataset(Dataset):
    def __init__(self, zarr_path, seq_len=SEQ_LEN):
        self.zarr_path = zarr_path
        self.seq_len = seq_len
        self.root = None
        
        temp_root = zarr.open(zarr_path, mode='r')
        self.length = temp_root['data/sparse_pc'].shape[0]
        
        self.episode_ends = temp_root['meta/episode_ends'][:]
        self.ep_starts = np.insert(self.episode_ends[:-1], 0, 0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.root is None:
            self.root = zarr.open(self.zarr_path, mode='r')
            
        ep_idx = np.searchsorted(self.episode_ends, idx, side='right')
        ep_start = self.ep_starts[ep_idx]
        
        start_idx = max(ep_start, idx - self.seq_len + 1)
        valid_len = idx - start_idx + 1
        
        sparse_history = torch.tensor(self.root['data/sparse_pc'][start_idx:idx+1], dtype=torch.float32)
        
        if valid_len < self.seq_len:
            pad_len = self.seq_len - valid_len
            pad_tensor = sparse_history[0:1].repeat(pad_len, 1, 1)
            sparse_history = torch.cat([pad_tensor, sparse_history], dim=0)
            
        # 提取 Target
        hand_pc = torch.tensor(self.root['data/dense_pc'][idx, :, :3], dtype=torch.float32)
        obj_pc = torch.tensor(self.root['data/object_pc'][idx, :, :3], dtype=torch.float32)
        pose = torch.tensor(self.root['data/pose'][idx], dtype=torch.float32)
        qpos = torch.tensor(self.root['data/qpos'][idx], dtype=torch.float32)
        
        # 实时计算 Contact Boolean (基于当前帧的力)
        current_force = sparse_history[-1, :, 3] 
        contact_bool = (current_force > 0.001).float() # [44,]
        
        return sparse_history, hand_pc, obj_pc, pose, qpos, contact_bool

# ================= 4. 时空编码器 (Spatiotemporal Encoder) =================
class TemporalTactileEncoder(nn.Module):
    def __init__(self, in_dim=4, embed_dim=384, seq_len=SEQ_LEN): 
        super().__init__()
        self.point_embed = nn.Linear(in_dim, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 44, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, batch_first=True, dim_feedforward=embed_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, sparse_history):
        x = self.point_embed(sparse_history) # [B, 5, 44, C]
        x = x + self.pos_embed + self.time_embed 
        x = x.flatten(1, 2) # [B, 220, C]
        z_real = self.transformer(x)
        return z_real

# ================= 5. 多头特权几何解码器 (Multi-head Geometry Decoder) =================
# ================= 5. 多头特权几何解码器 (Patch 降维版) =================
class PureGeometryDecoder(nn.Module):
    def __init__(self, embed_dim=384): 
        super().__init__()
        
        # 🌟 核心修改：让 1 个 Token 预测 16 个点，Token 数量从 1024 缩减到 64！
        self.pts_per_token = 16
        self.num_hand_tokens = 1024 // self.pts_per_token  # 64
        self.num_obj_tokens = 1024 // self.pts_per_token   # 64
        
        self.hand_token = nn.Parameter(torch.zeros(1, self.num_hand_tokens, embed_dim))
        self.obj_token = nn.Parameter(torch.zeros(1, self.num_obj_tokens, embed_dim))
        self.pose_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.qpos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.contact_token = nn.Parameter(torch.zeros(1, 44, embed_dim))
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=6, batch_first=True, dim_feedforward=embed_dim*4), 
            num_layers=6
        )
        
        # 🌟 输出头修改：每个 Token 输出 16 * 3 = 48 维的数据
        self.hand_pc_predictor = nn.Linear(embed_dim, self.pts_per_token * 3)
        self.obj_pc_predictor = nn.Linear(embed_dim, self.pts_per_token * 3)
        
        self.pose_predictor = nn.Linear(embed_dim, 7)
        self.qpos_predictor = nn.Linear(embed_dim, 22)
        self.contact_predictor = nn.Linear(embed_dim, 1)

    def forward(self, z_real):
        B = z_real.shape[0]
        # 现在的总 Query 数量只有：64 + 64 + 1 + 1 + 44 = 174 个！(显存消耗暴跌 99%)
        tgt = torch.cat([
            self.hand_token.repeat(B, 1, 1),
            self.obj_token.repeat(B, 1, 1),
            self.pose_token.repeat(B, 1, 1),
            self.qpos_token.repeat(B, 1, 1),
            self.contact_token.repeat(B, 1, 1)
        ], dim=1)
        
        decoded = self.decoder(tgt=tgt, memory=z_real)
        
        # --- 还原手部点云 ---
        pred_hand_pc = self.hand_pc_predictor(decoded[:, :self.num_hand_tokens, :]) # [B, 64, 48]
        pred_hand_pc = pred_hand_pc.reshape(B, 1024, 3) # 🌟 展平回 [B, 1024, 3] 供 Loss 计算
        
        # --- 还原物体点云 ---
        start_obj = self.num_hand_tokens
        end_obj = start_obj + self.num_obj_tokens
        pred_obj_pc = self.obj_pc_predictor(decoded[:, start_obj:end_obj, :]) # [B, 64, 48]
        pred_obj_pc = pred_obj_pc.reshape(B, 1024, 3) # 🌟 展平回 [B, 1024, 3]
        
        # --- 提取特权信息 ---
        pred_pose = self.pose_predictor(decoded[:, end_obj, :])     
        pred_qpos = self.qpos_predictor(decoded[:, end_obj+1, :])     
        pred_contact = self.contact_predictor(decoded[:, end_obj+2:, :]).squeeze(-1) 
        
        return pred_hand_pc, pred_obj_pc, pred_pose, pred_qpos, pred_contact

# ================= 6. DDP 训练循环 =================
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.root = zarr.open(worker_info.dataset.zarr_path, mode='r')

def train_worker(rank, world_size):
    ddp_setup(rank, world_size)
    
    dataset = ZarrTemporalDataset(ZARR_PATH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=sampler, 
        num_workers=8, prefetch_factor=1, persistent_workers=True, 
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    encoder_ddp = DDP(TemporalTactileEncoder().to(rank), device_ids=[rank])
    decoder_ddp = DDP(PureGeometryDecoder().to(rank), device_ids=[rank])
    
    optimizer = torch.optim.AdamW(list(encoder_ddp.parameters()) + list(decoder_ddp.parameters()), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    
    if rank == 0:
        print(f"🚀 5头特权多模态 DDP 预训练启动！共 {world_size} 张卡并行。总 Batch Size: {BATCH_SIZE_PER_GPU * world_size}")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        encoder_ddp.train()
        decoder_ddp.train()
        
        # 统计指标
        tot_loss_hand, tot_loss_obj = 0, 0
        tot_loss_pose, tot_loss_qpos, tot_loss_contact = 0, 0, 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=rank != 0)
        
        for sparse_history, hand_pc, obj_pc, pose, qpos, contact_bool in pbar:
            # 🌟 1. 搬运数据到 GPU
            sparse_history = sparse_history.to(rank, non_blocking=True)
            hand_pc = hand_pc.to(rank, non_blocking=True)
            obj_pc = obj_pc.to(rank, non_blocking=True)
            pose = pose.to(rank, non_blocking=True)
            qpos = qpos.to(rank, non_blocking=True)
            contact_bool = contact_bool.to(rank, non_blocking=True)

            sparse_history = torch.nan_to_num(sparse_history, nan=0.0)
            
            # 🛡️ 物理防爆甲：力维度的 Log 压缩
            sparse_force = sparse_history[..., 3].abs()
            sparse_history[..., 3] = torch.log1p(sparse_force)
            sparse_history = torch.clamp(sparse_history, min=-10.0, max=10.0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                z_real = encoder_ddp(sparse_history)
                pred_hand, pred_obj, pred_pose, pred_qpos, pred_contact = decoder_ddp(z_real)
                
                # contact_bool 是 [B, 44]，只要有任意一个点力 > 0，就算碰到物体了
                # has_contact 变成一个 [B] 维度的 0/1 掩码
                has_contact = (contact_bool.sum(dim=1) > 0).float()
                
                # ================= 无条件计算的 Loss (自我认知) =================
                loss_contact = F.binary_cross_entropy_with_logits(pred_contact, contact_bool)
                # qpos 用的是 mean，所以默认是全 Batch 计算
                loss_qpos = F.mse_loss(pred_qpos, qpos)
                loss_hand = chamfer_distance_loss(pred_hand, hand_pc).mean()
                
                # ================= 🌟 受掩码保护的 Loss (世界认知) =================
                # 计算出每个样本的 Loss，但不急着求平均
                loss_pose_batch = F.mse_loss(pred_pose, pose, reduction='none').mean(dim=1) # [B]
                loss_obj_batch = chamfer_distance_loss(pred_obj, obj_pc) # [B]
                
                # 🔪 核心掩码操作：只把发生了接触的样本 Loss 加起来求平均！
                valid_contact_count = has_contact.sum() + 1e-6 # 防止除以 0
                loss_pose = (loss_pose_batch * has_contact).sum() / valid_contact_count
                loss_obj = (loss_obj_batch * has_contact).sum() / valid_contact_count
                
                # 加权聚合
                loss = (loss_contact * 5.0) + \
                       (loss_pose * 2.0) + \
                       (loss_qpos * 1.0) + \
                       (loss_obj * 2.0) + \
                       (loss_hand * 1.0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 指标累计
            if not torch.isnan(loss_contact): tot_loss_contact += loss_contact.item()
            if not torch.isnan(loss_pose): tot_loss_pose += loss_pose.item()
            if not torch.isnan(loss_qpos): tot_loss_qpos += loss_qpos.item()
            if not torch.isnan(loss_hand): tot_loss_hand += loss_hand.item()
            if not torch.isnan(loss_obj): tot_loss_obj += loss_obj.item()
            
            if rank == 0:
                pbar.set_postfix({
                    "Ctc": f"{loss_contact.item():.3f}", 
                    "Pose": f"{loss_pose.item():.4f}", 
                    "CD_O": f"{loss_obj.item():.4f}"
                })
                
        if rank == 0:
            n = len(dataloader)
            print(f"✅ Epoch {epoch+1} 总结 | "
                  f"Ctc(BCE): {tot_loss_contact/n:.3f} | "
                  f"Pose(MSE): {tot_loss_pose/n:.4f} | "
                  f"Qpos(MSE): {tot_loss_qpos/n:.4f} | "
                  f"Hand(CD): {tot_loss_hand/n:.4f} | "
                  f"Obj(CD): {tot_loss_obj/n:.4f}")
            
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1, 'encoder_state_dict': encoder_ddp.module.state_dict(),
                'decoder_state_dict': decoder_ddp.module.state_dict()
            }, f"{save_dir}/latest_geometry_ckpt.pth")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)