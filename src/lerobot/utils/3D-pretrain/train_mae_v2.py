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

# ================= 2. 时序 Zarr 数据集 (滑动窗口机制) =================
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
            
        hand_pc = torch.tensor(self.root['data/dense_pc'][idx, :, :3], dtype=torch.float32)
        obj_pc = torch.tensor(self.root['data/object_pc'][idx, :, :3], dtype=torch.float32)
        pose = torch.tensor(self.root['data/pose'][idx], dtype=torch.float32)
        qpos = torch.tensor(self.root['data/qpos'][idx], dtype=torch.float32)
        
        current_force = sparse_history[-1, :, 3] 
        contact_bool = (current_force > 0.001).float()
        
        return sparse_history, hand_pc, obj_pc, pose, qpos, contact_bool

# ================= 3. 时空编码器 =================
class TemporalTactileEncoder(nn.Module):
    def __init__(self, in_dim=4, embed_dim=384, seq_len=SEQ_LEN): 
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

# ================= 4. 多头特权几何解码器 =================
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
        start_obj = self.num_hand_tokens
        end_obj = start_obj + self.num_obj_tokens
        pred_obj_pc = self.obj_pc_predictor(decoded[:, start_obj:end_obj, :]).reshape(B, 1024, 3)
        
        pred_pose = self.pose_predictor(decoded[:, end_obj, :])     
        pred_qpos = self.qpos_predictor(decoded[:, end_obj+1, :])     
        pred_contact = self.contact_predictor(decoded[:, end_obj+2:, :]).squeeze(-1) 
        
        return pred_hand_pc, pred_obj_pc, pred_pose, pred_qpos, pred_contact

# ================= 5. 终极物理防护与掩码 Loss 计算引擎 =================
class UltimatePhysicsLoss(nn.Module):
    def __init__(self, alpha_focal=0.25, gamma_focal=2.0, margin_pen=0.005, radius_rep=0.015):
        super().__init__()
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.margin_pen = margin_pen   
        self.radius_rep = radius_rep   

    def focal_loss(self, pred_logits, gt_bool):
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_bool, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha_focal * (1 - pt)**self.gamma_focal * bce_loss
        return f_loss.mean()

    def decoupled_pose_loss(self, pred_pose, gt_pose):
        loss_trans = F.l1_loss(pred_pose[:, :3], gt_pose[:, :3], reduction='none').mean(dim=1)
        q_pred = F.normalize(pred_pose[:, 3:], p=2, dim=1)
        q_gt = F.normalize(gt_pose[:, 3:], p=2, dim=1)
        dot_product = torch.abs(torch.sum(q_pred * q_gt, dim=1))
        loss_rot = 1.0 - dot_product
        return loss_trans + 0.1 * loss_rot  # [B]

    def repulsion_loss(self, pc):
        B, N, _ = pc.shape
        dists = torch.cdist(pc, pc)
        mask = torch.eye(N, device=pc.device).bool().unsqueeze(0).expand(B, -1, -1)
        dists = dists.masked_fill(mask, float('inf')) # 避开梯度报错
        rep_loss = F.relu(self.radius_rep - dists)
        return rep_loss.mean(dim=(1, 2))  # [B]

    def penetration_loss(self, hand_pc, obj_pc):
        dists = torch.cdist(obj_pc, hand_pc)
        min_dists, _ = torch.min(dists, dim=2)
        pen_loss = F.relu(self.margin_pen - min_dists)
        return pen_loss.mean(dim=1)  # [B]

    def chamfer_distance(self, p1, p2):
        dists = torch.cdist(p1, p2)
        min_p1_to_p2 = torch.min(dists, dim=2)[0].mean(dim=1)
        min_p2_to_p1 = torch.min(dists, dim=1)[0].mean(dim=1)
        return min_p1_to_p2 + min_p2_to_p1  # [B]

    def forward(self, pred_hand, pred_obj, pred_pose, pred_qpos, pred_contact_logits, 
                gt_hand, gt_obj, gt_pose, gt_qpos, gt_contact, has_contact):
        
        # 1. 无论是否接触，手部自身的认知都必须计算
        loss_cd_hand = self.chamfer_distance(pred_hand, gt_hand).mean()
        loss_qpos = F.mse_loss(pred_qpos, gt_qpos)
        loss_contact = self.focal_loss(pred_contact_logits, gt_contact)
        
        # 2. 获取每个 Batch 的几何与物理 Loss (维度皆为 [B])
        batch_cd_obj = self.chamfer_distance(pred_obj, gt_obj)
        batch_pose = self.decoupled_pose_loss(pred_pose, gt_pose)
        batch_rep = self.repulsion_loss(pred_obj)
        batch_pen = self.penetration_loss(pred_hand, pred_obj)

        # 3. 🔪 核心掩码操作：只对发生接触的样本计算外部世界 Loss
        valid_count = has_contact.sum() + 1e-6
        loss_cd_obj = (batch_cd_obj * has_contact).sum() / valid_count
        loss_pose = (batch_pose * has_contact).sum() / valid_count
        loss_repulsion = (batch_rep * has_contact).sum() / valid_count
        loss_penetration = (batch_pen * has_contact).sum() / valid_count

        total_loss = (
            5.0 * loss_cd_hand + 
            5.0 * loss_cd_obj + 
            2.0 * loss_pose + 
            1.0 * loss_qpos + 
            2.0 * loss_contact + 
            0.5 * loss_repulsion + 
            0.5 * loss_penetration
        )
        
        return total_loss, {
            "CD_H": loss_cd_hand.item(), "CD_O": loss_cd_obj.item(),
            "Pose": loss_pose.item(), "Qpos": loss_qpos.item(),
            "Ctc_FL": loss_contact.item(), "Rep": loss_repulsion.item(),
            "Pen": loss_penetration.item()
        }

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
    criterion = UltimatePhysicsLoss().to(rank)
    
    if rank == 0:
        print(f"🚀 终极六神装 DDP 预训练启动！共 {world_size} 张卡。总 Batch Size: {BATCH_SIZE_PER_GPU * world_size}")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        encoder_ddp.train()
        decoder_ddp.train()
        
        metrics = {k: 0.0 for k in ["CD_H", "CD_O", "Pose", "Qpos", "Ctc_FL", "Rep", "Pen"]}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=rank != 0)
        
        for sparse_history, hand_pc, obj_pc, pose, qpos, contact_bool in pbar:
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
            
            # 提取 0/1 接触掩码
            has_contact = (contact_bool.sum(dim=1) > 0).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                z_real = encoder_ddp(sparse_history)
                pred_hand, pred_obj, pred_pose, pred_qpos, pred_contact_logits = decoder_ddp(z_real)
                
                loss, loss_dict = criterion(
                    pred_hand, pred_obj, pred_pose, pred_qpos, pred_contact_logits,
                    hand_pc, obj_pc, pose, qpos, contact_bool, has_contact
                )
            
            scaler.scale(loss).backward()
            
            # 🌟 极度重要的梯度裁剪，防止 EMD 斥力奇异点引发梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder_ddp.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder_ddp.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            for k, v in loss_dict.items():
                if not torch.isnan(torch.tensor(v)):
                    metrics[k] += v
            
            if rank == 0:
                pbar.set_postfix({
                    "Ctc": f"{loss_dict['Ctc_FL']:.3f}", 
                    "Pose": f"{loss_dict['Pose']:.4f}", 
                    "CD_O": f"{loss_dict['CD_O']:.4f}",
                    "Pen": f"{loss_dict['Pen']:.4f}"
                })
                
        if rank == 0:
            n = len(dataloader)
            print(f"✅ Epoch {epoch+1} | "
                  f"Ctc(FL): {metrics['Ctc_FL']/n:.3f} | "
                  f"Pose(L1/Cos): {metrics['Pose']/n:.4f} | "
                  f"Qpos: {metrics['Qpos']/n:.4f} | "
                  f"CD_O: {metrics['CD_O']/n:.4f} | "
                  f"Rep: {metrics['Rep']/n:.4f} | "
                  f"Pen: {metrics['Pen']/n:.4f}")
            
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1, 
                'encoder_state_dict': encoder_ddp.module.state_dict(),
                'decoder_state_dict': decoder_ddp.module.state_dict()
            }, f"{save_dir}/latest_geometry_ckpt.pth")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)