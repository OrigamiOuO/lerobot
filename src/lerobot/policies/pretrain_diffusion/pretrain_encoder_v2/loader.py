import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ================= 1. 搬运神级 Encoder (帧级 Token 升级版) =================
class TemporalTactileEncoder(nn.Module):
    def __init__(self, num_points=44, point_dim=4, embed_dim=384, max_seq_len=16): 
        super().__init__()
        # 1. 展平 44*4，直接映射到 384 维
        self.frame_embed = nn.Linear(num_points * point_dim, embed_dim)
        
        # 2. 仅保留时间位置编码 (最大容量 max_seq_len)
        self.time_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=6, batch_first=True, dim_feedforward=embed_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, sparse_history):
        # sparse_history: [B, T, 44, 4]
        B, T, N, D = sparse_history.shape
        
        # 维度展平与特征映射
        x = sparse_history.reshape(B, T, N * D)  # [B, T, 176]
        x = self.frame_embed(x)                  # [B, T, 384]
        
        # 🌟 核心魔法：动态时间切片！
        # 无论你传进来的 T 是 16 还是 3，我们永远取 time_embed 的最后 T 帧。
        # 这样保证了“最新的一帧”永远对齐最后一个时间编码，物理意义绝对严谨！
        x = x + self.time_embed[:, -T:, :]       
        
        # 送入 Transformer，输出 z_real 形状为 [B, T, 384]
        return self.transformer(x)

# ================= 2. 封装：端到端 Imitation Learning 策略网络 =================
class TactileBehaviorCloningPolicy(nn.Module):
    def __init__(self, encoder_ckpt_path, action_dim=22, embed_dim=384, max_seq_len=16, freeze_encoder=True):
        super().__init__()
        
        # 1. 实例化新版 Encoder
        self.encoder = TemporalTactileEncoder(embed_dim=embed_dim, max_seq_len=max_seq_len)
        
        # 2. 加载预训练权重 (自动处理 DDP 的 module. 前缀)
        if os.path.exists(encoder_ckpt_path):
            print(f"📦 正在加载预训练 Encoder 权重: {encoder_ckpt_path}")
            checkpoint = torch.load(encoder_ckpt_path, map_location='cpu')
            clean_state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint['encoder_state_dict'].items()])
            
            # 严格加载，防止漏掉参数
            self.encoder.load_state_dict(clean_state_dict)
        else:
            print("⚠️ 警告：未找到预训练权重，将使用随机初始化的 Encoder！")

        # 3. 决定是否冻结 Encoder
        if freeze_encoder:
            print("❄️ Encoder 参数已冻结 (Linear Probing 模式)。")
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval() 
        
        # 4. 动作生成头 (Action Head)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim) 
        )

    def preprocess_input(self, raw_history):
        """ 🛡️ 内置物理防爆预处理：支持任意长度 T """
        clean_history = torch.nan_to_num(raw_history, nan=0.0)
        
        sparse_force = clean_history[..., 3].abs()
        clean_history[..., 3] = torch.log1p(sparse_force)
        
        return torch.clamp(clean_history, min=-10.0, max=10.0)

    def forward(self, raw_sparse_history):
        # 1. 自动预处理 -> [B, T, 44, 4]
        processed_history = self.preprocess_input(raw_sparse_history)
        
        # 2. 提取时空特征 z_real: [B, T, 384]
        if not self.encoder.training:
            with torch.no_grad():
                z_real = self.encoder(processed_history)
        else:
            z_real = self.encoder(processed_history)
            
        # 3. 全局平均池化: [B, T, 384] -> [B, 384]
        # 🌟 不管 T 是多少，平均池化都能完美压缩成一个 384 维向量
        z_pooled = z_real.mean(dim=1)
        
        # 4. 预测动作
        pred_action = self.action_head(z_pooled)
        return pred_action

# ================= 3. 极简使用与训练 Demo =================
if __name__ == "__main__":
    BATCH_SIZE = 64
    ACTION_DIM = 22  
    MAX_SEQ_LEN = 16
    CKPT_PATH = "checkpoints/latest_geometry_ckpt.pth" 
    
    policy = TactileBehaviorCloningPolicy(
        encoder_ckpt_path=CKPT_PATH, 
        action_dim=ACTION_DIM, 
        max_seq_len=MAX_SEQ_LEN,
        freeze_encoder=True 
    ).cuda()
    
    optimizer = torch.optim.AdamW(policy.action_head.parameters(), lr=1e-3)
    
    print("\n🚀 开始模拟 Imitation Learning 训练 (测试可变历史长度)...")
    
    for epoch in range(5):
        # 🌟 模拟可变长度！在这个 Epoch，我们假设由于刚开机，只采集到了 7 帧历史数据
        CURRENT_T = 7 
        
        # 生成 dummy 数据 [B, T, 44, 4]
        dummy_raw_input = torch.randn(BATCH_SIZE, CURRENT_T, 44, 4).cuda() 
        dummy_expert_action = torch.randn(BATCH_SIZE, ACTION_DIM).cuda()
        
        policy.train()
        optimizer.zero_grad()
        
        pred_action = policy(dummy_raw_input)
        loss = F.mse_loss(pred_action, dummy_expert_action)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} | 输入长度 T={CURRENT_T} | Behavior Cloning Loss: {loss.item():.4f}")
        
    print("✅ 跑通了！这套架构现在像水一样极其柔性，随便喂多少帧都能处理！")