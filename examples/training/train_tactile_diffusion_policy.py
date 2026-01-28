#!/usr/bin/env python
"""
使用示例：训练 My Custom Policy

这个脚本展示了如何使用 my_custom_policy 进行训练。
"""

import sys
from pathlib import Path

# 添加 src 到路径（如果需要）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.tactile_diffusion import MyCustomPolicy, MyCustomPolicyConfig


def train_example():
    """
    训练示例（简化版）
    
    实际训练请使用:
    lerobot-train --policy.type my_custom_policy ...
    """
    
    print("=" * 80)
    print("My Custom Policy 训练示例")
    print("=" * 80)
    
    # 1. 加载数据集
    print("\n【1】加载数据集")
    dataset = LeRobotDataset(
        repo_id="Opendrawer",
        root="./datasets/test_data/xarm_leap_tactile_lift_blind"
    )
    print(f"✓ 数据集加载成功")
    print(f"  总 Episodes: {dataset.num_episodes}")
    print(f"  总 Frames: {dataset.num_frames}")
    
    # 2. 创建配置
    print("\n【2】创建Policy配置")
    from lerobot.datasets.utils import dataset_to_policy_features
    
    # 从数据集特征创建policy特征
    policy_features = dataset_to_policy_features(dataset.features)
    
    # 分离输入和输出特征
    input_features = {k: v for k, v in policy_features.items() if k.startswith("observation")}
    output_features = {k: v for k, v in policy_features.items() if k.startswith("action")}
    
    config = MyCustomPolicyConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile_features=True,
        tactile_encoder_hidden_dim=64,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"✓ 配置创建成功")
    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")
    print(f"  Device: {config.device}")
    
    # 3. 实例化Policy
    print("\n【3】实例化Policy")
    policy = MyCustomPolicy(config)
    print(f"✓ Policy创建成功")
    print(f"  参数总数: {sum(p.numel() for p in policy.parameters()):,}")
    
    # 4. 准备训练
    print("\n【4】训练准备")
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)
    
    # 获取一个batch示例
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    
    # 将数据移到正确的设备
    batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    print(f"✓ 训练数据准备完成")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  observation.state shape: {batch['observation.state'].shape}")
    if 'observation.tactile' in batch:
        print(f"  observation.tactile shape: {batch['observation.tactile'].shape}")
    print(f"  action shape: {batch['action'].shape}")
    
    # 5. 执行一次训练步骤
    print("\n【5】执行训练步骤")
    policy.train()
    
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"✓ 训练步骤完成")
    print(f"  Loss: {loss.item():.6f}")
    
    # 6. 执行推理
    print("\n【6】执行推理测试")
    policy.eval()
    policy.reset()
    
    with torch.no_grad():
        # 准备单个观察
        obs = {
            k: v[0:1] for k, v in batch.items() 
            if k.startswith("observation")
        }
        
        # 选择动作
        action = policy.select_action(obs)
        
    print(f"✓ 推理测试完成")
    print(f"  Action shape: {action.shape}")
    
    print("\n" + "=" * 80)
    print("✅ 所有步骤成功完成！")
    print("\n💡 实际训练请使用:")
    print("   lerobot-train \\")
    print("       --policy.type my_custom_policy \\")
    print("       --policy.use_tactile_features=true \\")
    print("       --dataset.repo_id Opendrawer \\")
    print("       --dataset.root ./datasets/test_data/xarm_leap_tactile_lift_blind \\")
    print("       --steps 200000")
    print("=" * 80)


if __name__ == "__main__":
    try:
        train_example()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)