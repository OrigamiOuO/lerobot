基于encoder+action expert的训练过程
## terminal order
lerobot-train --dataset.repo_id=grasp_multi --dataset.root=datasets/luo_proj/Blind_Grasping_LeRobot_Multimodal --policy.type=pretrain_diffusion --output_dir=./checkpoints/luo_proj/pretrain_dp_v1 --policy.repo_id=pretrain_diffusion --batch_size=128 --num_workers=8 --policy.use_amp=true --steps=60000 --wandb.enable=true --wandb.mode=offline --policy.push_to_hub=false --save_freq=10000 --job_name=pretrain_dp_v1

## baseline order
lerobot-train --dataset.repo_id=grasp_multi --dataset.root=datasets/luo_proj/Blind_Grasping_LeRobot_Multimodal --policy.type=diffusion_baseline --output_dir=./checkpoints/luo_proj/pretrain_dpbase_v1 --policy.repo_id=diffusion_baseline --batch_size=128 --num_workers=8 --policy.use_amp=true --steps=60000 --wandb.enable=true --wandb.mode=offline --policy.push_to_hub=false --save_freq=10000 --job_name=pretrain_dpbase_v1