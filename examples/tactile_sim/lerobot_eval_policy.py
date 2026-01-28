import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import mujoco
from dm_control import mujoco as dm_mujoco

# --- 引入 Tactile Sim ---
try:
    from tactile_sim.models import models
except ImportError:
    # 如果 tactile_sim 没有安装，尝试添加路径
    sys.path.append("/home/kefei/Code/tactile-sim") 
    from tactile_sim.models import models

# --- 引入 LeRobot ---
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    sys.path.append("/home/kefei/Code/lerobot/src")
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

# =========================================================================
# 1. 触觉模拟器封装类 (负责实时计算)
# =========================================================================
class TactileOnlineSim:
    def __init__(self, model, data, hand_urdf, hand_mesh_dir, obj_mesh_path, device="cuda"):
        self.model = model
        self.data = data
        self.device = device
        
        # --- 初始化 Tactile Model ---
        print(f"[Tactile] Initializing simulator on {device}...")
        self.hand_model = models(
            hand_file_path=hand_urdf,
            hand_mesh_path=hand_mesh_dir,
            object_root_path=obj_mesh_path,
            device=device
        )
        # 加载标定点 (假设路径是相对当前运行目录或固定路径)
        # 请根据你的实际路径修改这里的 .txt 路径
        base_pts_dir = Path("/home/kefei/Code/tactile-sim/tactile_sim/pts") # 假设在这里
        if not base_pts_dir.exists():
             # 回退到当前目录
             base_pts_dir = Path("pts")
             
        self.hand_model.setup_models(
            str(base_pts_dir / "leappts-4000.txt"),
            str(base_pts_dir / "leaptaxel-4000.txt"),
            str(base_pts_dir / "leapfsr-4000.txt")
        )
        self.hand_model.set_object(obj_mesh_path)
        
        # --- ID 映射配置 ---
        self.HAND_ROOT_NAME = "palm"
        self.OBJECT_BODY_NAME = "cube"
        self.hand_dof_indices = range(6, 6 + 16) # 假设前6个是xarm，后面16个是手
        
        self.LINK_MAP = {
            "palm_lower": "palm", "fingertip": "if_ds", "dip": "if_md", "pip": "if_px", "mcp_joint": "if_bs",
            "fingertip_2": "mf_ds", "dip_2": "mf_md", "pip_2": "mf_px", "mcp_joint_2": "mf_bs",
            "fingertip_3": "rf_ds", "dip_3": "rf_md", "pip_3": "rf_px", "mcp_joint_3": "rf_bs",
            "thumb_temp_base":"th_mp", "thumb_pip":"th_bs", "thumb_dip":"th_px", "thumb_fingertip":"th_ds",
        }

    def get_tactile_observation(self):
        """
        从当前的 MuJoCo Data 中提取状态，计算触觉，并返回符合 LeRobot 格式的 Tensor (44维)
        """
        # 1. 提取 MuJoCo 状态 (复刻 _get_sim_state)
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.HAND_ROOT_NAME)
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.OBJECT_BODY_NAME)
        
        # 获取 Hand State
        h_pos = self.data.xpos[hand_id]
        h_rot = self.data.xquat[hand_id]
        h_vel = self.data.cvel[hand_id]
        h_s = np.concatenate([h_pos, h_rot, h_vel[3:6], h_vel[0:3]])
        
        # 获取 Object State
        o_pos = self.data.xpos[obj_id]
        o_rot = self.data.xquat[obj_id]
        o_vel = self.data.cvel[obj_id]
        o_s = np.concatenate([o_pos, o_rot, o_vel[3:6], o_vel[0:3]])
        
        # 获取 DOF Pos
        d_p = self.data.qpos[self.hand_dof_indices]
        
        # 获取 Link States
        link_states = []
        for t_name in self.hand_model.ind2link.keys():
            mj_name = self.LINK_MAP.get(t_name)
            if not mj_name: continue
            lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mj_name)
            l_pos = self.data.xpos[lid]
            l_rot = self.data.xquat[lid]
            l_vel = self.data.cvel[lid]
            link_states.append(np.concatenate([l_pos, l_rot, l_vel[3:6], l_vel[0:3]]))
            
        # 转 Tensor
        t_h_s = torch.tensor(h_s, dtype=torch.float, device=self.device).unsqueeze(0)
        t_o_s = torch.tensor(o_s, dtype=torch.float, device=self.device).unsqueeze(0)
        t_d_p = torch.tensor(d_p, dtype=torch.float, device=self.device).unsqueeze(0)
        t_l_s = torch.tensor(np.stack(link_states), dtype=torch.float, device=self.device).unsqueeze(0)
        
        # 2. 计算触觉
        # fsr: [1, 12], taxel: [1, 32]
        fsr, taxel = self.hand_model.cal_tactile_info(t_h_s, t_o_s, t_d_p, t_l_s)
        
        # 3. 格式化数据 (复刻 save_and_plot 的重排逻辑)
        # Taxel: 原始顺序可能是 Index, Middle, Ring, Thumb (取决于 leappts 顺序)
        # 你代码里的切片逻辑：
        # tac_index=0:8, tac_middle=8:16, tac_ring=16:24, tac_thumb=24:32
        # 目标顺序：[Thumb, Index, Middle, Ring]
        
        # 注意：这里直接操作 Tensor 避免 CPU/GPU 拷贝
        t_tac = taxel.squeeze(0) # [32]
        tac_index  = t_tac[0:8]
        tac_middle = t_tac[8:16]
        tac_ring   = t_tac[16:24]
        tac_thumb  = t_tac[24:32]
        
        # FSR: [12]
        # fsr_index=0:2, fsr_middle=2:4, fsr_ring=4:6, fsr_thumb=6:8, fsr_palm=8:12
        t_fsr = fsr.squeeze(0)
        fsr_index  = t_fsr[0:2]
        fsr_middle = t_fsr[2:4]
        fsr_ring   = t_fsr[4:6]
        fsr_thumb  = t_fsr[6:8]
        fsr_palm   = t_fsr[8:12]
        
        # 4. 最终拼接 (44维)
        # 顺序: Taxel(32) -> FSR(8) -> Palm(4)
        # Taxel 内部顺序: Thumb, Index, Middle, Ring
        tac_structured = torch.cat([tac_thumb, tac_index, tac_middle, tac_ring]) # 32
        
        # FSR 内部顺序: Thumb, Index, Middle, Ring
        fsr_structured = torch.cat([fsr_thumb, fsr_index, fsr_middle, fsr_ring]) # 8
        
        # Palm
        palm_structured = fsr_palm # 4
        
        final_tactile = torch.cat([tac_structured, fsr_structured, palm_structured]) # 44
        
        return final_tactile

# =========================================================================
# 2. Main Logic
# =========================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Policy with Online Tactile Sim")
    # 基础参数
    parser.add_argument("-p", "--policy", type=str, required=True, help="LeRobot Policy Path")
    parser.add_argument("-x", "--xml", type=str, required=True, help="MuJoCo Scene XML")
    parser.add_argument("-o", "--output", type=str, default="eval_tactile.mp4")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pc", action="store_true", help="Use PointCloud for rendering")
    
    # 触觉相关参数
    parser.add_argument("--urdf", type=str, required=True, help="Hand URDF for Tactile Sim")
    parser.add_argument("--mesh-dir", type=str, required=True, help="Hand Mesh Directory")
    parser.add_argument("--obj-path", type=str, required=True, help="Object OBJ Path")
    
    return parser.parse_args()

def load_policy_explicitly(policy_path, device):
    config_path = policy_path / "config.json"
    if not config_path.exists():
        # 尝试找 checkpoint
        policy_path = policy_path / "checkpoints" / "last" / "pretrained_model"
        config_path = policy_path / "config.json"
        
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    if cfg.get("type") == "act":
        policy = ACTPolicy.from_pretrained(policy_path)
    elif cfg.get("type") == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_path)
    else:
        raise ValueError(f"Unknown type: {cfg.get('type')}")
        
    policy.to(device)
    policy.eval()
    return policy, policy_path

def get_pc_from_camera(physics, cam_name, num_points=1024):
    """复刻采集端的反投影逻辑"""
    h, w = 480, 640 
    depth = physics.render(height=h, width=w, camera_id=cam_name, depth=True)
    
    # 2. 计算内参 K (复刻 _get_intrinsics)
    cam_id = physics.model.name2id(cam_name, 'camera')
    fovy = physics.model.cam_fovy[cam_id]
    f = (h / 2) / np.tan(np.deg2rad(fovy) / 2)
    
    # 3. 反投影 (复刻 _depth_to_point_cloud)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid_mask = (depth > 0.01) & (depth < 2.5) # 过滤背景
    
    z = depth[valid_mask]
    u_val = u[valid_mask]
    v_val = v[valid_mask]
    
    x = (u_val - w/2) * z / f
    y = (v_val - h/2) * z / f
    # 注意：采集端存的是 pc_cam = np.stack([x, y, z], axis=1)
    pc_cam = np.stack([x, y, z], axis=1)
    
    # 4. 下采样到 Policy 要求的固定点数
    if len(pc_cam) >= num_points:
        idx = np.random.choice(len(pc_cam), num_points, replace=False)
        pc_final = pc_cam[idx]
    elif len(pc_cam) > 0:
        idx = np.random.choice(len(pc_cam), num_points, replace=True)
        pc_final = pc_cam[idx]
    else:
        pc_final = np.zeros((num_points, 3))
        
    return torch.from_numpy(pc_final).float()

def main():
    args = get_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 1. Init MuJoCo
    print(f"[System] Loading MuJoCo: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    physics = dm_mujoco.Physics.from_xml_path(args.xml)

    # 2. Init Tactile Sim (Online)
    print(f"[System] Initializing Online Tactile Simulator...")
    tactile_sim = TactileOnlineSim(
        model=model, 
        data=data, # 传入 data 引用，实时读取
        hand_urdf=args.urdf, 
        hand_mesh_dir=args.mesh_dir, 
        obj_mesh_path=args.obj_path,
        device=args.device
    )

    # 3. Load Policy
    print(f"[System] Loading Policy...")
    policy, policy_path = load_policy_explicitly(Path(args.policy), args.device)
    
    print("[System] Building processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path
    )

    # 4. Video Setup
    save_dir = Path(args.policy) / "eval_videos"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(save_dir / args.output)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, 50, (640*2, 480))

    # 5. Inference Loop
    print(f"[System] Start Inference ({args.steps} steps)...")
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    physics.data.qpos[:] = data.qpos[:]
    physics.forward()
    
    # 初始化 tactile 内部状态 (如果有的话)
    if hasattr(tactile_sim.hand_model, 'init'):
        tactile_sim.hand_model.init = False

    with torch.inference_mode():
        for step in range(args.steps):
            # A. 同步物理状态
            physics.data.qpos[:] = data.qpos[:]
            physics.data.qvel[:] = data.qvel[:]
            physics.forward()
            
            # B. 获取视觉观测
            img_l = physics.render(height=480, width=640, camera_id='left_cam')
            img_r = physics.render(height=480, width=640, camera_id='right_cam')
            
            # 记录视频
            frame_vis = np.hstack([
                cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
            ])
            video_writer.write(frame_vis)
            
            # C. 准备 Policy 输入
            # 1. 图片
            t_img_l = torch.from_numpy(img_l.copy()).permute(2,0,1).unsqueeze(0).float().to(args.device)
            t_img_r = torch.from_numpy(img_r.copy()).permute(2,0,1).unsqueeze(0).float().to(args.device)
            
            # 2. 机器人状态 (22维)
            # 注意：这里我们只取前22维 (机器人的qpos)，忽略掉物体的7维
            # 必须和训练数据对齐
            robot_qpos = torch.from_numpy(data.qpos[:22]).float().to(args.device)
            
            # 3. 触觉状态 (44维) - 实时计算！
            tactile_vec = tactile_sim.get_tactile_observation() # 返回 Tensor(44)
            
            # 4. 拼接状态 [22] + [44] = [66]
            full_state = torch.cat([robot_qpos, tactile_vec], dim=0).unsqueeze(0) # (1, 66)

            # 5. 拼接点云（如果需要）
            if args.pc:
                # 获取点云
                pc_l = get_pc_from_camera(physics, 'left_cam', num_points=1024).to(args.device).unsqueeze(0)
                # 获取右路点云
                pc_r = get_pc_from_camera(physics, 'right_cam', num_points=1024).to(args.device).unsqueeze(0)
                
                # 构建符合训练格式的字典
                obs_dict = {
                    "observation.images.cam_left_rgb": t_img_l,
                    "observation.images.cam_right_rgb": t_img_r,
                    "observation.state": full_state,
                    "observation.point_cloud.left_cam": pc_l,
                    "observation.point_cloud.right_cam": pc_r
                }
            else:
                obs_dict = {
                    "observation.images.cam_left_rgb": t_img_l,
                    "observation.images.cam_right_rgb": t_img_r,
                    "observation.state": full_state
                }
            # D. 推理
            # 预处理 -> 推理 -> 后处理
            obs_processed = preprocessor(obs_dict)
            action = policy.select_action(obs_processed)
            action = postprocessor(action)
            
            # E. 执行动作
            action_np = action.squeeze(0).cpu().numpy()
            data.ctrl[:] = action_np
            mujoco.mj_step(model, data)
            
            if step % 50 == 0:
                print(f"Step {step}/{args.steps}")

    video_writer.release()
    print(f"✅ Done! Saved to: {out_path}")

if __name__ == "__main__":
    main()