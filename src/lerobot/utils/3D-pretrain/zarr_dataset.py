import os
import glob
import pickle
import json
import zarr
import numpy as np
import mujoco
from numcodecs import Blosc
from tqdm import tqdm
import cv2

# ================= 1. 核心路径配置 =================
ROOT_DIR = "/home/luoshch/Data/3D-pretrain"
ZARR_PATH = os.path.join(ROOT_DIR, "multimodal_pretrain.zarr")
XML_PATH = "/home/luoshch/Code/mujoco_playground/mujoco_playground/_src/manipulation/xarm_leap_hand/xmls/scene_mjx_cube.xml" 

HAND_PT_PATH = "./pts/leappts-4000.txt"
TAXEL_PATH = "./pts/leaptaxel-4000.txt"
FSR_PATH = "./pts/leapfsr-4000.txt"

OBJECTS = ["Ball", "Cube", "Egg", "H"]

# ================= 2. 传感器映射与配置 =================
LINK_MAP = {
    "fingertip": "if_ds", "dip": "if_md", "pip": "if_px", "mcp_joint": "if_bs",
    "fingertip_2": "mf_ds", "dip_2": "mf_md", "pip_2": "mf_px", "mcp_joint_2": "mf_bs",
    "fingertip_3": "rf_ds", "dip_3": "rf_md", "pip_3": "rf_px", "mcp_joint_3": "rf_bs",
    "thumb_fingertip": "th_ds", "thumb_dip": "th_px", "thumb_pip": "th_bs", "thumb_temp_base": "th_mp",
    "palm_lower": "palm",
}

def load_strict_configs():
    with open(HAND_PT_PATH, 'r') as f: hand_pts = json.loads(f.read())
    with open(TAXEL_PATH, 'r') as f: taxel_ranges = json.loads(f.read())
    with open(FSR_PATH, 'r') as f: fsr_clusters = json.loads(f.read())
    
    configs = []
    TAXEL_ORDER = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
    for link in TAXEL_ORDER:
        if link not in taxel_ranges: continue
        start_idx, end_idx = int(taxel_ranges[link][0]), int(taxel_ranges[link][1])
        link_surface_pts = np.array(hand_pts[link])
        for i in range(start_idx, end_idx):
            pt_data = link_surface_pts[i]
            local_pos = pt_data[0:3]
            local_normal = pt_data[3:6] if len(pt_data) >= 6 else local_pos / (np.linalg.norm(local_pos) + 1e-6)
            configs.append({"body_name": LINK_MAP.get(link, link), "local_pos": local_pos, "local_normal": local_normal, "type": "taxel"})
            
    FSR_ORDER = ["dip", "pip", "dip_2", "pip_2", "dip_3", "pip_3", "thumb_fingertip", "thumb_dip", "palm_lower"]
    for link in FSR_ORDER:
        if link not in fsr_clusters: continue
        for cluster_indices in fsr_clusters[link]:
            mean_pt_data = np.mean(np.array(hand_pts[link])[cluster_indices], axis=0)
            local_pos = mean_pt_data[0:3]
            local_normal = mean_pt_data[3:6] if len(mean_pt_data) >= 6 else local_pos / (np.linalg.norm(local_pos) + 1e-6)
            local_normal = local_normal / (np.linalg.norm(local_normal) + 1e-6)
            configs.append({"body_name": LINK_MAP.get(link, link), "local_pos": local_pos, "local_normal": local_normal, "type": "fsr"})
    return configs

# ================= 3. 初始化 MuJoCo =================
print("⚙️ 初始化 MuJoCo 物理引擎用于离线预处理...")
sensor_configs = load_strict_configs()
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
for cfg in sensor_configs:
    cfg["body_id"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body_name"])

# ================= 4. 初始化 Zarr 库 (极速重构版) =================
print(f"📦 正在创建极速版 Zarr 数据集: {ZARR_PATH}")
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
root = zarr.open(ZARR_PATH, mode='w')

data_group = root.create_group('data')
meta_group = root.create_group('meta')

z_rgb = data_group.zeros('rgb', shape=(0, 3, 128, 128), chunks=(1, 3, 128, 128), dtype='float32', compressor=compressor)
z_mask = data_group.zeros('seg_mask', shape=(0, 1, 128, 128), chunks=(1, 1, 128, 128), dtype='bool', compressor=compressor)
z_sparse_pc = data_group.zeros('sparse_pc', shape=(0, 44, 4), chunks=(1, 44, 4), dtype='float32', compressor=compressor)
z_dense_pc = data_group.zeros('dense_pc', shape=(0, 1024, 7), chunks=(1, 1024, 7), dtype='float32', compressor=compressor)
# 🌟 核心修改 1：为物体点云分配 Zarr 存储空间 (假设提取 3D xyz，固定 1024 个点)
z_obj_pc = data_group.zeros('object_pc', shape=(0, 1024, 3), chunks=(1, 1024, 3), dtype='float32', compressor=compressor)
z_pose = data_group.zeros('pose', shape=(0, 7), chunks=(1, 7), dtype='float32', compressor=compressor)
z_qpos = data_group.zeros('qpos', shape=(0, 22), chunks=(1, 22), dtype='float32', compressor=compressor)

episode_ends = []
episode_obj_ids = []
current_total_steps = 0
total_episodes_saved = 0
total_dirty_frames_dropped = 0  

# ================= 5. 开始多目录穿越与清洗萃取 =================
for obj_idx, obj_name in enumerate(OBJECTS):
    obj_dir = os.path.join(ROOT_DIR, obj_name)
    pkl_files = sorted(glob.glob(os.path.join(obj_dir, "*.pkl")))
    
    if len(pkl_files) == 0:
        continue
        
    print(f"\n🚀 开始处理 [{obj_name}] 数据，共 {len(pkl_files)} 个文件...")
    
    for file_path in tqdm(pkl_files, desc=f"Processing {obj_name}"):
        try:
            with open(file_path, 'rb') as f:
                ep_data = pickle.load(f)
        except Exception as e:
            continue
            
        valid_frames = ep_data
        
        if len(valid_frames) == 0:
            continue 
            
        clean_ep_rgb = []
        clean_ep_mask = []
        clean_ep_sparse = []
        clean_ep_dense = []
        clean_ep_obj = [] # 🌟 收集物体点云的容器
        clean_ep_pose = []
        clean_ep_qpos = []
        
        for step in valid_frames:
            # --- 处理手部致密点云 ---
            dense = step["hand_pc"]
            if len(dense) > 1024:
                dense = dense[np.random.choice(len(dense), 1024, replace=False)]
            elif 0 < len(dense) < 1024:
                dense = dense[np.random.choice(len(dense), 1024, replace=True)]
            elif len(dense) == 0:
                dense = np.zeros((1024, 7))
                
            if np.isnan(dense).any() or np.isinf(dense).any():
                total_dirty_frames_dropped += 1
                continue
                
            obj_pc_raw = step.get("obj_pc", np.zeros((1024, 3)))
            
            # 剥离可能存在的多余维度，只保留 (x, y, z)
            obj_pc = obj_pc_raw[:, :3] 
            
            # 对齐到 1024 个点
            if len(obj_pc) > 1024:
                obj_pc = obj_pc[np.random.choice(len(obj_pc), 1024, replace=False)]
            elif 0 < len(obj_pc) < 1024:
                obj_pc = obj_pc[np.random.choice(len(obj_pc), 1024, replace=True)]
            elif len(obj_pc) == 0:
                obj_pc = np.zeros((1024, 3))
                
            if np.isnan(obj_pc).any() or np.isinf(obj_pc).any():
                total_dirty_frames_dropped += 1
                continue

            # --- 处理稀疏点云 (调用 MuJoCo) ---
            frame_sparse_pc = np.zeros((44, 4), dtype=np.float32)
            data.qpos[:22] = step["qpos"]
            mujoco.mj_kinematics(model, data)
            real_forces = np.concatenate([step["tactile_taxel"], step["tactile_fsr"]])
            
            for j, cfg in enumerate(sensor_configs):
                b_id = cfg["body_id"]
                if b_id == -1: continue
                pos = data.xpos[b_id] + data.xmat[b_id].reshape(3, 3) @ cfg["local_pos"]
                
                if cfg["type"] == "fsr":
                    normal = -(data.xmat[b_id].reshape(3, 3) @ cfg["local_normal"])
                    pos = pos + normal * 0.0015
                    
                frame_sparse_pc[j, 0:3] = pos
                frame_sparse_pc[j, 3] = real_forces[j]
                
            if np.isnan(frame_sparse_pc).any() or np.isinf(frame_sparse_pc).any():
                total_dirty_frames_dropped += 1
                continue

            # --- 处理视觉和 Mask ---
            rgb = cv2.resize(step["rgb_img"], (128, 128)) / 255.0
            rgb_transposed = np.transpose(rgb, (2, 0, 1))
            
            mask = cv2.resize(step["seg_mask"], (128, 128)) > 128
            mask_expanded = np.expand_dims(mask, axis=0)

            pose = step.get("cube_pos", np.zeros(7))
            qpos = step.get("qpos", np.zeros(22))
            
            clean_ep_pose.append(pose)
            clean_ep_qpos.append(qpos)
            clean_ep_rgb.append(rgb_transposed)
            clean_ep_mask.append(mask_expanded)
            clean_ep_dense.append(dense)
            clean_ep_sparse.append(frame_sparse_pc)
            clean_ep_obj.append(obj_pc) # 🌟 存入容器

        if len(clean_ep_rgb) > 0:
            z_rgb.append(np.array(clean_ep_rgb, dtype=np.float32))
            z_mask.append(np.array(clean_ep_mask, dtype=bool))
            z_sparse_pc.append(np.array(clean_ep_sparse, dtype=np.float32))
            z_dense_pc.append(np.array(clean_ep_dense, dtype=np.float32))
            z_obj_pc.append(np.array(clean_ep_obj, dtype=np.float32)) # 🌟 批量追加进 Zarr
            z_pose.append(np.array(clean_ep_pose, dtype=np.float32))
            z_qpos.append(np.array(clean_ep_qpos, dtype=np.float32))
            
            current_total_steps += len(clean_ep_rgb)
            episode_ends.append(current_total_steps)
            episode_obj_ids.append(obj_idx)
            total_episodes_saved += 1

# ================= 6. 写入 Meta 元数据 =================
meta_group.array('episode_ends', np.array(episode_ends, dtype='int64'), compressor=None)
meta_group.array('episode_obj_ids', np.array(episode_obj_ids, dtype='int64'), compressor=None)
meta_group.attrs['object_mapping'] = {i: obj for i, obj in enumerate(OBJECTS)}

print(f"\n🎉 完美收工！包含 Object PC 的 Zarr 库构建完成！")
print(f"📊 统计信息:")
print(f"   - 总共融合的有效轨迹数: {total_episodes_saved}")
print(f"   - 总计接触有效帧数: {current_total_steps}")
print(f"   - Zarr 库路径: {ZARR_PATH}")
print("==================================================")