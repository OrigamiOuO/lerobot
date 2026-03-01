#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for recording a dataset with SO Follower robot including tactile sensor data.

This script demonstrates:
1. How to configure a robot with tactile sensors
2. How to create a dataset that includes tactile features (depth, normals, marker displacement)
3. How to use pipelines for feature transformation
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import IdentityProcessorStep, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ============================================================================
# Configuration
# ============================================================================
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "Grasp the object with tactile feedback"
HF_REPO_ID = "<hf_username>/so_follower_tactile_dataset"

# Robot and camera ports (adjust for your setup)
FOLLOWER_PORT = "/dev/ttyUSB0"
LEADER_PORT = "/dev/ttyUSB1"
CAMERA_INDEX = 0
TACTILE_CAMERA_INDEX = 2


def main():
    """
    Main function to record a dataset with tactile sensor data.
    
    The dataset will include:
    - observation.state: Motor positions (6 joints)
    - observation.images.front: RGB camera image
    - observation.tactile_depth: Tactile depth map (H, W, 1)
    - observation.tactile_normal: Tactile normal vectors (H, W, 3)
    - observation.marker_displacement: Marker displacements (N, 2)
    - action: Target motor positions
    """
    
    # ========================================================================
    # Step 1: Create robot configuration with tactile sensor
    # ========================================================================
    
    # Regular RGB camera configuration
    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=640,
            height=480,
            fps=FPS
        )
    }
    
    # Tactile camera configuration (GelSight-style sensor)
    tactile_config = TactileCameraConfig(
        index_or_path=TACTILE_CAMERA_INDEX,
        width=640,
        height=480,
        fps=FPS,
        exposure=1500,
        auto_exposure=False,
        wb_temperature=4000,
        auto_wb=False,
    )
    
    # Create follower robot configuration
    follower_config = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        id="tactile_follower",
        cameras=camera_config,
        tactile_camera=tactile_config,
        num_markers=35,  # Number of markers to track
        use_degrees=False,  # Use normalized range [-100, 100]
    )
    
    # Create leader teleoperator configuration
    leader_config = SO100LeaderConfig(
        port=LEADER_PORT,
        id="leader_arm"
    )
    
    # ========================================================================
    # Step 2: Initialize robot and teleoperator
    # ========================================================================
    
    follower = SOFollower(follower_config)
    leader = SO100Leader(leader_config)
    
    # ========================================================================
    # Step 3: Create processing pipelines
    # ========================================================================
    
    # Identity processor for observations (no transformation needed)
    observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    
    # Identity processor for teleop actions
    teleop_action_processor = RobotProcessorPipeline(
        steps=[IdentityProcessorStep()],
        to_transition=lambda x: {"action": x[0], "observation": x[1]},
        to_output=lambda t: t.get("action", {}),
    )
    
    # Identity processor for robot actions
    robot_action_processor = RobotProcessorPipeline(
        steps=[IdentityProcessorStep()],
        to_transition=lambda x: {"action": x[0], "observation": x[1]},
        to_output=lambda t: t.get("action", {}),
    )
    
    # ========================================================================
    # Step 4: Create the dataset
    # ========================================================================
    
    print("=" * 60)
    print("Creating dataset with features:")
    print("=" * 60)
    
    # Get observation features from follower (includes tactile)
    obs_features = aggregate_pipeline_dataset_features(
        pipeline=observation_processor,
        initial_features=create_initial_features(observation=follower.observation_features),
        use_videos=True,  # RGB images stored as video
    )
    
    # Get action features from leader
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=leader.action_features),
        use_videos=True,
    )
    
    # Combine all features
    dataset_features = combine_feature_dicts(action_features, obs_features)
    
    # Print feature summary
    print("\nDataset features:")
    for key, info in dataset_features.items():
        print(f"  {key}: dtype={info['dtype']}, shape={info['shape']}")
    print()
    
    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )
    
    print(f"Dataset created at: {dataset.root}")
    print()
    
    # ========================================================================
    # Step 5: Connect devices
    # ========================================================================
    
    print("Connecting devices...")
    leader.connect()
    follower.connect()
    
    if not leader.is_connected or not follower.is_connected:
        raise ValueError("Failed to connect robot or teleoperator!")
    
    print("All devices connected successfully!")
    print()
    
    # ========================================================================
    # Step 6: Initialize keyboard listener and visualization
    # ========================================================================
    
    listener, events = init_keyboard_listener()
    init_rerun(session_name="tactile_recording")
    
    # ========================================================================
    # Step 7: Record episodes
    # ========================================================================
    
    print("=" * 60)
    print("Starting recording session")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' to stop recording")
    print("  - Press 'r' to re-record current episode")
    print("  - Press Space to skip to next episode")
    print()
    
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
        
        # Main recording loop
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop=leader,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=observation_processor,
        )
        
        # Handle episode completion
        if events["rerecord_episode"]:
            events["rerecord_episode"] = False
            log_say("Re-recording episode")
            continue
        
        # Save the episode
        if not events["exit_early"]:
            dataset.save_episode()
            episode_idx += 1
            log_say(f"Episode saved. Total episodes: {episode_idx}")
        
        events["exit_early"] = False
        
        # Reset environment between episodes
        if episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say("Reset the environment")
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop=leader,
                control_time_s=RESET_TIME_SEC,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=observation_processor,
            )
    
    # ========================================================================
    # Step 8: Finalize and cleanup
    # ========================================================================
    
    print()
    print("=" * 60)
    print("Recording complete!")
    print("=" * 60)
    print(f"Total episodes recorded: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"Dataset location: {dataset.root}")
    
    # Finalize dataset (close parquet writers)
    dataset.finalize()
    
    # Disconnect devices
    leader.disconnect()
    follower.disconnect()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
