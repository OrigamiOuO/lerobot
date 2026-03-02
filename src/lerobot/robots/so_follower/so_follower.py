#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from typing import TypeAlias

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.gelsight_marker_tracker import GelSightMarkerTracker
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Tactile sensor components - support both single and multiple tactile cameras
        self.tactile_cameras: dict[str, TactileCamera] = {}
        self.tactile_processors: dict[str, any] = {}
        self.marker_trackers: dict[str, GelSightMarkerTracker] = {}
        self._tactile_initialized: dict[str, bool] = {}
        self._num_markers = config.num_markers
        
        # Legacy single tactile camera support (backward compatible)
        if config.tactile_camera is not None:
            self.tactile_cameras["tactile"] = TactileCamera(config.tactile_camera)
        
        # Multiple tactile cameras support
        if hasattr(config, 'tactile_cameras') and config.tactile_cameras:
            for name, tactile_cfg in config.tactile_cameras.items():
                self.tactile_cameras[name] = TactileCamera(tactile_cfg)
        
        # Initialize tracking dicts for all tactile cameras
        for name in self.tactile_cameras:
            self._tactile_initialized[name] = False
        
        # Legacy single camera reference (for backward compatibility)
        self.tactile_camera = self.tactile_cameras.get("tactile")
        self.tactile_processor = None
        self.marker_tracker = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def _tactile_ft(self) -> dict[str, tuple]:
        """Tactile sensor features: depth, normal, and marker displacement for all tactile cameras."""
        if not self.tactile_cameras:
            return {}
        
        features = {}
        for name in self.tactile_cameras:
            # For single legacy camera "tactile", use original names for backward compatibility
            if name == "tactile" and len(self.tactile_cameras) == 1:
                features["tactile_depth"] = (480, 640, 1)
                features["tactile_normal"] = (480, 640, 3)
                features["marker_displacement"] = (self._num_markers, 2)
            else:
                # For multiple cameras, use prefixed names
                features[f"{name}_depth"] = (480, 640, 1)
                features[f"{name}_normal"] = (480, 640, 3)
                features[f"{name}_marker_displacement"] = (self._num_markers, 2)
        
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._tactile_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        base_connected = self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
        if self.tactile_cameras:
            tactile_connected = all(cam.is_connected for cam in self.tactile_cameras.values())
            return base_connected and tactile_connected
        return base_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        # Connect all tactile sensors
        if self.tactile_cameras:
            import os
            from lerobot.cameras.tactile_cam import MLPProcessor
            
            # Get tactile_cam directory for loading calibration files
            tactile_cam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "..", "..", "cameras", "tactile_cam")
            tactile_cam_dir = os.path.abspath(tactile_cam_dir)
            
            # Paths to model and calibration files (与6_3_test_mlp_v2.py一致)
            model_path = os.path.join(tactile_cam_dir, "load", "nnmodel_v2.pth")
            calib_file = os.path.join(tactile_cam_dir, "calibration_data", "homography_matrix.npz")
            
            # Load ppmm from calibration file
            mm_per_pixel_file = os.path.join(tactile_cam_dir, "calibration_data", "mm_per_pixel.npz")
            if os.path.exists(mm_per_pixel_file):
                import numpy as np
                data = np.load(mm_per_pixel_file)
                mm_per_pixel = float(data['mm_per_pixel'])
                ppmm = 1.0 / mm_per_pixel
            else:
                ppmm = 7.6
            
            for name, tactile_cam in self.tactile_cameras.items():
                tactile_cam.connect()
                
                # Initialize marker tracker for each tactile camera
                self.marker_trackers[name] = GelSightMarkerTracker()
                self.marker_trackers[name].IsDisplay = False  # 禁止在worker线程中弹出Qt窗口
                
                # Initialize MLP processor for depth and normal reconstruction
                self.tactile_processors[name] = MLPProcessor(
                    model_path=model_path if os.path.exists(model_path) else None,
                    pad=20,
                    calib_file=calib_file if os.path.exists(calib_file) else None,
                    device=None,  # Auto-detect cuda/cpu
                    ppmm=ppmm
                )
                logger.info(f"{self} tactile sensor '{name}' connected with MLP processor (ppmm={ppmm:.2f}).")
            
            # Legacy single camera references (for backward compatibility)
            if "tactile" in self.tactile_cameras:
                self.tactile_camera = self.tactile_cameras["tactile"]
                self.tactile_processor = self.tactile_processors.get("tactile")
                self.marker_tracker = self.marker_trackers.get("tactile")

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position with retries for stability
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", num_retry=3)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Parallel capture from all cameras and tactile sensors
        start_all = time.perf_counter()
        
        def read_camera(cam_key, cam):
            """Helper to read a single camera."""
            start = time.perf_counter()
            frame = cam.async_read(timeout_ms=1000)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            return cam_key, frame
        
        def read_tactile(name, tactile_cam):
            """Helper to read a single tactile camera."""
            start = time.perf_counter()
            tactile_data = self._read_tactile_observation_single(name, tactile_cam)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read tactile '{name}': {dt_ms:.1f}ms")
            return tactile_data
        
        # Use ThreadPoolExecutor for parallel reads
        max_workers = len(self.cameras) + len(self.tactile_cameras)
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit camera read tasks
                for cam_key, cam in self.cameras.items():
                    futures.append(executor.submit(read_camera, cam_key, cam))
                
                # Submit tactile camera read tasks
                for name, tactile_cam in self.tactile_cameras.items():
                    futures.append(executor.submit(read_tactile, name, tactile_cam))
                
                # Collect results
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, tuple) and len(result) == 2:
                        # Camera result: (key, frame)
                        cam_key, frame = result
                        obs_dict[cam_key] = frame
                    elif isinstance(result, dict):
                        # Tactile result: dict
                        obs_dict.update(result)
        
        dt_all_ms = (time.perf_counter() - start_all) * 1e3
        logger.debug(f"{self} parallel read all cameras: {dt_all_ms:.1f}ms")

        return obs_dict

    def _read_tactile_observation(self) -> dict:
        """
        Legacy method: Read tactile sensor data from the default 'tactile' camera.
        For backward compatibility.
        """
        if "tactile" in self.tactile_cameras:
            return self._read_tactile_observation_single("tactile", self.tactile_cameras["tactile"])
        return {}

    def _read_tactile_observation_single(self, name: str, tactile_cam: TactileCamera) -> dict:
        """
        Read tactile sensor data from a single tactile camera using MLP model.
        
        Args:
            name: Name of the tactile camera (e.g., "tactile", "left_finger", "right_finger")
            tactile_cam: TactileCamera instance
            
        Returns:
            dict with keys depending on camera name:
            - For single camera "tactile": tactile_depth, tactile_normal, marker_displacement
            - For multiple cameras: {name}_depth, {name}_normal, {name}_marker_displacement
        """
        import cv2
        
        # Get processor and tracker for this specific tactile camera
        processor = self.tactile_processors.get(name)
        tracker = self.marker_trackers.get(name)
        
        # Read raw frame from tactile camera
        frame = tactile_cam.async_read(timeout_ms=200)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        h, w = frame_bgr.shape[:2]
        
        # Apply MLP processing to get depth and normals
        if processor is not None:
            # process_frame returns: depth_colored, normal_colored, raw_depth (H,W), raw_normals (H,W,3)
            depth_colored, normal_colored, raw_depth, raw_normals = processor.process_frame(
                frame_bgr, apply_warp=True
            )
            
            # Get warped frame for marker tracking
            warped_frame = processor.warp_perspective(frame_bgr)
        else:
            warped_frame = frame_bgr
            raw_depth = np.zeros((h, w), dtype=np.float32)
            raw_normals = np.zeros((h, w, 3), dtype=np.float32)
        
        h, w = warped_frame.shape[:2]
        
        # Initialize marker tracker on first valid frame (after background is collected)
        if not self._tactile_initialized.get(name, False) and processor is not None:
            # Check if background has been collected (bg_image is set in new MLPProcessor)
            bg_collected = hasattr(processor, 'bg_image') and processor.bg_image is not None
            if bg_collected and tracker is not None:
                # Use background frame from processor for marker initialization
                tracker.reinit(processor.bg_image)
                logger.info(f"Tactile marker tracker '{name}' initialized with {tracker.MarkerCount} markers")
                self._tactile_initialized[name] = True
            elif tracker is None:
                # No tracker, mark as initialized anyway
                self._tactile_initialized[name] = True
        
        # Convert depth to (H, W, 1) format
        # raw_depth is (H, W) where each pixel value is the depth at that point
        if raw_depth is not None:
            depth = raw_depth[..., np.newaxis].astype(np.float32)  # (H, W, 1)
        else:
            depth = np.zeros((h, w, 1), dtype=np.float32)
        
        # Normal vectors are already (H, W, 3) with (nx, ny, nz) components
        if raw_normals is not None and raw_normals.shape[-1] == 3:
            normal = raw_normals.astype(np.float32)  # (H, W, 3)
        else:
            normal = np.zeros((h, w, 3), dtype=np.float32)
        
        # Update marker tracking and get displacements
        marker_displacement = np.zeros((self._num_markers, 2), dtype=np.float32)
        if tracker is not None and self._tactile_initialized.get(name, False):
            tracker.update_markerMotion(warped_frame)
            displacements = tracker.get_marker_displacements()
            if displacements is not None:
                # Pad or truncate to match expected number of markers
                n = min(len(displacements), self._num_markers)
                marker_displacement[:n] = displacements[:n]
        
        # Use appropriate key names based on whether this is single or multiple camera setup
        if name == "tactile" and len(self.tactile_cameras) == 1:
            return {
                "tactile_depth": depth,
                "tactile_normal": normal,
                "marker_displacement": marker_displacement,
            }
        else:
            return {
                f"{name}_depth": depth,
                f"{name}_normal": normal,
                f"{name}_marker_displacement": marker_displacement,
            }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", num_retry=3)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm with retries for stability
        self.bus.sync_write("Goal_Position", goal_pos, num_retry=3)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect all tactile sensors
        for name, tactile_cam in self.tactile_cameras.items():
            if tactile_cam.is_connected:
                tactile_cam.disconnect()
                logger.info(f"Tactile sensor '{name}' disconnected.")

        logger.info(f"{self} disconnected.")


SO100Follower: TypeAlias = SOFollower
SO101Follower: TypeAlias = SOFollower
