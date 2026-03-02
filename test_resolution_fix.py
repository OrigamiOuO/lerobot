#!/usr/bin/env python3
"""
Test script to verify the resolution fix for OpenCV cameras.
This script checks if the v4l2-ctl fallback mechanism works correctly.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def test_resolution_fix():
    """Test the resolution fix by attempting to connect to /dev/video2 at 640x480."""
    
    print("=" * 60)
    print("Testing OpenCV Camera Resolution Fix")
    print("=" * 60)
    
    # Create config for 640x480
    config = OpenCVCameraConfig(
        index_or_path="/dev/video2",
        width=640,
        height=480,
        fps=30,
        warmup_s=1
    )
    
    print(f"\n[CONFIG] Device: /dev/video2")
    print(f"[CONFIG] Requested resolution: {config.width}x{config.height}")
    print(f"[CONFIG] Requested FPS: {config.fps}")
    
    try:
        # Create and connect camera
        camera = OpenCVCamera(config)
        print(f"\n[CONNECTING] {camera}...")
        camera.connect(warmup=False)
        
        print(f"\n[SUCCESS] Camera connected!")
        print(f"[RESULT] Actual resolution: {camera.width}x{camera.height}")
        print(f"[RESULT] Actual FPS: {camera.fps}")
        
        # Check if resolution matches expected
        if camera.width == 640 and camera.height == 480:
            print("\n✅ RESOLUTION FIX SUCCESSFUL: Camera is now running at 640x480!")
            print("This should significantly reduce CPU usage during data collection.")
            success = True
        else:
            print(f"\n⚠️  WARNING: Resolution is {camera.width}x{camera.height}, not 640x480")
            print("The v4l2-ctl fallback may not have worked. Check the logs above.")
            success = False
        
        # Grab a test frame to verify it's working
        print("\n[TEST] Attempting to read a frame...")
        frame = camera.read()
        print(f"[SUCCESS] Frame shape: {frame.shape}")
        
        # Disconnect
        camera.disconnect()
        print(f"\n[DISCONNECT] Camera disconnected safely")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_resolution_fix()
    sys.exit(0 if success else 1)
