#!/usr/bin/env python3
"""
Diagnostic script to understand the camera resolution mismatch issue.
"""

import cv2
import time
import subprocess

device = "/dev/video2"

print("=" * 70)
print("Camera Resolution Diagnostic")
print("=" * 70)

# Step 1: Check V4L2 level settings
print("\n[Step 1] Current V4L2 settings:")
result = subprocess.run(["v4l2-ctl", "-d", device, "--get-fmt-video"], capture_output=True, text=True)
print(result.stdout)

# Step 2: Open with default backend
print("[Step 2] OpenCV with default backend:")
cap = cv2.VideoCapture(device)
print(f"  Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
cap.release()
time.sleep(0.5)

# Step 3: Try V4L2 backend (CAP_V4L2)
print("[Step 3] OpenCV with V4L2 backend (CAP_V4L2):")
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
print(f"  Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
cap.release()
time.sleep(0.5)

# Step 4: Set resolution AFTER opening
print("[Step 4] OpenCV with manual resolution setting:")
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
print(f"  Before set - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.5)
print(f"  After set - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Step 5: Try reading a frame
print("[Step 5] Reading a frame:")
ret, frame = cap.read()
if ret:
    print(f"  Frame shape: {frame.shape}")
else:
    print("  Failed to read frame")

cap.release()
time.sleep(0.5)

# Step 6: Check V4L2 settings again
print("\n[Step 6] V4L2 settings after OpenCV operations:")
result = subprocess.run(["v4l2-ctl", "-d", device, "--get-fmt-video"], capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
