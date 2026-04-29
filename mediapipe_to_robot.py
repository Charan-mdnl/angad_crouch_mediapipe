#!/usr/bin/env python3
"""
MediaPipe → Angad Robot Upper-Body Retargeting (v4 FINAL)
═════════════════════════════════════════════════════════
FIXES in v4:
  1. CORRECT coordinate system: MP world y=DOWN, z=AWAY from camera
  2. RealSense D435 camera preference (falls back to laptop cam)
  3. L/R labels drawn on skeleton for verification
  4. 5-second standing phase → auto-calibrate → tracking
  5. All angles in degrees

Run:
  cd /home/charan/Documents/Angad_with_limits
  /home/charan/.gemini/antigravity/scratch/mp_venv/bin/python3 mediapipe_to_robot.py
"""
import numpy as np
import cv2
import time
import threading
import os
import mujoco
from mujoco import viewer
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode
)

os.chdir("/home/charan/Documents/Angad_with_limits")

# ═══════════════════════════════════════════════════════════════════
#  CAMERA SELECTION — Prefer RealSense D435 RGB
# ═══════════════════════════════════════════════════════════════════
def find_best_camera():
    """Try cameras in order: RealSense RGB (usually index 2), then fallback."""
    # RealSense D435: typically /dev/video2 is the RGB stream
    # Try indices in preference order
    for idx in [2, 1, 0, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0:
                backend = cap.getBackendName()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Using camera index {idx}: {w}x{h} ({backend})")
                return cap
            cap.release()
    return None

# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE POSE LANDMARKER
# ═══════════════════════════════════════════════════════════════════
MODEL_PATH = "pose_landmarker_lite.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
)
landmarker = PoseLandmarker.create_from_options(options)

LM_LEFT_SHOULDER  = 11;  LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW     = 13;  LM_RIGHT_ELBOW    = 14
LM_LEFT_WRIST     = 15;  LM_RIGHT_WRIST    = 16
LM_LEFT_HIP       = 23;  LM_RIGHT_HIP      = 24

# ═══════════════════════════════════════════════════════════════════
#  MUJOCO MODEL
# ═══════════════════════════════════════════════════════════════════
m = mujoco.MjModel.from_xml_path("XP_robot_with_actuators.xml")
d = mujoco.MjData(m)
nu = m.nu
gear = np.array([m.actuator_gear[i][0] for i in range(nu)])
act_names = [m.actuator(i).name for i in range(nu)]
IDX = {n: i for i, n in enumerate(act_names)}

JOINT_LIMITS = {
    "arm_pitch_r": (-3.14, 3.14),   "arm_roll_r": (-0.087, 3.316),  "elbow_r": (-2.53, 0.0),
    "arm_pitch_l": (-3.14, 3.14),   "arm_roll_l": (-3.316, 0.087),  "elbow_l": (-2.53, 0.0),
}
ARM_KEYS = ["arm_pitch_r","arm_roll_r","elbow_r","arm_pitch_l","arm_roll_l","elbow_l"]

# ═══════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════
lock = threading.Lock()
shared = {k: 0.0 for k in ARM_KEYS}
shared["gesture"] = "NEUTRAL"
shared["running"] = True
shared["tracking_active"] = False

calib_offset = {k: 0.0 for k in ARM_KEYS}

# ═══════════════════════════════════════════════════════════════════
#  ANGLE COMPUTATION — CORRECT COORDINATE SYSTEM
#
#  MediaPipe world landmarks:
#    x = to subject's LEFT (positive)
#    y = DOWNWARD (positive)
#    z = AWAY from camera (positive)
#
#  When arms hang down: upper_arm ≈ (0, +0.28, 0)
# ═══════════════════════════════════════════════════════════════════
def compute_joint_angles(world_landmarks):
    lm = world_landmarks
    def g(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

    r_shoulder, r_elbow_pt, r_wrist = g(LM_RIGHT_SHOULDER), g(LM_RIGHT_ELBOW), g(LM_RIGHT_WRIST)
    l_shoulder, l_elbow_pt, l_wrist = g(LM_LEFT_SHOULDER),  g(LM_LEFT_ELBOW),  g(LM_LEFT_WRIST)

    r_upper = r_elbow_pt - r_shoulder
    l_upper = l_elbow_pt - l_shoulder

    # PITCH: angle from hanging-down toward forward (toward camera)
    # y=down (positive), z=away (positive), toward camera = -z
    # 0 = arm down, +90° = arm horizontal forward
    r_pitch = np.arctan2(-r_upper[2], r_upper[1])
    l_pitch = np.arctan2(-l_upper[2], l_upper[1])

    # ROLL: lateral spread
    # Right arm outward = -x direction (to subject's right)
    r_roll = np.arctan2(-r_upper[0], r_upper[1])  # +ve = spread right
    # Left arm outward = +x direction (to subject's left)
    l_roll = -np.arctan2(l_upper[0], l_upper[1])   # -ve = spread left

    # ELBOW bend
    r_lower = r_wrist - r_elbow_pt
    r_cos = np.dot(r_upper, r_lower) / (np.linalg.norm(r_upper)*np.linalg.norm(r_lower)+1e-8)
    r_elbow_a = -(np.pi - np.arccos(np.clip(r_cos, -1, 1)))

    l_lower = l_wrist - l_elbow_pt
    l_cos = np.dot(l_upper, l_lower) / (np.linalg.norm(l_upper)*np.linalg.norm(l_lower)+1e-8)
    l_elbow_a = -(np.pi - np.arccos(np.clip(l_cos, -1, 1)))

    result = {
        "arm_pitch_r": r_pitch, "arm_roll_r": r_roll, "elbow_r": r_elbow_a,
        "arm_pitch_l": l_pitch, "arm_roll_l": l_roll, "elbow_l": l_elbow_a,
    }
    for key in result:
        lo, hi = JOINT_LIMITS[key]
        result[key] = np.clip(result[key], lo, hi)

    # COM-safety
    if result["arm_pitch_r"] > 0.5 and result["arm_pitch_l"] > 0.5:
        result["arm_pitch_r"] = min(result["arm_pitch_r"], 0.9)
        result["arm_pitch_l"] = min(result["arm_pitch_l"], 0.9)

    return result


def detect_gesture(world_landmarks):
    lm = world_landmarks
    def g(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

    l_shoulder, r_shoulder = g(LM_LEFT_SHOULDER), g(LM_RIGHT_SHOULDER)
    l_wrist, r_wrist = g(LM_LEFT_WRIST), g(LM_RIGHT_WRIST)
    l_hip, r_hip = g(LM_LEFT_HIP), g(LM_RIGHT_HIP)

    chest = (l_shoulder + r_shoulder) / 2.0
    wrist_dist = np.linalg.norm(l_wrist - r_wrist)
    wrist_mid = (l_wrist + r_wrist) / 2.0

    # y is DOWN → wrist above hip means wrist.y < hip.y
    if (wrist_dist < 0.15 and np.linalg.norm(wrist_mid - chest) < 0.25 and
        l_wrist[1] < l_hip[1] and r_wrist[1] < r_hip[1]):
        return "NAMASTE"

    r_ext = np.linalg.norm(r_wrist - r_shoulder)
    l_ext = np.linalg.norm(l_wrist - l_shoulder)
    if r_wrist[1] < r_hip[1] and r_ext > 0.35 and l_ext < 0.30:
        return "HANDSHAKE"

    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════
#  SKELETON DRAWING WITH L/R LABELS
# ═══════════════════════════════════════════════════════════════════
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]

def draw_skeleton(frame, norm_landmarks, flipped=False):
    h, w = frame.shape[:2]
    pts = {}
    for i, lm in enumerate(norm_landmarks):
        # If the display is flipped, mirror the x coordinates for drawing
        px = int((1.0 - lm.x) * w) if flipped else int(lm.x * w)
        py = int(lm.y * h)
        pts[i] = (px, py)
        if lm.visibility > 0.5:
            cv2.circle(frame, (px, py), 4, (0, 255, 128), -1)
    for (a, b) in POSE_CONNECTIONS:
        if a in pts and b in pts:
            if norm_landmarks[a].visibility > 0.5 and norm_landmarks[b].visibility > 0.5:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)

    # Draw L/R labels at wrists for verification
    if LM_LEFT_WRIST in pts and norm_landmarks[LM_LEFT_WRIST].visibility > 0.5:
        cv2.putText(frame, "L", (pts[LM_LEFT_WRIST][0]+10, pts[LM_LEFT_WRIST][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if LM_RIGHT_WRIST in pts and norm_landmarks[LM_RIGHT_WRIST].visibility > 0.5:
        cv2.putText(frame, "R", (pts[LM_RIGHT_WRIST][0]+10, pts[LM_RIGHT_WRIST][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ═══════════════════════════════════════════════════════════════════
#  WEBCAM THREAD
# ═══════════════════════════════════════════════════════════════════
def webcam_loop():
    cap = find_best_camera()
    if cap is None:
        print("ERROR: Cannot open any camera!")
        shared["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    global calib_offset
    frame_ts = 0
    start_time = time.time()
    calibrated = False

    while shared["running"]:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start_time

        # Process ORIGINAL frame — no flip before MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        # Flip for display
        display = cv2.flip(frame, 1)

        gesture = "NEUTRAL"
        has_pose = (result.pose_landmarks and len(result.pose_landmarks) > 0 and
                    result.pose_world_landmarks and len(result.pose_world_landmarks) > 0)

        if has_pose:
            draw_skeleton(display, result.pose_landmarks[0], flipped=True)
            world_lm = result.pose_world_landmarks[0]

            if elapsed < 5.0:
                countdown = int(5.0 - elapsed) + 1
                cv2.putText(display, f"STAND STILL - Starting in {countdown}s", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if not calibrated:
                    raw = compute_joint_angles(world_lm)
                    calib_offset = {k: raw[k] for k in ARM_KEYS}
                    calibrated = True
                    shared["tracking_active"] = True
                    print(f"AUTO-CALIBRATED at {elapsed:.1f}s")
                    for k in ARM_KEYS:
                        print(f"  {k}: offset = {np.degrees(calib_offset[k]):+.1f} deg")

                gesture = detect_gesture(world_lm)
                raw_angles = compute_joint_angles(world_lm)

                angles = {}
                for k in ARM_KEYS:
                    angles[k] = raw_angles[k] - calib_offset[k]
                    lo, hi = JOINT_LIMITS[k]
                    angles[k] = np.clip(angles[k], lo, hi)

                with lock:
                    shared["gesture"] = gesture
                    for key, val in angles.items():
                        shared[key] = val

                y0 = 80
                for key in ARM_KEYS:
                    deg = np.degrees(angles[key])
                    cv2.putText(display, f"{key}: {deg:+6.1f} deg", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y0 += 18

        colors = {"NAMASTE": (0,255,0), "HANDSHAKE": (0,165,255), "NEUTRAL": (200,200,200)}
        cv2.putText(display, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors.get(gesture, (255,255,255)), 3)

        if calibrated:
            cv2.putText(display, "TRACKING | Q=quit  C=recalibrate", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        elif elapsed < 5.0:
            cv2.putText(display, "Stand in ATTENTION...", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("MediaPipe -> Angad Robot", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            shared["running"] = False
            break
        elif key == ord('c') and has_pose:
            raw = compute_joint_angles(result.pose_world_landmarks[0])
            calib_offset = {k: raw[k] for k in ARM_KEYS}
            with lock:
                for k in ARM_KEYS: shared[k] = 0.0
            print("RECALIBRATED!")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


# ═══════════════════════════════════════════════════════════════════
#  MUJOCO CONTROLLER — EXACT FROM angad_poses.py
# ═══════════════════════════════════════════════════════════════════
smooth_t = {k: 0.0 for k in ARM_KEYS}
ALPHA = 0.08

fp, fr = 0.0, 0.0

def controller(m_, d_):
    global fp, fr

    # 1. Base Standing Pose (Athletic Crouch for stability)
    q_des = np.zeros(m_.nu)
    for i, name in enumerate(act_names):
        if "knee" in name: q_des[i] = 0.15 # Slightly bent
        elif "hip_pitch" in name: q_des[i] = 0.05
        elif "ankle_pitch" in name: q_des[i] = 0.05

    # 2. Add Arm Pose from MediaPipe
    if shared["tracking_active"]:
        with lock:
            for k in smooth_t:
                smooth_t[k] += ALPHA * (shared[k] - smooth_t[k])
        for k, v in smooth_t.items():
            if k in IDX: q_des[IDX[k]] = v

    # 3. IMU Balancing (Essential to prevent falling during handshake)
    quat = d_.qpos[3:7]
    pitch = 2.0 * (quat[0]*quat[2] - quat[3]*quat[1])
    roll  = 2.0 * (quat[0]*quat[1] + quat[2]*quat[3])

    # PD feedback on torso tilt
    raw_p = -(2500.0 * pitch + 350.0 * d_.qvel[4])
    raw_r = -(1500.0 * roll + 200.0 * d_.qvel[3])
    fp += 0.1 * (raw_p - fp)
    fr += 0.1 * (raw_r - fr)

    # PD Control for joints
    q_current = d_.qpos[7:]
    q_vel = d_.qvel[6:]
    kp = 400.0 # Increased stiffness
    kd = 20.0
    torque = kp * (q_des - q_current[:m_.nu]) - kd * q_vel[:m_.nu]

    # Apply balance to hips and ankles
    for i, name in enumerate(act_names):
        if "hip_pitch" in name: torque[i] += 0.6 * fp
        if "ankle_pitch" in name: torque[i] += 0.4 * fp
        if "hip_roll" in name: torque[i] += 0.6 * fr
        if "ankle_roll" in name: torque[i] += 0.4 * fr

    gears = np.array([m_.actuator_gear[i][0] for i in range(m_.nu)])
    d_.ctrl[:] = np.clip(torque / gears, -1.0, 1.0)

mujoco.set_mjcb_control(controller)

# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MediaPipe → Angad Robot Controller (v4)")
print("  ─────────────────────────────────────")
print("  1. Robot stands ATTENTION for 5 seconds")
print("  2. Auto-calibrates at 5s (stand still!)")
print("  3. Move arms → robot mirrors you")
print("  Camera: prefers RealSense D435, falls back to laptop")
print("  C = recalibrate | Q = quit")
print("=" * 60)

cam_thread = threading.Thread(target=webcam_loop, daemon=True)
cam_thread.start()
time.sleep(1.0)

with viewer.launch_passive(m, d) as v:
    while v.is_running() and shared["running"]:
        step_start = time.time()
        mujoco.mj_step(m, d)
        v.sync()
        dt = time.time() - step_start
        if dt < m.opt.timestep:
            time.sleep(m.opt.timestep - dt)

shared["running"] = False
cam_thread.join(timeout=2.0)
print("Done.")
