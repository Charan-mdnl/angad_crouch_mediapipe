# Angad Humanoid — Real-Time MediaPipe Teleoperation in Deep Crouch

<p align="center">
  <b>Live webcam-based upper-body teleoperation combined with a mathematically verified 12cm deep crouch pose.</b>
</p>

---

## 🤖 What Is This?

This repository contains a full, real-time control pipeline for the **Angad humanoid robot** (65 kg, 21-DOF). It combines two advanced robotic control techniques:

1. **Vision-based Teleoperation:** Uses a webcam and Google's MediaPipe Pose Landmarker to track a human operator's upper body in 3D space, translating human joint angles into robot actuator targets.
2. **Kinematically Stable Deep Crouch:** The lower body of the robot is rigidly locked into a mathematically optimized 12cm deep crouch pose. This pose utilizes a 6-DOF Inverse Kinematics solver to safely handle the robot's unique 15-degree oblique hip-pitch axis, preventing the lateral drift that usually causes the robot to fall over.

By combining these, the robot achieves an ultra-stable center of mass (COM) while the upper body dynamically mirrors the human operator!

---

## ✨ Features

- **Live MediaPipe Tracking:** Detects human shoulders, elbows, and wrists, computing accurate pitch and roll angles with proper spatial coordinate conversions.
- **Smart Camera Selection:** Automatically prioritizes RealSense D435 depth cameras (`/dev/video2`) for higher fidelity, while gracefully falling back to standard laptop webcams.
- **Auto-Calibration:** Features a 5-second "Attention" phase at startup. The user stands still, and the script zeros out the joint offsets, ensuring the robot accurately tracks the human regardless of the person's exact anatomical proportions.
- **Dynamic IMU Balancing:** Moving the arms heavily shifts the robot's Center of Mass. A high-frequency PD balance controller constantly monitors the torso's pitch and roll via simulated IMU, injecting corrective torques into the hip and ankle actuators to prevent falling.
- **Gesture Detection:** Recognizes predefined gestures like "Namaste" and "Handshake" based on spatial wrist/hip relationships.

---

## 🚀 How to Run

### 1. Prerequisites

You must have Python 3.8+ installed along with a functional webcam.

```bash
# Clone the repository
git clone https://github.com/Charan-mdnl/angad_crouch_mediapipe.git
cd angad_crouch_mediapipe

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execution

Run the main Python script:

```bash
python3 angad_mediapipe_crouch.py
```

### 3. Usage Instructions

1. **Step Back:** When the window opens, ensure your entire upper body (hips to head) is visible to the camera.
2. **Stand Still (Calibration):** The screen will show a 5-second countdown. Stand completely still with your arms resting at your sides.
3. **Tracking:** Once the screen says `AUTO-CALIBRATED`, the robot will begin tracking your arms in real-time.
4. **Controls:**
   - Press **`C`** to recalibrate your resting pose if the tracking drifts.
   - Press **`Q`** or **`ESC`** to safely terminate the simulation.

---

## 📁 Repository Structure

```
angad_crouch_mediapipe/
├── angad_mediapipe_crouch.py  # Main script — runs the webcam, MediaPipe, and MuJoCo viewer
├── pose_landmarker_lite.task  # Pre-trained Google MediaPipe Lite model file
├── XP_robot_walking.xml       # MuJoCo MJCF physics model for the Angad robot
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
└── meshes/                    # 22 STL files defining the visual & collision geometry
    ├── pelvis.stl
    ├── torso.stl
    ├── hip_pitch_l.stl
    ├── ... (and other limbs)
```

---

## 🧠 Technical Details

### The Deep Crouch (Base Pose)
The lower body is locked into the following verified joint configuration to lower the COM by 12cm:
- `hip_pitch`: `-0.5543 rad`
- `hip_roll`: `±0.0425 rad` (compensates for 15° oblique axis swing)
- `thigh_yaw`: `±0.0032 rad`
- `knee`: `-1.1537 rad` (66° bend)
- `ankle_pitch`: `-0.6024 rad`
- `ankle_roll`: `0.0 rad`

### The Teleoperation Controller
- Runs a dual-thread architecture: Thread 1 grabs webcam frames and runs ML inference at ~30Hz. Thread 2 steps the MuJoCo physics engine at 1000Hz.
- Arm targets are passed between threads using thread-safe locks and smoothed via a low-pass filter (`ALPHA = 0.08`) to prevent robotic jerking or self-collision.
- Balance torques are calculated as: `fp = -(2000*pitch + 300*pitch_velocity)`. This feedback is aggressively injected into the ankles (40%) and hips (60%).

---

## 👤 Author

**Charan** — XP Robotics  
Developing advanced humanoid control and vision systems for the Angad bipedal platform.
