# Angad Humanoid — Real-Time MediaPipe Teleoperation

<p align="center">
  <b>Live webcam-based upper-body teleoperation for the Angad humanoid robot.</b>
</p>

---

## 🤖 What Is This?

This repository focuses on **Vision-based Teleoperation** for the Angad humanoid robot (65 kg, 21-DOF). 

It uses a webcam and Google's MediaPipe Pose Landmarker to track a human operator's upper body in 3D space in real-time. It accurately translates human joint angles (shoulders, elbows, wrists) into robot actuator targets, allowing the robot to mirror your movements instantly.

*Note: As a bonus, this repository also includes a stabilized "Deep Crouch" version of the teleoperation script, where the robot holds a mathematically verified 12cm crouch while mirroring your arms.*

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

**To run the main standing teleoperation script:**
```bash
python3 mediapipe_to_robot.py
```

**To run the deep-crouch teleoperation script:**
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
├── mediapipe_to_robot.py      # MAIN SCRIPT: Standing MediaPipe teleoperation
├── angad_mediapipe_crouch.py  # ALTERNATE SCRIPT: MediaPipe teleoperation while in deep crouch
├── pose_landmarker_lite.task  # Pre-trained Google MediaPipe Lite model file
├── XP_robot_walking.xml       # MuJoCo MJCF physics model for the Angad robot
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
├── meshes/                    # 22 STL files defining the visual & collision geometry
└── msp2/                      # ROS 2 & MoveIt package for advanced upper-body collision avoidance and pre-planned poses (Namaste/Handshake)
```

---

## 🧠 Technical Details

### Advanced MoveIt Integration (`msp2`) — *For Future Work*
While the pure Python `mediapipe_to_robot.py` scripts provide ultra-fast 1000Hz direct-to-MuJoCo physics teleoperation without needing ROS, this repository also includes a full ROS 2 package (`msp2/`). 

**Note: This MoveIt package is NOT used by the current running MediaPipe scripts.** It is included for future advanced work, such as performing complex upper-body trajectory planning, self-collision avoidance, and precise keyframe generation within a ROS 2 framework.

### The Teleoperation Controller
- Runs a dual-thread architecture: Thread 1 grabs webcam frames and runs ML inference at ~30Hz. Thread 2 steps the MuJoCo physics engine at 1000Hz.
- Arm targets are passed between threads using thread-safe locks and smoothed via a low-pass filter (`ALPHA = 0.08`) to prevent robotic jerking or self-collision.
- Balance torques are calculated as: `fp = -(2000*pitch + 300*pitch_velocity)`. This feedback is aggressively injected into the ankles (40%) and hips (60%).

---

## 👤 Author

**Charan** — XP Robotics  
Developing advanced humanoid control and vision systems for the Angad bipedal platform.
