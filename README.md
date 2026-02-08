# 🧘 Atlas: A Smart Posture Monitor

![Atlas Logo](images/banner.png)

Atlas is a real-time posture monitoring and analysis system designed for edge devices, featuring AI-powered pose detection on Raspberry Pi and a comprehensive mobile dashboard for tracking your posture health. Appropriately named after the Titan of Greek mythology who carried the world upon his shoulders, the system serves as a digital guardian for spinal alignment and ergonomic health.

![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-c51a4a)
![Model](https://img.shields.io/badge/Model-MoveNet%20Lightning-orange)
![Framework](https://img.shields.io/badge/Framework-Flutter-02569B)
![Python](https://img.shields.io/badge/Python-3.9-3776AB)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Goal](#-project-goal)
- [Dataset & Data Collection](#-dataset--data-collection)
- [Model Pipeline & Workflow](#-model-pipeline--workflow)
- [Deployment Details](#-deployment-details)
- [Features](#-features)
- [System Architecture](#️-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Instructions to Run](#-instructions-to-run)
- [API Documentation](#-api-documentation)
- [Configuration](#️-configuration)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Team Members](#-team-members)

---

## 🎯 Problem Statement

In today's digital age, people spend an average of 8-12 hours daily sitting at desks, often with poor posture. This leads to:

- **Chronic back and neck pain** affecting 80% of office workers
- **Reduced productivity** due to discomfort and fatigue
- **Long-term health issues** including spinal problems, reduced lung capacity, and circulation issues
- **Lack of awareness** - people don't realize their posture is deteriorating until pain develops

Existing solutions are either:
- **Too expensive**: Professional posture analysis requires specialized equipment and clinics
- **Not real-time**: Manual ergonomic assessments are one-time evaluations
- **Intrusive**: Wearable sensors are uncomfortable for extended use
- **Not actionable**: Simple reminders don't provide quantitative feedback

**The Need**: An affordable, non-intrusive, real-time posture monitoring system that provides actionable insights and runs on edge devices for privacy and accessibility.

---

## 🚀 Project Goal

Develop an **intelligent posture monitoring system** that:

1. **Detects posture in real-time** using computer vision and AI on affordable edge hardware (Raspberry Pi)
2. **Provides quantitative feedback** through scientifically-based scoring of neck and torso angles
3. **Tracks user behavior patterns** including focus sessions and prolonged sitting periods
4. **Visualizes trends over time** via a mobile dashboard with actionable analytics
5. **Maintains privacy** by processing all data locally without cloud dependencies
6. **Operates autonomously** with automatic side detection and calibration-free setup

**Success Metrics**:
- Real-time pose detection at ≥15 FPS on Raspberry Pi
- Posture scoring accuracy within 5° of manual measurements
- Mobile app with <2 second data refresh latency
- Complete session tracking with CSV logging every 10 seconds
- Zero-configuration deployment (plug-and-play)

---

## 📊 Dataset & Data Collection

This project uses **real-time data collection** rather than a pre-existing dataset:

### Data Sources

#### 1. **Pose Keypoints** (Primary Input)
- **Source**: Google's MoveNet Lightning TFLite model
- **Format**: 17 COCO keypoints per frame (y, x, confidence)
- **Keypoints**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **Capture Rate**: 30 FPS from Raspberry Pi Camera Module
- **Resolution**: 640x480 pixels (YUV420 format)

#### 2. **Calculated Features** (Derived)
- **Neck Angle**: `ear → shoulder → hip` (at shoulder vertex) - measures forward/backward head lean
- **Torso Angle**: `shoulder → hip → knee` (at hip vertex) - measures slouching/upright position
- **Focus Angle**: `eye → ear → shoulder` (at ear vertex) - tracks head movement for attention detection
- **Idle Angle**: `ankle → knee → hip` (at knee vertex) - detects body position changes

#### 3. **Session Data** (Tracked Over Time)
- **Focus Sessions**: Periods where head remains stable (angle variation <10°)
- **Idle Sessions**: Periods where body remains stationary (angle variation <10°)
- **Minimum Durations**: 5 minutes (focus), 30 minutes (idle)

### Data Logging

All data is stored locally in CSV format:

| File | Frequency | Fields | Purpose |
|------|-----------|--------|---------|
| `logs.csv` | Every 10 seconds | timestamp, overall_score, neck_score, torso_score | Time-series posture analysis |
| `focus.csv` | Session completion | start_time, end_time, time_period | Productivity tracking |
| `idle.csv` | Session completion | start_time, end_time, time_period | Sedentary behavior monitoring |

**Privacy**: All data processing happens on-device. No images or videos are stored - only numerical keypoint coordinates and derived angles.

---

## 🔄 Model Pipeline & Workflow

### Complete System Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI EDGE DEVICE                      │
└──────────────────────────────────────────────────────────────────┘

1️⃣ VIDEO CAPTURE
   ┌─────────────────────┐
   │  Camera Module      │  → Captures 640x480 @ 30fps
   │  (rpicam-vid)       │  → YUV420 format stream
   └──────────┬──────────┘
              │
              ▼
2️⃣ PREPROCESSING
   ┌─────────────────────┐
   │  Frame Processing   │  → Convert YUV420 to BGR
   │                     │  → Resize to 192x192
   └──────────┬──────────┘  → Normalize to uint8
              │
              ▼
3️⃣ POSE ESTIMATION
   ┌─────────────────────┐
   │  MoveNet Lightning  │  → TFLite inference
   │  (TFLite Runtime)   │  → 17 keypoint detection
   └──────────┬──────────┘  → Confidence scores
              │
              ▼
4️⃣ SIDE DETECTION
   ┌─────────────────────┐
   │  Automatic Camera   │  → Sum confidence scores
   │  Position Detection │  → Left vs Right side
   └──────────┬──────────┘  → Select keypoints
              │
              ▼
5️⃣ ANGLE CALCULATION
   ┌─────────────────────┐
   │  Geometric Analysis │  → Neck: ear-shoulder-hip
   │                     │  → Torso: shoulder-hip-knee
   └──────────┬──────────┘  → Focus: eye-ear-shoulder
              │              → Idle: ankle-knee-hip
              ▼
6️⃣ POSTURE SCORING
   ┌─────────────────────┐
   │  Linear Scoring     │  → Compare to optimal angles
   │  Algorithm          │  → Score = f(deviation)
   └──────────┬──────────┘  → Weighted combination
              │
              ▼
7️⃣ SESSION TRACKING
   ┌─────────────────────┐
   │  Temporal Analysis  │  → Track angle stability
   │                     │  → Detect session boundaries
   └──────────┬──────────┘  → Calculate durations
              │
              ▼
8️⃣ DATA PERSISTENCE
   ┌─────────────────────┐
   │  CSV Logging        │  → logs.csv (every 10s)
   │                     │  → focus.csv (sessions)
   └──────────┬──────────┘  → idle.csv (sessions)
              │
              ▼
9️⃣ API SERVING
   ┌─────────────────────┐
   │  Flask REST API     │  → /data endpoint
   │  (Port 5000)        │  → JSON response
   └──────────┬──────────┘  → Safe CSV reading
              │
              └──────────────────┐
                                 │
┌────────────────────────────────▼───────────────────────────────┐
│                    MOBILE APP (FLUTTER)                        │
└────────────────────────────────────────────────────────────────┘

🔟 DATA VISUALIZATION
   ┌─────────────────────┐
   │  HTTP Request       │  → Fetch JSON data
   │                     │  → Parse logs/sessions
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Chart Rendering    │  → Time-series line charts
   │  (fl_chart)         │  → Bar charts for sessions
   └──────────┬──────────┘  → Summary statistics
              │
              ▼
   ┌─────────────────────┐
   │  User Dashboard     │  → Real-time scores
   │                     │  → Trend analysis
   └─────────────────────┘  → Session insights
```

### Algorithm Details

**Posture Scoring Formula**:
```
For each angle (neck, torso):
  if angle < optimal:
    score = (angle - min) / (optimal - min)
  else:
    score = 1.0 - (angle - optimal) / (max - optimal)
  
  score = clamp(score, 0.0, 1.0) × 100

Overall Score = (W_neck × neck_score + W_torso × torso_score)
```

**Session Detection**:
```
Focus Session:
  - Start: When eye-ear-shoulder angle stabilizes
  - Maintain: While angle variation < 10°
  - End: When angle changes > 10° or user moves away
  - Log: If duration ≥ 5 minutes

Idle Session:
  - Start: When ankle-knee-hip angle stabilizes
  - Maintain: While angle variation < 10°
  - End: When user changes position
  - Log: If duration ≥ 30 minutes
```

---

## 🚢 Deployment Details

### Edge Deployment (Raspberry Pi)

**Hardware Configuration**:
- **Device**: Raspberry Pi 5 (8GB RAM)
- **Camera**: Official Camera Module v3 (imx708 sensor)
- **Power**: 5V 3A USB-C power supply
- **Storage**: 32GB microSD card (Class 10)
- **Cooling**: Passive heatsink (optional fan for continuous operation)

**Software Stack**:
- **OS**: Raspberry Pi OS Bookworm (64-bit)
- **Python**: 3.9.19 (virtual environment)
- **ML Runtime**: TFLite Runtime (optimized for ARM)
- **Camera Interface**: rpicam-vid (hardware-accelerated)
- **Display**: X11 (DISPLAY=:0)
- **API Server**: Flask (multi-threaded)

**Performance Metrics**:
- **Inference Speed**: ~20-25 FPS with MoveNet Lightning
- **CPU Usage**: 60-70% on single core
- **Memory**: ~400MB RAM usage
- **Latency**: <50ms per frame (capture → inference → display)
- **Power Consumption**: ~5W idle, ~8W during inference

**Deployment Steps**:
1. Flash Raspberry Pi OS to SD card
2. Enable camera and configure network
3. Clone repository and install dependencies
4. Download pre-trained MoveNet model
5. Run `python main.py` - auto-starts Flask API
6. System displays local IP for mobile app connection

### Mobile Deployment (Android)

**Target Platform**:
- **OS**: Android 5.0+ (API Level 21+)
- **Architecture**: ARM64-v8a, ARMv7, x86_64
- **Screen**: Minimum 5" (responsive layout)

**Build Configuration**:
```bash
flutter build apk --release         # Single APK (all architectures)
flutter build appbundle --release   # For Google Play Store
```

**Deployment Options**:
1. **Development**: `flutter run` for hot-reload testing
2. **Ad-hoc**: Transfer APK via USB/email for direct installation
3. **Enterprise**: Host on internal app distribution platform
4. **Public**: Publish to Google Play Store (future)

**Network Requirements**:
- Mobile device and Raspberry Pi on same WiFi network
- Port 5000 accessible (no firewall blocking)
- Typical latency: <100ms on local network

---

## 🌟 Features

### Raspberry Pi Posture Monitor
- **Real-time Pose Estimation**: Uses Google's MoveNet Lightning model for accurate skeletal tracking
- **Automatic Side Detection**: Intelligently detects camera position (left/right side view)
- **Linear Scoring System**: Sophisticated scoring based on deviation from optimal posture angles
- **Focus Session Tracking**: Monitors concentration periods via head stability detection
- **Idle Session Tracking**: Identifies prolonged sitting/standing periods
- **CSV Data Logging**: Comprehensive logging of posture scores and sessions every 10 seconds
- **Flask REST API**: Serves data to mobile app in real-time
- **Visual Feedback**: Live video display with skeleton overlay and angle annotations
- **Time Acceleration**: Built-in testing mode (30x speed) for rapid development

### Flutter Mobile Dashboard
- **Real-time Analytics**: Beautiful charts visualizing posture trends over time
- **Session Analysis**: Track focus and idle sessions with duration graphs
- **Score Monitoring**: Average overall, neck, and torso score summaries
- **Color-coded Alerts**: Instant visual feedback on posture quality
- **Pull-to-Refresh**: Easy data updates with swipe gesture
- **Configurable API**: Simple setup for connecting to your Raspberry Pi

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│     Raspberry Pi 5                  │
│  ┌──────────────────────────────┐   │
│  │   Camera Module (rpicam-vid) │   │
│  └────────────┬─────────────────┘   │
│               │ YUV420 Stream       │
│               ▼                     │
│  ┌──────────────────────────────┐   │
│  │   MoveNet Pose Estimation    │   │
│  │   (TFLite Runtime)           │   │
│  └────────────┬─────────────────┘   │
│               │ 17 Keypoints        │
│               ▼                     │
│  ┌──────────────────────────────┐   │
│  │   Posture Analysis Engine    │   │
│  │   - Angle calculations       │   │
│  │   - Linear scoring           │   │
│  │   - Session tracking         │   │
│  └────────────┬─────────────────┘   │
│               │                     │
│         ┌─────┴─────┐               │
│         ▼           ▼               │
│   ┌─────────┐  ┌──────────┐         │
│   │   CSV   │  │  Flask   │         │
│   │ Logging │  │   API    │         │
│   └─────────┘  └─────┬────┘         │
│                      │ HTTP         │
└──────────────────────┼──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Mobile App    │
              │  (Flutter)     │
              │  - Charts      │
              │  - Analytics   │
              └────────────────┘
```

## 🔧 Hardware Requirements

### Raspberry Pi System
- **Board**: Raspberry Pi 5 (or Raspberry Pi 4)
- **Camera**: Raspberry Pi Camera Module (tested with imx708 sensor)
- **OS**: Raspberry Pi OS (64-bit recommended)
- **RAM**: Minimum 4GB
- **Storage**: 16GB+ SD card

### Mobile Device
- **Platform**: Android device (iOS support coming soon)
- **OS**: Android 5.0 (API 21) or higher
- **Network**: WiFi connectivity on same network as Raspberry Pi

## 📦 Installation

### 1. Raspberry Pi Setup

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git

# Enable camera
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
```

#### Clone Repository
```bash
cd ~
git clone https://github.com/yourusername/ACM-Hackathon.git
cd ACM-Hackathon
```

#### Create Virtual Environment
```bash
cd src
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install --upgrade pip
pip install opencv-contrib-python
pip install "numpy<2"
pip install tflite-runtime
pip install flask
```

#### Download MoveNet Model
The `model.tflite` file should already be in the `src/` directory. If not, download it:
```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/movenet_lightning_int8.tflite -O model.tflite
```

#### Configure Display (for GUI)
```bash
export DISPLAY=:0
```

Add to `~/.bashrc` for persistence:
```bash
echo 'export DISPLAY=:0' >> ~/.bashrc
```

### 2. Mobile App Setup

#### Prerequisites
- Install [Flutter SDK](https://docs.flutter.dev/get-started/install) (version 3.7.2+)
- Setup Android development environment
- Connect Android device or setup emulator

#### Install Dependencies
```bash
cd ~/ACM-Hackathon/app/posture
flutter pub get
```

#### Configure API URL (First Run)
You'll set this in the app, but note your Raspberry Pi's IP address:
```bash
# On Raspberry Pi, find IP address
hostname -I
```

---

## 📖 Instructions to Run

### Complete Setup and Execution Guide

#### **Step 1: Raspberry Pi Initial Setup**

1. **Flash Raspberry Pi OS**:
   ```bash
   # Download Raspberry Pi Imager from raspberrypi.com
   # Flash Raspberry Pi OS (64-bit) to microSD card
   # Insert card and boot Raspberry Pi
   ```

2. **Enable Camera**:
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options → Camera → Enable → Reboot
   ```

3. **Update System**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3-pip python3-venv git
   ```

#### **Step 2: Clone and Setup Raspberry Pi Project**

1. **Clone Repository**:
   ```bash
   cd ~
   git clone https://github.com/yourusername/ACM-Hackathon.git
   cd ACM-Hackathon/src
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install opencv-contrib-python "numpy<2" tflite-runtime flask
   ```

4. **Verify Model Exists**:
   ```bash
   ls model.tflite  # Should show the file
   ```

#### **Step 3: Run Posture Monitor**

1. **Start the Application**:
   ```bash
   cd ~/ACM-Hackathon/src
   source venv/bin/activate
   python main.py
   ```

2. **Note the API URL** displayed in terminal:
   ```
   ============================================================
     Flask API Server Running
     Access API at: http://192.168.1.XXX:5000/data
     Home: http://192.168.1.XXX:5000/
   ============================================================
   ```

3. **Position Yourself**:
   - Sit sideways to the camera (profile view)
   - Ensure visibility: eye, ear, shoulder, hip, knee, ankle
   - System auto-detects which side (LEFT/RIGHT)

4. **Monitor Live Feedback**:
   - Green overlay = Good posture (score ≥75)
   - Orange overlay = Fair posture (50-74)
   - Red overlay = Poor posture (<50)

5. **Keyboard Controls**:
   - `q` - Quit application
   - `s` - Save screenshot
   - `Ctrl+C` - Stop gracefully

#### **Step 4: Setup Mobile App**

1. **Install Flutter** (if not already installed):
   ```bash
   # Follow: https://docs.flutter.dev/get-started/install
   # Verify installation
   flutter doctor
   ```

2. **Navigate to App Directory**:
   ```bash
   cd ~/ACM-Hackathon/app/posture
   ```

3. **Install Dependencies**:
   ```bash
   flutter pub get
   ```

4. **Connect Android Device**:
   ```bash
   # Enable Developer Options and USB Debugging on Android
   # Connect via USB cable
   # Verify connection
   flutter devices
   ```

#### **Step 5: Run Mobile App**

1. **Launch in Development Mode**:
   ```bash
   flutter run
   ```

2. **Configure API Connection** (in app):
   - Tap ⚙️ Settings icon
   - Enter: `http://192.168.1.XXX:5000/data` (use IP from Step 3)
   - Tap "Connect"

3. **View Dashboard**:
   - Summary cards show average scores
   - Scroll to see time-series charts
   - Pull down to refresh data

#### **Step 6: Build Production APK** (Optional)

```bash
cd ~/ACM-Hackathon/app/posture
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

**Install on Device**:
```bash
adb install build/app/outputs/flutter-apk/app-release.apk
```

#### **Troubleshooting Common Issues**

| Issue | Solution |
|-------|----------|
| Camera not working | Run `libcamera-hello` to test, ensure enabled in raspi-config |
| Import errors | Activate venv: `source venv/bin/activate` |
| Low FPS | Reduce resolution to 320x240 in `main.py` |
| Cannot connect to API | Verify same WiFi network, check IP address |
| Flutter build fails | Run `flutter clean && flutter pub get` |

---

## 🚀 Usage

### Starting the Raspberry Pi Monitor

1. **Activate virtual environment:**
   ```bash
   cd ~/ACM-Hackathon/src
   source venv/bin/activate
   ```

2. **Run the posture monitor:**
   ```bash
   python main.py
   ```

3. **Note the API URL displayed:**
   ```
   ============================================================
     Flask API Server Running
     Access API at: http://192.168.1.XXX:5000/data
     Home: http://192.168.1.XXX:5000/
   ============================================================
   ```

4. **Position yourself:**
   - Sit in your chair in profile view (side view)
   - Ensure camera can see: eye, ear, shoulder, hip, knee, ankle
   - System will auto-detect which side (LEFT/RIGHT)

5. **Monitor your posture:**
   - Green score (≥75): Good posture ✅
   - Orange score (50-74): Fair posture ⚠️
   - Red score (<50): Poor posture ❌

6. **Keyboard controls:**
   - Press `q`: Quit application
   - Press `s`: Save screenshot
   - Press `Ctrl+C`: Stop gracefully

### Running the Mobile App

1. **Launch app:**
   ```bash
   cd ~/ACM-Hackathon/app/posture
   flutter run
   ```

2. **First-time setup:**
   - Tap the ⚙️ (Settings) icon in top-right
   - Enter the API URL from Raspberry Pi: `http://192.168.1.XXX:5000/data`
   - Tap "Connect"

3. **View your data:**
   - Summary cards show average scores
   - Scroll down to see time-series charts
   - Pull down to refresh data
   - Tap refresh icon for manual updates

### Building Android APK

```bash
cd ~/ACM-Hackathon/app/posture
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

Install on device:
```bash
adb install build/app/outputs/flutter-apk/app-release.apk
```

## 📡 API Documentation

### Base URL
```
http://<raspberry-pi-ip>:5000
```

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```
Posture Monitor API Running 🚀
```

#### `GET /data`
Retrieve all posture monitoring data.

**Response Format:**
```json
{
  "logs": [
    {
      "timestamp": "2026-01-04 14:30:00",
      "overall_score": "85.50",
      "neck_score": "82.30",
      "torso_score": "88.70"
    }
  ],
  "focus": [
    {
      "start_time": "2026-01-04 14:00:00",
      "end_time": "2026-01-04 14:15:30",
      "time_period": "930.0"
    }
  ],
  "idle": [
    {
      "start_time": "2026-01-04 13:00:00",
      "end_time": "2026-01-04 13:45:00",
      "time_period": "2700.0"
    }
  ]
}
```

**Fields:**
- `logs`: Posture scores logged every 10 seconds
  - `timestamp`: When the measurement was taken
  - `overall_score`: Combined posture score (0-100)
  - `neck_score`: Neck angle score (0-100)
  - `torso_score`: Torso angle score (0-100)

- `focus`: Completed focus sessions (≥5 minutes)
  - `start_time`: Session start timestamp
  - `end_time`: Session end timestamp
  - `time_period`: Duration in seconds (accelerated by TIME_RATE)

- `idle`: Completed idle sessions (≥30 minutes)
  - `start_time`: Session start timestamp
  - `end_time`: Session end timestamp
  - `time_period`: Duration in seconds (accelerated by TIME_RATE)

## ⚙️ Configuration

### Raspberry Pi Configuration

Edit `src/main.py` to customize:

```python
# Video Settings
WIDTH = 640          # Camera resolution width
HEIGHT = 480         # Camera resolution height
FRAMERATE = 30       # Frames per second

# Posture Angle Bounds
NECK_OPTIMAL = 180.0   # Perfect neck angle (ear-shoulder-hip)
NECK_MIN = 130.0       # Minimum acceptable
NECK_MAX = 200.0       # Maximum acceptable

TORSO_OPTIMAL = 90.0   # Perfect torso angle (shoulder-hip-knee)
TORSO_MIN = 45.0       # Minimum acceptable
TORSO_MAX = 135.0      # Maximum acceptable

# Scoring Weights
W_NECK = 0.5           # Neck score weight (0.0-1.0)
W_TORSO = 0.5          # Torso score weight (0.0-1.0)
SCORE_THRESHOLD = 75.0 # Good posture threshold

# Session Tracking
FOCUS_MIN_TIME = 5 * 60        # Minimum focus session (seconds)
IDLE_ALERT_TIME = 30 * 60      # Idle alert threshold (seconds)
BAD_POSTURE_ALERT_TIME = 10.0  # Bad posture alert (seconds)

# Testing
TIME_RATE = 30.0       # Time acceleration (1.0 = real-time, 30.0 = 30x faster)
```

### Mobile App Configuration

The API URL can be changed in-app via the Settings icon, or hardcode in `lib/main.dart`:

```dart
String apiUrl = 'http://192.168.1.100:5000/data';
```

## 📊 Data Logging

All data is stored in CSV format in `src/outputs/`:

### logs.csv
Posture measurements every 10 seconds:
```csv
timestamp,overall_score,neck_score,torso_score
2026-01-04 14:30:00,85.50,82.30,88.70
2026-01-04 14:30:10,87.20,85.10,89.30
```

### focus.csv
Completed focus sessions (≥5 minutes):
```csv
start_time,end_time,time_period
2026-01-04 14:00:00,2026-01-04 14:15:30,930.0
```

### idle.csv
Completed idle sessions (≥30 minutes):
```csv
start_time,end_time,time_period
2026-01-04 13:00:00,2026-01-04 13:45:00,2700.0
```

## 🔬 Technical Details

### Pose Estimation
- **Model**: MoveNet Lightning (192x192 input)
- **Output**: 17 COCO keypoints with confidence scores
- **Keypoints**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **Inference**: TFLite Runtime (optimized for edge devices)

### Posture Analysis Algorithm

1. **Side Detection**: Automatically determines camera position by comparing keypoint confidence scores

2. **Angle Calculation**: 
   - Neck angle: `ear → shoulder → hip` (at shoulder vertex)
   - Torso angle: `shoulder → hip → knee` (at hip vertex)
   - Focus angle: `eye → ear → shoulder` (at ear vertex)

3. **Linear Scoring**:
   ```
   If angle < optimal:
     score = (angle - min) / (optimal - min)
   
   If angle > optimal:
     score = 1.0 - (angle - optimal) / (max - optimal)
   
   Overall = (W_NECK × neck_score + W_TORSO × torso_score) × 100
   ```

4. **Session Tracking**:
   - **Focus**: Detected when eye-ear-shoulder angle varies < 10° (head stable)
   - **Idle**: Detected when ankle-knee-hip angle varies < 10° (body stable)
   - Sessions logged when ≥ minimum duration and broken by angle change

### Mobile App Architecture
- **State Management**: StatefulWidget with setState
- **HTTP Client**: `http` package for API calls
- **Charts**: `fl_chart` for beautiful visualizations
- **UI**: Material Design 3 with adaptive theming

## 🐛 Troubleshooting

### Raspberry Pi Issues

**Camera not detected:**
```bash
# Check camera is enabled
sudo raspi-config
# Interface Options -> Camera -> Enable

# Test camera
libcamera-hello
```

**Low FPS:**
- Reduce resolution in `main.py` (try 320x240)
- Close other applications
- Ensure adequate power supply (5V 3A for Pi 5)

**Import errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall opencv-contrib-python "numpy<2" tflite-runtime flask
```

**Display not working:**
```bash
# Set display variable
export DISPLAY=:0

# Or run headless (comment out cv2.imshow and cv2.waitKey lines)
```

### Mobile App Issues

**Cannot connect to API:**
- Verify both devices on same WiFi network
- Check Raspberry Pi IP address: `hostname -I`
- Test API in browser: `http://<pi-ip>:5000/data`
- Disable firewall on Raspberry Pi: `sudo ufw disable`

**No data showing:**
- Ensure `main.py` is running on Raspberry Pi
- Check CSV files exist: `ls ~/ACM-Hackathon/src/outputs/`
- Try manual refresh (pull down gesture)

**Build errors:**
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter build apk
```

### Network Configuration

If using different network setups:

**Port forwarding (for external access):**
```bash
# On router, forward port 5000 to Raspberry Pi IP
```

**Change Flask port** (if 5000 is occupied):
Edit `src/main.py`:
```python
app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
```

## 📈 Future Enhancements

- [ ] iOS app support
- [ ] Real-time notifications on mobile
- [ ] Historical trend analysis (weekly/monthly)
- [ ] Posture correction exercises recommendations
- [ ] Multi-user support with profiles
- [ ] Cloud sync for cross-device access
- [ ] Machine learning for personalized baselines
- [ ] Haptic feedback alerts
- [ ] Integration with fitness trackers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Google MoveNet team for the pose estimation model
- Raspberry Pi Foundation for the amazing hardware
- Flutter team for the cross-platform framework
- fl_chart library for beautiful visualizations

---

## 👥 Team Members

1. Pranav M R
2. Shreyas Sankaran
3. Ashwin Prasanth Hariharan
4. Veningston K
5. Ronalda M

---

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for better posture and healthier work habits**
