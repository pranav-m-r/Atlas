import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import math
import os
import csv
from datetime import datetime
from flask import Flask, jsonify
import threading

# ============================================================
# ========================= CONFIG ============================
# ============================================================

WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

MIN_KP_CONF = 0.3

# Output directory for CSV logs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
LOGS_CSV = os.path.join(OUTPUT_DIR, 'logs.csv')
FOCUS_CSV = os.path.join(OUTPUT_DIR, 'focus.csv')
IDLE_CSV = os.path.join(OUTPUT_DIR, 'idle.csv')

# Time acceleration for testing (1.0 = real-time, 10.0 = 10x faster)
TIME_RATE = 30.0

BAD_POSTURE_ALERT_TIME = 10.0
IDLE_ALERT_TIME = 30 * 60  # 30 minutes of continuous sitting/standing
FOCUS_MIN_TIME = 5 * 60

# Absolute angle bounds (180° = perfect posture, min/max = score of 0)
# Neck angle bounds (ear-shoulder-hip)
NECK_MIN = 130.0   # Below this = score 0
NECK_MAX = 200.0   # Above this = score 0
NECK_OPTIMAL = 180.0  # Perfect posture

# Torso angle bounds (shoulder-hip-knee)
TORSO_MIN = 45.0   # Below this = score 0
TORSO_MAX = 135.0  # Above this = score 0
TORSO_OPTIMAL = 90.0  # Perfect posture

EYE_EAR_SHOULDER_ANGLE_THRESH = 10.0  # degrees change to count as head movement
ANKLE_KNEE_HIP_ANGLE_THRESH = 10.0  # degrees change to count as position change

# weights for scoring
W_NECK = 0.5
W_TORSO = 0.5
SCORE_THRESHOLD = 75.0  # Overall score threshold for GOOD posture

os.environ['DISPLAY'] = ':0'

# Keypoint indices
# Left side: eye=1, ear=3, shoulder=5, hip=11, knee=13, ankle=15
# Right side: eye=2, ear=4, shoulder=6, hip=12, knee=14, ankle=16
LEFT_INDICES = {'eye': 1, 'ear': 3, 'shoulder': 5, 'hip': 11, 'knee': 13, 'ankle': 15}
RIGHT_INDICES = {'eye': 2, 'ear': 4, 'shoulder': 6, 'hip': 12, 'knee': 14, 'ankle': 16}

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# ============================================================
# ========================= UTILS =============================
# ============================================================

def draw_text(frame, text, x, y, color=(255,255,255), scale=0.6, thickness=2):
    """Draw text with translucent black background for better visibility"""
    # Get text size to create background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Create semi-transparent black background
    padding = 4
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                (0, 0, 0), -1)
    
    # Blend overlay with original frame (0.6 = 60% transparency)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text on top
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)

def draw_skeleton(frame, keypoints, connections=SKELETON_CONNECTIONS, conf_thresh=MIN_KP_CONF):
    h, w = frame.shape[:2]
    for start, end in connections:
        if keypoints[start][2] > conf_thresh and keypoints[end][2] > conf_thresh:
            y0, x0 = keypoints[start][:2]
            y1, x1 = keypoints[end][:2]
            cv2.line(frame, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), (255,0,0), 2)
    for i, (y, x, c) in enumerate(keypoints):
        if c > conf_thresh:
            cv2.circle(frame, (int(x*w), int(y*h)), 5, (0,255,0), -1)
    return frame

def calculate_angle(p1, p2, p3):
    """
    Calculate signed angle at p2 formed by points p1-p2-p3
    Returns angle in degrees (0-360)
    
    For side view (ear-shoulder-hip):
    - Small angles (0-90): Forward lean (ear in front)
    - ~180: Neutral/straight posture
    - Large angles (270-360): Backward lean (ear behind)
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Calculate angle using atan2 for full 360-degree range
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    
    # Calculate the angle difference
    angle = angle2 - angle1
    
    # Normalize to 0-360 range
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

def get_pixel_coords(kp, w, h):
    """Convert normalized coords to pixel coords"""
    return (int(kp[0] * w), int(kp[1] * h))  # (y_px, x_px)

# ============================================================
# ================== SIDE DETECTION ===========================
# ============================================================

def detect_side(keypoints):
    """
    Determine if camera is on LEFT or RIGHT side of person.
    If camera is on person's LEFT, we see their LEFT eye/ear/shoulder/hip better.
    If camera is on person's RIGHT, we see their RIGHT eye/ear/shoulder/hip better.
    """
    left_conf = (keypoints[LEFT_INDICES['eye']][2] + 
                keypoints[LEFT_INDICES['ear']][2] + 
                keypoints[LEFT_INDICES['shoulder']][2] + 
                keypoints[LEFT_INDICES['hip']][2] +
                keypoints[LEFT_INDICES['knee']][2])
    
    right_conf = (keypoints[RIGHT_INDICES['eye']][2] + 
                keypoints[RIGHT_INDICES['ear']][2] + 
                keypoints[RIGHT_INDICES['shoulder']][2] + 
                keypoints[RIGHT_INDICES['hip']][2] +
                keypoints[RIGHT_INDICES['knee']][2])
    
    return 'LEFT' if left_conf >= right_conf else 'RIGHT'

def get_side_keypoints(keypoints, side):
    """Get the 6 key points (eye, ear, shoulder, hip, knee, ankle) for the detected side"""
    indices = LEFT_INDICES if side == 'LEFT' else RIGHT_INDICES
    return {
        'eye': keypoints[indices['eye']],
        'ear': keypoints[indices['ear']],
        'shoulder': keypoints[indices['shoulder']],
        'hip': keypoints[indices['hip']],
        'knee': keypoints[indices['knee']],
        'ankle': keypoints[indices['ankle']]
    }

# ============================================================
# ================== MONITOR CLASS ============================
# ============================================================

class PostureMonitor:
    def __init__(self):
        self.bad_start = None
        self.last_eye_ear_shoulder_angle = None
        self.focus_session_start = None  # Track when current focus session started
        self.last_ankle_knee_hip_angle = None
        self.idle_session_start = None  # Track when current idle session started
        self.detected_side = None
        
        # Track completed sessions for CSV logging
        self.last_completed_focus = None  # (start_time, end_time, duration)
        self.last_completed_idle = None  # (start_time, end_time, duration)

    def update(self, keypoints):
        now = time.time()
        
        # Detect which side the camera is on
        self.detected_side = detect_side(keypoints)
        side_kps = get_side_keypoints(keypoints, self.detected_side)
        
        # Check if we have valid keypoints
        if (side_kps['ear'][2] < MIN_KP_CONF or 
            side_kps['shoulder'][2] < MIN_KP_CONF or 
            side_kps['hip'][2] < MIN_KP_CONF or
            side_kps['knee'][2] < MIN_KP_CONF):
            return None, False, False, False, side_kps
        
        ear = side_kps['ear'][:2]
        eye = side_kps['eye'][:2]
        shoulder = side_kps['shoulder'][:2]
        hip = side_kps['hip'][:2]
        knee = side_kps['knee'][:2]
        
        # Calculate current 3-point angles (absolute values)
        # 1) Neck: ear-shoulder-hip (angle at shoulder)
        neck_angle = calculate_angle(ear, shoulder, hip)
        
        # 2) Torso: shoulder-hip-knee (angle at hip)
        torso_angle = calculate_angle(shoulder, hip, knee)
        
        # 3) Focus: eye-ear-shoulder (angle at ear)
        eye_ear_shoulder_angle = calculate_angle(eye, ear, shoulder)
        
        # Linear scoring based on deviation from optimal angle (180° for neck, 90° for torso)
        # Score = 1.0 at optimal, 0.0 at min/max bounds
        
        # Neck score: linear interpolation from min/optimal/max
        if neck_angle <= NECK_MIN or neck_angle >= NECK_MAX:
            s_neck = 0.0
        elif neck_angle < NECK_OPTIMAL:
            # Between min and optimal: linear scale from 0 to 1
            s_neck = (neck_angle - NECK_MIN) / (NECK_OPTIMAL - NECK_MIN)
        else:
            # Between optimal and max: linear scale from 1 to 0
            s_neck = 1.0 - (neck_angle - NECK_OPTIMAL) / (NECK_MAX - NECK_OPTIMAL)
        
        # Torso score: linear interpolation from min/optimal/max
        if torso_angle <= TORSO_MIN or torso_angle >= TORSO_MAX:
            s_torso = 0.0
        elif torso_angle < TORSO_OPTIMAL:
            # Between min and optimal: linear scale from 0 to 1
            s_torso = (torso_angle - TORSO_MIN) / (TORSO_OPTIMAL - TORSO_MIN)
        else:
            # Between optimal and max: linear scale from 1 to 0
            s_torso = 1.0 - (torso_angle - TORSO_OPTIMAL) / (TORSO_MAX - TORSO_OPTIMAL)
        
        # Clamp scores to [0, 1]
        s_neck = max(0.0, min(1.0, s_neck))
        s_torso = max(0.0, min(1.0, s_torso))
        
        # Overall score
        score = (W_NECK * s_neck + W_TORSO * s_torso) * 100
        classification = "GOOD" if score >= SCORE_THRESHOLD else "BAD"
        
        # Determine reasons for bad posture (only show if outside bounds)
        reasons = []
        if neck_angle < NECK_MIN:
            reasons.append(f"Neck Too Forward (angle: {neck_angle:.1f}°)")
        elif neck_angle > NECK_MAX:
            reasons.append(f"Neck Too Backward (angle: {neck_angle:.1f}°)")
        elif neck_angle < NECK_OPTIMAL - 25:  # Significant deviation forward
            reasons.append(f"Neck Forward (angle: {neck_angle:.1f}°)")
        elif neck_angle > NECK_OPTIMAL + 10:  # Significant deviation backward
            reasons.append(f"Neck Backward (angle: {neck_angle:.1f}°)")
        
        if torso_angle < TORSO_MIN:
            reasons.append(f"Torso Too Slouched (angle: {torso_angle:.1f}°)")
        elif torso_angle > TORSO_MAX:
            reasons.append(f"Torso Too Far Back (angle: {torso_angle:.1f}°)")
        elif torso_angle < TORSO_OPTIMAL - 20:  # Significant deviation forward
            reasons.append(f"Torso Slouched (angle: {torso_angle:.1f}°)")
        elif torso_angle > TORSO_OPTIMAL + 20:  # Significant deviation backward
            reasons.append(f"Torso Leaning Back (angle: {torso_angle:.1f}°)")
        
        # Bad posture alert
        bad = score < SCORE_THRESHOLD
        if bad:
            self.bad_start = self.bad_start or now
        else:
            self.bad_start = None
        bad_alert = self.bad_start and ((now - self.bad_start) * TIME_RATE > BAD_POSTURE_ALERT_TIME)
        
        # Idle detection based on ankle-knee-hip angle stability
        # If angle variation is WITHIN threshold, user is idle (sitting/standing still)
        # If angle variation EXCEEDS threshold, idle session is broken (position changed)
        idle_duration = 0.0
        
        # Check if we have valid ankle keypoint for idle tracking
        if side_kps['ankle'][2] >= MIN_KP_CONF:
            ankle = side_kps['ankle'][:2]
            ankle_knee_hip_angle = calculate_angle(ankle, knee, hip)
            
            if self.last_ankle_knee_hip_angle is not None:
                angle_variation = abs(ankle_knee_hip_angle - self.last_ankle_knee_hip_angle)
                
                if angle_variation > ANKLE_KNEE_HIP_ANGLE_THRESH:
                    # Position changed significantly - break idle session
                    if self.idle_session_start is not None:
                        # Calculate real duration (without TIME_RATE acceleration)
                        real_duration = now - self.idle_session_start
                        # Check if session was long enough to log (using accelerated time)
                        if real_duration * TIME_RATE >= IDLE_ALERT_TIME:
                            self.last_completed_idle = (self.idle_session_start, now, real_duration)
                    self.idle_session_start = None
                else:
                    # Position stable - maintain or start idle session
                    if self.idle_session_start is None:
                        self.idle_session_start = now
            else:
                # First detection - start tracking
                if self.idle_session_start is None:
                    self.idle_session_start = now
            
            self.last_ankle_knee_hip_angle = ankle_knee_hip_angle
        else:
            # Ankle not detected - cannot track idle state
            # Don't break session, just don't update (user might be temporarily occluded)
            pass
        
        # Calculate current idle session duration (accelerated by TIME_RATE)
        if self.idle_session_start is not None:
            idle_duration = (now - self.idle_session_start) * TIME_RATE
        
        # Focus detection based on eye-ear-shoulder angle stability
        # If angle variation is WITHIN threshold, user is focusing
        # If angle variation EXCEEDS threshold, focus session is broken
        focus_duration = 0.0
        if self.last_eye_ear_shoulder_angle is not None:
            angle_variation = abs(eye_ear_shoulder_angle - self.last_eye_ear_shoulder_angle)
            
            if angle_variation > EYE_EAR_SHOULDER_ANGLE_THRESH:
                # Eye level changed significantly - break focus session
                if self.focus_session_start is not None:
                    # Calculate real duration (without TIME_RATE acceleration)
                    real_duration = now - self.focus_session_start
                    # Check if session was long enough to log (using accelerated time)
                    if real_duration * TIME_RATE >= FOCUS_MIN_TIME:
                        self.last_completed_focus = (self.focus_session_start, now, real_duration)
                self.focus_session_start = None
            else:
                # Eye level stable - maintain or start focus session
                if self.focus_session_start is None:
                    self.focus_session_start = now
        else:
            # First detection - start tracking
            if self.focus_session_start is None:
                self.focus_session_start = now
        
        self.last_eye_ear_shoulder_angle = eye_ear_shoulder_angle
        
        # Calculate current focus session duration (accelerated by TIME_RATE)
        if self.focus_session_start is not None:
            focus_duration = (now - self.focus_session_start) * TIME_RATE
        
        data = {
            "score": score,
            "classification": classification,
            "subscores": {"Neck": s_neck * 100, "Torso": s_torso * 100},
            "reasons": reasons,
            "neck_angle": neck_angle,
            "torso_angle": torso_angle,
            "eye_ear_shoulder_angle": eye_ear_shoulder_angle,
            "focus_duration": focus_duration,
            "idle_duration": idle_duration
        }
        
        return data, bad_alert, idle_duration >= IDLE_ALERT_TIME, focus_duration >= FOCUS_MIN_TIME, side_kps

# ============================================================
# =================== CSV LOGGING FUNCTIONS ==================
# ============================================================

def init_csv_files():
    """Initialize CSV files with headers if they don't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize logs.csv
    if not os.path.exists(LOGS_CSV):
        with open(LOGS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'overall_score', 'neck_score', 'torso_score'])
    
    # Initialize focus.csv
    if not os.path.exists(FOCUS_CSV):
        with open(FOCUS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'time_period'])
    
    # Initialize idle.csv
    if not os.path.exists(IDLE_CSV):
        with open(IDLE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'time_period'])

def log_scores(timestamp, overall_score, neck_score, torso_score):
    """Append a score entry to logs.csv"""
    with open(LOGS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            f"{overall_score:.2f}",
            f"{neck_score:.2f}",
            f"{torso_score:.2f}"
        ])

def log_focus_session(start_time, end_time, duration):
    """Append a completed focus session to focus.csv
    duration should already be the real duration (not accelerated)
    """
    with open(FOCUS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        # Scale duration by TIME_RATE for logging
        scaled_duration = duration * TIME_RATE
        writer.writerow([
            datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            f"{scaled_duration:.1f}"  # Duration in seconds (scaled)
        ])

def log_idle_session(start_time, end_time, duration):
    """Append a completed idle session to idle.csv
    duration should already be the real duration (not accelerated)
    """
    with open(IDLE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        # Scale duration by TIME_RATE for logging
        scaled_duration = duration * TIME_RATE
        writer.writerow([
            datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            f"{scaled_duration:.1f}"  # Duration in seconds (scaled)
        ])

# ============================================================
# ===================== FLASK API SERVER =====================
# ============================================================

app = Flask(__name__)

@app.route("/")
def home():
    return "Posture Monitor API Running 🚀"

@app.route("/data")
def get_data():
    """Return all CSV data as JSON"""
    data = {
        "logs": [],
        "focus": [],
        "idle": []
    }
    
    def safe_read_csv(file_path):
        """Safely read CSV file, handling concurrent access and corrupted data"""
        rows = []
        try:
            if os.path.exists(file_path):
                # Read file content first to avoid concurrent access issues
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Remove any NUL bytes
                content = content.replace('\x00', '')
                
                # Parse CSV from cleaned content
                lines = content.strip().split('\n')
                if len(lines) > 1:  # Has header + at least one row
                    reader = csv.DictReader(lines)
                    for row in reader:
                        try:
                            # Only add valid rows (non-empty)
                            if any(row.values()):
                                rows.append(row)
                        except Exception:
                            continue  # Skip malformed rows
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return rows
    
    # Read all CSV files safely
    data["logs"] = safe_read_csv(LOGS_CSV)
    data["focus"] = safe_read_csv(FOCUS_CSV)
    data["idle"] = safe_read_csv(IDLE_CSV)
    
    return jsonify(data)

def run_flask_server():
    """Run Flask server in background thread"""
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ============================================================
# ===================== MOVENET INFERENCE ====================
# ============================================================

def load_model(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def preprocess(frame, size):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[np.newaxis].astype(np.uint8)

def infer(interp, inp):
    i = interp.get_input_details()[0]["index"]
    o = interp.get_output_details()[0]["index"]
    interp.set_tensor(i, inp)
    interp.invoke()
    return interp.get_tensor(o)[0][0]

# ============================================================
# ========================== MAIN ============================
# ============================================================

def main():
    print("Loading MoveNet model...")
    interpreter = load_model("model.tflite")
    input_size = interpreter.get_input_details()[0]["shape"][1]
    print(f"Model input size: {input_size}x{input_size}")
    
    # Initialize CSV files
    print("Initializing CSV logging...")
    init_csv_files()
    print(f"Logs directory: {OUTPUT_DIR}")
    
    # Start Flask API server in background thread
    print("Starting Flask API server on port 5000...")
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    # Get local IP address for API access
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"\n{'='*60}")
        print(f"  Flask API Server Running")
        print(f"  Access API at: http://{local_ip}:5000/data")
        print(f"  Home: http://{local_ip}:5000/")
        print(f"{'='*60}\n")
    except:
        print("  API available at http://localhost:5000/data")
    
    monitor = PostureMonitor()
    
    # Start rpicam-vid process
    print(f"Starting video stream at {WIDTH}x{HEIGHT} @ {FRAMERATE}fps...")
    cmd = [
        "rpicam-vid", "-t", "0", "--inline", "--nopreview",
        "--codec", "yuv420", "--width", str(WIDTH), "--height", str(HEIGHT),
        "--framerate", str(FRAMERATE), "-o", "-"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=WIDTH * HEIGHT * 3
    )
    frame_size = WIDTH * HEIGHT * 3 // 2  # YUV420
    
    print("\nSide Camera Posture Monitor started")
    print("Linear scoring based on deviation from optimal angles")
    print(f"Neck: optimal={NECK_OPTIMAL}°, range={NECK_MIN}-{NECK_MAX}°")
    print(f"Torso: optimal={TORSO_OPTIMAL}°, range={TORSO_MIN}-{TORSO_MAX}°")
    print("Press Ctrl+C to stop\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
    last_log_time = start_time  # Track last time we logged to CSV
    fps = 0
    
    try:
        while True:
            # Read YUV420 frame
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                print("Warning: Incomplete frame")
                break
            
            # Convert YUV to BGR
            yuv = np.frombuffer(raw, np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            # Run pose estimation
            keypoints = infer(interpreter, preprocess(frame, input_size))
            
            # Draw full skeleton
            frame = draw_skeleton(frame, keypoints)
            
            # Update monitor
            result = monitor.update(keypoints)
            data, bad_alert, idle_alert, focused, side_kps = result
            
            # ==== CSV LOGGING ====
            # Log scores to logs.csv every 10 seconds (if valid data)
            if data and (current_time - last_log_time >= 10.0):
                log_scores(
                    current_time,
                    data['score'],
                    data['subscores']['Neck'],
                    data['subscores']['Torso']
                )
                last_log_time = current_time
            
            # Log completed focus sessions
            if monitor.last_completed_focus:
                start, end, duration = monitor.last_completed_focus
                log_focus_session(start, end, duration)
                scaled_duration = duration * TIME_RATE
                print(f"[LOG] Focus session completed: {scaled_duration:.1f}s (real: {duration:.1f}s)")
                monitor.last_completed_focus = None  # Clear after logging
            
            # Log completed idle sessions
            if monitor.last_completed_idle:
                start, end, duration = monitor.last_completed_idle
                log_idle_session(start, end, duration)
                scaled_duration = duration * TIME_RATE
                print(f"[LOG] Idle session completed: {scaled_duration:.1f}s (real: {duration:.1f}s)")
                monitor.last_completed_idle = None  # Clear after logging
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time
            
            # ==== DRAW SIDE INDICATOR ====
            side_color = (255, 255, 0)  # Cyan
            draw_text(frame, f"View: {monitor.detected_side} SIDE", WIDTH - 180, 30, side_color, 0.7)
            
            # ==== ANNOTATE THE 6 KEY POINTS ====
            h, w = frame.shape[:2]
            kp_colors = {
                'eye': (255, 0, 255),      # Magenta
                'ear': (0, 255, 255),      # Yellow
                'shoulder': (255, 128, 0), # Orange
                'hip': (0, 128, 255),      # Light blue
                'knee': (128, 0, 255),     # Purple
                'ankle': (255, 192, 203)   # Pink
            }
            
            for name, kp in side_kps.items():
                if kp[2] > MIN_KP_CONF:
                    px = int(kp[1] * w)
                    py = int(kp[0] * h)
                    color = kp_colors[name]
                    # Draw larger circle for key points
                    cv2.circle(frame, (px, py), 10, color, 3)
                    # Label the point
                    label = f"{name.upper()}"
                    cv2.putText(frame, label, (px + 15, py + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw lines connecting key points
            # Neck line: ear-shoulder-hip
            if (side_kps['ear'][2] > MIN_KP_CONF and 
                side_kps['shoulder'][2] > MIN_KP_CONF and 
                side_kps['hip'][2] > MIN_KP_CONF):
                ear_px = (int(side_kps['ear'][1] * w), int(side_kps['ear'][0] * h))
                shoulder_px = (int(side_kps['shoulder'][1] * w), int(side_kps['shoulder'][0] * h))
                hip_px = (int(side_kps['hip'][1] * w), int(side_kps['hip'][0] * h))
                cv2.line(frame, ear_px, shoulder_px, (0, 255, 255), 3)  # Ear-shoulder (yellow)
                cv2.line(frame, shoulder_px, hip_px, (0, 255, 255), 3)  # Shoulder-hip (yellow)
                
            # Torso line: shoulder-hip-knee
            if (side_kps['shoulder'][2] > MIN_KP_CONF and 
                side_kps['hip'][2] > MIN_KP_CONF and 
                side_kps['knee'][2] > MIN_KP_CONF):
                shoulder_px = (int(side_kps['shoulder'][1] * w), int(side_kps['shoulder'][0] * h))
                hip_px = (int(side_kps['hip'][1] * w), int(side_kps['hip'][0] * h))
                knee_px = (int(side_kps['knee'][1] * w), int(side_kps['knee'][0] * h))
                cv2.line(frame, shoulder_px, hip_px, (255, 128, 0), 3)  # Shoulder-hip (orange)
                cv2.line(frame, hip_px, knee_px, (255, 128, 0), 3)  # Hip-knee (orange)
                
            # Focus line: eye-ear-shoulder
            if (side_kps['eye'][2] > MIN_KP_CONF and 
                side_kps['ear'][2] > MIN_KP_CONF and 
                side_kps['shoulder'][2] > MIN_KP_CONF):
                eye_px = (int(side_kps['eye'][1] * w), int(side_kps['eye'][0] * h))
                ear_px = (int(side_kps['ear'][1] * w), int(side_kps['ear'][0] * h))
                shoulder_px = (int(side_kps['shoulder'][1] * w), int(side_kps['shoulder'][0] * h))
                cv2.line(frame, eye_px, ear_px, (255, 0, 255), 2)  # Eye-ear (magenta)
                cv2.line(frame, ear_px, shoulder_px, (255, 0, 255), 2)  # Ear-shoulder (magenta)
            
            # ==== ANNOTATIONS ====
            if data:
                color = (0, 255, 0) if data["classification"] == "GOOD" else (0, 0, 255)
                draw_text(frame, f"Score: {int(data['score'])}", 10, 30, color, 0.8)
                draw_text(frame, f"Status: {data['classification']}", 10, 60, color, 0.7)
                
                # Show angles
                draw_text(frame, f"Neck: {data['neck_angle']:.1f}deg (optimal: {NECK_OPTIMAL}, range: {NECK_MIN}-{NECK_MAX})", 10, 95, (255, 255, 255), 0.5)
                draw_text(frame, f"Torso: {data['torso_angle']:.1f}deg (optimal: {TORSO_OPTIMAL}, range: {TORSO_MIN}-{TORSO_MAX})", 10, 115, (255, 255, 255), 0.5)
                draw_text(frame, f"Focus: {data['eye_ear_shoulder_angle']:.1f}deg", 10, 135, (255, 255, 255), 0.5)
                
                # Show subscores
                y = 165
                for k, v in data["subscores"].items():
                    draw_text(frame, f"{k}: {int(v)}", 10, y)
                    y += 25
                
                # Show issues
                if data["reasons"]:
                    draw_text(frame, "Issues:", 10, y, (0, 0, 255))
                    for i, r in enumerate(data["reasons"]):
                        draw_text(frame, f"- {r}", 20, y + 25 * (i + 1), (0, 0, 255))
                
                # Alerts
                if bad_alert:
                    draw_text(frame, "BAD POSTURE ALERT", WIDTH - 250, 60, (0, 0, 255), 0.7)
                
                # Focus session duration (only show if >= 5 minutes)
                if focused:
                    minutes = int(data["focus_duration"] // 60)
                    seconds = int(data["focus_duration"] % 60)
                    draw_text(frame, f"FOCUS: {minutes}m {seconds}s", WIDTH - 250, 90, (0, 255, 255), 0.7)
                
                # Idle session duration (only show if >= 30 minutes)
                if idle_alert:
                    minutes = int(data["idle_duration"] // 60)
                    seconds = int(data["idle_duration"] % 60)
                    draw_text(frame, f"IDLE: {minutes}m {seconds}s", WIDTH - 250, 120, (255, 165, 0), 0.7)
            else:
                draw_text(frame, "Waiting for valid pose...", 10, 30, (0, 255, 255), 0.7)
            
            # FPS
            draw_text(frame, f"FPS: {fps:.1f}", 10, HEIGHT - 20, (0, 255, 255), 0.6)
            
            # Key point coordinates (bottom right)
            y_coord = HEIGHT - 120
            draw_text(frame, "Coordinates (y,x):", WIDTH - 200, y_coord, (200, 200, 200), 0.5)
            for name, kp in side_kps.items():
                y_coord += 18
                conf_str = f"{kp[2]:.2f}" if kp[2] > MIN_KP_CONF else "LOW"
                draw_text(frame, f"{name}: ({kp[0]:.2f},{kp[1]:.2f}) [{conf_str}]", 
                            WIDTH - 200, y_coord, kp_colors[name], 0.4)
            
            # Display frame
            cv2.imshow("Side Camera Posture Monitor", frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"side_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord("c"):
                monitor.calibrated = False
                print("Recalibrating on next frame...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        proc.terminate()
        proc.wait()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        final_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nDone! Average FPS: {final_fps:.1f}")
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()