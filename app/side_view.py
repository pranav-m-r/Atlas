"""
Live Desk Posture + Focus Monitor (Side Camera)
3-point angle measurements: ear-shoulder-hip for neck, shoulder-hip-knee for torso
Eye-ear-shoulder angle variation for focus/attention metric
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import math
import os

# ============================================================
# ========================= CONFIG ============================
# ============================================================

WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

MIN_KP_CONF = 0.3

BAD_POSTURE_ALERT_TIME = 10.0
SEATED_ALERT_TIME = 45 * 60
FOCUS_MIN_TIME = 5 * 60

NECK_ANGLE_THRESH = 10.0  # degrees deviation from calibrated
TORSO_ANGLE_THRESH = 8.0
EYE_EAR_SHOULDER_ANGLE_THRESH = 3.0  # degrees change to count as head movement

# weights for scoring
W_NECK = 0.5
W_TORSO = 0.5

os.environ['DISPLAY'] = ':0'

# Keypoint indices
# Left side: eye=1, ear=3, shoulder=5, hip=11, knee=13
# Right side: eye=2, ear=4, shoulder=6, hip=12, knee=14
LEFT_INDICES = {'eye': 1, 'ear': 3, 'shoulder': 5, 'hip': 11, 'knee': 13}
RIGHT_INDICES = {'eye': 2, 'ear': 4, 'shoulder': 6, 'hip': 12, 'knee': 14}

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

def draw_text(frame, text, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

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
    Calculate angle at p2 formed by points p1-p2-p3
    Returns angle in degrees (0-180)
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Angle in radians then degrees
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
    angle = math.acos(cos_angle)
    return math.degrees(angle)

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
    """Get the 5 key points (eye, ear, shoulder, hip, knee) for the detected side"""
    indices = LEFT_INDICES if side == 'LEFT' else RIGHT_INDICES
    return {
        'eye': keypoints[indices['eye']],
        'ear': keypoints[indices['ear']],
        'shoulder': keypoints[indices['shoulder']],
        'hip': keypoints[indices['hip']],
        'knee': keypoints[indices['knee']]
    }

# ============================================================
# ================== MONITOR CLASS ============================
# ============================================================

class PostureMonitor:
    def __init__(self):
        self.bad_start = None
        self.seated_start = time.time()
        self.last_eye_ear_shoulder_angle = None
        self.last_move = time.time()
        self.calibrated = False
        self.neck_angle0 = None
        self.torso_angle0 = None
        self.detected_side = None

    def calibrate(self, side_kps):
        """Calibrate using 3-point angles"""
        ear = side_kps['ear'][:2]
        shoulder = side_kps['shoulder'][:2]
        hip = side_kps['hip'][:2]
        knee = side_kps['knee'][:2]
        
        # Neck angle: ear-shoulder-hip (angle at shoulder)
        self.neck_angle0 = calculate_angle(ear, shoulder, hip)
        
        # Torso angle: shoulder-hip-knee (angle at hip)
        self.torso_angle0 = calculate_angle(shoulder, hip, knee)
        
        self.calibrated = True
        print(f"Calibrated - Neck angle: {self.neck_angle0:.1f}°, Torso angle: {self.torso_angle0:.1f}°")

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
        
        # Calibrate on first valid frame
        if not self.calibrated:
            self.calibrate(side_kps)
        
        ear = side_kps['ear'][:2]
        eye = side_kps['eye'][:2]
        shoulder = side_kps['shoulder'][:2]
        hip = side_kps['hip'][:2]
        knee = side_kps['knee'][:2]
        
        # Calculate current 3-point angles
        # 1) Neck: ear-shoulder-hip (angle at shoulder)
        neck_angle = calculate_angle(ear, shoulder, hip)
        neck_deviation = neck_angle - self.neck_angle0
        
        # 2) Torso: shoulder-hip-knee (angle at hip)
        torso_angle = calculate_angle(shoulder, hip, knee)
        torso_deviation = torso_angle - self.torso_angle0
        
        # 3) Focus: eye-ear-shoulder (angle at ear)
        eye_ear_shoulder_angle = calculate_angle(eye, ear, shoulder)
        
        # Calculate subscores
        s_neck = max(0, 1 - abs(neck_deviation) / NECK_ANGLE_THRESH)
        s_torso = max(0, 1 - abs(torso_deviation) / TORSO_ANGLE_THRESH)
        score = (W_NECK * s_neck + W_TORSO * s_torso) * 100
        classification = "GOOD" if score >= 60 else "BAD"
        
        # Determine reasons for bad posture
        reasons = []
        if abs(neck_deviation) > NECK_ANGLE_THRESH:
            reasons.append(f"Neck Slouch ({abs(neck_deviation):.1f}°)")
        if abs(torso_deviation) > TORSO_ANGLE_THRESH:
            reasons.append(f"Torso Slouch ({abs(torso_deviation):.1f}°)")
        
        # Bad posture alert
        bad = score < 60
        if bad:
            self.bad_start = self.bad_start or now
        else:
            self.bad_start = None
        bad_alert = self.bad_start and (now - self.bad_start > BAD_POSTURE_ALERT_TIME)
        
        # Seated alert
        seated_alert = (now - self.seated_start) > SEATED_ALERT_TIME
        
        # Focus detection based on eye-ear-shoulder angle variation
        focused = False
        if self.last_eye_ear_shoulder_angle is not None:
            if abs(eye_ear_shoulder_angle - self.last_eye_ear_shoulder_angle) > EYE_EAR_SHOULDER_ANGLE_THRESH:
                self.last_move = now
        self.last_eye_ear_shoulder_angle = eye_ear_shoulder_angle
        focused = (now - self.last_move) > FOCUS_MIN_TIME
        
        data = {
            "score": score,
            "classification": classification,
            "subscores": {"Neck": s_neck * 100, "Torso": s_torso * 100},
            "reasons": reasons,
            "neck_angle": neck_angle,
            "neck_deviation": neck_deviation,
            "torso_angle": torso_angle,
            "torso_deviation": torso_deviation,
            "eye_ear_shoulder_angle": eye_ear_shoulder_angle
        }
        
        return data, bad_alert, seated_alert, focused, side_kps

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
    print("Press 'q' to quit, 's' to save screenshot, 'c' to recalibrate\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_update = start_time
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
            data, bad_alert, seated_alert, focused, side_kps = result
            
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
            
            # ==== ANNOTATE THE 5 KEY POINTS ====
            h, w = frame.shape[:2]
            kp_colors = {
                'eye': (255, 0, 255),      # Magenta
                'ear': (0, 255, 255),      # Yellow
                'shoulder': (255, 128, 0), # Orange
                'hip': (0, 128, 255),      # Light blue
                'knee': (128, 0, 255)      # Purple
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
                draw_text(frame, f"Neck: {data['neck_angle']:.1f}deg ({data['neck_deviation']:+.1f})", 10, 95, (255, 255, 255), 0.5)
                draw_text(frame, f"Torso: {data['torso_angle']:.1f}deg ({data['torso_deviation']:+.1f})", 10, 115, (255, 255, 255), 0.5)
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
                if seated_alert:
                    draw_text(frame, "TIME TO STAND UP", WIDTH - 250, 90, (255, 0, 0), 0.7)
                if focused:
                    draw_text(frame, "FOCUSED", WIDTH - 250, 120, (0, 255, 255), 0.7)
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
            
            # Check for key press - use longer wait and check explicitly
            key = cv2.waitKey(30) & 0xFF  # Increased to 30ms
            
            # Debug: print key value if any key is pressed
            if key != 255:  # 255 means no key pressed
                print(f"Key pressed: {key} (chr: {chr(key) if key < 128 else 'N/A'})")
            
            if key == ord("q") or key == ord("Q") or key == 27:  # 27 is ESC
                print("Quitting...")
                break
            elif key == ord("s") or key == ord("S"):
                filename = f"side_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord("c") or key == ord("C"):
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