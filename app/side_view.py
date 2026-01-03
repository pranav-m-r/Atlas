"""
Single Side-Camera Desk Posture + Focus Monitor
Features:
- Neck slouch (forward head)
- Torso slouch
- Focus based on head stillness
Displays skeleton, torso lines, score, subscores, classification, and alerts
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import math
import os

# Set display
os.environ['DISPLAY'] = ':0'

# ============================================================
# ========================= CONFIG ============================
# ============================================================

WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

MIN_KP_CONF = 0.4

BAD_POSTURE_ALERT_TIME = 10.0      # seconds
SEATED_ALERT_TIME = 45*60          # seconds
FOCUS_MIN_TIME = 5*60              # seconds

# Thresholds
NECK_FLEX_BAD = 18.0       # degrees
TORSO_COMP_BAD = 1.6       # torso height / shoulder width
HEAD_MOVEMENT_THRESH = 3.0 # degrees

# Weights
W_NECK = 0.5
W_TORSO = 0.5

# Keypoints
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
# ===================== UTILITY FUNCTIONS =====================
# ============================================================

def valid(kp): 
    return kp[2] > MIN_KP_CONF

def midpoint(p1,p2): 
    return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

def draw_text(frame,text,x,y,color=(255,255,255),scale=0.6):
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2)

def draw_skeleton(frame,keypoints,connections=SKELETON_CONNECTIONS,conf_thresh=MIN_KP_CONF):
    h,w=frame.shape[:2]
    for start,end in connections:
        if keypoints[start][2]>conf_thresh and keypoints[end][2]>conf_thresh:
            y0,x0=keypoints[start][:2]
            y1,x1=keypoints[end][:2]
            cv2.line(frame,(int(x0*w),int(y0*h)),(int(x1*w),int(y1*h)),(255,0,0),2)
    for i,(y,x,c) in enumerate(keypoints):
        if c>conf_thresh: 
            cv2.circle(frame,(int(x*w),int(y*h)),5,(0,255,0),-1)
    return frame

# ============================================================
# ===================== POSTURE FEATURES =====================
# ============================================================

def neck_flexion(keypoints):
    nose = keypoints[0][:2]
    shoulder_mid = midpoint(keypoints[5][:2], keypoints[6][:2])
    if keypoints[0][2]<MIN_KP_CONF or keypoints[5][2]<MIN_KP_CONF or keypoints[6][2]<MIN_KP_CONF:
        return None
    dy = (nose[1]-shoulder_mid[1])*HEIGHT
    dx = (nose[0]-shoulder_mid[0])*WIDTH
    angle = abs(math.degrees(math.atan2(dx,dy)))
    return angle

def torso_compression(keypoints):
    l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
    l_hip, r_hip = keypoints[11][:2], keypoints[12][:2]
    if keypoints[5][2]<MIN_KP_CONF or keypoints[6][2]<MIN_KP_CONF or keypoints[11][2]<MIN_KP_CONF or keypoints[12][2]<MIN_KP_CONF:
        return None
    shoulder_w = np.linalg.norm(np.array(r_sh)-np.array(l_sh))
    torso_h = np.linalg.norm(np.array(midpoint(l_sh,r_sh))-np.array(midpoint(l_hip,r_hip)))
    return torso_h/shoulder_w if shoulder_w>0 else None

def subscore(val,thresh,invert=False):
    if val is None: return 0.5
    if invert: return min(1.0,val/thresh)
    return max(0.0,1.0 - abs(val)/thresh)

def posture_score(keypoints):
    neck = neck_flexion(keypoints)
    torso = torso_compression(keypoints)
    s_neck = subscore(neck,NECK_FLEX_BAD)
    s_torso = subscore(torso,TORSO_COMP_BAD,invert=True)
    score = (W_NECK*s_neck + W_TORSO*s_torso)*100
    classification = "GOOD" if score>=60 else "BAD"
    reasons=[]
    if neck and neck>NECK_FLEX_BAD: reasons.append("Forward Head")
    if torso and torso<TORSO_COMP_BAD: reasons.append("Slouching")
    return {
        "score":score,
        "classification":classification,
        "subscores":{"Neck":s_neck*100,"Torso":s_torso*100},
        "reasons":reasons,
        "neck_angle": neck
    }

# ============================================================
# ================== MONITOR CLASS ============================
# ============================================================

class PostureMonitor:
    def __init__(self):
        self.bad_start = None
        self.seated_start = time.time()
        self.last_head = None
        self.last_move = time.time()
        
    def update(self,score_data):
        now = time.time()
        bad = score_data["score"]<60
        if bad: 
            self.bad_start=self.bad_start or now
        else: 
            self.bad_start=None
            
        bad_alert = self.bad_start and (now - self.bad_start>BAD_POSTURE_ALERT_TIME)
        seated_alert = (now - self.seated_start)>SEATED_ALERT_TIME
        
        head = score_data.get("neck_angle")
        focused=False
        if head is not None:
            if self.last_head and abs(head-self.last_head)>HEAD_MOVEMENT_THRESH:
                self.last_move=now
            self.last_head=head
            focused=(now-self.last_move)>FOCUS_MIN_TIME
            
        return bad_alert,seated_alert,focused

# ============================================================
# ===================== MOVENET INFERENCE ====================
# ============================================================

def load_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(frame,size):
    img = cv2.resize(frame,(size,size))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img[np.newaxis].astype(np.uint8)

def infer(interpreter,inp):
    i=interpreter.get_input_details()[0]["index"]
    o=interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(i,inp)
    interpreter.invoke()
    return interpreter.get_tensor(o)[0][0]

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
        "rpicam-vid","-t","0","--inline","--nopreview",
        "--codec","yuv420","--width",str(WIDTH),"--height",str(HEIGHT),
        "--framerate",str(FRAMERATE),"-o","-"
    ]
    
    proc = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.DEVNULL,
        bufsize=WIDTH * HEIGHT * 3
    )
    frame_size = WIDTH * HEIGHT * 3 // 2  # YUV420

    print("\nSide Camera Posture Monitor started")
    print("Press 'q' to quit, 's' to save screenshot\n")

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
            score_data = posture_score(keypoints)
            bad_alert, seated_alert, focused = monitor.update(score_data)

            # Draw skeleton
            frame = draw_skeleton(frame, keypoints)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time

            # ==== ANNOTATIONS ====
            color = (0, 255, 0) if score_data["classification"] == "GOOD" else (0, 0, 255)
            draw_text(frame, f"Score: {int(score_data['score'])}", 10, 30, color, 0.8)
            draw_text(frame, f"Status: {score_data['classification']}", 10, 60, color, 0.7)
            draw_text(frame, f"FPS: {fps:.1f}", 10, HEIGHT - 20, (0, 255, 255), 0.6)
            
            y = 100
            for k, v in score_data["subscores"].items():
                draw_text(frame, f"{k}: {int(v)}", 10, y)
                y += 25
                
            if score_data["reasons"]:
                draw_text(frame, "Issues:", 10, y, (0, 0, 255))
                for i, r in enumerate(score_data["reasons"]):
                    draw_text(frame, f"- {r}", 20, y + 25 * (i + 1), (0, 0, 255))
                    
            if bad_alert: 
                draw_text(frame, "BAD POSTURE ALERT", 350, 40, (0, 0, 255), 0.7)
            if seated_alert: 
                draw_text(frame, "TIME TO STAND UP", 350, 70, (255, 0, 0), 0.7)
            if focused: 
                draw_text(frame, "FOCUSED", 350, 100, (0, 255, 255), 0.7)

            # Display frame
            cv2.imshow("Side Camera Posture", frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"side_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
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