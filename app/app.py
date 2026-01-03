"""
Live Desk Posture + Focus Monitor
Camera-placement robust (body-centric torso frame)
"""


"""
Live Desk Posture + Focus Monitor with 3D pseudo-coordinates
Works for oblique camera angles (~45°)
Displays skeleton and annotations
"""
os.environ['DISPLAY']=':0'

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

MIN_KP_CONF = 0.4

BAD_POSTURE_ALERT_TIME = 10.0
SEATED_ALERT_TIME = 45 * 60
FOCUS_MIN_TIME = 5 * 60

NECK_FLEX_BAD = 18.0          # degrees
TORSO_COMP_BAD = 1.6          # torso height / shoulder width
TORSO_ROLL_BAD = 12.0         # degrees

HEAD_MOVEMENT_THRESH = 3.0    # degrees

W_NECK = 0.4
W_TORSO = 0.4
W_ROLL = 0.2

SHOULDER_WIDTH_REAL = 0.4     # meters, approximate shoulder width



KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6),                               # shoulders
    (5, 7), (7, 9),                        # left arm
    (6, 8), (8, 10),                       # right arm
    (5, 11), (6, 12),                      # torso sides
    (11, 12),                              # hips
    (11, 13), (13, 15),                    # left leg
    (12, 14), (14, 16)                     # right leg
]

# ============================================================
# ========================= UTILS =============================
# ============================================================

def valid(kp):
    return kp[2] > MIN_KP_CONF

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def draw_text(frame, text, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

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

# ============================================================
# =================== PSEUDO 3D BODY ==========================
# ============================================================

def to_3d(keypoints, shoulder_width_real=SHOULDER_WIDTH_REAL):
    l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
    shoulder_width_px = np.linalg.norm(np.array(r_sh) - np.array(l_sh))
    if shoulder_width_px == 0: return None
    scale = shoulder_width_real / shoulder_width_px
    kp3d = []
    for x, y, c in keypoints:
        kp3d.append(np.array([(x-0.5)*scale, (y-0.5)*scale, scale]))
    return kp3d

def torso_axes(kp3d):
    l_sh, r_sh = kp3d[5], kp3d[6]
    l_hip, r_hip = kp3d[11], kp3d[12]
    origin = (l_hip + r_hip)/2
    y_axis = ((l_sh+r_sh)/2 - origin)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = r_sh - l_sh
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return origin, x_axis, y_axis, z_axis

def project_to_torso(point, origin, x_axis, y_axis, z_axis):
    vec = point - origin
    return np.array([np.dot(vec, x_axis), np.dot(vec, y_axis), np.dot(vec, z_axis)])

# ============================================================
# ================== POSTURE FEATURES ========================
# ============================================================

def neck_flexion(kp3d, origin, x_axis, y_axis, z_axis):
    nose = kp3d[0]
    local = project_to_torso(nose, origin, x_axis, y_axis, z_axis)
    return math.degrees(math.atan2(local[2], local[1]))

def torso_compression(kp3d):
    l_sh, r_sh = kp3d[5], kp3d[6]
    l_hip, r_hip = kp3d[11], kp3d[12]
    shoulder_w = np.linalg.norm(r_sh - l_sh)
    torso_h = np.linalg.norm(midpoint(l_sh,r_sh)-midpoint(l_hip,r_hip))
    return torso_h / shoulder_w if shoulder_w>0 else None

def torso_roll(kp3d, origin, x_axis, y_axis, z_axis):
    nose = kp3d[0]
    local = project_to_torso(nose, origin, x_axis, y_axis, z_axis)
    return math.degrees(math.atan2(local[0], local[1]))

def subscore(val, thresh, invert=False):
    if val is None: return 0.5
    if invert: return min(1.0, val/thresh)
    return max(0.0, 1.0 - val/thresh)

def posture_score(keypoints):
    kp3d = to_3d(keypoints)
    if kp3d is None: return {"score":50,"classification":"BAD","subscores":{"Neck":50,"Torso":50,"Roll":50},"reasons":["Invalid"]}
    origin, x_axis, y_axis, z_axis = torso_axes(kp3d)

    neck = neck_flexion(kp3d, origin, x_axis, y_axis, z_axis)
    comp = torso_compression(kp3d)
    roll = torso_roll(kp3d, origin, x_axis, y_axis, z_axis)

    s_neck = subscore(neck, NECK_FLEX_BAD)
    s_torso = subscore(comp, TORSO_COMP_BAD, invert=True)
    s_roll = subscore(roll, TORSO_ROLL_BAD)
    score = (W_NECK*s_neck + W_TORSO*s_torso + W_ROLL*s_roll)*100
    classification = "GOOD" if score>=60 else "BAD"
    reasons=[]
    if neck>NECK_FLEX_BAD: reasons.append("Forward Head")
    if comp<TORSO_COMP_BAD: reasons.append("Slouching")
    if roll>TORSO_ROLL_BAD: reasons.append("Lateral Lean")
    return {"score":score,"classification":classification,"subscores":{"Neck":s_neck*100,"Torso":s_torso*100,"Roll":s_roll*100},"reasons":reasons}

# ============================================================
# ====================== MONITOR ==============================
# ============================================================

class PostureMonitor:
    def __init__(self):
        self.bad_start = None
        self.seated_start = time.time()
        self.last_head = None
        self.last_move = time.time()
    def update(self, keypoints):
        now = time.time()
        data = posture_score(keypoints)
        bad = data["score"] < 60
        if bad: self.bad_start = self.bad_start or now
        else: self.bad_start = None
        bad_alert = self.bad_start and (now - self.bad_start > BAD_POSTURE_ALERT_TIME)
        seated_alert = (now - self.seated_start) > SEATED_ALERT_TIME
        neck = neck_flexion(to_3d(keypoints), *torso_axes(to_3d(keypoints))) if to_3d(keypoints) else None
        focused=False
        if neck is not None:
            if self.last_head and abs(neck - self.last_head) > HEAD_MOVEMENT_THRESH:
                self.last_move = now
            self.last_head = neck
            focused = (now - self.last_move) > FOCUS_MIN_TIME
        return data, bad_alert, seated_alert, focused

# ============================================================
# ====================== MOVENET =============================
# ============================================================

def load_model(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def preprocess(frame, size):
    img = cv2.resize(frame, (size,size))
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
    interpreter = load_model("model.tflite")
    input_size = interpreter.get_input_details()[0]["shape"][1]
    monitor = PostureMonitor()

    cmd = [
        "rpicam-vid","-t","0","--inline","--nopreview",
        "--codec","yuv420","--width",str(WIDTH),"--height",str(HEIGHT),
        "--framerate",str(FRAMERATE),"-o","-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = WIDTH*HEIGHT*3//2

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw)!=frame_size: break
        yuv = np.frombuffer(raw,np.uint8).reshape((HEIGHT*3//2,WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        keypoints = infer(interpreter, preprocess(frame,input_size))
        frame = draw_skeleton(frame, keypoints)
        data, bad_alert, seat_alert, focused = monitor.update(keypoints)

        # ==== ANNOTATIONS ====
        color = (0,255,0) if data["classification"]=="GOOD" else (0,0,255)
        draw_text(frame, f"Score: {int(data['score'])}", 10, 30, color, 0.8)
        draw_text(frame, f"Status: {data['classification']}", 10, 60, color, 0.7)
        y=100
        for k,v in data["subscores"].items():
            draw_text(frame,f"{k}: {int(v)}",10,y)
            y+=25
        if data["reasons"]:
            draw_text(frame,"Issues:",10,y, (0,0,255))
            for i,r in enumerate(data["reasons"]):
                draw_text(frame,f"- {r}",20,y+25*(i+1),(0,0,255))
        if bad_alert: draw_text(frame,"BAD POSTURE ALERT",350,40,(0,0,255),0.7)
        if seat_alert: draw_text(frame,"TIME TO STAND UP",350,70,(255,0,0),0.7)
        if focused: draw_text(frame,"FOCUSED",350,100,(0,255,255),0.7)

        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(1) & 0xFF==ord("q"): break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
