"""
Single Side-Camera Desk Posture + Focus Monitor
Features:
- Neck slouch (forward head)
- Torso slouch
- Focus based on head stillness
Displays skeleton, torso lines, score, subscores, classification, and alerts
"""
#os.environ['DISPLAY']=':0'

"""
Single Side-Camera Desk Posture + Focus Monitor with Live Bars
Features:
- Neck slouch (forward head)
- Torso slouch
- Focus based on head stillness
- Live visual bars for Neck, Torso, and Total Score
"""

"""
Live Desk Posture + Focus Monitor (Side Camera)
Simple angular displacement from calibrated straight posture
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
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

NECK_ANGLE_THRESH = 15.0  # degrees deviation from calibrated
TORSO_ANGLE_THRESH = 15.0

HEAD_MOVEMENT_THRESH = 3.0

# weights for scoring
W_NECK = 0.5
W_TORSO = 0.5
os.environ['DISPLAY']=':0'

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

def midpoint(p1,p2): return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

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

def angle_from_vertical(p1,p2):
    """Return angle (degrees) between line p1->p2 and vertical"""
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    if dy==0: return 0
    return math.degrees(math.atan2(dx,dy))

# ============================================================
# ================== MONITOR CLASS ============================
# ============================================================

class PostureMonitor:
    def __init__(self):
        self.bad_start = None
        self.seated_start = time.time()
        self.last_head = None
        self.last_move = time.time()
        self.calibrated = False
        self.neck_angle0 = None
        self.torso_angle0 = None

    def calibrate(self,keypoints):
        nose = keypoints[0][:2]
        shoulder_mid = midpoint(keypoints[5][:2], keypoints[6][:2])
        hip_mid = midpoint(keypoints[11][:2], keypoints[12][:2])
        self.neck_angle0 = angle_from_vertical(shoulder_mid,nose)
        self.torso_angle0 = angle_from_vertical(hip_mid,shoulder_mid)
        self.calibrated = True

    def update(self,keypoints):
        now = time.time()
        if not self.calibrated:
            self.calibrate(keypoints)

        nose = keypoints[0][:2]
        shoulder_mid = midpoint(keypoints[5][:2], keypoints[6][:2])
        hip_mid = midpoint(keypoints[11][:2], keypoints[12][:2])

        neck_angle = angle_from_vertical(shoulder_mid,nose) - self.neck_angle0
        torso_angle = angle_from_vertical(hip_mid,shoulder_mid) - self.torso_angle0

        s_neck = max(0,1 - abs(neck_angle)/NECK_ANGLE_THRESH)
        s_torso = max(0,1 - abs(torso_angle)/TORSO_ANGLE_THRESH)
        score = (W_NECK*s_neck + W_TORSO*s_torso)*100
        classification = "GOOD" if score>=60 else "BAD"

        reasons=[]
        if abs(neck_angle)>NECK_ANGLE_THRESH: reasons.append("Forward Head")
        if abs(torso_angle)>TORSO_ANGLE_THRESH: reasons.append("Slouching")

        bad = score<60
        if bad: self.bad_start = self.bad_start or now
        else: self.bad_start = None
        bad_alert = self.bad_start and (now - self.bad_start>BAD_POSTURE_ALERT_TIME)
        seated_alert = (now - self.seated_start)>SEATED_ALERT_TIME

        focused=False
        if self.last_head and abs(neck_angle - self.last_head)>HEAD_MOVEMENT_THRESH:
            self.last_move=now
        self.last_head=neck_angle
        focused=(now-self.last_move)>FOCUS_MIN_TIME

        return {"score":score,"classification":classification,"subscores":{"Neck":s_neck*100,"Torso":s_torso*100},"reasons":reasons}, bad_alert, seated_alert, focused

# ============================================================
# ===================== MOVENET INFERENCE ====================
# ============================================================

def load_model(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def preprocess(frame,size):
    img = cv2.resize(frame,(size,size))
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return img[np.newaxis].astype(np.uint8)

def infer(interp,inp):
    i=interp.get_input_details()[0]["index"]
    o=interp.get_output_details()[0]["index"]
    interp.set_tensor(i,inp)
    interp.invoke()
    return interp.get_tensor(o)[0][0]

# ============================================================
# ========================== MAIN ============================
# ============================================================

def main():
    interpreter = load_model("model.tflite")
    input_size = interpreter.get_input_details()[0]["shape"][1]
    monitor = PostureMonitor()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret: break

        keypoints = infer(interpreter, preprocess(frame,input_size))
        frame = draw_skeleton(frame,keypoints)
        data,bad_alert,seat_alert,focused = monitor.update(keypoints)

        # ==== ANNOTATIONS ====
        color = (0,255,0) if data["classification"]=="GOOD" else (0,0,255)
        draw_text(frame,f"Score: {int(data['score'])}",10,30,color,0.8)
        draw_text(frame,f"Status: {data['classification']}",10,60,color,0.7)

        y=100
        for k,v in data["subscores"].items():
            draw_text(frame,f"{k}: {int(v)}",10,y)
            y+=25

        if data["reasons"]:
            draw_text(frame,"Issues:",10,y,(0,0,255))
            for i,r in enumerate(data["reasons"]):
                draw_text(frame,f"- {r}",20,y+25*(i+1),(0,0,255))

        if bad_alert: draw_text(frame,"BAD POSTURE ALERT",350,40,(0,0,255),0.7)
        if seat_alert: draw_text(frame,"TIME TO STAND UP",350,70,(255,0,0),0.7)
        if focused: draw_text(frame,"FOCUSED",350,100,(0,255,255),0.7)

        cv2.imshow("Side Camera Posture Monitor",frame)
        if cv2.waitKey(1)&0xFF==ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
