"""
Live Desk Posture + Focus Monitor
Camera-placement robust (body-centric torso frame)
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import math

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

NECK_FLEX_BAD = 18.0
TORSO_COMP_BAD = 1.6
TORSO_ROLL_BAD = 12.0

HEAD_MOVEMENT_THRESH = 3.0

W_NECK = 0.4
W_TORSO = 0.4
W_ROLL = 0.2

os.environ['DISPLAY']=':0'

# ============================================================
# ========================= UTILS =============================
# ============================================================

def valid(kp):
    return kp[2] > MIN_KP_CONF


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def draw_text(frame, text, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# ============================================================
# ==================== BODY FRAME =============================
# ============================================================

def torso_frame(keypoints):
    l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
    l_hip, r_hip = keypoints[11][:2], keypoints[12][:2]

    shoulder_mid = midpoint(l_sh, r_sh)
    hip_mid = midpoint(l_hip, r_hip)

    y_axis = np.array(shoulder_mid) - np.array(hip_mid)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.array(r_sh) - np.array(l_sh)
    x_axis /= np.linalg.norm(x_axis)

    return hip_mid, x_axis, y_axis


def to_torso_coords(point, origin, x_axis, y_axis):
    v = np.array(point) - np.array(origin)
    return np.array([np.dot(v, x_axis), np.dot(v, y_axis)])


# ============================================================
# =================== POSTURE FEATURES ========================
# ============================================================

def neck_flexion(keypoints):
    if not all(valid(keypoints[i]) for i in [0,5,6,11,12]):
        return None
    origin, x_axis, y_axis = torso_frame(keypoints)
    nose = keypoints[0][:2]
    v = to_torso_coords(nose, origin, x_axis, y_axis)
    return abs(math.degrees(math.atan2(v[0], v[1])))


def torso_compression(keypoints):
    l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
    l_hip, r_hip = keypoints[11][:2], keypoints[12][:2]
    shoulder_w = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
    torso_h = np.linalg.norm(
        np.array(midpoint(l_sh, r_sh)) -
        np.array(midpoint(l_hip, r_hip))
    )
    return torso_h / shoulder_w if shoulder_w > 0 else None


def torso_roll(keypoints):
    l_sh, r_sh = keypoints[5][:2], keypoints[6][:2]
    dx = r_sh[1] - l_sh[1]
    dy = r_sh[0] - l_sh[0]
    return abs(math.degrees(math.atan2(dy, dx)))


def subscore(val, thresh, invert=False):
    if val is None:
        return 0.5
    if invert:
        return min(1.0, val / thresh)
    return max(0.0, 1.0 - val / thresh)


def posture_score(keypoints):
    neck = neck_flexion(keypoints)
    comp = torso_compression(keypoints)
    roll = torso_roll(keypoints)

    s_neck = subscore(neck, NECK_FLEX_BAD)
    s_torso = subscore(comp, TORSO_COMP_BAD, invert=True)
    s_roll = subscore(roll, TORSO_ROLL_BAD)

    score = (W_NECK*s_neck + W_TORSO*s_torso + W_ROLL*s_roll) * 100
    classification = "GOOD" if score >= 60 else "BAD"

    reasons = []
    if neck and neck > NECK_FLEX_BAD:
        reasons.append("Forward Head")
    if comp and comp < TORSO_COMP_BAD:
        reasons.append("Slouching")
    if roll > TORSO_ROLL_BAD:
        reasons.append("Lateral Lean")

    return {
        "score": score,
        "classification": classification,
        "subscores": {
            "Neck": s_neck*100,
            "Torso": s_torso*100,
            "Roll": s_roll*100
        },
        "reasons": reasons
    }


# ============================================================
# ===================== MONITOR ==============================
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
        if bad:
            self.bad_start = self.bad_start or now
        else:
            self.bad_start = None

        bad_alert = self.bad_start and (now - self.bad_start > BAD_POSTURE_ALERT_TIME)
        seated_alert = (now - self.seated_start) > SEATED_ALERT_TIME

        neck = neck_flexion(keypoints)
        focused = False
        if neck is not None:
            if self.last_head and abs(neck - self.last_head) > HEAD_MOVEMENT_THRESH:
                self.last_move = now
            self.last_head = neck
            focused = (now - self.last_move) > FOCUS_MIN_TIME

        return data, bad_alert, seated_alert, focused


# ============================================================
# ====================== MOVENET ==============================
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
# ========================= MAIN ==============================
# ============================================================

def main():
    interpreter = load_model("model.tflite")
    input_size = interpreter.get_input_details()[0]["shape"][1]
    monitor = PostureMonitor()

    cmd = [
        "rpicam-vid", "-t", "0", "--inline", "--nopreview",
        "--codec", "yuv420", "--width", str(WIDTH),
        "--height", str(HEIGHT), "--framerate", str(FRAMERATE),
        "-o", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = WIDTH * HEIGHT * 3 // 2

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            break

        yuv = np.frombuffer(raw, np.uint8).reshape((HEIGHT*3//2, WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        keypoints = infer(interpreter, preprocess(frame, input_size))
        data, bad_alert, seat_alert, focused = monitor.update(keypoints)

        # ===== LIVE OVERLAY =====
        color = (0,255,0) if data["classification"]=="GOOD" else (0,0,255)

        draw_text(frame, f"Score: {int(data['score'])}", 10, 30, color, 0.8)
        draw_text(frame, f"Status: {data['classification']}", 10, 60, color, 0.7)

        y = 100
        for k,v in data["subscores"].items():
            draw_text(frame, f"{k}: {int(v)}", 10, y)
            y += 25

        if data["reasons"]:
            draw_text(frame, "Issues:", 10, y+10, (0,0,255))
            for i,r in enumerate(data["reasons"]):
                draw_text(frame, f"- {r}", 20, y+40+25*i, (0,0,255))

        if bad_alert:
            draw_text(frame, "BAD POSTURE ALERT", 350, 40, (0,0,255), 0.7)
        if seat_alert:
            draw_text(frame, "TIME TO STAND UP", 350, 70, (255,0,0), 0.7)
        if focused:
            draw_text(frame, "FOCUSED", 350, 100, (0,255,255), 0.7)

        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    proc.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
