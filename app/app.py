import numpy as np
import time
import math

# ===================== CONFIG ===================== #

# ---- Confidence ----
MIN_KP_CONF = 0.4

# ---- Timing (seconds) ----
POSTURE_EVAL_WINDOW = 5.0          # posture evaluated over rolling window
BAD_POSTURE_ALERT_TIME = 10.0      # alert if bad posture persists
SEATED_ALERT_TIME = 45 * 60        # 45 minutes
FOCUS_MIN_TIME = 5 * 60            # 5 minutes of stable head

# ---- Angle thresholds (degrees) ----
NECK_FORWARD_BAD = 20              # forward head angle
TORSO_PITCH_BAD = 15               # slouch angle
TORSO_ROLL_BAD = 10                # lateral lean

# ---- Focus thresholds ----
HEAD_MOVEMENT_THRESH = 3.0         # degrees
FOCUS_RESET_TIME = 2.0             # seconds of movement to reset focus

# ---- Scoring weights (sum = 1.0) ----
W_NECK = 0.4
W_TORSO_PITCH = 0.4
W_TORSO_ROLL = 0.2

# ---- Score mapping ----
GOOD_POSTURE_SCORE = 100
BAD_POSTURE_SCORE = 40

# ================================================= #
def angle_between(p1, p2):
    """Angle of line p1->p2 w.r.t vertical (degrees)."""
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]  # inverted y-axis
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def valid(kp):
    return kp[2] > MIN_KP_CONF
def extract_posture_features(keypoints):
    """
    Returns:
        neck_angle
        torso_pitch
        torso_roll
    """

    nose = keypoints[0]
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]

    if not all(map(valid, [nose, l_sh, r_sh, l_hip, r_hip])):
        return None

    shoulder_mid = midpoint(l_sh[:2], r_sh[:2])
    hip_mid = midpoint(l_hip[:2], r_hip[:2])

    # ---- Neck forward angle ----
    neck_angle = angle_between(shoulder_mid, nose[:2])

    # ---- Torso pitch (slouch) ----
    torso_pitch = angle_between(hip_mid, shoulder_mid)

    # ---- Torso roll (lateral lean) ----
    dx = r_sh[1] - l_sh[1]
    dy = r_sh[0] - l_sh[0]
    torso_roll = abs(math.degrees(math.atan2(dy, dx)))

    return neck_angle, torso_pitch, torso_roll
def subscore(angle, bad_thresh):
    """Maps angle to [0,1]"""
    return max(0.0, 1.0 - angle / bad_thresh)


def compute_posture_score(features):
    neck, pitch, roll = features

    s_neck = subscore(neck, NECK_FORWARD_BAD)
    s_pitch = subscore(pitch, TORSO_PITCH_BAD)
    s_roll = subscore(roll, TORSO_ROLL_BAD)

    score = (
        W_NECK * s_neck +
        W_TORSO_PITCH * s_pitch +
        W_TORSO_ROLL * s_roll
    ) * 100

    reasons = []
    if neck > NECK_FORWARD_BAD:
        reasons.append("Forward Head")
    if pitch > TORSO_PITCH_BAD:
        reasons.append("Slouching")
    if roll > TORSO_ROLL_BAD:
        reasons.append("Lateral Lean")

    return score, reasons
class PostureMonitor:
    def __init__(self):
        self.bad_posture_start = None
        self.seated_start = time.time()
        self.focus_start = time.time()
        self.last_head_angle = None
        self.last_head_move_time = time.time()

    def update(self, keypoints):
        now = time.time()

        features = extract_posture_features(keypoints)
        if features is None:
            return None

        score, reasons = compute_posture_score(features)

        # ---- Bad posture tracking ----
        bad_posture = score < 60
        if bad_posture:
            if self.bad_posture_start is None:
                self.bad_posture_start = now
        else:
            self.bad_posture_start = None

        bad_posture_alert = (
            self.bad_posture_start is not None and
            now - self.bad_posture_start > BAD_POSTURE_ALERT_TIME
        )

        # ---- Seated time ----
        seated_time = now - self.seated_start
        seated_alert = seated_time > SEATED_ALERT_TIME

        # ---- Focus detection (head stability) ----
        nose = keypoints[0]
        l_ear, r_ear = keypoints[3], keypoints[4]

        focus = False
        if valid(nose) and valid(l_ear) and valid(r_ear):
            ear_mid = midpoint(l_ear[:2], r_ear[:2])
            head_angle = angle_between(ear_mid, nose[:2])

            if self.last_head_angle is not None:
                if abs(head_angle - self.last_head_angle) > HEAD_MOVEMENT_THRESH:
                    self.last_head_move_time = now

            self.last_head_angle = head_angle

            if now - self.last_head_move_time > FOCUS_MIN_TIME:
                focus = True

        return {
            "score": score,
            "bad_posture": bad_posture,
            "bad_posture_alert": bad_posture_alert,
            "reasons": reasons,
            "seated_time": seated_time,
            "seated_alert": seated_alert,
            "focus_time": now - self.last_head_move_time,
            "focused": focus
        }

def main():
    monitor = PostureMonitor()

# inside while True:
    while True:

        status = monitor.update(keypoints)

        if status:
            print(
                f"Score: {status['score']:.1f} | "
                f"Bad: {status['bad_posture']} | "
                f"Reasons: {status['reasons']} | "
                f"Focused: {status['focused']}"
            )
        time.sleep_ms(200)
