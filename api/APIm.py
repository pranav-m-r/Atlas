"""
MoveNet Pose Estimation - Fast Video Stream Version
+ API packaging
- /video : live MJPEG video
- /pose  : latest keypoints JSON

CORE LOGIC: UNCHANGED
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

# ===================== API GLOBALS =====================
app = FastAPI(title="MoveNet Pose API")

latest_frame = None
latest_keypoints = None
data_lock = threading.Lock()

# ===================== VIDEO SETTINGS =====================
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30

# ===================== MODEL UTILS =====================
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame, input_size):
    img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.uint8)


def run_inference(interpreter, input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores[0][0]


def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    h, w = frame.shape[:2]
    for y, x, confidence in keypoints:
        if confidence > confidence_threshold:
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame


def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    h, w = frame.shape[:2]

    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    for s, e in connections:
        if keypoints[s][2] > confidence_threshold and keypoints[e][2] > confidence_threshold:
            sy, sx = keypoints[s][:2]
            ey, ex = keypoints[e][:2]
            cv2.line(
                frame,
                (int(sx * w), int(sy * h)),
                (int(ex * w), int(ey * h)),
                (255, 0, 0),
                2
            )
    return frame


# ===================== MAIN PIPELINE (UNCHANGED) =====================
def main():
    global latest_frame, latest_keypoints

    model_path = "model.tflite"
    interpreter = load_model(model_path)

    input_size = interpreter.get_input_details()[0]['shape'][1]

    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--inline",
        "--nopreview",
        "--codec", "yuv420",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FRAMERATE),
        "-o", "-"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=WIDTH * HEIGHT * 3
    )

    frame_size = WIDTH * HEIGHT * 3 // 2
    frame_count = 0
    start_time = time.time()

    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) != frame_size:
            continue

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        input_image = preprocess_frame(frame, input_size)
        keypoints = run_inference(interpreter, input_image)

        frame = draw_skeleton(frame, keypoints)
        frame = draw_keypoints(frame, keypoints)

        with data_lock:
            latest_frame = frame.copy()
            latest_keypoints = keypoints.copy()

        frame_count += 1
        fps = frame_count / (time.time() - start_time + 1e-6)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("MoveNet Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ===================== API ENDPOINTS =====================
@app.get("/pose")
def get_pose():
    with data_lock:
        if latest_keypoints is None:
            return JSONResponse({"status": "warming_up"})
        return {
            "timestamp": time.time(),
            "keypoints": latest_keypoints.tolist()
        }


def mjpeg_generator():
    while True:
        with data_lock:
            if latest_frame is None:
                continue
            _, jpeg = cv2.imencode(".jpg", latest_frame)
            frame = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    threading.Thread(target=main, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
