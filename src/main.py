"""
MoveNet Pose Estimation - Raspberry Pi Camera Test
Displays live camera feed with pose estimation annotations
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# Try to import picamera2 for modern RPi camera support
# try:
PICAMERA2_AVAILABLE = True
# except ImportError:
#     PICAMERA2_AVAILABLE = False
#     print("picamera2 not available, will use OpenCV for camera access")

# MoveNet keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Skeleton connections for drawing lines between keypoints
SKELETON_CONNECTIONS = [
    ('left_ear', 'left_eye'),
    ('left_eye', 'nose'),
    ('nose', 'right_eye'),
    ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

# Colors for visualization (BGR format)
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (255, 0, 0)  # Blue
CONFIDENCE_THRESHOLD = 0.3


def load_model(model_path):
    """Load TFLite model and allocate tensors."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame, input_size):
    """Preprocess frame for MoveNet model."""
    # Resize to model input size
    img = cv2.resize(frame, (input_size, input_size))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Add batch dimension and convert to int32 (MoveNet expects int32)
    img = np.expand_dims(img, axis=0).astype(np.int32)
    return img


def run_inference(interpreter, input_image):
    """Run pose estimation inference."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    
    # Output shape: [1, 1, 17, 3] - batch, person, keypoints, (y, x, confidence)
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints[0, 0, :, :]  # Return [17, 3] array


def draw_keypoints(frame, keypoints, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw keypoints on frame."""
    height, width, _ = frame.shape
    
    for i, (y, x, confidence) in enumerate(keypoints):
        if confidence > confidence_threshold:
            # Convert normalized coordinates to pixel coordinates
            px = int(x * width)
            py = int(y * height)
            cv2.circle(frame, (px, py), 6, KEYPOINT_COLOR, -1)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
    
    return frame


def draw_skeleton(frame, keypoints, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw skeleton connections on frame."""
    height, width, _ = frame.shape
    
    for connection in SKELETON_CONNECTIONS:
        start_idx = KEYPOINT_DICT[connection[0]]
        end_idx = KEYPOINT_DICT[connection[1]]
        
        start_kp = keypoints[start_idx]
        end_kp = keypoints[end_idx]
        
        # Only draw if both keypoints have sufficient confidence
        if start_kp[2] > confidence_threshold and end_kp[2] > confidence_threshold:
            start_point = (int(start_kp[1] * width), int(start_kp[0] * height))
            end_point = (int(end_kp[1] * width), int(end_kp[0] * height))
            cv2.line(frame, start_point, end_point, SKELETON_COLOR, 3)
    
    return frame


def main():
    """Main function to run pose estimation on camera feed."""
    # Path to the TFLite model
    model_path = "model.tflite"
    
    print("Loading MoveNet model...")
    interpreter = load_model(model_path)
    
    # Get model input size
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]  # Assuming square input (e.g., 192x192 or 256x256)
    print(f"Model input size: {input_size}x{input_size}")
    
    # Initialize camera
    print("Initializing camera...")
    use_picamera2 = False
    cap = None
    picam2 = None
    
    # Try picamera2 first (modern RPi camera support)
    if PICAMERA2_AVAILABLE:
        try:
            print("Attempting to initialize with picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            use_picamera2 = True
            print("Successfully initialized camera with picamera2")
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
            if picam2:
                picam2.close()
            picam2 = None
    
    # Fall back to OpenCV if picamera2 didn't work
    if not use_picamera2:
        print("Trying OpenCV camera access...")
        # Try different camera indices with V4L2 backend (common for RPi)
        for cam_index in [0, 1, 2, -1]:
            print(f"Trying camera index {cam_index} with V4L2 backend...")
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"Successfully opened camera at index {cam_index}")
                break
            cap.release()
        
        # If V4L2 didn't work, try default backend
        if cap is None or not cap.isOpened():
            for cam_index in [0, 1, 2]:
                print(f"Trying camera index {cam_index} with default backend...")
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    print(f"Successfully opened camera at index {cam_index}")
                    break
                cap.release()
        
        if cap is None or not cap.isOpened():
            print("Error: Could not open camera with any method.")
            print("\nFor modern Raspberry Pi OS, install picamera2:")
            print("  sudo apt install -y python3-picamera2")
            print("\nOr check:")
            print("  1. Camera is properly connected")
            print("  2. User has permissions to access /dev/video*")
            return
        
        # Set camera resolution (optional, adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    print("Starting pose estimation... Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame based on camera type
            if use_picamera2:
                frame = picam2.capture_array()
                # picamera2 gives RGB, convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
            
            # Preprocess frame
            input_image = preprocess_frame(frame, input_size)
            
            # Run inference
            keypoints = run_inference(interpreter, input_image)
            
            # Draw annotations on frame
            frame = draw_skeleton(frame, keypoints)
            frame = draw_keypoints(frame, keypoints)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('MoveNet Pose Estimation', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        if use_picamera2 and picam2:
            picam2.stop()
            picam2.close()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
