import cv2
from ultralytics import YOLO
import sys
import os

# 1. Load your custom trained model
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

model = YOLO(model_path)

# 2. Open the input video stream
video_path = "ab.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    sys.exit(1)

print("Press 'q' to exit.")

fps = cap.get(cv2.CAP_PROP_FPS)
delay = max(1, int(1000 / fps)) if fps and fps > 0 else 30

# Fraction to scale the annotated frame for display (e.g. 0.5 = half size).
# Set to 1.0 to keep original size.
display_scale = 0.5

while True:
    ret, frame = cap.read()

    # Loop video when it ends
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame after reset.")
            break

    # Optional: resize large frames for faster inference
    # frame = cv2.resize(frame, (640, 480))

    try:
        results = model.predict(source=frame, show=False, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
    except Exception as e:
        print(f"Inference error: {e}")
        annotated_frame = frame  # Fall back to raw frame on error

    # Resize annotated frame for display (keeps model input unchanged)
    if display_scale is not None and display_scale > 0 and display_scale < 1:
        h, w = annotated_frame.shape[:2]
        disp_w, disp_h = int(w * display_scale), int(h * display_scale)
        display_frame = cv2.resize(annotated_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    else:
        display_frame = annotated_frame

    cv2.imshow("YOLOv8 Traffic Segmentation", display_frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()