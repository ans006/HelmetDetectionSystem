import torch
import cv2
import time
import os
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import winsound


# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("✅ Available classes:", model.names)

print("✅ Model loaded successfully!")

# Set class names (change according to your dataset)
# Don't overwrite model.names, just extract them
class_names = model.names


# Create folder to save violations
if not os.path.exists('violations'):
    os.makedirs('violations')

# CSV Log file
log_file = 'violation_log.csv'
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=['Time', 'Violation'])
    df.to_csv(log_file, index=False)

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)

print("🚦 Helmet Detection Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Parse results
    detections = results.xyxy[0]  # Get detections for the current frame

    for *xyxy, conf, cls in detections.tolist():
            class_id = int(cls)
            label = model.names[class_id]
    
            if label not in ['helmet', 'no_helmet']:
                continue  # Ignore rider, license_plate, background
    
            x1, y1, x2, y2 = map(int, xyxy)
    
            # Determine bounding box area to detect 'close to camera'
            area = (x2 - x1) * (y2 - y1)
    
            # Skip tiny boxes (avoids false detections far away)
            if area < 1000:
                continue
    
            color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
            text = '✅ HELMET' if label == 'helmet' else '❌ NO HELMET'
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
            print(f"Detected class: {label}, Area: {area}, Confidence: {conf:.2f}")
    
            if label == 'no_helmet':
                winsound.PlaySound("mixkit-data-scaner-2847.wav", winsound.SND_FILENAME)
                # Draw red box for no helmet
                color = (0, 0, 255)
                # Save violation
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'violations/violation_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
    
                # Log to CSV
                df = pd.read_csv(log_file)
                df.loc[len(df.index)] = [timestamp, 'No Helmet Detected']
                df.to_csv(log_file, index=False)
    
            # Display live feed
            cv2.imshow("Helmet Detection", frame)
    
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🛑 Detection Stopped.")
