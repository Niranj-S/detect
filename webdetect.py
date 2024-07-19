import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('tankss.pt')

# Load the data.yaml file
model.yaml_file = 'data.yaml'

# Initialize the webcam
cap = cv2.VideoCapture(0)
confidence = 0.6
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, conf = confidence)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
