import cv2
import torch
from ultralytics import YOLO
import datetime

print("Libraries imported successfully")

# Load the YOLOv8 model
model = YOLO('tankss.pt')
print("YOLO model loaded")

# Load the data.yaml file
model.yaml_file = 'data.yaml'
print("YAML file loaded")

# Input and output file paths
input_file = 'input_video.mp4'
output_file = f'{datetime.datetime.now()}.mp4'

# Open the input video file
cap = cv2.VideoCapture(input_file)
print(f"Video capture opened: {cap.isOpened()}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # We'll initialize this after processing the first frame

confidence = 0.4

frame_count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Run YOLOv8 inference on the frame
    results = model(frame, conf=confidence)
    print(f"YOLO inference completed for frame {frame_count}")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Initialize the VideoWriter with the first frame's dimensions
    if out is None:
        height, width = annotated_frame.shape[:2]
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        print(f"VideoWriter initialized with dimensions: {width}x{height}")

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame (optional, for debugging)
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted")
        break

print("Main loop completed")

# Release the video capture and writer objects
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print("Resources released and windows closed")
