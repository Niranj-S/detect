import cv2
import torch
import numpy as np
from PIL import Image
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Set input and output video paths
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the classes you want to detect
classes = ['Drone']

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.time()

while True:
    # Read frame from video source
    ret, frame = cap.read()
    if not ret:
        break

    # Get current frame number and calculate timestamp
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = current_frame / fps
    
    # Convert the frame to a format that YOLOv5 can process
    img = Image.fromarray(frame[...,::-1])
    
    # Run inference on the frame
    results = model(img, size=640)
    
    # Process the results and draw bounding boxes on the frame
    drone_detected = False
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and classes[int(cls)] in classes:
            drone_detected = True
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # Display the confidence score above the box
            text_conf = "{:.2f}%".format(conf * 100)
            cv2.putText(frame, text_conf, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Display the bounding box coordinates below the box
            text_coords = "({}, {})".format(int((x1 + x2) / 2), int(y2))
            cv2.putText(frame, text_coords, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display a warning message if a drone is detected
    if drone_detected:
        cv2.putText(frame, "Warning: Drone Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add timestamp to the frame
    cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Calculate and display progress
    progress = (current_frame / total_frames) * 100
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / (progress / 100)
    remaining_time = estimated_total_time - elapsed_time
    print(f"\rProgress: {progress:.2f}% | Estimated time remaining: {remaining_time:.2f}s", end="")

    # Wait for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video source, writer, and close the window
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nProcessing completed.")
