import cv2
import torch
import numpy as np
from PIL import Image
import serial

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Set video source (webcam or video file)
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
classes = ['Drone']

# Initialize serial connection (adjust port as needed)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

while True:
    # Read frame from video source
    ret, frame = cap.read()
    
    # Convert the frame to a format that YOLOv5 can process
    img = Image.fromarray(frame[...,::-1])
    
    # Run inference on the frame
    results = model(img, size=640)
    
    drone_detected = False
    
    # Process the results and draw bounding boxes on the frame
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and classes[int(cls)] in classes:
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Display the confidence score above the box
            text_conf = "{:.2f}%".format(conf * 100)
            cv2.putText(frame, text_conf, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display the bounding box coordinates below the box
            text_coords = "({}, {})".format(int((x1 + x2) / 2), int(y2))
            cv2.putText(frame, text_coords, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Check if confidence is above 90%
            if conf > 0.7:
                drone_detected = True
                cv2.putText(frame, "Drone Detected! Confidence > 90%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Send serial data based on detection
    if drone_detected:
        ser.write(b'1')
    else:
        ser.write(b'0')
    print(drone_detected)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Wait for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video source, close the window, and close serial connection
cap.release()
cv2.destroyAllWindows()
ser.close()
