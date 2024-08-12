# from ultralytics import YOLO
# import cv2 as cv

# # Load the YOLOv5 model 
# model = YOLO('yolov8m.pt')
# results=model.track('videos/people.mp4',save=True,persist=True,classes=[0],show=True)


import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8m.pt')  # Use 'yolov8n.pt' for YOLOv8 Nano or 'yolov8s.pt' for YOLOv8 Small, etc.

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=70, n_init=3, nn_budget=100)

# Load video
video_path = 'videos/people.mp4'
cap = cv2.VideoCapture(video_path)

# Output video writer
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define bounding box shrinkage factor
shrink_factor = 0.3  # Adjust this value to shrink the boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 detection
    results = model(frame)[0]
    
    # Extract person detections (YOLO class 0 is person)
    detections = []
    for box in results.boxes:
        if box.cls == 0:  # 0 is the class index for person in YOLO
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            
            # Apply shrinkage to the bounding box
            width = x2 - x1
            height = y2 - y1
            new_width = int(width * shrink_factor)
            new_height = int(height * shrink_factor)
            new_x1 = x1 + (width - new_width) // 2
            new_y1 = y1 + (height - new_height) // 2
            new_x2 = new_x1 + new_width
            new_y2 = new_y1 + new_height
            
            detections.append(([new_x1, new_y1, new_x2, new_y2], conf))
    
    # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track ID
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Optionally display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
