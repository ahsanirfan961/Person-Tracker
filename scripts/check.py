from ultralytics import YOLO
import torch
# Load YOLOv8 model
model = YOLO("models/yolov8n.pt").to('cuda')  # Ensure the model file path is correct

# Track objects in a video file
results=model.track(rf'videos/cctv.mp4',save=True,classes=[0],persist=True, tracker="bytetrack.yaml",device='cuda')

