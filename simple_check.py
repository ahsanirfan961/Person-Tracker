from ultralytics import YOLO
import cv2 as cv

# Load the YOLOv5 model 
model = YOLO('yolov8m.pt') # change the model path here.
results=model.track('videos/people.mp4',save=True,persist=True,classes=[0],show=True)