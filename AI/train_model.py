from ultralytics import YOLO
import os

PRETRAINED_MODEL = "yolov8n.pt"
DATASET_DIR = input("Enter your dataset directory path: ")  # replace with the actual path

# Load model (pretrained)
model = YOLO(PRETRAINED_MODEL)

# Train
model.train(
    data=os.path.join(DATASET_DIR, "data.yaml"),  # full path to your YAML
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
)
