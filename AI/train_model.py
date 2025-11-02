from ultralytics import YOLO
import os
import shutil

PRETRAINED_MODEL = "yolov8n.pt"
DATASET_DIR = input("Enter your dataset directory path: ")
OUTPUT_DIR = input("Enter output directory (default: train_output): ").strip() or "train_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(PRETRAINED_MODEL)

model.train(
    data=os.path.join(DATASET_DIR, "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    project=OUTPUT_DIR,
    name="run",
)

best_model_path = os.path.join(OUTPUT_DIR, "run", "weights", "best.pt")
final_best_path = os.path.join(OUTPUT_DIR, "best.pt")

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, final_best_path)
    print(f"✓ Training complete!")
    print(f"  Best model: {final_best_path}")
else:
    print("Warning: best.pt not found in expected location")
