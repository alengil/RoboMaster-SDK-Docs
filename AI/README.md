# Chair Detection Project

This project implements a chair detection system using YOLOv8 object detection model with robot control capabilities.

## Overview

The project consists of three main components:
1. **Automated data capture** - Script for robot control and image capture
2. **Model training** - YOLOv8 model training using Roboflow dataset
3. **Real-time detection** - Live chair detection through robot camera

## Project Structure

```
AI/
├── auto_capture.py     # Robot control and automatic image capture
├── train_model.py      # Model training script
└── yolo_camera.py      # Real-time chair detection
```

## Setup Instructions

### Prerequisites
- Python virtual environment (Robomaster environment recommended)
- Roboflow account
- Robot with camera capabilities

### Installation

**Install dependencies:**
```bash
pip install roboflow
```

## Usage

### 1. Data Capture
Run the automatic capture script to collect training images:
```bash
python AI/auto_capture.py
```
- Control robot using keyboard
- Captures one image per second automatically

### 2. Dataset Preparation
1. Log into Roboflow
2. Create new project
3. Select "Object Detection" option
4. Upload captured images
5. Annotate chairs in images (use AI assistance for speed)
6. Create dataset version with:
   - **Split:** 80% train, 20% validation
   - **Preprocessing:** Resize - Stretch to 640x640
   - **Augmentation:** Rotation, Shear, Blur
7. Download dataset in YOLOv8 format
8. Use provided download code in your project directory

### 3. Model Training
Train the YOLOv8 model:
```bash
python AI/train_model.py
```
- DATASET_DIR needs to be adjusted first
- The trained model will be saved at: `<DATASET_DIR>/runs/detect/train/weights/best.pt`

### 4. Real-time Detection
Run live chair detection:
```bash
python AI/yolo_camera.py
```
- MODEL_PATH needs to be adjusted first
- This will display the robot camera feed with chair detection overlays.

## Features

- **Automated data collection** with robot control
- **Custom dataset creation** with proper train/validation split
- **Data augmentation** for improved model robustness
- **Real-time detection** capabilities
- **YOLOv8 integration** for state-of-the-art object detection

## Dataset Configuration

- **Training split:** 80% training, 20% validation
- **Image resolution:** 640x640 pixels
- **Augmentations:** Rotation, shear, and blur for data diversity
- **Annotation format:** YOLOv8 compatible

## Notes

- Ensure varied training images for better model generalization
- The built-in AI annotation feature in Roboflow can significantly speed up the annotation process
- Virtual environment setup is recommended for dependency management
