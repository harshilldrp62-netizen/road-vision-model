# 🚧 AI-Based Road Damage Detection

An AI-powered road damage detection system that uses **YOLO (You Only Look Once)** for detecting and classifying different types of road damage from images.

The system allows users to upload or capture a road image, processes the image using a trained YOLO object detection model, identifies road damage, estimates its severity, and provides an interpretable visualization of the model's predictions.

---

## 📌 Project Overview

Road damage such as potholes, unpaved roads, and speed breakers can affect vehicle safety, driving comfort, and road infrastructure management.

Traditional road inspection methods generally require manual inspection, which can be time-consuming and difficult to scale.

This project aims to automate the initial road inspection process using **Computer Vision and Deep Learning**.

The system takes a road image as input and performs:

1. Image preprocessing
2. Road damage detection using YOLO
3. Damage classification
4. Bounding-box localization
5. Confidence estimation
6. Severity estimation
7. Explainable AI visualization using EigenCAM

### Supported Road Damage Classes

The current model detects the following classes:

* 🕳️ **Pothole**
* 🛣️ **Unpaved Road**
* ⚠️ **Speed Breaker**

---

# 🎯 Objectives

The main objectives of this project are:

* Detect road damage automatically from images.
* Identify the type of road damage.
* Locate detected damage using bounding boxes.
* Estimate the severity of detected damage.
* Provide confidence scores for predictions.
* Provide visual explanations of model predictions.
* Create a system that can eventually be integrated into a web or mobile application.
* Reduce the dependency on manual road inspection for preliminary assessment.

---

# 🧠 Technology Stack

| Component               | Technology                 |
| ----------------------- | -------------------------- |
| Programming Language    | Python                     |
| Deep Learning Framework | PyTorch                    |
| Object Detection        | YOLO                       |
| YOLO Implementation     | Ultralytics                |
| Computer Vision         | OpenCV                     |
| Numerical Processing    | NumPy                      |
| Explainable AI          | EigenCAM                   |
| Model Visualization     | PyTorch Grad-CAM           |
| Development             | Python / Jupyter / VS Code |
| Version Control         | Git & GitHub               |

---

# 🏗️ System Architecture

The overall pipeline of the system is:

```text
                ┌─────────────────────┐
                │    Input Image      │
                │  Road Image / Photo │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Image Preprocessing │
                │ Resize / Normalize  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   YOLO Detection    │
                │   Model Inference   │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Class Label   Confidence   Bounding Box
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Severity Estimation │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   EigenCAM / XAI    │
                │ Prediction Heatmap  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Final Result / UI   │
                └─────────────────────┘
```

---

# 🔍 How the System Works

## 1. Image Input

The user provides an image containing a road.

The image can come from:

* A mobile camera
* A mobile gallery
* A desktop/laptop
* A web application
* Any other compatible image source

---

## 2. Image Preprocessing

Before inference, the image is prepared for the YOLO model.

Typical preprocessing includes:

* Reading the image using OpenCV
* Converting the image into the appropriate format
* Resizing according to the model's requirements
* Preparing the image for model inference

The preprocessing pipeline ensures that the input is compatible with the trained model.

---

## 3. YOLO Object Detection

The processed image is passed to the trained YOLO model.

YOLO performs object detection by predicting:

* Class
* Bounding box
* Confidence score

For example:

```text
Detected Object:
Class: Pothole
Confidence: 0.91
Bounding Box: [x1, y1, x2, y2]
```

Multiple road damages can be detected in a single image.

---

## 4. Damage Classification

The model classifies the detected object into one of the supported categories:

```text
Pothole
Unpaved Road
Speed Breaker
```

The predicted class is displayed along with the detection confidence.

---

## 5. Bounding Box Localization

For every detected damage, YOLO generates a bounding box.

The bounding box identifies the approximate location of the damage within the image.

Example:

```text
┌────────────────────────────────────┐
│                                    │
│       Road                         │
│                                    │
│          ┌──────────────┐          │
│          │   Pothole    │          │
│          └──────────────┘          │
│                                    │
└────────────────────────────────────┘
```

---

# 📊 Confidence Score

The YOLO model provides a confidence score for every detection.

The confidence score represents how strongly the model believes that the detected region belongs to a particular class.

Example:

```text
Pothole       → 0.94
Speed Breaker → 0.87
Unpaved Road  → 0.91
```

A configurable confidence threshold can be used to filter low-confidence predictions.

---

# ⚠️ Severity Estimation

The project also includes a severity estimation stage.

The purpose of severity estimation is to categorize detected road damage according to its estimated impact.

A simplified severity classification can be represented as:

```text
Low Severity
     ↓
Moderate Severity
     ↓
High Severity
```

The severity estimation can use visual characteristics such as:

* Size of detected region
* Bounding-box dimensions
* Relative area of the damage
* Detection confidence
* Damage category

### Important Note

The severity estimation is an **image-based heuristic**, not an engineering-grade measurement of road damage.

Actual road damage assessment would require additional information such as:

* Real-world dimensions
* Depth
* Road geometry
* Camera calibration
* Distance from camera
* LiDAR/depth information
* Multiple viewpoints

Therefore, severity predictions should be considered an **approximate preliminary assessment**.

---

# 🧩 Explainable AI — EigenCAM

To make the model's predictions more interpretable, the project uses **EigenCAM**.

Instead of only showing the predicted bounding box, the system can generate a heatmap showing the regions of the image that contribute strongly to the model's prediction.

### Why Explainability?

Deep learning models are often considered black-box systems.

For road damage detection, it is useful to understand whether the model is actually focusing on the damaged road region.

EigenCAM helps visualize this behavior.

The pipeline is approximately:

```text
Input Image
     ↓
YOLO Model
     ↓
Feature Activations
     ↓
EigenCAM
     ↓
Activation Heatmap
     ↓
Overlay on Original Image
```

This provides a visual explanation of the model's attention.

---

# 📁 Project Structure

A recommended repository structure is:

```text
road-damage-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── model_pipeline.py
│   ├── inference.py
│   ├── severity.py
│   └── utils.py
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── model_training.ipynb
│
├── outputs/
│   ├── predictions/
│   └── explainability/
│
├── screenshots/
│   ├── detection_result.png
│   └── eigen_cam_result.png
│
└── dataset/
    └── README.md
```

> Adjust the structure according to the actual files in your project. Do not upload unnecessary generated files just to match this example.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/road-damage-detection.git
```

Move into the project directory:

```bash
cd road-damage-detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the major dependencies include:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install torch
pip install torchvision
pip install grad-cam
```

---

# 🧪 Running the Model

After installing the dependencies and placing the trained model in the appropriate location, run the inference pipeline.

Example:

```bash
python src/inference.py
```

Depending on the implementation, you may need to modify the input image path:

```python
image_path = "path/to/your/image.jpg"
```

The trained YOLO model can then be loaded using:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
```

---

# 🖼️ Example Inference

The model takes a road image:

```text
Input Image
     ↓
YOLO Model
     ↓
Detection
     ↓
┌─────────────────────────────┐
│ Pothole                     │
│ Confidence: 0.91            │
│ Severity: High              │
└─────────────────────────────┘
```

The output image contains the detected road damage along with its bounding box and prediction information.

---

# 📈 Model Training

The YOLO model was trained on a road damage dataset containing images representing different types of road conditions.

The training process generally consists of:

```text
Dataset
   ↓
Data Annotation
   ↓
Train / Validation Split
   ↓
YOLO Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Best Model Selection
```

---

# 🏷️ Dataset Annotation

Object detection datasets require annotations containing the class and bounding-box coordinates of each object.

YOLO annotation format:

```text
class_id x_center y_center width height
```

All coordinates are normalized between `0` and `1`.

Example:

```text
0 0.521 0.614 0.231 0.184
```

---

# 📂 Dataset Organization

A typical YOLO dataset follows this structure:

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

Example `data.yaml`:

```yaml
path: /path/to/dataset

train: images/train
val: images/val

names:
  0: pothole
  1: unpaved_road
  2: speed_breaker
```

---

# 🤖 YOLO Training Example

A typical Ultralytics training command is:

```bash
yolo detect train model=yolo_model.pt data=data.yaml epochs=50 imgsz=640
```

The exact model, number of epochs, image size, batch size, and other hyperparameters depend on the experiment.

---

# 📊 Model Evaluation

The trained model can be evaluated using standard object detection metrics such as:

* Precision
* Recall
* mAP@50
* mAP@50-95

### Precision

Precision measures how many of the detected objects were actually correct.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

Recall measures how many actual objects were successfully detected.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### mAP

Mean Average Precision is commonly used to evaluate object detection models across different classes and confidence thresholds.

---

# 🔬 Model Pipeline

The main processing pipeline can be summarized as:

```python
Input Image
    ↓
OpenCV Image Loading
    ↓
Image Preprocessing
    ↓
YOLO Inference
    ↓
Detection Results
    ↓
Bounding Boxes
    ↓
Class + Confidence
    ↓
Severity Estimation
    ↓
EigenCAM
    ↓
Visualization
```

---

# 🧠 Model Pipeline Implementation

The project contains a model pipeline responsible for loading the trained YOLO model, performing inference, processing detections, and generating explainability visualizations.

The pipeline uses components such as:

```python
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
```

The exact implementation is available in the source files included in this repository.

---

# 🔥 Why YOLO?

YOLO was selected because it provides:

* Fast inference
* Real-time object detection capability
* Good detection performance
* Bounding-box localization
* Support for multiple objects in one image
* Easy deployment
* Strong ecosystem through Ultralytics

These characteristics make YOLO suitable for a road damage detection application where fast prediction can be useful.

---

# 💡 Key Features

### 🚧 Road Damage Detection

Detects different types of road damage from an input image.

### 🎯 Object Localization

Draws bounding boxes around detected road damage.

### 📊 Confidence Estimation

Displays model confidence for each detection.

### ⚠️ Severity Estimation

Provides an approximate severity category based on visual characteristics.

### 🔥 Explainable AI

Uses EigenCAM to visualize areas influencing the model's predictions.

### 📱 Application Ready

The detection pipeline can be integrated into a web or mobile application.

### 🖼️ Image Upload

Supports processing road images provided by the user.

---

# 🌐 Possible Application Workflow

The complete application can be designed as:

```text
User
 │
 ▼
Upload / Capture Road Image
 │
 ▼
Frontend
 │
 ▼
Backend / ML Pipeline
 │
 ▼
Image Preprocessing
 │
 ▼
YOLO Model
 │
 ▼
Road Damage Detection
 │
 ├── Damage Type
 ├── Confidence
 ├── Bounding Box
 └── Severity
 │
 ▼
EigenCAM Visualization
 │
 ▼
Result Display
```

---

# 🛡️ Limitations

Although the project demonstrates the use of deep learning for automated road damage detection, it has several limitations.

### 1. Image-Based Detection

The model only analyzes the visual information available in the input image.

It cannot directly measure physical damage depth.

### 2. Lighting Conditions

Performance may decrease under:

* Very low light
* Strong shadows
* Overexposure
* Night-time conditions

### 3. Camera Angle

Unusual camera angles may affect detection performance.

### 4. Occlusion

Vehicles, pedestrians, water, mud, or other objects may partially hide road damage.

### 5. Severity Approximation

The severity estimation is not a substitute for professional road inspection.

### 6. Dataset Dependency

Model performance depends heavily on:

* Dataset quality
* Dataset size
* Class distribution
* Annotation accuracy
* Environmental diversity

---

# 🚀 Future Improvements

The project can be further improved by adding:

## 1. Real-Time Video Detection

Instead of processing individual images, the model can process live camera video.

```text
Camera
  ↓
Video Frames
  ↓
YOLO
  ↓
Real-Time Detection
```

---

## 2. Mobile Application

The model can be integrated into an Android/iOS application where users can capture road images directly.

---

## 3. GPS-Based Damage Mapping

Each detected road damage can be associated with GPS coordinates.

This could enable:

```text
Road Damage
     ↓
GPS Location
     ↓
Database
     ↓
Interactive Map
```

This would allow authorities to visualize damaged roads geographically.

---

## 4. Road Damage Reporting System

Users could submit detected road damage to a centralized database.

A report could contain:

```text
Damage Type
Severity
Image
Location
Timestamp
Confidence
```

---

## 5. Improved Severity Estimation

Future versions could use:

* Depth estimation
* Stereo cameras
* LiDAR
* Camera calibration
* Multiple images
* Real-world measurements

to estimate actual damage dimensions more accurately.

---

## 6. More Damage Classes

Additional classes could be added, such as:

* Cracks
* Road depressions
* Rutting
* Surface deterioration
* Manhole damage
* Faded road markings

---

## 7. Larger and More Diverse Dataset

The model can be improved by collecting road images under:

* Different weather conditions
* Different lighting conditions
* Different road types
* Different camera angles
* Different geographical regions

---

## 8. Cloud-Based Deployment

The model could be deployed using a cloud backend and exposed through an API.

Possible architecture:

```text
Mobile/Web App
      ↓
REST API
      ↓
Cloud Server
      ↓
YOLO Model
      ↓
Prediction
      ↓
API Response
      ↓
Application
```

---

# 🔐 Privacy Considerations

The application should avoid collecting unnecessary personal information.

If images are uploaded to a server, appropriate security and privacy measures should be implemented.

Potential considerations include:

* Secure image transmission
* Secure storage
* User consent
* Limited data retention
* Removal of unnecessary metadata

---

# 📌 Project Status

**Current Status:** Completed Prototype / Academic Project

The current version demonstrates the core road damage detection pipeline using YOLO along with severity estimation and explainability visualization.

Future development can focus on deployment, real-time detection, GPS integration, and improved severity estimation.

---

# 📚 Learning Outcomes

This project provided practical experience with:

* Python
* Computer Vision
* Deep Learning
* Object Detection
* YOLO
* PyTorch
* OpenCV
* Model Inference
* Dataset Preparation
* Object Detection Metrics
* Explainable AI
* Grad-CAM / EigenCAM
* Git & GitHub
* ML Model Deployment Concepts

---

# 🧑‍💻 Repository Guidelines

The repository should contain the source code and configuration required to understand and reproduce the project.

Recommended files:

```text
road-damage-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── model_pipeline.py
│   ├── inference.py
│   ├── severity.py
│   └── utils.py
│
├── notebooks/
│   └── model_training.ipynb
│
├── models/
│   └── best.pt
│
├── screenshots/
│   ├── detection_result.png
│   └── eigen_cam_result.png
│
└── dataset/
    └── README.md
```

---

# 🚫 Files That Should NOT Be Uploaded

Do **not** upload unnecessary generated files or private/local files.

Examples:

```text
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
*.log
temporary files
large generated output folders
personal images
IDE configuration files
```

If your trained model file is very large, consider using **Git LFS** or a model hosting service rather than committing a large binary directly to the normal Git history.

---

# 📄 Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
env/
.venv/

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
.env.*

# Logs
*.log

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# Generated outputs
runs/
temp/
tmp/

# Python cache
.pytest_cache/

# Large local datasets
dataset/raw/
```

---

# 📦 requirements.txt

A basic dependency file can contain:

```text
ultralytics
torch
torchvision
opencv-python
numpy
grad-cam
```

It is recommended to generate the final `requirements.txt` from the actual environment used to run the project so that dependency versions are reproducible.

---

# 🧪 Reproducibility

To reproduce the project:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install dependencies.
4. Download or provide the trained model.
5. Provide a compatible input image.
6. Run the inference script.
7. Inspect the detection and explainability outputs.

---

# 📜 License

This project is developed for educational and academic purposes.

If you plan to distribute the project publicly or use third-party datasets/models, verify and comply with their respective licenses and terms of use.

---

# ⚠️ Disclaimer

This system is intended as an **AI-assisted road damage detection prototype**.

The predictions generated by the model should not be treated as an official engineering assessment or a replacement for professional road inspection.

Severity estimates are approximate and depend on image quality, camera perspective, dataset characteristics, and model performance.

---

# 👨‍💻 Author

**Harshil Prajapati**

Computer Engineering Student
India

---

# ⭐ Acknowledgements

This project makes use of open-source technologies and libraries including:

* Ultralytics YOLO
* PyTorch
* OpenCV
* NumPy
* PyTorch Grad-CAM / EigenCAM

Special thanks to the open-source computer vision and deep learning community for providing the tools and resources used in this project.

---

# ⭐ If You Find This Project Useful

If this project helped you understand road damage detection, YOLO, or explainable AI, consider giving the repository a ⭐ on GitHub.

---

## 🔎 Project Summary

**AI-Based Road Damage Detection** is a computer vision project that uses YOLO to automatically detect **potholes, unpaved roads, and speed breakers** from road images.

The system combines:

```text
YOLO
+
Computer Vision
+
Severity Estimation
+
Explainable AI
+
EigenCAM
```

to create an AI-assisted road inspection prototype.

The project demonstrates how deep learning and computer vision can be applied to real-world infrastructure problems and provides a foundation for future development into a mobile/web-based road monitoring and reporting system.
