# 🧠 YOLO11 FINE-TUNING FOR VISUAL DAMAGE DETECTION

## 🚀 Pretrained Model → Fine-Tuning → Evaluation → ONNX → FastAPI → Docker → Cloud

### 🤖 End-to-End Computer Vision & Machine Learning Engineering Project

<p align="center">
  <img src="docs/images/architecture.png" alt="YOLO11 Fine-Tuning Architecture" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![ONNX](https://img.shields.io/badge/ONNX-Inference-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue?logo=docker)
![Cloud](https://img.shields.io/badge/Cloud-Deployment-4285F4)

</p>

---

## 📖 About This Project

This project demonstrates the process of taking a **pretrained YOLO11 object detection model and fine-tuning it for a specialized visual damage detection task**.

The project covers the complete machine learning lifecycle:

```text
Pretrained YOLO11
        ↓
Dataset Engineering
        ↓
Image Curation
        ↓
Human Annotation
        ↓
YOLO Dataset Preparation
        ↓
Train / Validation / Test Split
        ↓
Transfer Learning
        ↓
YOLO11 Fine-Tuning
        ↓
Model Evaluation
        ↓
Inference Testing
        ↓
ONNX Export
        ↓
ONNX Runtime
        ↓
FastAPI Model Serving
        ↓
Docker
        ↓
Cloud Deployment
        ↓
Production Inference
```

The main objective is to demonstrate practical experience with:

- Transfer Learning
- Fine-Tuning pretrained models
- Computer Vision
- YOLO Object Detection
- Deep Learning
- Model Evaluation
- ONNX
- ONNX Runtime
- FastAPI
- Docker
- Cloud Deployment
- MLOps

---

# 🎯 Project Objective

The goal is to take a **pretrained YOLO11 model**, adapt it to a specialized computer vision problem through fine-tuning, evaluate the resulting model, and build a complete inference system around it.

The model produces:

```text
Damage Class
      +
Bounding Box
      +
Confidence Score
```

Example:

```json
{
  "detections": [
    {
      "damage_type": "screen_crack",
      "confidence": 0.94,
      "bbox": [120, 80, 650, 720]
    }
  ]
}
```

The machine learning model focuses on **visual perception and object detection**.

Application-specific business decisions remain outside the model.

---

# 🧠 Core Skill — Fine-Tuning

The primary skill demonstrated by this project is **fine-tuning a pretrained object detection model**.

The model is not trained from random initialization.

Instead:

```text
             PRETRAINED YOLO11
                     │
                     ▼
              Learned Weights
                     │
                     ▼
            Domain-Specific Data
                     │
                     ▼
                 Fine-Tuning
                     │
                     ▼
          Specialized YOLO11 Model
```

Fine-tuning adapts the pretrained model to a specialized visual detection problem.

---

# 🔬 Fine-Tuning Workflow

```text
                    YOLO11 PRETRAINED MODEL
                              │
                              ▼
                       Dataset Preparation
                              │
                              ▼
                         CVAT Annotation
                              │
                              ▼
                       YOLO Dataset Format
                              │
                              ▼
                    Train / Validation / Test
                              │
                              ▼
                        YOLO11 Fine-Tuning
                              │
                              ▼
                         Model Evaluation
                              │
                              ▼
                         Trained Weights
                              │
                              ▼
                          ONNX Export
                              │
                              ▼
                       ONNX Runtime
                              │
                              ▼
                           FastAPI
                              │
                              ▼
                       REST Inference API
                              │
                              ▼
                            Docker
                              │
                              ▼
                       Cloud Deployment
```

---

# ⚖️ Fine-Tuning vs Training From Scratch

## Training From Scratch

```text
Random Initialization
        ↓
Large Dataset
        ↓
Learn Visual Representations
        ↓
Learn Detection Task
        ↓
Trained Model
```

## Fine-Tuning

```text
Pretrained Weights
        ↓
Domain-Specific Dataset
        ↓
Fine-Tuning
        ↓
Specialized Detection Model
```

Fine-tuning allows an existing pretrained model to be adapted to a specialized domain without rebuilding the entire model from random initialization.

---

# 🧠 Transfer Learning

Transfer learning means using knowledge learned by a model from a previous training task as a starting point for a new task.

In this project:

```text
Source
General Pretrained YOLO11
        │
        ▼
Transfer Learning
        │
        ▼
Target
Specialized Visual Damage Detection
```

The pretrained model provides an initial set of learned parameters which are adapted during fine-tuning.

---

# 👁️ Why Object Detection?

A classification model answers:

```text
Does this image contain damage?
```

A practical visual inspection system requires more information:

```text
What type of damage?
Where is the damage?
How many damaged regions exist?
How confident is the model?
```

Object detection provides:

```text
Class
  +
Bounding Box
  +
Confidence
```

This makes YOLO suitable for applications where both **classification and localization** are required.

---

# 🏗️ End-to-End Architecture

```text
                         DATA LAYER
                              │
                              ▼
                         Raw Images
                              │
                              ▼
                       Data Curation
                              │
                              ▼
                           CVAT
                     Human Annotation
                              │
                              ▼
                       YOLO Dataset
                              │
                              ▼
                  Train / Validation / Test
                              │
                              ▼
                       YOLO11 Fine-Tuning
                              │
                              ▼
                       Model Evaluation
                              │
                              ▼
                            best.pt
                              │
                              ▼
                         ONNX Export
                              │
                              ▼
                           best.onnx
                              │
                              ▼
                        ONNX Runtime
                              │
                              ▼
                           FastAPI
                              │
                              ▼
                       REST Inference API
                              │
                              ▼
                            Docker
                              │
                              ▼
                      Cloud Infrastructure
                              │
                              ▼
                       Production Inference
```

---

# 📊 Dataset Engineering

Model quality starts with dataset quality.

The dataset pipeline follows:

```text
Raw Images
    ↓
Validation
    ↓
Curation
    ↓
Duplicate Checking
    ↓
Quality Review
    ↓
Human Annotation
    ↓
YOLO Conversion
    ↓
Dataset Splitting
    ↓
Training
```

The dataset is treated as a versioned machine learning artifact rather than simply a collection of images.

---

# 🏷️ Human Annotation

CVAT is used for visual annotation.

The annotation process provides the ground truth required for supervised object detection training.

```text
Image
  ↓
Human Review
  ↓
Object / Damage Identification
  ↓
Bounding Box
  ↓
Class Assignment
  ↓
Ground Truth
```

Human-reviewed annotations are used as the reference for model training and evaluation.

---

# 📦 YOLO Dataset Format

YOLO object detection labels contain:

```text
class_id
center_x
center_y
width
height
```

Example:

```text
0 0.512 0.431 0.245 0.318
```

Coordinates are normalized relative to the image dimensions.

---

# 🔐 Data Leakage Prevention

Data leakage can make model evaluation appear better than the model's actual ability to generalize.

When multiple images represent the same physical object or related sample, they should not be distributed independently across training and testing datasets.

### ❌ Incorrect

```text
Sample A → Image 1 → Train
Sample A → Image 2 → Validation
Sample A → Image 3 → Test
```

### ✅ Correct

```text
Sample A → Train

Sample B → Validation

Sample C → Test
```

The objective is:

```text
Train
  ↓
Learn

Validation
  ↓
Develop / Tune

Test
  ↓
Final Evaluation
```

The test set should remain isolated from model training and repeated tuning.

---

# 🏋️ YOLO11 Fine-Tuning

The training lifecycle follows:

```text
Input Image
     ↓
Preprocessing
     ↓
Forward Pass
     ↓
Prediction
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Gradient Calculation
     ↓
Optimizer
     ↓
Weight Update
     ↓
Next Training Step
```

Important concepts involved include:

- Parameters
- Weights
- Biases
- Forward Pass
- Loss
- Backpropagation
- Gradients
- Optimizer
- Learning Rate
- Epoch
- Batch
- Iteration
- AdamW
- Mixed Precision
- GPU Acceleration

---

# ⚡ GPU Training

The model is trained using GPU acceleration.

The training environment uses:

```text
NVIDIA GPU
     ↓
CUDA
     ↓
PyTorch
     ↓
Ultralytics YOLO
     ↓
YOLO11 Fine-Tuning
```

GPU acceleration enables the model to perform deep learning tensor operations efficiently.

---

# 📈 Model Evaluation

The model is evaluated using standard object detection metrics:

- Precision
- Recall
- IoU
- AP
- mAP50
- mAP50-95

### Precision

Measures how many predicted detections are correct.

```text
Precision =
True Positives
-------------------------
True Positives + False Positives
```

### Recall

Measures how many actual objects are successfully detected.

```text
Recall =
True Positives
-------------------------
True Positives + False Negatives
```

### IoU

Intersection over Union measures overlap between the predicted bounding box and the ground-truth bounding box.

```text
IoU =
Intersection Area
-----------------
Union Area
```

### mAP50

Mean Average Precision at IoU threshold 0.50.

### mAP50-95

Average Precision averaged across IoU thresholds from 0.50 to 0.95.

---

# 📉 Overfitting & Generalization

A model can perform well on training data while performing poorly on unseen data.

This is called **overfitting**.

```text
Training Performance
        ↑
        ↑
        ↑

Validation Performance
        ↓
```

The objective is not to memorize the training dataset.

The objective is to learn visual patterns that generalize to unseen images.

Therefore:

```text
Training Performance
        ≠
Real-World Performance
```

Proper validation, testing, dataset diversity, and visual inspection are required before treating a model as production-ready.

---

# 🧪 Inference

Inference means using a trained model to generate predictions on new images.

## Training

```text
Images + Labels
      ↓
Learn Model Parameters
```

## Inference

```text
New Image
    ↓
Trained Model
    ↓
Prediction
```

Inference normally does not update the model weights.

---

# 🎯 Confidence Score

A detection contains a confidence score.

Example:

```json
{
  "damage_type": "screen_crack",
  "confidence": 0.94
}
```

Confidence thresholds can be used to filter predictions.

Production thresholds should be selected using validation data and the consequences of false positives and false negatives rather than choosing an arbitrary value.

---

# 📦 Model Artifacts

Training produces model checkpoints such as:

```text
best.pt
last.pt
```

The selected model can then be exported for deployment:

```text
best.pt
   ↓
ONNX Export
   ↓
best.onnx
```

---

# ⚡ ONNX

ONNX provides a standardized representation of a trained machine learning model.

The deployment flow is:

```text
YOLO11 / PyTorch
        ↓
      best.pt
        ↓
    ONNX Export
        ↓
     best.onnx
        ↓
   ONNX Runtime
        ↓
      Inference
```

This creates a separation between:

```text
Training Environment
        │
        │
        ▼
Inference Environment
```

The model can therefore be deployed using an inference runtime without requiring the complete training workflow.

---

# 🚀 FastAPI Model Serving

The fine-tuned model is exposed through a FastAPI inference service.

```text
Client
   │
   │ HTTP Request
   ▼
FastAPI
   │
   ▼
ONNX Runtime
   │
   ▼
Fine-Tuned YOLO11
   │
   ▼
Post Processing
   │
   ▼
JSON Response
```

Example:

```json
{
  "detections": [
    {
      "damage_type": "screen_crack",
      "confidence": 0.94,
      "bbox": [120, 80, 650, 720]
    }
  ]
}
```

---

# 📚 OpenAPI & Swagger

FastAPI automatically provides an OpenAPI specification for the inference service.

This provides interactive API documentation and makes the model-serving service easier to integrate with other applications.

```text
FastAPI
   ↓
OpenAPI Schema
   ↓
Swagger UI
   ↓
API Testing
```

---

# 🐳 Docker

The inference service can be packaged together with its runtime dependencies:

```text
FastAPI
+
ONNX Runtime
+
Model
+
Python Dependencies
        ↓
Docker Image
```

Docker provides a reproducible runtime environment for deployment.

---

# ☁️ Cloud Deployment Architecture

The production deployment architecture follows:

```text
GitHub
   ↓
CI/CD
   ↓
Docker Build
   ↓
Container Registry
   ↓
Cloud Compute
   ↓
FastAPI
   ↓
ONNX Runtime
   ↓
Fine-Tuned YOLO11
   ↓
Inference API
```

This separates model development from production inference infrastructure.

---

# 🔄 Model Lifecycle

```text
Pretrained YOLO11
        ↓
Fine-Tuning
        ↓
Evaluation
        ↓
Inference Testing
        ↓
Model Version
        ↓
ONNX Export
        ↓
Deployment
        ↓
Monitoring
        ↓
New Data
        ↓
Human Review
        ↓
Next Fine-Tuning Cycle
```

Model improvements should happen through controlled dataset and model versions rather than uncontrolled automatic retraining.

---

# 🧠 Active Learning

A future model improvement workflow can use the existing model to assist annotation:

```text
Unlabelled Images
       ↓
Current Model
       ↓
Predictions
       ↓
Human Review
       ↓
Corrected Annotations
       ↓
Approved Dataset
       ↓
Next Fine-Tuning Cycle
```

Human validation remains part of the process.

---

# 🛡️ Production Engineering Principles

The project follows:

- Human-reviewed ground truth
- Leakage-safe dataset splitting
- Isolated test data
- Versioned datasets
- Versioned models
- Reproducible experiments
- Visual prediction inspection
- Confidence-based filtering
- Human review for uncertain predictions
- Controlled retraining
- Separation of model inference and business logic
- Secure handling of private data

---

# 🧩 Technology Stack

## Computer Vision

- YOLO11
- Ultralytics
- OpenCV
- Pillow

## Deep Learning

- PyTorch
- CUDA
- NVIDIA GPU
- AMP / Mixed Precision
- AdamW

## Dataset & Annotation

- CVAT
- Python
- YOLO Dataset Format
- CSV
- JSON

## Model Deployment

- ONNX
- ONNX Runtime
- FastAPI
- Pydantic
- OpenAPI

## Engineering

- Python
- Git
- Docker
- REST API

## Cloud / MLOps

- Azure
- Container Registry
- Kubernetes
- CI/CD
- Model Versioning
- Dataset Versioning
- Monitoring

---

# 📁 Repository Structure

```text
yolo11-fine-tuning-visual-damage-detection/
│
├── README.md
│
├── docs/
│   ├── images/
│   │   └── architecture.png
│   │
│   ├── dataset.md
│   ├── annotation.md
│   ├── fine-tuning.md
│   ├── evaluation.md
│   ├── inference.md
│   └── deployment.md
│
├── src/
│   ├── data/
│   ├── training/
│   ├── inference/
│   └── serving/
│
├── configs/
│   ├── dataset.yaml
│   └── training.yaml
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
│
├── deployment/
│   ├── Dockerfile
│   └── k8s/
│
├── examples/
│   └── sample_predictions/
│
├── requirements.txt
│
└── .gitignore
```

---

# 📌 Project Status

```text
[✓] Dataset Engineering
[✓] Image Curation
[✓] Human Annotation Workflow
[✓] YOLO Dataset Preparation
[✓] Leakage-Safe Dataset Splitting
[✓] Transfer Learning
[✓] YOLO11 Fine-Tuning
[✓] GPU Training
[✓] Model Evaluation
[✓] Local Inference
[✓] ONNX Export
[✓] ONNX Runtime
[✓] FastAPI Model Serving
[✓] OpenAPI / Swagger Validation
[✓] Docker Deployment Architecture
[→] Cloud Production Deployment
[→] Production Monitoring
[→] Continuous Model Improvement
```

---

# 🗺️ Roadmap

## Phase 1 — Dataset & Fine-Tuning

- Dataset engineering
- Annotation
- Fine-tuning
- Experiment tracking
- Model evaluation

## Phase 2 — Model Improvement

- Expand dataset
- Improve data diversity
- Improve annotation quality
- Analyze false positives
- Analyze false negatives
- Improve generalization

## Phase 3 — Inference Engineering

- ONNX optimization
- Inference benchmarking
- FastAPI
- Docker
- API testing

## Phase 4 — Cloud

- Container Registry
- Cloud Compute
- CI/CD
- Secure configuration
- Model versioning
- Monitoring

## Phase 5 — Continuous Improvement

- Active learning
- Model-assisted annotation
- Dataset expansion
- Controlled fine-tuning
- Model comparison
- Production model approval

---

# 🏆 Skills Demonstrated

```text
Transfer Learning
Fine-Tuning
YOLO11
Object Detection
Computer Vision
Dataset Engineering
CVAT
Ground Truth
Data Leakage Prevention
Train / Validation / Test Splitting
PyTorch
CUDA
GPU Training
AMP
AdamW
Model Evaluation
Precision / Recall
IoU
mAP
Overfitting
Generalization
Inference
ONNX
ONNX Runtime
FastAPI
Pydantic
OpenAPI
Docker
Azure
Kubernetes
CI/CD
MLOps
Model Versioning
Dataset Versioning
```

---

# 🎯 End-to-End ML Engineering

The primary objective of this project is not simply to train a YOLO model.

It demonstrates the complete engineering journey:

```text
                    PRETRAINED MODEL
                           │
                           ▼
                    TRANSFER LEARNING
                           │
                           ▼
                       FINE-TUNING
                           │
                           ▼
                        EVALUATION
                           │
                           ▼
                        INFERENCE
                           │
                           ▼
                          ONNX
                           │
                           ▼
                     ONNX RUNTIME
                           │
                           ▼
                        FASTAPI
                           │
                           ▼
                         DOCKER
                           │
                           ▼
                    CLOUD DEPLOYMENT
                           │
                           ▼
                   PRODUCTION INFERENCE
                           │
                           ▼
                       MONITORING
                           │
                           ▼
                  CONTINUOUS IMPROVEMENT
```

---

# 👨‍💻 Engineering Portfolio

This project is part of my personal **AI / ML Engineering Portfolio**.

The primary engineering skill demonstrated is:

> **Taking a pretrained YOLO11 model, fine-tuning it for a specialized object detection problem, evaluating the model, exporting it to ONNX, and building a production-oriented API inference system around it.**

The project combines:

**Computer Vision + Transfer Learning + Fine-Tuning + Deep Learning + Model Optimization + API Development + Docker + Cloud + MLOps**

---

## ⭐ Project Focus

```text
PRETRAINED YOLO11
        ↓
TRANSFER LEARNING
        ↓
FINE-TUNING
        ↓
MODEL EVALUATION
        ↓
ONNX
        ↓
FASTAPI
        ↓
DOCKER
        ↓
CLOUD
        ↓
PRODUCTION INFERENCE
```

> **A production-oriented computer vision engineering project demonstrating the complete lifecycle from pretrained model fine-tuning to deployable inference.**
