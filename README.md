# 🧬 AIBuster Image Detection System

A comprehensive AI image detection system designed to identify synthetic and deepfake imagery.

![AI Image Detector](https://github.com/moustaphahassaan584-create/AIBusterArt/assets/detector-banner.jpg)

## 📌 Project Overview

This repository contains the graduation project implementation for detecting AI-generated images. The system evolved through two distinct methodologies, both of which are preserved here to demonstrate the project's progression from a single-model approach to a highly advanced ensemble.

### Method 1: The ViT-Only Approach (Original)
Found in `method_1_vit/`. This was our foundational approach.
- **Architecture:** A single, standalone Vision Transformer (ViT) model.
- **Approach:** This method relied entirely on fine-tuning a ViT model on a specific dataset to classify images as real or AI-generated.
- **Components:** Contains the original training scripts (`train_vit.py`, `capcheck_vit_training.ipynb`), the inference server (`server.py`), and the prediction logic (`predict.py`).

### Method 2: The Advanced Ensemble Approach (Current)
Found in `method_2_ensemble/`. This is the finalized, highly robust approach designed in our system diagrams, representing a massive upgrade in detection capability.
- **Architecture:** A sophisticated weighted ensemble combining deep learning models and mathematical forensic analysis.
- **Components:** 
  - **Fine-Tuned Model (50%):** A completely new fine-tuned model, trained on *different data* than Method 1, serving as the core detector.
  - **Mathematical/Forensic Engines (20%):** Physics-based Noise Pattern Forensics, Fast Fourier Transform (FFT), and Error Level Analysis (ELA).
  - **Pre-trained Semantic Models (30%):** Backup deep learning models (SigLIP, SMOGY) for edge cases.
- **Test-Time Augmentation (TTA):** Analyzes original, flipped, and cropped views for robustness.
- **Pros:** Extremely high accuracy (95%+), immune to simple perturbations, and robust against modern diffusion models.

## 📂 Repository Structure

```text
AIBusterArt/
├── method_1_vit/              ← The original single fine-tuned ViT pipeline
├── method_2_ensemble/         ← The advanced mathematical & multi-model ensemble (Gradio + FastAPI)
│   └── experimental/          ← Early 9-model Gradio prototype
├── training/                  ← Jupyter notebook for the new fine-tuning pipeline
├── benchmarking/              ← Scripts to evaluate model accuracy
├── docs/                      ← Detailed methodology documentation
├── android_app/               ← The Android Application source code
└── Diagrams/                  ← System architecture and UML diagrams
```

## 🚀 Getting Started

### To run the Advanced Ensemble (Method 2)
```bash
cd method_2_ensemble
pip install -r requirements.txt
python app.py
```
*The Gradio interface will launch at `http://localhost:7860` and the REST API at `http://localhost:7860/analyze`.*

## 📊 Benchmarks & Documentation

- See the `benchmarking/` folder to run comparative tests between the approaches.
- Detailed explanations of the theory, weights, algorithms, and training datasets are available in the `docs/` folder.
- System diagrams (Use Case, Activity, Class) mapping out Method 2 are in the `Diagrams/` folder.
