# 🧬 AIBuster Ensemble Pipeline

A comprehensive AI image detection system utilizing dual methods to identify synthetic and deepfake imagery.

![AI Image Detector](https://github.com/moustaphahassaan584-create/AIBusterArt/assets/detector-banner.jpg)

## 📌 Project Overview

This repository contains the graduation project implementation for detecting AI-generated images. The system evolved through two distinct methodologies, both of which are preserved here for reference and comparison.

### Method 1: The ViT-Only Approach (Original)
Found in `method_1_vit/`. This was the initial approach using a Streamlit web interface and a combination of four distinct, pre-trained models.
- **Architecture:** Averages scores from 4 generic pre-trained models (ResNet, SigLIP, SDXL Detector, ViT Deepfake Detector).
- **Visuals:** Basic FFT and Error Level Analysis (ELA).
- **Pros:** Fast to set up, requires no training.
- **Cons:** Lower accuracy (~70%) on modern models like Midjourney v6 and Flux.

### Method 2: The Ensemble Approach (Current/Advanced)
Found in `method_2_ensemble/`. This is the finalized, high-accuracy approach designed in our system diagrams.
- **Architecture:** Weighted ensemble. 50% from our **fine-tuned ViT**, 30% from backup models (SigLIP, SMOGY), and 20% from physics-based **Noise Forensics**.
- **Test-Time Augmentation (TTA):** Analyzes original, flipped, and cropped views for robustness.
- **Visuals:** FFT, ELA, and advanced Noise Pattern visualization.
- **Pros:** Extremely high accuracy (95%+), robust against modern diffusion models.

## 📂 Repository Structure

```text
AIBusterArt/
├── method_1_vit/              ← Original 4-model averaging approach (Streamlit)
├── method_2_ensemble/         ← Advanced weighted ensemble (Gradio + FastAPI)
│   └── experimental/          ← Early 9-model Gradio prototype
├── training/                  ← Jupyter notebook for fine-tuning the ViT model
├── benchmarking/              ← Scripts to evaluate model accuracy
├── docs/                      ← Detailed methodology documentation
├── Diagrams/                  ← System architecture and UML diagrams
└── app/                       ← Android Application
```

## 🚀 Getting Started

### To run the Advanced Ensemble (Method 2)
```bash
cd method_2_ensemble
pip install -r requirements.txt
python app.py
```
*The Gradio interface will launch at `http://localhost:7860` and the REST API at `http://localhost:7860/analyze`.*

### To run the Original Approach (Method 1)
```bash
cd method_1_vit/web_app
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Fine-Tuning the Model

If you want to reproduce the fine-tuned model used in Method 2:
1. Open `training/finetune_notebook.py` in Google Colab.
2. Enable a T4 GPU.
3. Update the `HF_USERNAME` to your Hugging Face account.
4. Run all cells to train and push the model to the Hub.

## 📊 Benchmarks & Documentation

- See the `benchmarking/` folder to run comparative tests between Method 1 and Method 2.
- Detailed explanations of the theory, weights, and algorithms are available in the `docs/` folder.
- System diagrams (Use Case, Activity, Class) are in the `Diagrams/` folder.
