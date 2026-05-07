# AIBuster Detection System — Diagrams Explanation

> This document provides a comprehensive, detailed explanation of all UML diagrams used in the **AIBuster** graduation project. Each section breaks down every element in the diagram — actors, use cases, relationships, classes, activities, and messages — so you can fully understand how the system is designed and how it works end-to-end.

---

# 1. Use Case Diagram

## 1.1 Overview

The Use Case Diagram provides a high-level, functional view of the **AIBuster Detection System**. It answers the fundamental questions:
- **Who** interacts with the system? (Actors)
- **What** can they do? (Use Cases)
- **How** are these actions related to each other? (Relationships: include, extend, association)

The entire diagram is enclosed within a single **system boundary** labeled **"AIBuster Detection System"**, which visually separates the internal system functionality from the external actors that interact with it.

---

## 1.2 Actors

Actors represent any entity — human or system — that interacts with the AIBuster system from the outside. The diagram defines **four actors**, divided into two categories:

### 1.2.1 Primary (Human) Actors

| Actor | Stereotype | Role Description |
|-------|-----------|-----------------|
| **User** | `«Main Actor»` | The end-user who uploads images and consumes detection results. This is the primary consumer of the system. They interact through either the Gradio web interface or indirectly through the Android app. |
| **Developer** | `«Admin»` | The system administrator/ML engineer responsible for training models, deploying the application to HuggingFace Spaces, and pushing model weights to the HuggingFace Hub. This actor manages the lifecycle of the AI models. |

### 1.2.2 Secondary (System/External) Actors

| Actor | Stereotype | Role Description |
|-------|-----------|-----------------|
| **HuggingFace** | `«External»` | An external cloud platform that serves two critical roles: (1) hosting the model weights that the system downloads at startup, and (2) hosting the deployed Gradio application on HuggingFace Spaces, providing the inference endpoint. |
| **Android App** | `«Mobile Client»` | A mobile application that acts as an alternative client interface. It does not interact with the system through the Gradio UI but instead communicates directly via the REST API (`/analyze` endpoint) to send images and receive verdicts. |

### Why Four Actors?

The system is designed with a **multi-client architecture**. The User interacts through the web (Gradio), while the Android App interacts through the API. HuggingFace acts as the infrastructure provider. The Developer is separate because their use cases (training, deploying) are administrative and do not overlap with the detection workflow.

---

## 1.3 Use Cases — Detailed Breakdown

The use cases are organized into **three logical groups** based on their purpose and which actor primarily triggers them.

### 1.3.1 User-Facing Use Cases (Left Column — Blue)

These are the use cases that the **User** directly initiates or consumes. They represent the front-end experience of the system.

#### UC1: Upload Image
- **Description**: The user selects and uploads an image (JPEG, PNG, etc.) to the system for AI-generated content detection. This is the **entry point** of the entire detection pipeline.
- **Triggered by**: User (via Gradio UI) or Android App (via REST API).
- **What happens next**: Once the image is uploaded, the system automatically triggers preprocessing (UC7).

#### UC2: View Verdict (REAL / FAKE)
- **Description**: After the analysis is complete, the user sees the final binary classification result — either **"REAL"** (the image is authentic) or **"FAKE"** (the image is AI-generated).
- **Triggered by**: User or Android App.
- **Important**: This use case **extends** from the "Generate & Return Verdict" (UC13) system use case, meaning the verdict must first be generated internally before the user can view it.

#### UC3: View Confidence Score
- **Description**: Alongside the verdict, the user sees a **confidence score** (e.g., 92.5% confidence that the image is FAKE). This gives the user a sense of how certain the model is about its prediction.
- **Triggered by**: User or Android App.
- **Important**: Like UC2, this also **extends** from UC13. The confidence score is a byproduct of the ensemble aggregation step.

#### UC4: View Forensic Visuals (FFT · ELA · Noise map)
- **Description**: The user can view **three forensic visualization maps** generated during analysis:
  - **FFT (Fast Fourier Transform)**: Reveals frequency-domain artifacts. AI-generated images often have distinctive spectral patterns (e.g., grid-like artifacts from GAN architectures).
  - **ELA (Error Level Analysis)**: Highlights regions of the image that have been modified or have inconsistent compression levels. Useful for detecting local manipulations.
  - **Noise map**: Extracts and visualizes the noise residual of the image. AI-generated images tend to have unnaturally uniform noise patterns compared to real camera photos.
- **Triggered by**: User.
- **Important**: This use case **extends** from "Run FFT Analysis" (UC8), meaning the forensic visual display depends on the FFT analysis pipeline being run first.

#### UC5: Use Web Interface (Gradio)
- **Description**: The user accesses the system through a **Gradio-based web interface**. Gradio provides an interactive UI where users can drag-and-drop images, see results, and explore forensic visuals — all in a browser.
- **Triggered by**: User.
- **Relationship**: This use case **includes** UC1 (Upload Image), because using the web interface inherently involves uploading an image for analysis.

#### UC6: Use REST API /analyze
- **Description**: Instead of the Gradio UI, clients can submit images programmatically via a **REST API endpoint** (`POST /analyze`). The API accepts an image file and returns JSON results containing the verdict, confidence score, and forensic visual URLs.
- **Triggered by**: User (for testing/integration) or **Android App** (as its primary interaction method).
- **Relationship**: This use case **includes** UC1 (Upload Image), because calling the API inherently involves uploading an image.

---

### 1.3.2 Developer/Admin Use Cases (Left Column — Green)

These use cases are performed by the **Developer** actor and relate to the **model lifecycle management** — training, deploying, and distributing model weights.

#### UC-Dev1: Train & Fine-tune Model
- **Description**: The developer trains and fine-tunes the deep learning models (ViT, SigLIP, SMOGY) on datasets of real and AI-generated images. This includes:
  - Selecting training hyperparameters (learning rate, batch size, epochs)
  - Running training loops with data augmentation
  - Evaluating model performance on validation sets
  - Fine-tuning pre-trained models on domain-specific data
- **Triggered by**: Developer.
- **Note**: This is an offline process that happens before deployment. The trained weights are then pushed to HuggingFace Hub.

#### UC-Dev2: Deploy to HF Spaces
- **Description**: The developer deploys the complete Gradio application to **HuggingFace Spaces**, a cloud hosting platform. This makes the web interface publicly accessible without the user needing to install anything locally.
- **Triggered by**: Developer.
- **Note**: Deployment involves pushing the application code (including `app.py`, requirements, and configuration) to a HuggingFace Space repository.

#### UC-Dev3: Push Model Weights to Hub
- **Description**: After training, the developer uploads the trained model weight files (`.safetensors`, `.bin`, etc.) to the **HuggingFace Model Hub**. This serves as a centralized model registry that the deployed application can download from at startup.
- **Triggered by**: Developer.
- **Note**: This decouples model storage from application deployment, allowing model updates without redeploying the entire app.

---

### 1.3.3 System/Internal Use Cases (Right Column)

These use cases represent the **internal processing pipeline** of the AIBuster system. They are not directly triggered by external actors but are instead invoked through `«include»` and `«extend»` relationships from other use cases.

#### UC7: Preprocess Image (EXIF strip · CLAHE · resize)
- **Description**: The first step after image upload. The system performs several preprocessing operations:
  - **EXIF strip**: Removes metadata from the image (camera info, GPS coordinates, timestamps) to normalize the input and prevent metadata-based bias.
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast in the image, making subtle artifacts more detectable by the downstream models.
  - **Resize**: Scales the image to the required input dimensions for the deep learning models (e.g., 224×224 or 384×384 pixels).
- **Included by**: UC1 (Upload Image) — every uploaded image must be preprocessed.
- **Includes**: UC8, UC9, UC10, UC11 — after preprocessing, the image is passed to all four analysis modules in parallel.

#### UC8: Run FFT Analysis
- **Description**: Applies **Fast Fourier Transform** to convert the image from spatial domain to frequency domain. The resulting spectrum is analyzed for:
  - Grid-like artifacts (common in GAN-generated images)
  - Unusual frequency distributions
  - Periodic patterns that are invisible in spatial domain
- **Included by**: UC7 (Preprocess Image) and UC12 (Ensemble Aggregation).
- **Extended by**: UC4 (View Forensic Visuals) — the FFT output is used for the forensic visual display.

#### UC9: Run ELA Analysis
- **Description**: Applies **Error Level Analysis** by re-compressing the image at a known quality level and comparing it to the original. Areas with different error levels indicate:
  - Regions that were pasted from another image
  - AI-generated regions with different compression characteristics
  - Inconsistent editing across the image
- **Included by**: UC7 (Preprocess Image) and UC12 (Ensemble Aggregation).

#### UC10: Run Noise Forensics
- **Description**: Extracts the **noise residual** from the image by subtracting a denoised version from the original. This reveals:
  - The noise pattern of the image sensor (for real photos)
  - Artificially uniform or patterned noise (for AI-generated images)
  - Inconsistencies that suggest post-processing or generation
- **Included by**: UC7 (Preprocess Image) and UC12 (Ensemble Aggregation).

#### UC11: Run DL Classification (ViT · SigLIP · SMOGY + TTA)
- **Description**: This is the core **deep learning inference** step. Three models run on the preprocessed image:
  - **ViT (Vision Transformer)**: A transformer-based image classifier fine-tuned for AI detection.
  - **SigLIP**: A vision-language model adapted for binary classification of real vs. AI-generated images.
  - **SMOGY**: A custom or specialized model architecture used in the ensemble.
  - **TTA (Test-Time Augmentation)**: The input image is augmented (flipped, rotated, etc.) during inference, and predictions are averaged across augmentations for more robust results.
- **Included by**: UC7 (Preprocess Image) and UC12 (Ensemble Aggregation).
- **Also included by**: UC14 (Load Model Weights) — the DL models need their weights loaded before they can perform classification.

#### UC12: Ensemble Aggregation
- **Description**: This is the **decision fusion** step. It combines the outputs from all four analysis modules (FFT, ELA, Noise, DL) using a **weighted voting/aggregation scheme** (with weights like 50/15/15/20) to produce a single unified prediction. The ensemble approach is more robust than relying on any single model.
- **Includes**: UC8, UC9, UC10, UC11 (all analysis modules feed into the ensemble).
- **Includes**: UC13 (Generate & Return Verdict) — the aggregated result is passed to the verdict generation step.

#### UC13: Generate & Return Verdict
- **Description**: Takes the aggregated ensemble output and produces the final response:
  - Binary verdict: **REAL** or **FAKE**
  - Confidence score (as a percentage)
  - Forensic visual maps (as image URLs or base64 data)
  - The response is formatted as JSON (for API clients) or rendered in the Gradio UI (for web users).
- **Included by**: UC12 (Ensemble Aggregation).
- **Extended by**: UC2 (View Verdict) and UC3 (View Confidence Score) — these user-facing use cases are extensions of this system use case.
- **Associated with**: Android App — the mobile client receives this verdict through the API response.

#### UC14: Load Model Weights
- **Description**: At system startup (or on first request), the application downloads the trained model weight files from the **HuggingFace Hub** and loads them into GPU/CPU memory. This is a one-time initialization step per deployment.
- **Associated with**: HuggingFace (as the external provider of model weights).
- **Includes**: UC11 (Run DL Classification) — models must be loaded before inference can happen.

#### UC15: Serve Inference Endpoint
- **Description**: The system exposes an HTTP server (via Gradio + FastAPI) that listens for incoming requests. This is the runtime infrastructure that makes the web interface and REST API available.
- **Associated with**: HuggingFace (as the hosting platform for the endpoint).

---

## 1.4 Relationships Explained

The diagram uses three types of UML relationships to connect actors and use cases:

### 1.4.1 Association (Solid Lines)

An **association** is a simple solid line connecting an actor to a use case. It means "this actor participates in this use case."

| Actor | Associated Use Cases |
|-------|---------------------|
| **User** | UC1 (Upload Image), UC2 (View Verdict), UC3 (View Confidence Score), UC4 (View Forensic Visuals), UC5 (Use Web Interface), UC6 (Use REST API) |
| **Developer** | UC-Dev1 (Train & Fine-tune), UC-Dev2 (Deploy to HF Spaces), UC-Dev3 (Push Model Weights) |
| **HuggingFace** | UC14 (Load Model Weights), UC15 (Serve Inference Endpoint) |
| **Android App** | UC6 (Use REST API), UC2 (View Verdict), UC3 (View Confidence Score), UC13 (Generate & Return Verdict) |

### 1.4.2 Include Relationships (Dashed Arrow with `«include»`)

An **include** relationship means: "When use case A is executed, use case B is **always** executed as a mandatory part of A."

The include relationships form the **processing pipeline**:

```
UC5 (Web Interface) ──«include»──► UC1 (Upload Image)
UC6 (REST API)      ──«include»──► UC1 (Upload Image)

UC1 (Upload Image)  ──«include»──► UC7 (Preprocess Image)

UC7 (Preprocess)    ──«include»──► UC8  (FFT Analysis)
UC7 (Preprocess)    ──«include»──► UC9  (ELA Analysis)
UC7 (Preprocess)    ──«include»──► UC10 (Noise Forensics)
UC7 (Preprocess)    ──«include»──► UC11 (DL Classification)

UC12 (Ensemble)     ──«include»──► UC8  (FFT Analysis)
UC12 (Ensemble)     ──«include»──► UC9  (ELA Analysis)
UC12 (Ensemble)     ──«include»──► UC10 (Noise Forensics)
UC12 (Ensemble)     ──«include»──► UC11 (DL Classification)

UC12 (Ensemble)     ──«include»──► UC13 (Generate Verdict)

UC14 (Load Weights) ──«include»──► UC11 (DL Classification)
```

**Reading the pipeline**: When a user uses the Web Interface (UC5), it *includes* uploading an image (UC1), which *includes* preprocessing (UC7), which *includes* all four analysis modules (UC8–UC11). The ensemble aggregation (UC12) also *includes* all four analysis modules and the verdict generation (UC13).

### 1.4.3 Extend Relationships (Dashed Arrow with `«extend»`)

An **extend** relationship means: "Use case B **may optionally** extend use case A under certain conditions." The extending use case adds behavior to the base use case.

```
UC2 (View Verdict)          ──«extend»──► UC13 (Generate Verdict)
UC3 (View Confidence Score) ──«extend»──► UC13 (Generate Verdict)
UC4 (View Forensic Visuals) ──«extend»──► UC8  (FFT Analysis)
```

**Why extend and not include?**
- Viewing the verdict and confidence score is **optional from the system's perspective** — the system generates these values regardless, but whether a user actually views them depends on the client's UI.
- Viewing forensic visuals is also optional — some clients (like a minimal API consumer) may only want the verdict and skip the visual maps.

---

## 1.5 End-to-End Flow (How Everything Connects)

To understand the system holistically, here is the **complete flow** from a user's perspective:

### Scenario A: User via Web Interface
1. **User** opens the Gradio web interface → `UC5: Use Web Interface`
2. User drags and drops an image → `UC1: Upload Image` (included by UC5)
3. System strips EXIF, applies CLAHE, resizes → `UC7: Preprocess Image` (included by UC1)
4. System runs all analysis modules in parallel:
   - `UC8: Run FFT Analysis`
   - `UC9: Run ELA Analysis`
   - `UC10: Run Noise Forensics`
   - `UC11: Run DL Classification` (requires `UC14: Load Model Weights` at startup)
5. Results are combined → `UC12: Ensemble Aggregation`
6. Final response is created → `UC13: Generate & Return Verdict`
7. User sees the result → `UC2: View Verdict`, `UC3: View Confidence Score`, `UC4: View Forensic Visuals`

### Scenario B: Android App via REST API
1. **Android App** sends a POST request with the image → `UC6: Use REST API /analyze`
2. Image is uploaded → `UC1: Upload Image` (included by UC6)
3. Same processing pipeline as above (steps 3–6)
4. JSON response is returned → `UC13: Generate & Return Verdict`
5. App displays results → `UC2: View Verdict`, `UC3: View Confidence Score`

### Scenario C: Developer Workflow
1. **Developer** trains models locally → `UC-Dev1: Train & Fine-tune Model`
2. Developer uploads trained weights → `UC-Dev3: Push Model Weights to Hub`
3. Developer deploys app → `UC-Dev2: Deploy to HF Spaces`
4. **HuggingFace** hosts the app → `UC15: Serve Inference Endpoint`
5. On first request, app downloads weights → `UC14: Load Model Weights`

---

## 1.6 Color Coding Guide

The diagram uses a consistent color scheme to visually group related elements:

| Color | Hex Code | Used For |
|-------|----------|----------|
| 🔵 Blue | `#dce8fb` / `#1F3864` | User-facing use cases and the User actor |
| 🟢 Green | `#d4f0e0` / `#1a5c35` | Developer use cases and the Developer actor |
| 🟠 Orange | `#fff3e8` / `#f09040` | HuggingFace-related use cases and the HuggingFace actor |
| 🟣 Purple | `#f0eafc` / `#7050a8` | Android App actor and DL-related system use cases |
| 🟣 Dark Purple | `#e8d0f8` / `#4a2080` | Core system use cases (Ensemble Aggregation, Generate Verdict) — emphasized with bold text and thicker borders |
| ⚫ Gray | `#555555` / `#888888` | Include and extend relationship labels |

---

## 1.7 Key Design Decisions

1. **Multi-client support**: The system is designed to serve both web users (Gradio) and mobile users (Android App via REST API) from the same backend, avoiding code duplication.

2. **Decoupled model lifecycle**: Model training (Developer) and model inference (User) are completely separated. The HuggingFace Hub acts as the bridge, storing model weights that the deployed app can download.

3. **Parallel analysis pipeline**: All four analysis modules (FFT, ELA, Noise, DL) run independently after preprocessing. This enables potential parallel execution for faster inference.

4. **Ensemble-based decision**: Rather than relying on a single model, the system uses an ensemble of four different analysis techniques. This is a deliberate design choice to improve accuracy and robustness against different types of AI-generated images.

5. **Forensic transparency**: By exposing FFT, ELA, and Noise maps to the user, the system provides **explainability** — users can visually understand *why* the system classified an image as fake, rather than blindly trusting a score.

---

*Next section: Sequence Diagram →*
