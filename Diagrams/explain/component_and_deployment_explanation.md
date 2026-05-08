# AIBuster Detection System — Component & Deployment Diagrams Explanation

> This document provides a detailed explanation of the **Component Diagram** and **Deployment Diagram** used in the AIBuster graduation project. Each section breaks down every element — components, nodes, connections, and layers — so you can fully understand how the system is architecturally organized and where each piece physically runs.

---

# 1. Component Diagram

## 1.1 What Is a Component Diagram?

A **Component Diagram** is a type of UML structural diagram that shows the **software components** (modules, libraries, services) that make up a system and the **dependencies and connections** between them. Think of it as a map of all the "building blocks" of your software and how they plug into each other.

**Key concepts:**
- **Component**: A modular, self-contained unit of software that performs a specific function (e.g., "FFT Engine", "FastAPI Server"). Drawn as a rounded rectangle.
- **Dependency**: An arrow showing that one component relies on another to function (e.g., the TTA Manager *depends on* the Preprocessor to provide clean image data).
- **Interface**: A named contract through which components communicate (e.g., `IVerdict` — the interface the Ensemble Aggregator uses to send results back).
- **System Boundary**: A dashed rectangle enclosing all components that belong to the system, separating them from external actors.

**Why do we need it?** While the Class Diagram shows individual classes and their attributes, the Component Diagram operates at a *higher level of abstraction* — it groups related classes into components and shows how those components are wired together. It answers: *"What are the major software modules and how do they interact?"*

---

## 1.2 System Boundary

The entire diagram is enclosed in a dashed rounded rectangle labeled **"AIBuster Detection System (V3)"**. This is the **system boundary** — everything inside it is part of the AIBuster application. Components outside this boundary (Web Browser, Android App, HuggingFace Hub) are **external actors** that interact with the system but are not part of it.

---

## 1.3 Components — Detailed Breakdown

### 1.3.1 External Components (Outside the System Boundary)

These components exist outside the AIBuster system and interact with it from the outside:

| Component | Color | Description |
|-----------|-------|-------------|
| **Web Browser (Gradio UI Client)** | 🔵 Blue | The user's web browser that loads the Gradio interface. It sends images to the system and displays results. It is a *consumer* of the system, not part of it. |
| **Android App (Mobile Client)** | 🟢 Green | The mobile application that communicates with the system via REST API. It bypasses the Gradio UI entirely and talks directly to the FastAPI endpoint. |
| **HuggingFace Hub (Model Registry)** | 🟠 Orange | The external cloud service that stores pre-trained and fine-tuned model weight files. The DL engines download their weights from here at startup. |

### 1.3.2 API Layer (Top of the System)

These components handle incoming requests from clients:

#### Gradio UI Server (Web Interface)
- **Color**: Blue (`#dce8fb`)
- **Role**: The server-side Gradio application that generates the interactive web interface. When a user opens the browser, Gradio serves the HTML/JS frontend and handles the image upload widget, confidence bar, and forensic visual display.
- **Important**: Gradio is **mounted on** FastAPI — meaning Gradio's web server runs *inside* the FastAPI application as a sub-application, not as a separate server.

#### FastAPI Server (REST API)
- **Color**: Orange (`#fff3e8`)
- **Role**: The core HTTP server that exposes the `POST /analyze` endpoint. All requests — whether from Gradio or the Android App — ultimately pass through FastAPI. It is the single entry point into the processing pipeline.

### 1.3.3 Processing Layer (Middle)

These components prepare the raw image for analysis:

#### Preprocessor (EXIF strip, RGB, JPEG)
- **Color**: Light Blue (`#ddeeff`)
- **Role**: Takes the raw uploaded image and normalizes it. It strips EXIF metadata, converts to RGB format, and generates the specific image formats required by each downstream engine (grayscale for FFT, RGB+JPEG for ELA, RGB for Noise, resized tensors for DL models).

#### TTA Manager (3 Views: Orig, Flip, Crop)
- **Color**: Light Blue (`#ddeeff`)
- **Role**: Implements **Test-Time Augmentation**. It takes the preprocessed image and creates **three augmented views**: (1) the original image, (2) a horizontally flipped version, and (3) a center-cropped version. These three views are sent to each deep learning model, and their predictions are averaged for more robust results.

### 1.3.4 Engine Layer — Forensic Engines (Left Column)

These engines perform traditional computer-vision-based forensic analysis. They produce **visual outputs** (maps) rather than classification scores:

| Engine | Weight | Description |
|--------|--------|-------------|
| **FFT Engine (Visual Only)** | — | Applies Fast Fourier Transform to convert the image to the frequency domain. Produces a spectrum visualization that reveals GAN grid artifacts. Labeled "Visual Only" because its output is a forensic map shown to the user, not a numeric score fed into the ensemble. |
| **ELA Engine (Visual Only)** | — | Applies Error Level Analysis by re-compressing the image and amplifying the difference. Produces an error map highlighting manipulated or AI-generated regions. Also "Visual Only". |
| **Noise Forensic Engine (Weight: 20%)** | 20% | Extracts the noise residual and computes statistical metrics (variance, spatial correlation, entropy, channel correlation). Unlike FFT and ELA, this engine **does** produce a numeric score that feeds into the ensemble with a 20% weight. |

### 1.3.5 Engine Layer — Deep Learning Engines (Right Column)

These engines use neural networks for classification:

| Engine | Weight | Description |
|--------|--------|-------------|
| **Fine-Tuned ViT (PRIMARY - Weight: 50%)** | 50% | The primary classification model — a Vision Transformer fine-tuned on AI-generated vs. real image datasets. Receives 3 TTA views from the TTA Manager. Has the highest ensemble weight (50%) because it is the most accurate single model. Styled with a **thicker border** to emphasize its primary role. |
| **SigLIP Engine (Weight: 15%)** | 15% | A vision-language model adapted for binary classification. Receives 3 TTA views. Contributes 15% to the ensemble. |
| **SMOGY Engine (Weight: 15%)** | 15% | A specialized model architecture. Receives 3 TTA views. Contributes 15% to the ensemble. |

### 1.3.6 Aggregation Layer (Bottom)

#### Ensemble Aggregator (Threshold: 0.5 → FAKE/REAL)
- **Color**: Deep Purple (`#d8c0f0`)
- **Role**: Collects numeric scores from all scoring engines (Noise: 20%, ViT: 50%, SigLIP: 15%, SMOGY: 15%), computes the weighted sum, and applies a threshold of 0.5. If the aggregated score exceeds 0.5, the verdict is **FAKE**; otherwise, it is **REAL**.

---

## 1.4 Connections (Edges) — How Data Flows

Every arrow in the diagram represents a dependency or data flow. Here is the complete list:

### Client → API Layer

| From | To | Label | Meaning |
|------|----|-------|---------|
| Web Browser | Gradio UI Server | `uses` (dashed) | The browser loads and interacts with the Gradio interface. Dashed because it is a dependency — the browser *depends on* Gradio to render the UI. |
| Android App | FastAPI Server | `POST /analyze` (dashed) | The Android app sends HTTP POST requests directly to the REST API. Dashed because it crosses the system boundary. |

### API Layer Internal

| From | To | Label | Meaning |
|------|----|-------|---------|
| Gradio UI Server | FastAPI Server | `mounted on` (solid) | Gradio runs as a sub-application *inside* FastAPI. This is a strong coupling — Gradio is literally mounted on the FastAPI ASGI app. |

### API → Processing

| From | To | Label | Meaning |
|------|----|-------|---------|
| FastAPI Server | Preprocessor | `delegates` | FastAPI delegates the image processing work to the Preprocessor. The API layer does not process images itself. |

### Processing → Engines

| From | To | Label | Meaning |
|------|----|-------|---------|
| Preprocessor | TTA Manager | `feeds` | Preprocessed image data is passed to the TTA Manager for augmentation. |
| Preprocessor | FFT Engine | `grayscale` | The Preprocessor sends a **grayscale** version of the image to FFT (frequency analysis requires single-channel input). |
| Preprocessor | ELA Engine | `RGB+JPEG` | The Preprocessor sends both the RGB image and a JPEG-compressed version (ELA needs to compare original vs. re-compressed). |
| Preprocessor | Noise Forensic Engine | `RGB` | The Preprocessor sends the RGB image for noise residual extraction. |
| TTA Manager | Fine-Tuned ViT | `3 views` | The TTA Manager sends 3 augmented views to ViT for inference. |
| TTA Manager | SigLIP Engine | `3 views` | Same 3 views sent to SigLIP. |
| TTA Manager | SMOGY Engine | `3 views` | Same 3 views sent to SMOGY. |

### Engines → Aggregator

| From | To | Label | Meaning |
|------|----|-------|---------|
| Noise Forensic Engine | Ensemble Aggregator | `score` | Noise engine sends its probability score (0.0–1.0). |
| Fine-Tuned ViT | Ensemble Aggregator | `score` | ViT sends its averaged TTA score. |
| SigLIP Engine | Ensemble Aggregator | `score` | SigLIP sends its averaged TTA score. |
| SMOGY Engine | Ensemble Aggregator | `score` | SMOGY sends its averaged TTA score. |

### Aggregator → API Layer

| From | To | Label | Meaning |
|------|----|-------|---------|
| Ensemble Aggregator | FastAPI Server | `IVerdict` (dashed) | The aggregator returns the final verdict through the **IVerdict interface** — a contract that includes the verdict (FAKE/REAL), confidence score, and forensic maps. The dashed line and long path (going right and up) shows this is a return/callback flow. |

### External Dependencies

| From | To | Label | Meaning |
|------|----|-------|---------|
| HuggingFace Hub | Fine-Tuned ViT | `weights` (dashed) | ViT downloads its model weights from HuggingFace Hub at startup. |
| HuggingFace Hub | SigLIP Engine | `weights` (dashed) | SigLIP downloads its weights from the Hub. |
| HuggingFace Hub | SMOGY Engine | `weights` (dashed) | SMOGY downloads its weights from the Hub. |

---

## 1.5 Data Flow Summary (End-to-End)

```
Web Browser ──uses──► Gradio UI Server ──mounted on──► FastAPI Server
Android App ──POST /analyze──────────────────────────► FastAPI Server

FastAPI Server ──delegates──► Preprocessor
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                grayscale        RGB+JPEG          RGB          feeds
                    │               │               │             │
                    ▼               ▼               ▼             ▼
               FFT Engine     ELA Engine    Noise Engine    TTA Manager
              (visual only)  (visual only)   (20% weight)       │
                                                │          ┌────┼────┐
                                              score     3 views  3 views  3 views
                                                │         │      │      │
                                                │         ▼      ▼      ▼
                                                │       ViT    SigLIP  SMOGY
                                                │      (50%)   (15%)   (15%)
                                                │         │      │      │
                                                │       score  score  score
                                                │         │      │      │
                                                ▼         ▼      ▼      ▼
                                            ┌──────────────────────────────┐
                                            │    Ensemble Aggregator       │
                                            │  weighted sum → threshold    │
                                            │    → FAKE or REAL            │
                                            └──────────┬───────────────────┘
                                                       │
                                                   IVerdict
                                                       │
                                                       ▼
                                                FastAPI Server ──► Client
```

---

## 1.6 Key Design Decisions Visible in the Diagram

1. **Separation of visual-only vs. scoring engines**: FFT and ELA are labeled "Visual Only" — they produce forensic maps for the user but do not contribute numeric scores to the ensemble. Only Noise, ViT, SigLIP, and SMOGY produce scores.

2. **Single entry point**: Both Gradio and the Android App funnel through FastAPI. There is no separate backend for mobile clients.

3. **Preprocessor as a router**: The Preprocessor doesn't just clean the image — it also routes different *formats* to different engines (grayscale for FFT, RGB+JPEG for ELA, etc.).

4. **TTA only for DL models**: The TTA Manager feeds only the three deep learning models, not the forensic engines. Forensic engines don't benefit from augmentation.

5. **IVerdict interface**: The use of a named interface (`IVerdict`) between the Ensemble Aggregator and FastAPI shows a clean architectural boundary — the API layer doesn't need to know how the verdict was computed.

---
---

# 2. Deployment Diagram

## 2.1 What Is a Deployment Diagram?

A **Deployment Diagram** is a UML structural diagram that shows the **physical architecture** of a system — the actual hardware devices, servers, cloud services, and containers where the software runs, and the network connections between them.

**Key concepts:**
- **Node**: A physical or virtual computing resource where software runs. Drawn as a 3D box or a labeled rectangle. Examples: a laptop, a cloud server, a mobile phone.
- **Artifact**: A deployable piece of software that lives on a node (e.g., `app.py`, a Docker container, a model weight file).
- **Communication Path**: A line connecting two nodes, representing a network protocol or deployment relationship (e.g., HTTPS, REST API, git push).
- **Stereotype**: A label in `«guillemets»` that classifies a node (e.g., `«device»`, `«cloud node»`, `«client»`).

**Why do we need it?** The Component Diagram shows *what software modules exist*. The Deployment Diagram shows *where those modules physically run*. It answers: *"On which machines/servers does each piece of our system execute, and how do those machines communicate?"*

---

## 2.2 Nodes — The Physical/Virtual Environments

The diagram defines **six nodes**, each represented as a colored swimlane-style rectangle with a stereotype label.

### 2.2.1 Developer Machine `«device»`
- **Color**: Dark Blue header (`#1F3864`), light blue body (`#e8f0fb`)
- **What it is**: The developer's local computer (laptop/desktop) where code is written, models are trained, and deployments are triggered.
- **Contains four artifacts:**

| Artifact | Description |
|----------|-------------|
| **Source Code** (`train_vit.py`, `app.py`) | The Python source files — the training script and the main application entry point. |
| **Google Colab** (T4 GPU, 16GB, Model Training) | The cloud notebook environment used for model training. Although Colab is technically a cloud service, it is shown here because the developer initiates and controls training sessions from their machine. The T4 GPU with 16GB VRAM is the hardware accelerator used for fine-tuning. |
| **Python Env** (PyTorch, HuggingFace Transformers, Gradio, FastAPI) | The local Python environment with all dependencies installed. This is the development environment where the developer tests the application locally before deploying. |
| **Git Repository** (GitHub, requirements.txt, Dockerfile, CI/CD Actions) | The local Git repository that syncs with GitHub. Contains all configuration files needed for deployment. |

### 2.2.2 Hugging Face Spaces `«cloud node»`
- **Color**: Orange header (`#f09040`), light orange body (`#fff8f0`)
- **What it is**: The cloud hosting platform where the AIBuster application runs in production. HuggingFace Spaces provides free-tier hosting with automatic HTTPS and a public URL.
- **Contains a Docker Container** (dashed border, labeled `«Docker Container»`) with:

| Artifact | Description |
|----------|-------------|
| **Gradio UI** (port 7860, image-upload interface) | The web interface server, listening on port 7860. Provides the drag-and-drop image upload widget and result display. |
| **FastAPI** (POST /analyze, REST API endpoint) | The REST API server that handles programmatic requests from the Android App and internal Gradio calls. |
| **AI Engines** (FFT · ELA · Noise, SigLIP · SMOGY, Ensemble Aggregator) | All the analysis engines and the ensemble aggregator, running as Python modules inside the container. |
| **Fine-tuned ViT** (google/vit-base-patch16-224, mohamed9679/...) | The primary deep learning model, showing both its base architecture (`google/vit-base-patch16-224`) and the fine-tuned version hosted under the developer's HuggingFace namespace. |
| **Model Weights Cache** (HuggingFace Hub — pre-trained & fine-tuned checkpoints, loaded once via `@lru_cache`, served from GPU/CPU runtime, HTTPS public URL · auto TLS · free tier runtime) | The cached model weights, downloaded once from HuggingFace Hub and kept in memory using Python's `@lru_cache` decorator to avoid re-downloading on every request. |

### 2.2.3 Hugging Face Hub `«cloud registry»`
- **Color**: Purple header (`#7050a8`), light purple body (`#f5e8fb`)
- **What it is**: The model registry — a centralized cloud storage for machine learning model weights and datasets. Separate from HF Spaces (which runs the app).
- **Contains:**

| Artifact | Description |
|----------|-------------|
| **Model Registry** (Fine-tuned ViT, Pre-trained models, Versioned weights, Push via API token) | Stores all model weight files. Developers push trained weights here using an API token. The deployed app pulls weights from here at startup. Supports versioning so you can roll back to previous model versions. |
| **Datasets** (CIFAKE, Dima806/AI_vs_Real, Parveshiiii dataset, CapCheck repo) | The training datasets used to fine-tune the models. Stored on HuggingFace for easy access during Colab training sessions. |

### 2.2.4 Android Mobile App `«device»`
- **Color**: Green header (`#40a070`), light green body (`#e8f8ee`)
- **What it is**: The end user's Android phone running the AIBuster mobile app.
- **Contains:**

| Artifact | Description |
|----------|-------------|
| **UI Layer** (UploadFragment, ResultFragment) | The Android UI screens — one for uploading images and one for displaying results. Built using Android Fragments. |
| **ViewModel** (UploadViewModel, LiveData) | The ViewModel layer that holds UI state and survives configuration changes (like screen rotation). Uses LiveData for reactive data binding. |
| **Network Layer** (Ktor HTTP Client, CIO Engine, Bearer Token Auth, Kotlin Coroutines, Base64 encoding) | The networking stack: **Ktor** is the HTTP client library, **CIO** is its coroutine-based engine, requests are authenticated with **Bearer Token**, images are **Base64-encoded** before sending, and all network calls run on **Kotlin Coroutines** for async execution. |
| **MVVM Pattern** (View observes ViewModel state, Model handles API calls async) | A note explaining the architectural pattern: the app follows **MVVM (Model-View-ViewModel)** — Views observe ViewModel state reactively, and the Model layer handles API calls asynchronously. |

### 2.2.5 Web Browser `«client»`
- **Color**: Blue header (`#4a72c4`), light blue body (`#f0f8ff`)
- **What it is**: The end user's web browser (Chrome, Firefox, etc.) that loads the Gradio interface.
- **Contains:**

| Artifact | Description |
|----------|-------------|
| **Gradio UI Client** (Image upload widget, Confidence bar display, FFT · ELA · Noise visuals) | The client-side JavaScript/HTML that Gradio serves to the browser. Includes the image upload widget, the confidence score bar, and the three forensic visual displays. |

### 2.2.6 GitHub Repository `«version control»`
- **Color**: Dark Gray header (`#333333`), light gray body (`#f0f0f0`)
- **What it is**: The remote Git repository on GitHub that stores the project's source code and CI/CD configuration.
- **Contains:**

| Artifact | Description |
|----------|-------------|
| **Source Code** (app.py, train_vit.py, requirements.txt, Dockerfile) | The complete application source code and deployment configuration files. |
| **GitHub Actions** (CI/CD Pipeline, Auto-deploy to HF on push → main) | The continuous integration/continuous deployment pipeline. When code is pushed to the `main` branch, GitHub Actions automatically deploys the updated application to HuggingFace Spaces. |

---

## 2.3 Communication Paths — How Nodes Connect

Each arrow represents a network connection or deployment relationship between nodes:

| From | To | Label | Line Style | Description |
|------|----|-------|-----------|-------------|
| Developer Machine | GitHub Repository | `git push` | Dashed | The developer pushes source code changes to GitHub using Git. |
| GitHub Repository | HuggingFace Spaces | `auto-deploy (Actions)` | Dashed, orange | GitHub Actions automatically deploys the app to HF Spaces when code is pushed to `main`. This is the CI/CD pipeline. |
| HuggingFace Hub | HuggingFace Spaces | `pull weights` | Solid, purple | At startup, the Docker container on HF Spaces pulls model weights from the Hub. This is an internal HuggingFace-to-HuggingFace connection. |
| Developer Machine | HuggingFace Hub | `push trained model` | Dashed, purple | After training on Colab, the developer pushes the fine-tuned model weights to the Hub using the HuggingFace API. |
| Web Browser | HuggingFace Spaces | `HTTPS / WebSocket` | Solid, blue | The browser connects to HF Spaces over HTTPS. Gradio uses WebSocket for real-time updates during analysis. |
| Android Mobile App | HuggingFace Spaces | `REST API (HTTPS) POST /analyze Bearer Token Auth` | Solid, green | The Android app sends authenticated HTTPS POST requests to the `/analyze` endpoint on HF Spaces. Bearer token authentication ensures only authorized clients can access the API. |

---

## 2.4 The Docker Container

Inside the HuggingFace Spaces node, there is a **dashed rectangle** labeled `«Docker Container»`. This is a critical architectural element:

- The entire AIBuster application runs inside a **Docker container** — a lightweight, isolated virtual environment that packages the app with all its dependencies.
- Docker ensures that the application behaves identically regardless of the host machine. What works on the developer's laptop will work on HF Spaces.
- The container includes: the Gradio UI, FastAPI server, all AI engines, the ViT model, and the cached model weights.
- HuggingFace Spaces natively supports Docker-based deployments, which is why the `Dockerfile` is part of the source code.

---

## 2.5 End-to-End Deployment Flow

### Developer Workflow (Build & Deploy)
```
Developer Machine                GitHub                  HF Spaces
      │                            │                        │
      ├── git push ───────────────►│                        │
      │                            ├── auto-deploy ────────►│
      │                            │   (GitHub Actions)     │
      │                            │                        │ (Docker container starts)
      │                            │                        │
Developer Machine                HF Hub                  HF Spaces
      │                            │                        │
      ├── push trained model ─────►│                        │
      │                            │◄── pull weights ───────┤
      │                            │                        │ (models loaded into memory)
```

### User Workflow (Web)
```
Web Browser ──HTTPS/WebSocket──► HF Spaces (Docker Container)
                                      │
                                      ├─► Gradio UI (port 7860)
                                      ├─► FastAPI ─► Preprocessor ─► Engines ─► Ensemble
                                      │
                                      ◄── verdict + visuals ──
```

### User Workflow (Mobile)
```
Android App ──REST API (HTTPS)──► HF Spaces (Docker Container)
    │          POST /analyze            │
    │          Bearer Token             ├─► FastAPI ─► Preprocessor ─► Engines ─► Ensemble
    │                                   │
    ◄────── JSON { verdict } ───────────┤
```

---

## 2.6 Color Coding Guide

| Color | Used For |
|-------|----------|
| 🔵 Dark Blue (`#1F3864`) | Developer Machine node |
| 🟠 Orange (`#f09040`) | HuggingFace Spaces (cloud hosting) |
| 🟣 Purple (`#7050a8`) | HuggingFace Hub (model registry) |
| 🟢 Green (`#40a070`) | Android Mobile App |
| 🔵 Blue (`#4a72c4`) | Web Browser client |
| ⚫ Dark Gray (`#333333`) | GitHub Repository |

---

## 2.7 Legend (Bottom of the Diagram)

The diagram includes a legend explaining the visual notation:

| Symbol | Meaning |
|--------|---------|
| Colored header rectangle | **Node / Device** — a physical or virtual computing environment |
| White rounded rectangle | **Component / Artifact** — a deployable software piece inside a node |
| Dashed rounded rectangle | **Docker Container** — an isolated runtime environment |
| Solid arrow | **Communication / Deploy** — a network connection or deployment action |
| Dashed arrow | **Dependency** — one node depends on or pushes to another |

---

## 2.8 Key Design Decisions Visible in the Diagram

1. **Dockerized deployment**: The entire application is containerized, ensuring consistent behavior across development and production environments.

2. **CI/CD automation**: The `git push → GitHub Actions → HF Spaces` pipeline automates deployment. Developers never manually deploy — pushing to `main` triggers automatic deployment.

3. **Separated model storage**: Model weights are stored on HuggingFace Hub (a registry), not inside the Docker image or GitHub repo. This keeps the Docker image small and allows updating models without redeploying the app.

4. **Dual client architecture**: The system serves two completely different clients (Web Browser and Android App) from the same Docker container, using two different protocols (WebSocket for Gradio, REST for Android).

5. **Bearer Token authentication**: The Android App uses token-based authentication, adding a security layer that prevents unauthorized access to the API.

6. **Google Colab for training**: Model training happens on Google Colab (free T4 GPU), not on the developer's local machine. This is a practical decision — training deep learning models requires GPU resources that most developer laptops don't have.

---

## 2.9 What This Diagram Tells You (vs. the Component Diagram)

The Component Diagram shows *what software modules exist and how data flows between them*. The Deployment Diagram shows *where those modules physically run*:

| Question | Component Diagram | Deployment Diagram |
|----------|------------------|--------------------|
| Where does the Gradio UI run? | Shows it as a component | Shows it runs inside a Docker container on HuggingFace Spaces, on port 7860 |
| How does code get to production? | Not shown | `git push` → GitHub → GitHub Actions → HF Spaces |
| Where are model weights stored? | Shows HF Hub as an external actor | Shows HF Hub as a separate cloud registry node with versioned weights |
| What technology does the Android app use? | Shows it as a single external component | Shows the full stack: Ktor, CIO, MVVM, Coroutines, Bearer Auth |
| How is the app containerized? | Not shown | Shows the Docker container boundary inside HF Spaces |
| Where does training happen? | Not shown | Shows Google Colab with T4 GPU on the Developer Machine |
