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

---

# 2. Sequence Diagram

## 2.1 Overview

The Sequence Diagram shows the **time-ordered message exchange** between the system's participants during a single image detection request. While the Use Case Diagram shows *what* the system can do, the Sequence Diagram shows *how* it does it — step by step, message by message, from the moment the user uploads an image until the result is displayed on screen.

The diagram reads **top-to-bottom** (time flows downward) and **left-to-right** (from the user through increasingly deeper system layers).

---

## 2.2 Participants (Lifelines)

The diagram defines **six participants**, each represented as a colored header box at the top with a vertical **lifeline** (dashed line) extending downward. The lifelines show when each participant is "alive" and participating in the interaction.

| Participant | Description |
|-------------|-------------|
| **User** | The human actor who initiates the process by uploading an image. |
| **Frontend** (Gradio / Android) | The client interface layer — either the Gradio web UI or the Android app. It receives the user's input and forwards it to the backend. |
| **Backend** (FastAPI + Gradio) | The server-side application that orchestrates the entire detection pipeline. It receives HTTP requests, delegates work to other components, and assembles the final response. |
| **Preprocessor** (EXIF · CLAHE · Resize) | The preprocessing module that normalizes the raw image before analysis. |
| **AI Engines** (ViT · FFT · ELA · Noise) | The collection of analysis engines that run in parallel to produce individual scores and visual outputs. |
| **Ensemble** (Aggregator) | The decision-fusion component that combines all engine scores into a single verdict. |

### Activation Bars

Each lifeline has a semi-transparent **activation bar** — a thin colored rectangle overlaying the lifeline. This bar indicates the period during which that participant is actively processing. Notice how the User and Frontend bars span nearly the entire diagram (they're waiting for the response), while the Preprocessor bar is short (it finishes quickly) and the Ensemble bar only appears near the bottom (it acts last).

---

## 2.3 Message Flow — Step by Step

The messages are drawn as horizontal arrows between lifelines. **Solid arrows with filled heads** represent synchronous calls (requests), while **dashed arrows with open heads** represent return/response messages.

### Step 1 — `uploadImage(file)`
- **From**: User → Frontend
- **Type**: Synchronous call
- The user selects an image file and submits it through the interface.

### Step 2 — `POST /analyze (image bytes)`
- **From**: Frontend → Backend
- **Type**: Synchronous call
- The frontend packages the image and sends it to the backend as an HTTP POST request to the `/analyze` endpoint.

### Step 3 — `preprocess(image)`
- **From**: Backend → Preprocessor
- **Type**: Synchronous call
- The backend delegates image preparation to the Preprocessor module.

### Steps 3a–3c — Preprocessor Internal Operations
Inside the Preprocessor's activation period, three sequential operations are shown as labeled boxes:
1. **strip EXIF metadata** — removes camera/GPS metadata
2. **apply CLAHE enhancement** — improves local contrast
3. **resize to 224×224, generate formats** — scales to model-expected dimensions and creates format variants needed by different engines

### Step 4 — `preprocessed data` (return)
- **From**: Preprocessor → Backend
- **Type**: Response (dashed arrow)
- The Preprocessor returns the cleaned, normalized image data back to the Backend.

### Step 5 — `runParallelAnalysis(data)`
- **From**: Backend → AI Engines
- **Type**: Synchronous call
- The Backend sends the preprocessed data to the AI Engines for analysis.

### Steps 5a–5f — Parallel Analysis Pipeline (`par` frame)
A **`par` (parallel) combined fragment** — shown as a yellow-bordered rectangle — encloses six internal operations that execute simultaneously:
1. **compute 2D FFT spectrum** — frequency-domain analysis
2. **run ELA (JPEG diff ×15)** — error level analysis with 15× amplification
3. **extract noise residual** — noise pattern extraction
4. **compute var · corr · entropy · chan** — statistical noise metrics (variance, spatial correlation, entropy, channel correlation)
5. **ViT forward pass (×3 TTA)** — Vision Transformer inference with 3 test-time augmentation views
6. **SigLIP · SMOGY inference** — additional deep learning model predictions

### Step 6 — `scores[ ] + visual maps` (return)
- **From**: AI Engines → Backend
- **Type**: Response (dashed arrow)
- All engines return their individual probability scores plus any generated visual maps (FFT spectrum, ELA map, noise residual).

### Step 7 — `aggregate(scores{})`
- **From**: Backend → Ensemble
- **Type**: Synchronous call
- The Backend passes the collected scores to the Ensemble Aggregator.

### Steps 7a–7c — Ensemble Internal Operations
Three sequential operations inside the Ensemble:
1. **compute weighted sum** — applies the 50/15/15/20 weight distribution across engine scores
2. **apply threshold (0.5)** — compares the weighted score against the 0.5 decision boundary
3. **determine verdict: FAKE/REAL** — produces the binary classification result

### Step 8 — `verdict + confidence + evidence` (return)
- **From**: Ensemble → Backend
- **Type**: Response (dashed arrow)
- The Ensemble returns the final verdict, confidence score, and supporting evidence.

### Step 9 — `JSON { verdict, confidence, scores }` (return)
- **From**: Backend → Frontend
- **Type**: Response (dashed arrow)
- The Backend serializes everything into a JSON response and sends it back to the Frontend.

### Step 10 — `display result + visuals` (return)
- **From**: Frontend → User
- **Type**: Response (dashed arrow)
- The Frontend renders the verdict, confidence score, and forensic visual maps on screen for the user to see.

---

## 2.4 Key Diagram Elements

### The `par` Combined Fragment
The yellow `par [ Parallel Analysis Pipeline ]` box is a UML **combined fragment** that indicates all enclosed operations run concurrently, not sequentially. This is a critical performance feature — running all six analyses in parallel significantly reduces total inference time compared to running them one after another.

### Bottom Note
The note at the bottom of the diagram reinforces: *"All analysis engines (FFT, ELA, Noise, ViT, SigLIP, SMOGY) execute in parallel via async pipeline and results are collected before ensemble aggregation."*

### Legend
The diagram includes a legend distinguishing:
- **Solid arrows** → Synchronous calls (requests going forward)
- **Dashed arrows** → Returns/Responses (data coming back)
- **Blue boxes** → Internal process notes
- **Yellow boxes** → Loop/Parallel frames

---

## 2.5 What This Diagram Tells You (vs. the Use Case Diagram)

The Use Case Diagram says *"the system preprocesses the image, runs analysis, and returns a verdict."* The Sequence Diagram shows **exactly what messages flow between which components, in what order, and what data each message carries**. It answers questions like:
- Does the Frontend talk directly to the AI Engines? → **No**, everything goes through the Backend.
- Do the analysis engines run one after another? → **No**, they run in parallel inside a `par` frame.
- When does the Ensemble get involved? → **Only after all engine scores have been collected.**
- What does the final response look like? → **A JSON object with verdict, confidence, and individual scores.**

---
---

# 3. Activity Diagram

## 3.1 Overview

The Activity Diagram models the **workflow** of the detection process as a flow of activities. Think of it as a flowchart that shows every action, decision, and parallel branch from the moment a user uploads an image to the moment they see the result.

Unlike the Sequence Diagram (which focuses on *messages between participants*), the Activity Diagram focuses on **what work gets done and in what order**, including branching logic and concurrent execution paths.

---

## 3.2 Swimlanes

The diagram is divided into **four vertical swimlanes**, each representing a responsibility zone. Every activity node is placed inside the swimlane of the component responsible for executing it.

| Swimlane | Color | Responsible For |
|----------|-------|-----------------|
| **User / Frontend** | 🔵 Blue (`#e8f0fb`) | User actions and result display |
| **Backend (API)** | 🟠 Orange (`#fff3e8`) | Request handling, validation, preprocessing, and orchestration |
| **AI / ML Engines** | 🟢 Green (`#e8f8ee`) | Running the four analysis techniques |
| **Ensemble & Result** | 🟣 Purple (`#f5e8fb`) | Score aggregation, thresholding, verdict generation |

---

## 3.3 Flow of Activities

### Start Node
A solid black circle (**initial node**) in the User/Frontend lane marks the beginning of the workflow.

### Activity 1 — Upload Image
- **Lane**: User / Frontend
- The user selects and submits an image for analysis.

### Activity 2 — Receive & Validate Image
- **Lane**: Backend (API)
- The backend receives the incoming image and checks that it meets format and size requirements.

### Decision: Valid?
- **Lane**: Backend (API)
- A **diamond-shaped decision node** checks the validation result:
  - **No** → flow branches left to "Return Error to User" (red error node), and the process ends.
  - **Yes** → flow continues downward to preprocessing.

### Activity 3 — Preprocessing Pipeline
- **Lane**: Backend (API)
- Performs the three-step preprocessing: strip EXIF, apply CLAHE, resize to 224×224.

### Fork Bar — Parallel Execution
- A **horizontal bar** labeled `[ Fork — Parallel Execution ]` splits the flow into **four concurrent branches**. In UML, a fork bar means all outgoing paths execute simultaneously.

### Parallel Branch 1 — FFT Analysis → Frequency Map
- **Lane**: AI / ML Engines (overlapping with Backend)
- Runs the FFT analysis engine and produces a **Frequency Map** output artifact.

### Parallel Branch 2 — ELA Analysis → ELA Error Map
- **Lane**: AI / ML Engines
- Runs the ELA analysis engine and produces an **ELA Error Map**.

### Parallel Branch 3 — Noise Pattern Forensics → Noise Residual Map
- **Lane**: AI / ML Engines
- Runs the noise forensics engine (computing variance, spatial correlation, entropy, and channel correlation) and produces a **Noise Residual Map**.

### Parallel Branch 4 — Deep Learning Models → Test-Time Augmentation
- **Lane**: Ensemble & Result (purple styling)
- Runs the three fine-tuned DL models (ViT, SigLIP, SMOGY) and applies **TTA** (original, flip, center crop → average the predictions).

### Join Bar
- A second horizontal bar labeled `[ Join ]` **synchronizes** all four branches — the flow only continues once every parallel branch has completed.

### Activity 4 — Ensemble Aggregator (V3)
- **Lane**: Ensemble & Result
- Combines all scores using the weighted formula: `0.50·ViT + 0.20·Noise + 0.15·SigLIP + 0.15·SMOGY`

### Decision: Score > 0.5?
- A second **decision diamond** evaluates the aggregated score:
  - **Yes** (score > 0.5) → flow branches right to **FAKE** (AI-Generated), styled in red
  - **No** (score ≤ 0.5) → flow branches left to **REAL** (Authentic Photo), styled in green

### Activity 5 — Return Verdict + Confidence + Visual Evidence
- Both the FAKE and REAL paths **merge** into this single activity. The system packages the verdict, confidence percentage, and all forensic maps (FFT, ELA, Noise, Score) into the response.

### Activity 6 — Display Result + Forensic Visuals
- **Lane**: User / Frontend
- The frontend renders everything for the user to view.

### End Node
A **bull's-eye circle** (filled circle inside an open circle) marks the end of the workflow.

---

## 3.4 Key Diagram Elements

### Fork and Join Bars
The thick dark horizontal bars are **UML fork/join nodes**:
- **Fork**: One incoming flow splits into multiple outgoing concurrent flows.
- **Join**: Multiple incoming flows synchronize into one outgoing flow.

Together they express the parallelism that is fundamental to the system's performance — all four analysis engines run simultaneously, and the system waits for all of them to finish before proceeding to ensemble aggregation.

### Decision Diamonds
There are two decision points:
1. **Valid?** — guards against invalid input early in the pipeline (error path).
2. **Score > 0.5?** — the classification threshold that determines the final verdict.

### Color-Coded Node Types
The legend at the bottom maps colors to roles:
| Color | Meaning |
|-------|---------|
| 🔵 Blue | User action |
| 🟠 Orange | Backend process |
| 🟢 Green | AI / Forensic engine |
| 🟣 Purple | Deep learning model |
| 🟣 Dark Purple | Ensemble logic |
| 🟡 Yellow Diamond | Decision |
| 🔴 Red | Error / Result |

---

## 3.5 What This Diagram Tells You (vs. the Other Diagrams)

The Activity Diagram uniquely shows:
- **The validation gate**: invalid images are rejected early with an error — something not visible in the Sequence Diagram.
- **The explicit fork/join parallelism**: while the Sequence Diagram uses a `par` frame, the Activity Diagram makes the four concurrent branches visually distinct with their own paths and output artifacts.
- **The threshold decision**: the Score > 0.5 decision diamond makes the binary classification logic visually explicit — you can see exactly where and how FAKE vs. REAL is determined.
- **The swimlane responsibility mapping**: you can instantly see which component owns which activity, making it clear that the user never interacts directly with AI engines or the ensemble.

---
---

# 4. Class Diagram

## 4.1 Overview

The Class Diagram shows the **static structure** of the AIBuster system — the classes (blueprints for objects), their attributes (data), their methods (behavior), and the relationships between them. Unlike the previous diagrams that show *behavior over time*, the Class Diagram shows **how the code is organized and how components depend on each other**.

---

## 4.2 Classes

### 4.2.1 ImageInput
- **Color**: Dark Blue (`#1F3864`)
- **Role**: Represents the raw image uploaded by the user before any processing.

| Section | Content |
|---------|---------|
| **Attributes** | `- filePath: String` — path to the uploaded file |
| | `- format: String` — image format (JPEG, PNG, etc.) |
| | `- width: int` — original pixel width |
| | `- height: int` — original pixel height |
| **Methods** | `+ validate(): bool` — checks that the file is a valid, supported image |

### 4.2.2 Preprocessor
- **Color**: Light Blue (`#60a8d0`)
- **Role**: Normalizes the raw image to prepare it for all analysis engines.

| Section | Content |
|---------|---------|
| **Attributes** | `- claheFactor: float = 1.5` — intensity of the CLAHE contrast enhancement |
| | `- targetSize: int = 224` — the square pixel dimension models expect |
| **Methods** | `+ stripExif(img): Image` — removes metadata |
| | `+ applyClahe(img): Image` — enhances local contrast |
| | `+ resize(img): Image` — scales to target dimensions |
| | `+ generateFormats(img): dict` — creates format variants for different engines |

### 4.2.3 DetectionEngine (Abstract)
- **Color**: Gray (`#888888`), dashed border
- **Role**: The **abstract base class** that all analysis engines inherit from. It defines the common interface (contract) that every engine must implement.

| Section | Content |
|---------|---------|
| **Attributes** | `- name: String` — identifier of the engine (e.g., "FFT", "ViT") |
| | `- weight: float` — the engine's contribution weight in the ensemble |
| **Methods** | `+ analyze(img): float` — **[abstract]** processes the image and returns a probability score (0.0–1.0). Each subclass implements this differently. |
| | `+ getWeight(): float` — returns the engine's ensemble weight |

The `«abstract»` stereotype and dashed border indicate this class **cannot be instantiated directly** — it only exists to be inherited by concrete engine classes.

### 4.2.4 FFTEngine
- **Color**: Green (`#40a070`)
- **Inherits from**: DetectionEngine

| Section | Content |
|---------|---------|
| **Attributes** | `- epsilon: float = 1e-8` — small constant to avoid division by zero in spectrum computation |
| **Methods** | `+ analyze(img): float` — overrides the abstract method; runs FFT-based analysis |
| | `+ computeSpectrum(gray)` — converts grayscale image to frequency domain |
| | `+ detectGridPattern()` — looks for GAN-characteristic grid artifacts |
| | `+ getVisualOutput(): Image` — generates the FFT spectrum visualization |

### 4.2.5 ELAEngine
- **Color**: Green (`#40a070`)
- **Inherits from**: DetectionEngine

| Section | Content |
|---------|---------|
| **Attributes** | `- jpegQuality: int = 90` — the compression quality level used for re-saving |
| | `- ampFactor: float = 15.0` — amplification multiplier for the error difference |
| **Methods** | `+ analyze(img): float` — overrides the abstract method; runs ELA-based analysis |
| | `+ recompress(img): Image` — re-saves the image at the specified JPEG quality |
| | `+ amplifyDiff(a, b): Image` — computes and amplifies the pixel-wise difference |

### 4.2.6 NoiseForensicEngine
- **Color**: Green (`#40a070`)
- **Inherits from**: DetectionEngine

| Section | Content |
|---------|---------|
| **Attributes** | `- weights: float[4] = [0.25, 0.30, 0.25, 0.20]` — internal sub-weights for the four noise metrics |
| **Methods** | `+ analyze(img): float` — overrides the abstract method; combines all noise metrics |
| | `+ computeVariance(n): float` — measures noise variance |
| | `+ computeSpatialCorr(n): float` — measures spatial correlation patterns |
| | `+ computeChannelCorr(n): float` — measures cross-channel correlation |
| | `+ computeEntropy(n): float` — measures noise entropy (randomness) |
| | `+ extractResidual(img): arr` — subtracts denoised version to isolate noise |

### 4.2.7 DLModelEngine
- **Color**: Purple (`#9060c0`)
- **Inherits from**: DetectionEngine

| Section | Content |
|---------|---------|
| **Attributes** | `- modelId: String` — HuggingFace model repository identifier |
| | `- architecture: String` — model type (ViT, SigLIP, or SMOGY) |
| | `- ttaViews: int = 3` — number of test-time augmentation views |
| **Methods** | `+ analyze(img): float` — overrides the abstract method; runs DL inference with TTA |
| | `+ runTTA(img): float[]` — applies augmentations and collects predictions |
| | `+ extractScore(result): float` — extracts the "fake" probability from model output |
| | `+ loadPipeline(): void` — downloads and loads model weights from HuggingFace Hub |
| | `+ flipImage(img): Image` — horizontal flip augmentation for TTA |

### 4.2.8 EnsembleAggregator
- **Color**: Deep Purple (`#4a2080`)
- **Role**: Combines all engine scores into a single verdict using weighted aggregation.

| Section | Content |
|---------|---------|
| **Attributes** | `- threshold: float = 0.5` — the decision boundary for FAKE vs. REAL |
| | `- engineWeights: Map` — mapping of engine names to their weights (e.g., ViT→0.50, Noise→0.20, etc.) |
| **Methods** | `+ aggregate(scores): float` — computes the weighted sum of all engine scores |
| | `+ computeWeightedAvg(): float` — calculates the weighted average |
| | `+ applyThreshold(s): String` — converts the score to "FAKE" or "REAL" |
| | `+ getVerdict(): DetectionResult` — produces the complete result object |

### 4.2.9 DetectionResult
- **Color**: Dark Blue (`#1F3864`)
- **Role**: The data-transfer object that encapsulates the complete output of a detection run.

| Section | Content |
|---------|---------|
| **Attributes** | `- verdict: String` — "FAKE" or "REAL" |
| | `- confidence: float` — confidence percentage |
| | `- agreement: String` — engine agreement level |
| | `- engineScores: Map` — individual scores from each engine |
| | `- fftMap: Image` — FFT spectrum visualization |
| | `- elaMap: Image` — ELA error map visualization |
| | `- noiseMap: Image` — noise residual visualization |
| **Methods** | `+ toJSON(): String` — serializes the entire result for API responses |

### 4.2.10 DetectionService
- **Color**: Orange (`#f09040`)
- **Role**: The **orchestrator class** — the central service that wires everything together and controls the detection pipeline.

| Section | Content |
|---------|---------|
| **Attributes** | `- preprocessor: Preprocessor` — reference to the preprocessing component |
| | `- engines: List<Engine>` — collection of all detection engines |
| | `- aggregator: Ensemble` — reference to the ensemble aggregator |
| **Methods** | `+ detect(img): DetectionResult` — the main entry point; runs the full pipeline |
| | `+ runParallel(data): scores[]` — dispatches preprocessed data to all engines concurrently |
| | `+ buildResponse(r): JSON` — assembles the final JSON response |
| | `+ handleError(e): Response` — handles and formats error responses |

---

## 4.3 Relationships

### 4.3.1 Inheritance (Solid line with hollow triangle arrowhead)

Four concrete engine classes inherit from the abstract `DetectionEngine`:

```
FFTEngine          ──▷  DetectionEngine
ELAEngine          ──▷  DetectionEngine
NoiseForensicEngine ──▷  DetectionEngine
DLModelEngine      ──▷  DetectionEngine
```

This means each engine **must** implement the `analyze(img): float` method, but each does so using a completely different technique. This is the **polymorphism** at the heart of the system — the `DetectionService` can treat all engines uniformly through the `DetectionEngine` interface, regardless of whether an engine uses FFT, ELA, noise analysis, or deep learning.

### 4.3.2 Composition (Solid line with filled diamond)

```
DetectionService ◆───── DetectionEngine  (1 to *)
```

The filled diamond on the `DetectionService` end with multiplicity `1 *` means:
- One `DetectionService` **owns** multiple `DetectionEngine` instances.
- The engines **cannot exist independently** without the service — if the service is destroyed, so are its engines.
- This is the strongest form of association in UML.

### 4.3.3 Association (Solid line with open arrowhead)

```
Preprocessor ────► DetectionEngine       (labeled "feeds")
DetectionService ────► EnsembleAggregator (labeled "uses")
```

- The Preprocessor **feeds** processed image data to the detection engines.
- The DetectionService **uses** the EnsembleAggregator to combine scores.

### 4.3.4 Dependency (Dashed line with open arrowhead, `«uses»` / `«creates»`)

```
ImageInput - - -«uses»- - -► Preprocessor
EnsembleAggregator - - -«creates»- - -► DetectionResult
DetectionService - - -«creates»- - -► DetectionResult
```

- `ImageInput` **uses** the Preprocessor (sends itself for processing).
- Both the `EnsembleAggregator` and `DetectionService` **create** `DetectionResult` instances — the dashed arrow indicates a weaker, temporary relationship rather than permanent ownership.

---

## 4.4 Design Patterns Visible in the Diagram

1. **Strategy Pattern**: The abstract `DetectionEngine` with multiple concrete implementations (FFT, ELA, Noise, DL) is a textbook Strategy pattern — the algorithm varies, but the interface stays the same.

2. **Composition over Inheritance**: `DetectionService` holds a *list* of engines rather than inheriting from them. This allows adding or removing engines without changing the service class.

3. **Facade Pattern**: `DetectionService` acts as a simplified facade — external callers only need to call `detect(img)`, and the service handles all the internal complexity (preprocessing, parallel execution, aggregation, error handling).

---

## 4.5 What This Diagram Tells You (vs. the Other Diagrams)

The Class Diagram uniquely shows:
- **The inheritance hierarchy**: you can see that all four analysis engines share a common abstract base class, enforcing a consistent interface.
- **Attribute-level detail**: default values like `threshold = 0.5`, `ttaViews = 3`, `claheFactor = 1.5` reveal exact configuration parameters.
- **Multiplicity and ownership**: the composition relationship (`1 *`) makes it clear that one service manages many engines.
- **The complete method signatures**: you know exactly what inputs each method takes and what it returns — information that neither the sequence nor activity diagrams provide.
- **The data model**: `DetectionResult` shows every piece of data the system outputs, including all three forensic maps.
