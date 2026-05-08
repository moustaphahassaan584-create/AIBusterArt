# 🧬 AI Image Detector — API Documentation

## What Is an API?

An **API** (Application Programming Interface) is a way for different programs to talk to each other. Think of it like a waiter in a restaurant: you (the client) give your order to the waiter (the API), the waiter takes it to the kitchen (the server), and then brings your food (the response) back to you. You never go into the kitchen yourself — the waiter handles everything.

In our project, the API lets any program send an image and get back a verdict: **is this image AI-generated or real?**

---

## How Our API Is Built

The API is built using two Python frameworks that work together:

| Framework | Role |
|-----------|------|
| **FastAPI** | Handles the API endpoint — receives images and returns JSON results |
| **Gradio** | Provides a visual web interface so users can also interact through a browser |

Both are defined in a single file: `app.py`. FastAPI handles the programmatic (machine-to-machine) requests, while Gradio handles the visual (human-facing) interface. They are combined at the end of the file so both run on the same server.

---

## The API Endpoint

An **endpoint** is a specific URL that accepts requests. Our API has one endpoint:

```
POST /analyze
```

- **Method:** `POST` — this means you are *sending* data (the image) to the server.
- **URL path:** `/analyze` — this is the address you send the image to.
- **Input:** An image file (JPEG, PNG, etc.) uploaded as form data under the field name `file`.
- **Output:** A JSON object containing the analysis results.

### What Happens When You Send an Image

1. **Receive** — The server receives the uploaded image file.
2. **Convert** — The raw file bytes are converted into a format Python can work with (a PIL Image object).
3. **Analyze** — The image is passed through the full analysis pipeline, which includes:
   - **Fine-Tuned ViT Model (50% weight)** — Our primary AI model, a Vision Transformer fine-tuned to detect AI-generated images.
   - **SigLIP Model (15% weight)** — A backup AI model for cross-checking.
   - **SMOGY Model (15% weight)** — Another backup AI model for extra confidence.
   - **Noise Forensics (20% weight)** — A physics-based analysis that looks at the noise patterns in the image (AI images have different noise characteristics than real photos).
4. **Combine** — The scores from all four engines are combined using a weighted average to produce a final verdict.
5. **Respond** — The server sends back a JSON response with the results.

### Example Response

When the API finishes analyzing your image, it returns something like this:

```json
{
  "verdict": "FAKE",
  "confidence": 92.35,
  "agreement": "3 fake / 1 real",
  "scores": {
    "finetuned": 95.12,
    "siglip": 88.44,
    "smogy": 91.07,
    "noise": 72.30
  }
}
```

| Field | Meaning |
|-------|---------|
| `verdict` | The final decision — either `"FAKE"` (AI-generated) or `"REAL"` |
| `confidence` | How certain the system is about the verdict (0–100%) |
| `agreement` | How many of the 4 engines voted fake vs. real |
| `scores` | The individual score from each engine (0–100%, where higher = more likely fake) |

---

## The Gradio Web Interface

Besides the API endpoint, the application also provides a **web interface** at the root URL (`/`). This is a visual page where a user can:

1. Upload an image using a drag-and-drop area.
2. Click the "Analyze Image" button.
3. See the verdict, confidence, forensic images (FFT spectrum, ELA error map, noise pattern), and individual model scores — all displayed visually.

This interface is powered by **Gradio** and is useful for manual testing and demonstrations. It uses the same analysis logic as the API endpoint.

---

## How the Two Interfaces Are Connected

At the bottom of `app.py`, the Gradio app is mounted onto the FastAPI server:

```python
fastapi_app = FastAPI(title="AI Image Detector API")

@fastapi_app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    # ... handles API requests

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
```

This means:
- Visiting `http://server:7860/` opens the **Gradio web interface**.
- Sending a POST request to `http://server:7860/analyze` uses the **API endpoint**.
- Both share the same models and analysis logic — no code is duplicated.

---

## How to Use the API (Practical Example)

If you want to send an image to the API from your own code or a tool like `curl`, here is an example:

### Using curl (command line)
```bash
curl -X POST "http://localhost:7860/analyze" \
     -F "file=@my_image.png"
```

### Using Python
```python
import requests

with open("my_image.png", "rb") as f:
    response = requests.post("http://localhost:7860/analyze", files={"file": f})

result = response.json()
print(result["verdict"])      # "FAKE" or "REAL"
print(result["confidence"])   # e.g. 92.35
```

---

## Deployment

The application is containerized using **Docker**, which packages everything (Python, libraries, models, code) into a single portable unit. It is deployed on **Hugging Face Spaces**, a free cloud platform for hosting AI applications.

- The server listens on **port 7860**.
- Models are automatically downloaded from Hugging Face when the application starts for the first time.
- After the first load, models are cached so subsequent requests are fast.

---

## Summary

| Concept | Our Implementation |
|---------|--------------------|
| API Framework | FastAPI |
| Web Interface | Gradio |
| Endpoint | `POST /analyze` |
| Input | Image file |
| Output | JSON with verdict, confidence, agreement, and per-engine scores |
| Hosting | Docker container on Hugging Face Spaces |
| Port | 7860 |
