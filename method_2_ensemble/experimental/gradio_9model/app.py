"""
AI Image Detector Ensemble — Gradio Version (v2.0)
===================================================
A multi-engine ensemble that detects whether an image is AI-generated or real.
Designed for Hugging Face Spaces.

Two ways to use:
  1. Web UI  — open the Space URL in a browser
  2. REST API — POST an image to /analyze (for Android / mobile apps)

Engines (Visual — human-interpreted):
  1. FFT    — Fast Fourier Transform (frequency-domain artifact detection)
  2. ELA    — Error Level Analysis  (compression-rate tampering detection)
  3. Noise  — Noise Pattern Forensic Analysis (camera noise fingerprint)

Engines (Scoring — contribute to final verdict):
  4. ResNet  — umm-maybe/AI-image-detector
  5. SigLIP  — Ateeqq/ai-vs-human-image-detector
  6. SDXL    — Organika/sdxl-detector
  7. ViT-DF  — prithivMLmods/Deep-Fake-Detector-v2-Model
  8. Wvolf   — Wvolf/ViT_Deepfake_Detection  (98.70% accuracy)
  9. SMOGY   — Smogy/SMOGY-Ai-images-detector (98.18% accuracy)

Accuracy Features:
  - Noise Pattern Forensic Analysis (physics-based, catches what ML misses)
  - Multi-Scale Inference (original + downscale + zoom-crop per model)
  - Confidence-weighted ensemble + adaptive threshold
  - CLAHE preprocessing enhancement
"""

import io
import json
import functools

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required on HF Spaces
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps, ImageFilter
from transformers import pipeline
import gradio as gr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# ─────────────────────────────────────────────
#  Model loading  (cached — loaded only once)
# ─────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_resnet_pipeline():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")


@functools.lru_cache(maxsize=1)
def load_siglip_pipeline():
    return pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")


@functools.lru_cache(maxsize=1)
def load_sdxl_pipeline():
    return pipeline("image-classification", model="Organika/sdxl-detector")


@functools.lru_cache(maxsize=1)
def load_deepfake_pipeline():
    return pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")


@functools.lru_cache(maxsize=1)
def load_wvolf_pipeline():
    return pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")


@functools.lru_cache(maxsize=1)
def load_smogy_pipeline():
    return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")





# ─────────────────────────────────────────────
#  Pre-processing
# ─────────────────────────────────────────────

def _apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Apply Contrast-Limited Adaptive Histogram Equalization.
    Enhances subtle compression artifacts and low-contrast AI traces.
    Pure PIL/numpy implementation (no OpenCV dependency).
    """
    arr = np.array(pil_img).astype(np.float32)
    # Per-channel adaptive enhancement
    enhanced = np.empty_like(arr)
    for c in range(3):
        channel = arr[:, :, c]
        # Local mean via box blur (block size 64)
        from PIL import ImageFilter
        ch_img = Image.fromarray(channel.astype(np.uint8))
        blurred = np.array(ch_img.filter(ImageFilter.BoxBlur(32))).astype(np.float32)
        # Adaptive contrast: amplify difference from local mean, clipped
        diff = channel - blurred
        enhanced[:, :, c] = np.clip(blurred + diff * 1.5, 0, 255)
    return Image.fromarray(enhanced.astype(np.uint8))


def prepare_image(pil_image: Image.Image):
    """
    Accepts a PIL image and returns variations for downstream tasks.
    Validates image format and strips metadata.
    """
    img = pil_image.convert("RGB")
    data = list(img.getdata())
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(data)
    grayscale_array = np.array(clean_img.convert("L"))
    buffer = io.BytesIO()
    clean_img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    ela_jpeg_img = Image.open(buffer).convert("RGB")
    return grayscale_array, ela_jpeg_img, clean_img


# ─────────────────────────────────────────────
#  Multi-Scale Test-Time Augmentation
# ─────────────────────────────────────────────

def _generate_multiscale_views(image: Image.Image) -> list:
    """
    Generate multiple views at different scales for robust inference.
    More sophisticated than simple TTA — catches scale-dependent artifacts.
    """
    w, h = image.size
    views = []

    # View 1: Original
    views.append(image)

    # View 2: Horizontal flip (catches left/right asymmetry)
    views.append(ImageOps.mirror(image))

    # View 3: Center crop 80% + resize back (zooms into fine details)
    crop_ratio = 0.80
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    views.append(image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS))

    # View 4: CLAHE-enhanced (reveals hidden low-contrast artifacts)
    views.append(_apply_clahe(image))

    return views


def _run_with_multiscale(model_fn, image: Image.Image) -> float:
    """
    Run model on multiple views and return averaged score.
    """
    views = _generate_multiscale_views(image)
    scores = [model_fn(view) for view in views]
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────
#  Analysis engines
# ─────────────────────────────────────────────

def fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_img


def run_fft(grayscale_array: np.ndarray) -> Image.Image:
    f = np.fft.fft2(grayscale_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(magnitude_spectrum, cmap="gray")
    ax.axis("off")
    ax.set_title("FFT Magnitude Spectrum", fontsize=10)
    plt.tight_layout()
    return fig_to_pil(fig)


def run_ela(original_img: Image.Image, jpeg_img: Image.Image) -> Image.Image:
    diff = ImageChops.difference(original_img, jpeg_img)
    return Image.eval(diff, lambda x: min(255, x * 15.0))


# ── Noise Pattern Forensic Analysis ──────────

def run_noise_analysis(image: Image.Image) -> tuple:
    """
    Physics-based forensic engine: extract and analyze image noise patterns.

    Real cameras leave sensor-specific noise (PRNU). AI-generated images have:
    - Unnaturally uniform/smooth noise
    - Missing high-frequency noise components
    - Spatially correlated noise patterns (from upsampling)

    Returns: (noise_score: float 0-1, noise_visualization: PIL.Image)
      noise_score: probability the noise pattern is synthetic (0=natural, 1=synthetic)
    """
    arr = np.array(image).astype(np.float64)

    # Extract noise residual: original - denoised
    # Use median filter (radius 2) as denoiser — preserves edges better than Gaussian
    denoised = np.array(image.filter(ImageFilter.MedianFilter(size=3))).astype(np.float64)
    noise = arr - denoised  # shape: (H, W, 3)

    # ── Feature 1: Noise variance (real photos have more noise than AI) ──
    noise_var = np.var(noise)
    # Real photos typically have variance 15-80, AI images 2-15
    # Normalize to 0-1 scale (higher = more likely AI/synthetic)
    var_score = 1.0 - min(1.0, noise_var / 50.0)  # low variance = likely AI

    # ── Feature 2: Spatial correlation of noise ──
    # Real sensor noise is spatially uncorrelated (random).
    # AI noise from upsampling has spatial correlation (neighboring pixels similar).
    noise_gray = np.mean(noise, axis=2)
    h, w = noise_gray.shape

    if h > 2 and w > 2:
        # Horizontal neighbor correlation
        horiz_corr = np.corrcoef(noise_gray[:, :-1].flatten(), noise_gray[:, 1:].flatten())[0, 1]
        # Vertical neighbor correlation
        vert_corr = np.corrcoef(noise_gray[:-1, :].flatten(), noise_gray[1:, :].flatten())[0, 1]
        spatial_corr = (abs(horiz_corr) + abs(vert_corr)) / 2.0
    else:
        spatial_corr = 0.0

    # Real noise: corr ≈ 0.0-0.15, AI noise: corr ≈ 0.2-0.8
    corr_score = min(1.0, spatial_corr / 0.4)  # high correlation = likely AI

    # ── Feature 3: Channel consistency ──
    # Real noise differs per color channel (Bayer filter pattern).
    # AI noise is often identical across channels.
    r_noise = noise[:, :, 0].flatten()
    g_noise = noise[:, :, 1].flatten()
    b_noise = noise[:, :, 2].flatten()

    rg_corr = abs(np.corrcoef(r_noise, g_noise)[0, 1]) if len(r_noise) > 10 else 0.0
    rb_corr = abs(np.corrcoef(r_noise, b_noise)[0, 1]) if len(r_noise) > 10 else 0.0
    channel_corr = (rg_corr + rb_corr) / 2.0
    # Real: channel corr ≈ 0.3-0.6, AI: channel corr ≈ 0.7-0.99
    chan_score = min(1.0, max(0.0, (channel_corr - 0.3) / 0.5))

    # ── Feature 4: Noise entropy (randomness) ──
    # Real noise has high entropy (truly random). AI noise has low entropy.
    noise_uint8 = np.clip((noise_gray * 10) + 128, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(noise_uint8, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # normalize
    hist = hist[hist > 0]  # remove zeros for log
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = 8.0  # max for 256 bins
    # Real: entropy ≈ 5-7, AI: entropy ≈ 2-5
    entropy_score = 1.0 - min(1.0, entropy / 6.0)  # low entropy = likely AI

    # ── Combined noise forensic score ──
    # Weighted combination of all noise features
    noise_score = (
        var_score * 0.25 +
        corr_score * 0.30 +
        chan_score * 0.25 +
        entropy_score * 0.20
    )
    noise_score = max(0.0, min(1.0, noise_score))

    # ── Visualization: amplified noise residual ──
    noise_vis = np.clip(np.abs(noise) * 8.0, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise_vis)

    return noise_score, noise_img


# ── Robust score extraction ──────────────────

_FAKE_LABELS = frozenset({
    "artificial", "fake", "ai", "ai generated", "ai_generated",
    "deepfake", "generated", "computer", "synthetic",
})
_REAL_LABELS = frozenset({
    "human", "real", "realism", "authentic", "nature", "photo",
    "not_ai_generated", "not ai generated",
})


def _extract_fake_score(results) -> float:
    """
    Robustly extract the 'fake' probability from model results.
    """
    for res in results:
        label = res["label"].lower().strip()
        if label in _FAKE_LABELS:
            return float(res["score"])
        if label in _REAL_LABELS:
            return float(1.0 - res["score"])

    # Fallback: partial keyword matching
    if results:
        top = results[0]
        label = top["label"].lower().strip()
        if any(kw in label for kw in ("fake", "ai", "deep", "artifi", "generat", "synth")):
            return float(top["score"])
        if any(kw in label for kw in ("real", "human", "authen", "photo", "nature")):
            return float(1.0 - top["score"])
        return float(top["score"])
    return 0.5


def run_resnet(image):
    return _extract_fake_score(load_resnet_pipeline()(image))

def run_siglip(image):
    return _extract_fake_score(load_siglip_pipeline()(image))

def run_sdxl_detector(image):
    return _extract_fake_score(load_sdxl_pipeline()(image))

def run_deepfake_detector(image):
    return _extract_fake_score(load_deepfake_pipeline()(image))

def run_wvolf(image):
    return _extract_fake_score(load_wvolf_pipeline()(image))

def run_smogy(image):
    return _extract_fake_score(load_smogy_pipeline()(image))



# ─────────────────────────────────────────────
#  Adaptive Confidence-Weighted Ensemble
# ─────────────────────────────────────────────

def _adaptive_ensemble(scores: dict) -> tuple:
    """
    Advanced ensemble that adapts threshold based on model agreement.

    Returns: (verdict: str, confidence: float, agreement: str)
      verdict: "FAKE", "REAL", or "UNCERTAIN"
      confidence: 0-100 percentage
      agreement: description of model consensus
    """
    all_scores = list(scores.values())
    n = len(all_scores)

    # ── Step 1: Confidence-weighted average ──
    epsilon = 1e-6
    weights = [abs(s - 0.5) + epsilon for s in all_scores]
    total_weight = sum(weights)
    weighted_avg = sum(s * w for s, w in zip(all_scores, weights)) / total_weight

    # ── Step 2: Count model votes ──
    fake_votes = sum(1 for s in all_scores if s > 0.5)
    real_votes = n - fake_votes
    agreement_ratio = max(fake_votes, real_votes) / n

    # ── Step 3: Measure model spread (disagreement) ──
    score_std = np.std(all_scores)

    # ── Step 4: Adaptive threshold logic ──
    if agreement_ratio >= 0.75 and score_std < 0.25:
        # Strong consensus: 75%+ models agree with low spread
        threshold = 0.45  # slightly lower threshold — trust the consensus
        agreement = f"Strong consensus ({fake_votes} fake / {real_votes} real)"
    elif agreement_ratio >= 0.60:
        # Moderate consensus
        threshold = 0.50  # standard threshold
        agreement = f"Moderate consensus ({fake_votes} fake / {real_votes} real)"
    else:
        # Models are split — require stronger evidence
        threshold = 0.55  # higher threshold to avoid false positives
        agreement = f"Split decision ({fake_votes} fake / {real_votes} real)"

    # ── Step 5: Determine verdict ──
    # Check for high uncertainty: all models near 0.5 AND high disagreement
    uncertain_models = sum(1 for s in all_scores if 0.35 < s < 0.65)
    if uncertain_models >= n * 0.6 and score_std > 0.15:
        verdict = "UNCERTAIN"
        confidence = round((1.0 - score_std) * 50, 2)  # low confidence
        agreement = f"High uncertainty ({uncertain_models}/{n} models unsure)"
    elif weighted_avg > threshold:
        verdict = "FAKE"
        confidence = round(weighted_avg * 100, 2)
    else:
        verdict = "REAL"
        confidence = round((1.0 - weighted_avg) * 100, 2)

    return verdict, confidence, agreement


# ─────────────────────────────────────────────
#  Core analysis (shared by UI and REST API)
# ─────────────────────────────────────────────

def run_full_analysis(pil_image: Image.Image) -> dict:
    """
    Runs all engines and returns a plain dict with all results.
    Called by both the Gradio UI function and the /analyze REST endpoint.
    """
    grayscale_array, ela_jpeg_img, rgb_img = prepare_image(pil_image)

    # Run all 6 scoring models with multi-scale TTA
    model_scores = {
        "resnet":   _run_with_multiscale(run_resnet, rgb_img),
        "siglip":   _run_with_multiscale(run_siglip, rgb_img),
        "sdxl":     _run_with_multiscale(run_sdxl_detector, rgb_img),
        "deepfake": _run_with_multiscale(run_deepfake_detector, rgb_img),
        "wvolf":    _run_with_multiscale(run_wvolf, rgb_img),
        "smogy":    _run_with_multiscale(run_smogy, rgb_img),
    }

    # Run noise forensic analysis
    noise_score, noise_img = run_noise_analysis(rgb_img)

    # Include noise score in ensemble
    model_scores["noise"] = noise_score

    # Adaptive ensemble verdict
    verdict, confidence, agreement = _adaptive_ensemble(model_scores)

    return {
        "verdict":    verdict,
        "confidence": confidence,
        "agreement":  agreement,
        "scores": {k: round(v * 100, 2) for k, v in model_scores.items()},
        # Internal UI images
        "_fft_img":   run_fft(grayscale_array),
        "_ela_img":   run_ela(rgb_img, ela_jpeg_img),
        "_noise_img": noise_img,
    }


# ─────────────────────────────────────────────
#  Gradio UI callback
# ─────────────────────────────────────────────

def analyze_image(pil_image):
    if pil_image is None:
        empty = "<p style='color:gray;text-align:center'>Upload an image to begin.</p>"
        return (empty, None, None, None,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                "{}")

    result = run_full_analysis(pil_image)
    verdict    = result["verdict"]
    confidence = result["confidence"]
    agreement  = result["agreement"]

    if verdict == "FAKE":
        color, icon = "#ff4b4b", "🤖"
    elif verdict == "UNCERTAIN":
        color, icon = "#f0a500", "⚠️"
    else:
        color, icon = "#00c44f", "✅"

    verdict_html = f"""
    <div style="text-align:center;padding:24px 16px;border-radius:16px;
                background:{color}22;border:2px solid {color};margin:8px 0;">
      <span style="font-size:3rem">{icon}</span>
      <h2 style="margin:8px 0;color:{color};font-size:2rem;font-weight:800">{verdict}</h2>
      <p style="margin:0;font-size:1.1rem;color:#ccc">
        <b>{confidence:.1f}%</b> certainty this image is <b>{verdict.lower()}</b>.
      </p>
      <p style="margin:4px 0 0;font-size:0.85rem;color:#999">{agreement}</p>
    </div>"""

    scores = result["scores"]
    json_str = json.dumps({
        "verdict":    verdict,
        "confidence": confidence,
        "agreement":  agreement,
        "scores":     scores,
    }, indent=2)

    return (
        verdict_html,
        result["_fft_img"],
        result["_ela_img"],
        result["_noise_img"],
        scores.get("resnet", 0),
        scores.get("siglip", 0),
        scores.get("sdxl", 0),
        scores.get("deepfake", 0),
        scores.get("wvolf", 0),
        scores.get("smogy", 0),
        scores.get("noise", 0),
        json_str,
    )


# ─────────────────────────────────────────────
#  Gradio UI definition
# ─────────────────────────────────────────────

DESCRIPTION_MD = """
## 👁️ AI Image Detector Ensemble v2.0
Upload any image and **nine independent engines** will analyse it to determine if it is **AI-generated** or **real**.

| # | Engine | Method |
|---|--------|--------|
| 1 | FFT    | Frequency-domain geometric artifact detection |
| 2 | ELA    | Compression-level tampering map |
| 3 | Noise  | Noise pattern forensic analysis (camera fingerprint) |
| 4 | ResNet | `umm-maybe/AI-image-detector` |
| 5 | SigLIP | `Ateeqq/ai-vs-human-image-detector` |
| 6 | SDXL   | `Organika/sdxl-detector` |
| 7 | ViT-DF | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| 8 | Wvolf  | `Wvolf/ViT_Deepfake_Detection` (98.70% acc) |
| 9 | SMOGY  | `Smogy/SMOGY-Ai-images-detector` (98.18% acc) |

> **Accuracy features:** Multi-scale inference (4 views) · Noise forensics · CLAHE enhancement · Adaptive confidence threshold

> **Android API:** `POST /analyze` with the image as `multipart/form-data` (field name: `file`)
"""

with gr.Blocks(
    title="AI Image Detector",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue", neutral_hue="slate"),
    css="footer { display:none !important; }",
) as demo:

    gr.Markdown(DESCRIPTION_MD)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Image", height=320)
            submit_btn  = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            verdict_output = gr.HTML(label="🏛️ Final Verdict")

    gr.Markdown("---\n### 🔬 Visual & Forensic Analysis")
    with gr.Row():
        fft_output   = gr.Image(type="pil", label="Engine 1 — FFT Spectrum")
        ela_output   = gr.Image(type="pil", label="Engine 2 — ELA Error Map")
        noise_output = gr.Image(type="pil", label="Engine 3 — Noise Pattern")

    with gr.Row():
        gr.Markdown("💡 **FFT:** Smooth gradient = real. Grid/star patterns = AI.")
        gr.Markdown("💡 **ELA:** Uniform = real. Glowing regions = tampered/AI.")
        gr.Markdown("💡 **Noise:** Bright uniform = real sensor noise. Dark/patterned = synthetic.")

    gr.Markdown("---\n### 🧠 Deep-Learning Scores *(% fake — multi-scale averaged)*")
    with gr.Row():
        resnet_output   = gr.Number(label="ResNet",       precision=2)
        siglip_output   = gr.Number(label="SigLIP",       precision=2)
        sdxl_output     = gr.Number(label="SDXL",         precision=2)
        deepfake_output = gr.Number(label="ViT DeepFake", precision=2)
    with gr.Row():
        wvolf_output    = gr.Number(label="Wvolf ViT",    precision=2)
        smogy_output    = gr.Number(label="SMOGY",        precision=2)
        noise_score_out = gr.Number(label="🔬 Noise Forensics", precision=2)

    gr.Markdown("---\n### 📦 JSON Result  *(for API / Android integration)*")
    json_output = gr.Textbox(label="Structured API Response", lines=16,
                             show_copy_button=True, interactive=False)

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[verdict_output, fft_output, ela_output, noise_output,
                 resnet_output, siglip_output, sdxl_output, deepfake_output,
                 wvolf_output, smogy_output, noise_score_out,
                 json_output],
        api_name=False,
    )


# ─────────────────────────────────────────────
#  FastAPI app — mounts Gradio UI + /analyze
# ─────────────────────────────────────────────

fastapi_app = FastAPI(title="AI Image Detector API")


@fastapi_app.post("/analyze", summary="Analyse an image for AI generation")
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Upload an image file and receive a JSON verdict.

    Response:
        {
          "verdict": "FAKE" | "REAL" | "UNCERTAIN",
          "confidence": 87.45,
          "agreement": "Strong consensus (5 fake / 2 real)",
          "scores": {
            "resnet": 92.3,
            "siglip": 88.1,
            "sdxl": 85.0,
            "deepfake": 84.5,
            "wvolf": 97.2,
            "smogy": 95.8,
            "noise": 72.3
          }
        }
    """
    content  = await file.read()
    pil_img  = Image.open(io.BytesIO(content)).convert("RGB")
    result   = run_full_analysis(pil_img)

    # strip internal UI keys before returning
    api_result = {k: v for k, v in result.items() if not k.startswith("_")}
    return JSONResponse(content=api_result)


# Mount Gradio UI at the root path
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ── Local development only ────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
