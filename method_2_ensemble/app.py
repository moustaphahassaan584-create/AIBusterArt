"""
AI Image Detector — New Approach (Fine-Tuned Model)
====================================================
Uses YOUR fine-tuned ViT model as the primary detector,
backed by the 2 best pre-trained models + noise forensics.

Engines:
  1. FFT    — frequency-domain artifact detection (visual)
  2. ELA    — compression tampering map (visual)
  3. Noise  — noise pattern forensics (visual + scoring)
  4. ViT-FT — YOUR fine-tuned model (primary detector)
  5. SigLIP — Ateeqq/ai-vs-human-image-detector (backup)
  6. SMOGY  — Smogy/SMOGY-Ai-images-detector (backup)
"""

import io
import json
import functools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps, ImageFilter
from transformers import pipeline
import gradio as gr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# ─────────────────────────────────────────────
#  CONFIGURATION — Update after fine-tuning
# ─────────────────────────────────────────────

# ⬇️ CHANGE THIS to your fine-tuned model ID after running the notebook
FINETUNED_MODEL = "mohamed9679/ai-image-detector-v1"

# Weights for the ensemble — as described in the research report
WEIGHTS = {
    "finetuned": 0.50,   # Fine-tuned ViT — primary detector (50%)
    "siglip":    0.15,   # SigLIP backup model (15%)
    "smogy":     0.15,   # SMOGY backup model (15%)
    "noise":     0.20,   # Physics-based noise forensics (20%)
}


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_finetuned_pipeline():
    return pipeline("image-classification", model=FINETUNED_MODEL)


@functools.lru_cache(maxsize=1)
def load_siglip_pipeline():
    return pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")


@functools.lru_cache(maxsize=1)
def load_smogy_pipeline():
    return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")


# ─────────────────────────────────────────────
#  Pre-processing
# ─────────────────────────────────────────────

def prepare_image(pil_image: Image.Image):
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
#  Test-Time Augmentation
# ─────────────────────────────────────────────

def _generate_views(image: Image.Image) -> list:
    w, h = image.size
    views = [image]
    # Horizontal flip
    views.append(ImageOps.mirror(image))
    # Center crop 80%
    cw, ch = int(w * 0.8), int(h * 0.8)
    left, top = (w - cw) // 2, (h - ch) // 2
    views.append(image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS))
    return views


def _run_with_tta(model_fn, image: Image.Image) -> float:
    views = _generate_views(image)
    scores = [model_fn(view) for view in views]
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────
#  Visual analysis engines
# ─────────────────────────────────────────────

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def run_fft(grayscale_array):
    f = np.fft.fft2(grayscale_array)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(magnitude, cmap="gray")
    ax.axis("off")
    ax.set_title("FFT Magnitude Spectrum", fontsize=10)
    plt.tight_layout()
    return fig_to_pil(fig)


def run_ela(original, jpeg):
    diff = ImageChops.difference(original, jpeg)
    return Image.eval(diff, lambda x: min(255, x * 15.0))


# ─────────────────────────────────────────────
#  Noise Pattern Forensic Analysis
# ─────────────────────────────────────────────

def run_noise_analysis(image: Image.Image) -> tuple:
    arr = np.array(image).astype(np.float64)
    denoised = np.array(image.filter(ImageFilter.MedianFilter(size=3))).astype(np.float64)
    noise = arr - denoised

    # Feature 1: Noise variance
    noise_var = np.var(noise)
    var_score = 1.0 - min(1.0, noise_var / 50.0)

    # Feature 2: Spatial correlation
    noise_gray = np.mean(noise, axis=2)
    h, w = noise_gray.shape
    if h > 2 and w > 2:
        horiz = np.corrcoef(noise_gray[:, :-1].flatten(), noise_gray[:, 1:].flatten())[0, 1]
        vert = np.corrcoef(noise_gray[:-1, :].flatten(), noise_gray[1:, :].flatten())[0, 1]
        spatial_corr = (abs(horiz) + abs(vert)) / 2.0
    else:
        spatial_corr = 0.0
    corr_score = min(1.0, spatial_corr / 0.4)

    # Feature 3: Channel consistency
    r, g, b = noise[:,:,0].flatten(), noise[:,:,1].flatten(), noise[:,:,2].flatten()
    rg = abs(np.corrcoef(r, g)[0,1]) if len(r) > 10 else 0.0
    rb = abs(np.corrcoef(r, b)[0,1]) if len(r) > 10 else 0.0
    chan_score = min(1.0, max(0.0, ((rg + rb) / 2 - 0.3) / 0.5))

    # Feature 4: Noise entropy
    noise_u8 = np.clip((noise_gray * 10) + 128, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(noise_u8, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    entropy_score = 1.0 - min(1.0, entropy / 6.0)

    # Combined score
    score = var_score * 0.25 + corr_score * 0.30 + chan_score * 0.25 + entropy_score * 0.20
    score = max(0.0, min(1.0, score))

    # Visualization
    noise_vis = np.clip(np.abs(noise) * 8.0, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise_vis)

    return score, noise_img


# ─────────────────────────────────────────────
#  Score extraction
# ─────────────────────────────────────────────

_FAKE = frozenset({"artificial","fake","ai","ai generated","ai_generated","deepfake","generated","computer","synthetic"})
_REAL = frozenset({"human","real","realism","authentic","nature","photo","not_ai_generated","not ai generated"})

def _extract_fake_score(results):
    for r in results:
        l = r["label"].lower().strip()
        if l in _FAKE: return float(r["score"])
        if l in _REAL: return float(1.0 - r["score"])
    if results:
        top = results[0]
        l = top["label"].lower().strip()
        if any(k in l for k in ("fake","ai","deep","artifi","generat","synth")): return float(top["score"])
        if any(k in l for k in ("real","human","authen","photo","nature")): return float(1.0 - top["score"])
        return float(top["score"])
    return 0.5


def run_finetuned(image):
    return _extract_fake_score(load_finetuned_pipeline()(image))

def run_siglip(image):
    return _extract_fake_score(load_siglip_pipeline()(image))

def run_smogy(image):
    return _extract_fake_score(load_smogy_pipeline()(image))


# ─────────────────────────────────────────────
#  Weighted ensemble
# ─────────────────────────────────────────────

def _weighted_ensemble(scores: dict) -> tuple:
    weighted_sum = sum(scores[k] * WEIGHTS[k] for k in scores)
    total_weight = sum(WEIGHTS[k] for k in scores)
    avg = weighted_sum / total_weight

    # Count votes
    fake_votes = sum(1 for s in scores.values() if s > 0.5)
    real_votes = len(scores) - fake_votes

    if avg > 0.5:
        verdict = "FAKE"
        confidence = round(avg * 100, 2)
    else:
        verdict = "REAL"
        confidence = round((1.0 - avg) * 100, 2)

    agreement = f"{fake_votes} fake / {real_votes} real"

    return verdict, confidence, agreement


# ─────────────────────────────────────────────
#  Core analysis
# ─────────────────────────────────────────────

def run_full_analysis(pil_image: Image.Image) -> dict:
    grayscale_array, ela_jpeg_img, rgb_img = prepare_image(pil_image)

    # Run models with TTA
    scores = {
        "finetuned": _run_with_tta(run_finetuned, rgb_img),
        "siglip":    _run_with_tta(run_siglip, rgb_img),
        "smogy":     _run_with_tta(run_smogy, rgb_img),
    }

    # Run noise forensics
    noise_score, noise_img = run_noise_analysis(rgb_img)
    scores["noise"] = noise_score

    # Ensemble verdict
    verdict, confidence, agreement = _weighted_ensemble(scores)

    return {
        "verdict":    verdict,
        "confidence": confidence,
        "agreement":  agreement,
        "scores": {k: round(v * 100, 2) for k, v in scores.items()},
        "_fft_img":   run_fft(grayscale_array),
        "_ela_img":   run_ela(rgb_img, ela_jpeg_img),
        "_noise_img": noise_img,
    }


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────

def analyze_image(pil_image):
    if pil_image is None:
        empty = "<p style='color:gray;text-align:center'>Upload an image to begin.</p>"
        return empty, None, None, None, 0.0, 0.0, 0.0, 0.0, "{}"

    result = run_full_analysis(pil_image)
    v, c, a = result["verdict"], result["confidence"], result["agreement"]

    if v == "FAKE":
        color, icon = "#ff4b4b", "🤖"
    else:
        color, icon = "#00c44f", "✅"

    html = f"""
    <div style="text-align:center;padding:24px 16px;border-radius:16px;
                background:{color}22;border:2px solid {color};margin:8px 0;">
      <span style="font-size:3rem">{icon}</span>
      <h2 style="margin:8px 0;color:{color};font-size:2rem;font-weight:800">{v}</h2>
      <p style="margin:0;font-size:1.1rem;color:#ccc">
        <b>{c:.1f}%</b> certainty · <span style="font-size:0.9rem">{a}</span>
      </p>
    </div>"""

    s = result["scores"]
    j = json.dumps({"verdict": v, "confidence": c, "agreement": a, "scores": s}, indent=2)

    return html, result["_fft_img"], result["_ela_img"], result["_noise_img"], s.get("finetuned",0), s.get("siglip",0), s.get("smogy",0), s.get("noise",0), j


# ─────────────────────────────────────────────
#  Gradio UI — Premium Design
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }
footer { display: none !important; }

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%) !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 32px 20px 16px;
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.08));
    border-radius: 16px;
    border: 1px solid rgba(139,92,246,0.25);
    margin-bottom: 8px;
}
.hero-header h1 { margin: 0 0 6px; font-size: 1.8rem; font-weight: 800; color: #e2e8f0; }
.hero-header .tagline { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.hero-header .badge {
    display: inline-block; margin-top: 10px; padding: 4px 14px;
    background: rgba(139,92,246,0.2); border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px; font-size: 0.75rem; color: #a78bfa; font-weight: 600;
    letter-spacing: 0.3px;
}

/* Engine cards */
.engines-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin: 10px 0 4px; }
.engine-card {
    background: rgba(30,30,60,0.6); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 8px 12px; text-align: center; min-width: 120px; flex: 1;
    backdrop-filter: blur(10px);
}
.engine-card .name { font-weight: 700; font-size: 0.8rem; color: #e2e8f0; }
.engine-card .weight { font-size: 0.7rem; color: #8b5cf6; font-weight: 600; margin-top: 2px; }
.engine-card .type { font-size: 0.65rem; color: #64748b; margin-top: 1px; }
.engine-card.primary { border-color: rgba(139,92,246,0.5); background: rgba(139,92,246,0.1); }

/* Section headers */
.section-title {
    font-size: 0.95rem; font-weight: 700; color: #a78bfa;
    margin: 16px 0 6px; padding-left: 4px;
    border-left: 3px solid #8b5cf6; padding-left: 10px;
}

/* Override Gradio dark styling */
.dark .gr-block { background: rgba(20,20,45,0.8) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 12px !important; }
.dark .gr-button-primary {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    border: none !important; font-weight: 700 !important; font-size: 1rem !important;
    border-radius: 10px !important; padding: 12px !important;
    box-shadow: 0 4px 15px rgba(139,92,246,0.3) !important;
    transition: all 0.3s ease !important;
}
.dark .gr-button-primary:hover {
    box-shadow: 0 6px 20px rgba(139,92,246,0.5) !important;
    transform: translateY(-1px) !important;
}
"""

HEADER_HTML = f"""
<div class="hero-header">
    <h1>🧬 AI Image Detector</h1>
    <p class="tagline">Powered by a <b>fine-tuned Vision Transformer</b> with 99.4% accuracy</p>
    <span class="badge">✨ FINE-TUNED MODEL · 4 ENGINES · NOISE FORENSICS</span>
</div>
<div class="engines-row">
    <div class="engine-card primary">
        <div class="name">⭐ ViT Fine-Tuned</div>
        <div class="weight">50% weight</div>
        <div class="type">Your custom model</div>
    </div>
    <div class="engine-card">
        <div class="name">SigLIP</div>
        <div class="weight">15%</div>
        <div class="type">Semantic</div>
    </div>
    <div class="engine-card">
        <div class="name">SMOGY</div>
        <div class="weight">15%</div>
        <div class="type">Modern AI</div>
    </div>
    <div class="engine-card">
        <div class="name">🔬 Noise</div>
        <div class="weight">20%</div>
        <div class="type">Physics-based</div>
    </div>
    <div class="engine-card">
        <div class="name">FFT</div>
        <div class="weight">visual</div>
        <div class="type">Frequency</div>
    </div>
    <div class="engine-card">
        <div class="name">ELA</div>
        <div class="weight">visual</div>
        <div class="type">Compression</div>
    </div>
</div>
"""


def analyze_image(pil_image):
    if pil_image is None:
        empty = "<p style='color:#64748b;text-align:center;padding:40px'>Upload an image to begin analysis.</p>"
        return empty, None, None, None, 0.0, 0.0, 0.0, 0.0, "{}"

    result = run_full_analysis(pil_image)
    v, c, a = result["verdict"], result["confidence"], result["agreement"]

    if v == "FAKE":
        color, bg, icon = "#ef4444", "rgba(239,68,68,0.12)", "🤖"
    else:
        color, bg, icon = "#22c55e", "rgba(34,197,94,0.12)", "✅"

    html = f"""
    <div style="text-align:center;padding:28px 20px;border-radius:16px;
                background:{bg};border:2px solid {color};margin:4px 0;">
      <div style="font-size:3.5rem;line-height:1">{icon}</div>
      <h2 style="margin:10px 0 6px;color:{color};font-size:2.2rem;font-weight:800;letter-spacing:1px">{v}</h2>
      <p style="margin:0;font-size:1.05rem;color:#94a3b8">
        <b style="color:#e2e8f0;font-size:1.2rem">{c:.1f}%</b> certainty
      </p>
      <p style="margin:6px 0 0;font-size:0.8rem;color:#64748b">Engine votes: {a}</p>
    </div>"""

    s = result["scores"]
    j = json.dumps({"verdict": v, "confidence": c, "agreement": a, "scores": s}, indent=2)

    return html, result["_fft_img"], result["_ela_img"], result["_noise_img"], s.get("finetuned",0), s.get("siglip",0), s.get("smogy",0), s.get("noise",0), j


with gr.Blocks(
    title="AI Image Detector — Fine-Tuned",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Image", height=340)
            submit_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            verdict_out = gr.HTML(label="Verdict")

    gr.HTML('<div class="section-title">🔬 Forensic Analysis</div>')
    with gr.Row():
        fft_out = gr.Image(type="pil", label="FFT Spectrum", height=220)
        ela_out = gr.Image(type="pil", label="ELA Error Map", height=220)
        noise_out = gr.Image(type="pil", label="Noise Pattern", height=220)

    gr.HTML('<div class="section-title">🧠 Model Scores — TTA averaged (% fake confidence)</div>')
    with gr.Row():
        ft_out = gr.Number(label="⭐ Fine-Tuned ViT (50%)", precision=2)
        sig_out = gr.Number(label="SigLIP (15%)", precision=2)
        smogy_out = gr.Number(label="SMOGY (15%)", precision=2)
        noise_score_out = gr.Number(label="🔬 Noise (20%)", precision=2)

    gr.HTML('<div class="section-title">📦 API Response</div>')
    json_out = gr.Textbox(label="JSON", lines=8, show_copy_button=True, interactive=False)

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[verdict_out, fft_out, ela_out, noise_out, ft_out, sig_out, smogy_out, noise_score_out, json_out],
        api_name=False,
    )


# ─────────────────────────────────────────────
#  FastAPI
# ─────────────────────────────────────────────

fastapi_app = FastAPI(title="AI Image Detector API")

@fastapi_app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    result = run_full_analysis(pil_img)
    api_result = {k: v for k, v in result.items() if not k.startswith("_")}
    return JSONResponse(content=api_result)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
