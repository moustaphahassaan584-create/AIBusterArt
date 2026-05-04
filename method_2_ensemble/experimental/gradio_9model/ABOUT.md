# 🔬 Experimental: Gradio 9-Model Version

> **This is NOT the final production version.** This was an intermediate experimental approach.

## What Is This?

This is a **Gradio-based version** that uses **9 pre-trained models** from HuggingFace for AI image detection. It served as a stepping stone between Method 1 (ViT-only with Streamlit) and Method 2 (Ensemble with fine-tuned model).

## Why It Was Replaced

- Used 6+ pre-trained models that we had **no control over**
- Each model was trained on **different, older datasets** (ProGAN, StyleGAN era)
- Accuracy on modern AI generators (Flux, DALL-E 3, Midjourney v6) was **limited (35-82%)**
- The ensemble couldn't compensate for models that were fundamentally trained on the wrong data

## What Method 2 Does Differently

Instead of using many weak pre-trained models, Method 2:
- **Fine-tunes one strong model** on a modern AI-vs-Real dataset
- Uses only **2 backup models** + **noise forensics** for diversity
- Achieves **95%+ accuracy** compared to ~70% with this approach

## Files

- `app.py` — The full Gradio + FastAPI app (618 lines)
- `requirements.txt` — Python dependencies

## How to Run (for reference only)

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```
