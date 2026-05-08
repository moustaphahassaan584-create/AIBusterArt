# 🧬 AI Image Detection — New Approach (Fine-Tuned Model) Methods Guide

> **Purpose:** This document explains the methods used in the **new approach** (fine-tuned ViT model). For the gradio version with 9 pre-trained models, see `method explain 2.md`.

---

## Table of Contents

1. [Why a New Approach?](#1-why-a-new-approach)
2. [Fine-Tuning — The Core Innovation](#2-fine-tuning--the-core-innovation)
3. [Visual Engines — FFT, ELA, Noise Forensics](#3-visual-engines--fft-ela-noise-forensics)
4. [The Fine-Tuned ViT Model (50% weight)](#4-the-fine-tuned-vit-model-50-weight)
5. [Backup Models — SigLIP and SMOGY](#5-backup-models--siglip-and-smogy)
6. [Noise Pattern Forensics (20% weight)](#6-noise-pattern-forensics-20-weight)
7. [The Weighted Ensemble](#7-the-weighted-ensemble)
8. [Quick-Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. Why a New Approach?

### The Problem with Pre-Trained Models

The gradio version uses 6 pre-trained models from Hugging Face. Each was trained by different people on different datasets. Their individual accuracy on modern AI images (Flux, Midjourney v6, DALL-E 3) ranged from **35% to 82%**. Even as an ensemble, accuracy was limited because:

- Models were trained on **old generators** (ProGAN, StyleGAN) but tested on **new ones** (Flux, DALL-E 3)
- Each model has different **biases** — some always say "FAKE", others almost never do
- No amount of TTA or ensemble tricks can fix a model that was trained on the wrong data

### What Commercial Tools Do Differently

Services like **SightEngine** achieve 95%+ accuracy because they **train their own models** on massive, continuously updated datasets. They don't rely on other people's pre-trained models.

### Our Solution: Fine-Tune Your Own Model

Instead of using 6 weak models, we **fine-tune one strong model** specifically for this task, on a modern dataset. One well-trained model beats six weak ones.

```
OLD APPROACH:                           NEW APPROACH:
6 random pre-trained models             1 fine-tuned model (50%)
(each 35-82% accuracy)                  + 2 best backup models (15% each)
→ Ensemble: ~70% real-world accuracy    + Noise forensics (20%)
                                        → Expected: 95%+ accuracy
```

---

## 2. Fine-Tuning — The Core Innovation

### 🍎 The Simple Analogy

Imagine you hire a general security guard (the pre-trained ViT model) who knows how to recognize people, objects, and situations. Now you specifically train them for 2 weeks on what counterfeit bills look like at YOUR bank. After training, they're FAR better at catching counterfeits than a random guard who never received specific training.

**That's fine-tuning.** We take a model that already understands images (ViT pre-trained on ImageNet) and specifically train it to recognize AI-generated vs. real images.

### 🔬 The Science

#### What Is Fine-Tuning?

| Step | What Happens | Data |
|------|-------------|------|
| 1. Pre-training | ViT learns general image features (edges, textures, objects) | ImageNet-21k (14M images, 21,000 classes) |
| 2. Fine-tuning | ViT specializes in real vs. fake classification | AI-vs-Real dataset (8,000 images, 2 classes) |

During fine-tuning:
- The pre-trained weights are loaded
- The classification head is replaced (21,000 classes → 2 classes: real/fake)
- The model is trained with a **very small learning rate** (2×10⁻⁵) to gently adapt existing knowledge
- After ~5 epochs, the model learns the specific patterns of modern AI-generated images

#### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| **Learning Rate** | 2e-5 | Small enough to preserve pre-trained features, large enough to learn new ones |
| **Epochs** | 5 | Enough for transfer learning (we're not training from scratch) |
| **Batch Size** | 16 | Fits in free Colab T4 GPU (16GB VRAM) |
| **Warmup** | 10% of steps | Gradually increases LR to prevent jarring weight updates at the start |
| **Mixed Precision** | FP16 | Uses half-precision math for 2× speed without accuracy loss |
| **Weight Decay** | 0.01 | Prevents overfitting by penalizing large weights |

#### The Dataset: `Parveshiiii/AI-vs-Real`

- **~10,000 images** (balanced: ~5,000 real + ~5,000 AI)
- High resolution (not 32×32 like CIFAR)
- Contains modern AI-generated images
- Publicly accessible (no authentication needed)

#### Training Process

```
Epoch 1: Model adapts classification head → rapid improvement
Epoch 2: Begins learning AI-specific patterns → accuracy jumps
Epoch 3: Fine-tunes edge cases → accuracy plateaus
Epoch 4-5: Polishes decision boundaries → best model saved
```

The trainer automatically saves the **best checkpoint** based on F1 score.

### ❓ Possible Professor Questions

**Q: "Why not train from scratch?"**  
A: Training from scratch requires millions of images and hundreds of GPU-hours. With transfer learning, we repurpose 86 million learned parameters and only need to update them slightly. This is 100× more efficient.

**Q: "What is the difference between pre-training and fine-tuning?"**  
A: Pre-training learns general visual features from a massive diverse dataset. Fine-tuning adapts these features for a specific task on a smaller, task-specific dataset. It's like a medical doctor (pre-trained on all of medicine) going through a cardiology fellowship (fine-tuned for heart disease).

**Q: "Could you overfit with only 8,000 images?"**  
A: Transfer learning dramatically reduces this risk because the model already has a strong prior understanding of images. We use weight decay, warmup, and evaluate on a held-out test set. The pre-trained features act as a regularizer.

**Q: "Why ViT and not ResNet for fine-tuning?"**  
A: ViT uses self-attention which captures global image relationships (lighting consistency, spatial coherence). For AI detection, global analysis is often more important than local texture analysis, making ViT a better base architecture.

---

## 3. Visual Engines — FFT, ELA, Noise Forensics

These are **identical** to the gradio version. They produce images for human interpretation:

- **FFT** — Frequency-domain analysis (grid/cross patterns = AI)
- **ELA** — Compression error map (non-uniform glow = AI)
- **Noise** — Sensor noise fingerprint (dark/uniform = synthetic)

Noise forensics **also** contributes a numerical score to the ensemble (20% weight).

> For detailed explanations of each, see `method explain 2.md` sections 3, 4, and 5.

---

## 4. The Fine-Tuned ViT Model (50% weight)

### Architecture: `google/vit-base-patch16-224`

```
Input Image (224×224×3)
    │
    ├── Split into 196 patches (16×16 each)
    ├── Linear projection → 768-dim embeddings
    ├── Add positional encodings
    ├── Prepend [CLS] token
    │
    ├── 12 Transformer Encoder layers
    │     └── Each: Self-Attention + FFN + LayerNorm
    │
    ├── Take [CLS] output (768-dim vector)
    ├── Classification head: 768 → 2 (real/fake)
    └── Softmax → [P(real), P(fake)]
```

### Why 50% Weight?

This model is **specifically trained for our task**. The backup models (SigLIP, SMOGY) are generic pre-trained models. Our fine-tuned model should be the primary decision-maker because:

1. It was trained on a modern dataset with current AI generators
2. It has seen the exact type of images we're trying to classify
3. Its training was optimized for our specific use case

---

## 5. Backup Models — SigLIP and SMOGY

| Model | Weight | Role |
|-------|--------|------|
| SigLIP (`Ateeqq/...`) | 15% | Semantic analysis — catches meaning-level inconsistencies |
| SMOGY (`Smogy/...`) | 15% | Modern generator coverage — tested on Flux, DALL-E, SD |

These serve as **backup detectors** — they catch things the fine-tuned model might miss because they were trained on different data distributions.

---

## 6. Noise Pattern Forensics (20% weight)

A physics-based engine that requires **no machine learning training**. It extracts the image noise residual and computes 4 statistical features. See `method explain 2.md` section 5 for full details.

Weight of 20% because it's a fundamentally different signal (physics vs. ML), providing diversity to the ensemble.

---

## 7. The Weighted Ensemble

### The Architecture

```
Fine-Tuned ViT (50%) ─┐
SigLIP (15%)          ├─→ Weighted Sum → Verdict (FAKE/REAL)
SMOGY (15%)           │
Noise Forensics (20%) ─┘
```

### The Math

```python
weighted_score = (finetuned × 0.50 + siglip × 0.15 + smogy × 0.15 + noise × 0.20) / 1.00

if weighted_score > 0.5:  verdict = "FAKE"
else:                     verdict = "REAL"
```

### Why Fixed Weights Instead of Adaptive?

With a strong primary model (the fine-tuned ViT), we don't need adaptive thresholds. The fine-tuned model provides confident, well-calibrated scores. The backup models and noise forensics add robustness without needing complex voting logic.

### ❓ Possible Professor Questions

**Q: "Why not just use the fine-tuned model alone?"**  
A: Having backup models protects against edge cases where the fine-tuned model was wrong. The noise forensics adds a fundamentally different signal (physics-based vs. ML-based), which catches cases that NO neural network detects.

**Q: "How did you choose the weights?"**  
A: The fine-tuned model gets 50% because it's the most accurate (trained for this task). SigLIP and SMOGY get 15% each as complementary backup signals. Noise forensics gets 20% because it's a fundamentally different approach (physics vs. ML), making it a valuable diversity signal.

---

## 8. Quick-Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│              NEW APPROACH CHEAT SHEET                                │
├──────────────┬───────────────────────────────────────────────────────┤
│ Fine-Tuned   │ YOUR model: google/vit-base fine-tuned on AI-vs-Real │
│ ViT (50%)    │ Expected: 95%+ accuracy on modern generators         │
│              │ ✅ Primary detector                                   │
├──────────────┼───────────────────────────────────────────────────────┤
│ SigLIP (15%) │ Semantic analysis backup                              │
│              │ ✅ Catches meaning-level inconsistencies              │
├──────────────┼───────────────────────────────────────────────────────┤
│ SMOGY (15%)  │ Modern generator backup (Flux/DALL-E/SD tested)      │
│              │ ✅ Complementary coverage                              │
├──────────────┼───────────────────────────────────────────────────────┤
│ Noise (20%)  │ Physics-based: noise variance, spatial corr,         │
│              │ channel consistency, entropy. No ML needed.           │
│              │ ✅ Visual + Score                                      │
├──────────────┼───────────────────────────────────────────────────────┤
│ FFT          │ Frequency analysis. Visual only.                      │
├──────────────┼───────────────────────────────────────────────────────┤
│ ELA          │ Compression analysis. Visual only.                    │
├──────────────┼───────────────────────────────────────────────────────┤
│ ENSEMBLE     │ Weighted sum: FT×0.50 + SigLIP×0.15 +                │
│              │ SMOGY×0.15 + Noise×0.20. Threshold = 0.5.            │
└──────────────┴───────────────────────────────────────────────────────┘
```

### Key Terms

| Term | Definition |
|------|-----------|
| **Fine-Tuning** | Continuing to train a pre-trained model on new, task-specific data |
| **Transfer Learning** | Reusing features learned from one task for a different task |
| **ViT** | Vision Transformer — processes images as sequences of patches using self-attention |
| **Learning Rate** | Step size during optimization — too high destroys pre-trained weights, too low prevents learning |
| **Warmup** | Gradually increasing learning rate at the start of training to prevent instability |
| **FP16** | Half-precision floating point — uses 16 bits instead of 32 for 2× speed and 50% less memory |
| **F1 Score** | Harmonic mean of precision and recall — balances both false positives and false negatives |
| **Epoch** | One complete pass through the entire training dataset |

---

> [!TIP]
> **Key differentiation from the gradio version:** Instead of relying on 6 pre-trained models that we have no control over, we **trained our own model** specifically for this task. This is the same approach used by commercial services like SightEngine. The result is a simpler system (3 models + 1 forensic engine instead of 6 models) with higher accuracy.
