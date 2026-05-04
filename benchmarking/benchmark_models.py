import subprocess, sys
for pkg in ["transformers", "datasets", "scikit-learn"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── Step 1: Imports ───────────────────────────────────────────────────────────
import time
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("✅ All imports successful.\n")


# ── Step 2: Load labeled dataset ──────────────────────────────────────────────
# Using "Parveshiiii/AI-vs-Real" — fully PUBLIC, high-resolution images.
# ⚠️ IMPORTANT: In this dataset, label 0 = AI-generated, label 1 = Real
#    We FLIP them so our convention matches: 0 = REAL, 1 = FAKE(AI)

print("📦 Loading dataset from Hugging Face (no manual download needed)...")
dataset = load_dataset("Parveshiiii/AI-vs-Real", split="train", download_mode="force_redownload", verification_mode="no_checks")

# Shuffle and take a subset to keep runtime reasonable on free Colab GPU
# Increase NUM_SAMPLES for more precise results (but slower)
NUM_SAMPLES = 200  # ← Change this: 100=fast (~8min), 200=balanced (~15min), 400=thorough (~30min)

dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))

# Auto-detect column names
print(f"   Columns found: {dataset.column_names}")
# Find the label column (could be 'label', 'labels', 'class', 'target', etc.)
LABEL_COL = None
for candidate in ["label", "labels", "class", "target", "is_ai", "category"]:
    if candidate in dataset.column_names:
        LABEL_COL = candidate
        break
if LABEL_COL is None:
    # Pick the non-image column
    LABEL_COL = [c for c in dataset.column_names if c != "image"][0]

# Find the image column
IMG_COL = "image" if "image" in dataset.column_names else dataset.column_names[0]

print(f"   Using: image='{IMG_COL}', label='{LABEL_COL}'")

# Check what the label values look like
sample_labels = [dataset[i][LABEL_COL] for i in range(min(5, len(dataset)))]
print(f"   Sample labels: {sample_labels}")

# Build ground truth: we need 0=REAL, 1=FAKE
# Detect if labels are strings or ints
raw_labels = [sample[LABEL_COL] for sample in dataset]

if isinstance(raw_labels[0], str):
    # String labels — map them
    flipped_labels = []
    for lbl in raw_labels:
        lbl_lower = lbl.lower().strip()
        if any(kw in lbl_lower for kw in ("fake", "ai", "generated", "synthetic", "artificial")):
            flipped_labels.append(1)
        else:
            flipped_labels.append(0)
elif isinstance(raw_labels[0], int):
    # Check which numeric value means what by looking at class names if available
    if hasattr(dataset.features[LABEL_COL], 'names'):
        class_names = dataset.features[LABEL_COL].names
        print(f"   Class names: {class_names}")
        # Find which index is "fake/ai"
        fake_idx = None
        for idx, name in enumerate(class_names):
            if any(kw in name.lower() for kw in ("fake", "ai", "generated", "synthetic", "artificial")):
                fake_idx = idx
                break
        if fake_idx is not None:
            flipped_labels = [1 if lbl == fake_idx else 0 for lbl in raw_labels]
        else:
            # Assume 0=real, 1=fake (standard)
            flipped_labels = raw_labels
    else:
        # No class names available. Assume: 0=AI(fake), 1=Real based on dataset docs
        flipped_labels = [1 - lbl for lbl in raw_labels]
else:
    flipped_labels = [int(lbl) for lbl in raw_labels]

num_real = flipped_labels.count(0)
num_fake = flipped_labels.count(1)
print(f"   Final: {len(dataset)} images ({num_real} real, {num_fake} AI-generated)\n")


# ── Step 3: Load all 6 models ────────────────────────────────────────────────
MODELS = {
    "ResNet (umm-maybe)":     "umm-maybe/AI-image-detector",
    "SigLIP (Ateeqq)":        "Ateeqq/ai-vs-human-image-detector",
    "SDXL (Organika)":        "Organika/sdxl-detector",
    "ViT-DF (prithivMLmods)": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "Wvolf ViT":              "Wvolf/ViT_Deepfake_Detection",
    "SMOGY":                  "Smogy/SMOGY-Ai-images-detector",
}

pipelines = {}
print("🧠 Loading models (this takes a few minutes the first time)...")
for name, model_id in MODELS.items():
    print(f"   Loading {name}...", end=" ", flush=True)
    try:
        pipelines[name] = pipeline("image-classification", model=model_id)
        print("✅")
    except Exception as e:
        print(f"❌ Failed: {e}")

print(f"\n   {len(pipelines)}/{len(MODELS)} models loaded successfully.\n")


# ── Step 4: Score extraction (same logic as app.py) ──────────────────────────

FAKE_KEYWORDS = {"artificial", "fake", "ai", "ai generated", "ai_generated",
                 "deepfake", "generated", "computer", "synthetic"}
REAL_KEYWORDS = {"human", "real", "realism", "authentic", "nature", "photo",
                 "not_ai_generated", "not ai generated"}

def extract_fake_score(results):
    """Extract the probability that the image is fake/AI-generated."""
    for res in results:
        label = res["label"].lower().strip()
        if label in FAKE_KEYWORDS:
            return float(res["score"])
        if label in REAL_KEYWORDS:
            return float(1.0 - res["score"])

    # Fallback: partial keyword match
    if results:
        top = results[0]
        label = top["label"].lower().strip()
        if any(kw in label for kw in ("fake", "ai", "deep", "artifi", "generat", "synth")):
            return float(top["score"])
        if any(kw in label for kw in ("real", "human", "authen", "photo", "nature")):
            return float(1.0 - top["score"])
        return float(top["score"])
    return 0.5


# ── Step 5: Run benchmark ───────────────────────────────────────────────────
print("🏁 Running benchmark — each dot is one image processed...")
print(f"   Total: {len(dataset)} images × {len(pipelines)} models = {len(dataset) * len(pipelines)} inferences\n")

# Store predictions: {model_name: [predicted_label_0_or_1, ...]}
all_predictions = {name: [] for name in pipelines}
ground_truth = []

for i, sample in enumerate(dataset):
    image = sample[IMG_COL].convert("RGB")
    true_label = flipped_labels[i]  # Use our flipped labels (0=real, 1=fake)
    ground_truth.append(true_label)

    for name, pipe in pipelines.items():
        try:
            results = pipe(image)
            fake_score = extract_fake_score(results)
            predicted = 1 if fake_score > 0.5 else 0
        except Exception:
            predicted = 0  # on error, default to real (conservative)

        all_predictions[name].append(predicted)

    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"   [{i+1}/{len(dataset)}] processed", flush=True)

print(f"\n✅ Benchmark complete! Processed {len(dataset)} images.\n")


# ── Step 6: Calculate metrics ────────────────────────────────────────────────
print("=" * 75)
print(f"{'MODEL':<28} {'ACCURACY':>8}  {'PREC.':>7}  {'RECALL':>7}  {'F1':>7}")
print("=" * 75)

results_table = []

for name in pipelines:
    preds = all_predictions[name]
    gt = ground_truth

    acc  = accuracy_score(gt, preds) * 100
    prec = precision_score(gt, preds, zero_division=0) * 100
    rec  = recall_score(gt, preds, zero_division=0) * 100
    f1   = f1_score(gt, preds, zero_division=0) * 100

    results_table.append((name, acc, prec, rec, f1))
    print(f"{name:<28} {acc:>7.2f}%  {prec:>6.2f}%  {rec:>6.2f}%  {f1:>6.2f}%")

print("=" * 75)

# Sort by F1 score (best overall metric)
results_table.sort(key=lambda x: x[4], reverse=True)

print("\n🏆 RANKING (by F1 score — best overall metric):\n")
for rank, (name, acc, prec, rec, f1) in enumerate(results_table, 1):
    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"   {emoji} #{rank}  {name:<28}  F1={f1:.2f}%  Acc={acc:.2f}%")

print("\n" + "-" * 75)
print("📋 RECOMMENDATION:")
print(f"   ✅ KEEP the top 4 models (best F1 scores)")
print(f"   ❌ REMOVE the bottom 2 models (they're dragging down accuracy)")
print(f"\n   Bottom 2: {results_table[-1][0]} and {results_table[-2][0]}")
print("-" * 75)


# ── Step 7: Ensemble comparison ──────────────────────────────────────────────
print("\n\n📊 BONUS: Ensemble Accuracy Comparison\n")

# Current 6-model ensemble
all_6_scores = np.array([all_predictions[name] for name in pipelines])
ensemble_6 = (all_6_scores.mean(axis=0) > 0.5).astype(int)
acc_6 = accuracy_score(ground_truth, ensemble_6) * 100

# Top 4 ensemble
top_4_names = [r[0] for r in results_table[:4]]
top_4_scores = np.array([all_predictions[name] for name in top_4_names])
ensemble_top4 = (top_4_scores.mean(axis=0) > 0.5).astype(int)
acc_top4 = accuracy_score(ground_truth, ensemble_top4) * 100

print(f"   All 6 models ensemble accuracy:  {acc_6:.2f}%")
print(f"   Top 4 models ensemble accuracy:  {acc_top4:.2f}%")
diff = acc_top4 - acc_6
if diff > 0:
    print(f"   → Top 4 is BETTER by {diff:.2f}% ✅ (drop the weak models!)")
elif diff < 0:
    print(f"   → All 6 is better by {-diff:.2f}% (keep all models)")
else:
    print(f"   → Same accuracy (drop weak models to save inference time)")

print("\n🎯 Done! Copy the results and share them with me so we can update app.py.")