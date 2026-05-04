"""
===============================================================================
🔬 AI Image Detector — Fine-Tuning Notebook
===============================================================================
Run this in Google Colab with GPU enabled:
  Runtime → Change runtime type → T4 GPU

This notebook will:
  1. Load a balanced dataset of real + AI-generated images
  2. Fine-tune google/vit-base-patch16-224 for binary classification
  3. Evaluate accuracy on a held-out test set
  4. Push the trained model to your Hugging Face account

Expected results: 95-98% accuracy in ~20 minutes on free Colab GPU.
===============================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

packages = [
    "transformers>=4.38.0",
    "datasets",
    "accelerate",
    "evaluate",
    "scikit-learn",
    "huggingface_hub",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All packages installed.\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Configuration — EDIT THESE VALUES
# ══════════════════════════════════════════════════════════════════════════════

# Your Hugging Face username (the model will be pushed to YOUR account)
HF_USERNAME = "mohamed9679"  # ← Change this to your HF username

# Model name on HF Hub (will be: HF_USERNAME/MODEL_NAME)
MODEL_NAME = "ai-image-detector-v1"

# Training hyperparameters (optimized for this task)
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
BATCH_SIZE = 16          # 16 fits comfortably on free T4 GPU (16GB VRAM)
WARMUP_RATIO = 0.1       # 10% of steps for learning rate warmup
WEIGHT_DECAY = 0.01
NUM_TRAIN_SAMPLES = 8000  # Use 8000 training images (4000 real + 4000 AI)
NUM_TEST_SAMPLES = 2000   # Use 2000 test images (1000 real + 1000 AI)

# Base model — ViT pretrained on ImageNet-21k
BASE_MODEL = "google/vit-base-patch16-224"

print(f"📋 Config:")
print(f"   Model: {BASE_MODEL}")
print(f"   Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
print(f"   Train: {NUM_TRAIN_SAMPLES} images, Test: {NUM_TEST_SAMPLES} images")
print(f"   Will push to: {HF_USERNAME}/{MODEL_NAME}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load and prepare dataset
# ══════════════════════════════════════════════════════════════════════════════
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import ViTImageProcessor

print("📦 Loading dataset (cached after first download)...")
dataset = load_dataset(
    "Parveshiiii/AI-vs-Real",
    split="train",
    verification_mode="no_checks",
)

# Detect column names
print(f"   Columns: {dataset.column_names}")
IMG_COL = "image" if "image" in dataset.column_names else dataset.column_names[0]
LABEL_COL = None
for c in ["label", "labels", "binary_label", "class", "target", "category"]:
    if c in dataset.column_names:
        LABEL_COL = c
        break
if LABEL_COL is None:
    LABEL_COL = [c for c in dataset.column_names if c != IMG_COL][0]
print(f"   Using: image='{IMG_COL}', label='{LABEL_COL}'")

# Detect and normalize labels
sample_labels = [dataset[i][LABEL_COL] for i in range(5)]
print(f"   Sample labels: {sample_labels}")

# Determine label mapping
if hasattr(dataset.features[LABEL_COL], 'names'):
    class_names = dataset.features[LABEL_COL].names
    print(f"   Class names: {class_names}")

# For this dataset: 0 = AI-generated, 1 = Real
# We want: 0 = Real, 1 = Fake/AI → so we flip
# The model will output: label 0 = "real", label 1 = "fake"

def normalize_label(example):
    """Flip labels: dataset's 0(AI) → 1(Fake), dataset's 1(Real) → 0(Real)"""
    raw = example[LABEL_COL]
    if isinstance(raw, str):
        raw_lower = raw.lower().strip()
        if any(kw in raw_lower for kw in ("fake", "ai", "generated", "synthetic")):
            example["label"] = 1
        else:
            example["label"] = 0
    else:
        # Numeric: 0=AI→Fake(1), 1=Real→Real(0)
        example["label"] = 1 - int(raw)
    return example

dataset = dataset.map(normalize_label)

# Balance the dataset: equal real and fake
real_ds = dataset.filter(lambda x: x["label"] == 0)
fake_ds = dataset.filter(lambda x: x["label"] == 1)
print(f"   Total real: {len(real_ds)}, Total fake: {len(fake_ds)}")

# Adapt to minority class — use 80% train / 20% test from the smaller class
minority_size = min(len(real_ds), len(fake_ds))
n_per_class_train = int(minority_size * 0.80)
n_per_class_test  = minority_size - n_per_class_train

print(f"   Minority class has {minority_size} images → using {n_per_class_train} train + {n_per_class_test} test per class")

# Shuffle and split
real_ds = real_ds.shuffle(seed=42)
fake_ds = fake_ds.shuffle(seed=42)

real_train = real_ds.select(range(n_per_class_train))
fake_train = fake_ds.select(range(n_per_class_train))

real_test = real_ds.select(range(n_per_class_train, n_per_class_train + n_per_class_test))
fake_test = fake_ds.select(range(n_per_class_train, n_per_class_train + n_per_class_test))

train_dataset = concatenate_datasets([real_train, fake_train]).shuffle(seed=42)
test_dataset = concatenate_datasets([real_test, fake_test]).shuffle(seed=42)

print(f"\n✅ Dataset ready:")
print(f"   Train: {len(train_dataset)} images ({len(real_train)} real + {len(fake_train)} fake)")
print(f"   Test:  {len(test_dataset)} images ({len(real_test)} real + {len(fake_test)} fake)\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Preprocessing — ViT Image Processor
# ══════════════════════════════════════════════════════════════════════════════
print("🖼️ Setting up image preprocessing...")

processor = ViTImageProcessor.from_pretrained(BASE_MODEL)

def preprocess(examples):
    """Convert PIL images to model inputs with data augmentation."""
    images = []
    for img in examples[IMG_COL]:
        if img is None:
            continue
        img = img.convert("RGB")
        images.append(img)

    if not images:
        return {"pixel_values": [], "label": []}

    inputs = processor(images=images, return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"],
        "label": examples["label"][:len(images)],
    }

# Apply preprocessing
print("   Processing training set...")
train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    batch_size=32,
    remove_columns=[c for c in train_dataset.column_names if c not in ["pixel_values", "label"]],
)
train_dataset.set_format("torch")

print("   Processing test set...")
test_dataset = test_dataset.map(
    preprocess,
    batched=True,
    batch_size=32,
    remove_columns=[c for c in test_dataset.column_names if c not in ["pixel_values", "label"]],
)
test_dataset.set_format("torch")

print(f"✅ Preprocessing complete.\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Load pre-trained ViT and set up for fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import evaluate

print("🧠 Loading pre-trained ViT model...")

# id2label / label2id for the classification head
id2label = {0: "real", 1: "fake"}
label2id = {"real": 0, "fake": 1}

model = ViTForImageClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # we're replacing the classification head
)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"✅ Model loaded.\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Training
# ══════════════════════════════════════════════════════════════════════════════
print("🏋️ Starting fine-tuning...\n")

# Metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, zero_division=0)["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, zero_division=0)["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }

training_args = TrainingArguments(
    output_dir="./vit-ai-detector",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # mixed precision on GPU
    dataloader_num_workers=2,
    remove_unused_columns=False,
    push_to_hub=False,  # we push manually after training
    report_to="none",   # disable wandb etc.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train!
train_result = trainer.train()

print(f"\n✅ Training complete!")
print(f"   Training loss: {train_result.training_loss:.4f}")
print(f"   Training time: {train_result.metrics['train_runtime']:.0f} seconds\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Evaluate on test set
# ══════════════════════════════════════════════════════════════════════════════
print("📊 Evaluating on test set...\n")

metrics = trainer.evaluate()
print("=" * 60)
print(f"  FINAL TEST RESULTS")
print("=" * 60)
print(f"  Accuracy:  {metrics['eval_accuracy'] * 100:.2f}%")
print(f"  Precision: {metrics['eval_precision'] * 100:.2f}%")
print(f"  Recall:    {metrics['eval_recall'] * 100:.2f}%")
print(f"  F1 Score:  {metrics['eval_f1'] * 100:.2f}%")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Push to Hugging Face Hub
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n🚀 Pushing model to Hugging Face Hub...")
print(f"   Destination: {HF_USERNAME}/{MODEL_NAME}")
print(f"\n⚠️  You need to be logged in to Hugging Face.")
print(f"   Run this in a separate cell first:")
print(f"   from huggingface_hub import notebook_login; notebook_login()")

try:
    # Push model and processor
    repo_id = f"{HF_USERNAME}/{MODEL_NAME}"
    model.push_to_hub(repo_id, commit_message="Fine-tuned ViT for AI image detection")
    processor.push_to_hub(repo_id, commit_message="Add image processor")
    print(f"\n✅ Model pushed successfully!")
    print(f"   🔗 https://huggingface.co/{repo_id}")
    print(f"\n   Use in your app with:")
    print(f'   pipeline("image-classification", model="{repo_id}")')
except Exception as e:
    print(f"\n❌ Push failed: {e}")
    print(f"\n   To fix, run this in a NEW cell FIRST:")
    print(f"   from huggingface_hub import notebook_login")
    print(f"   notebook_login()")
    print(f"\n   Then re-run ONLY the push step (Step 7) by copying these lines:")
    print(f'   repo_id = "{HF_USERNAME}/{MODEL_NAME}"')
    print(f'   model.push_to_hub(repo_id)')
    print(f'   processor.push_to_hub(repo_id)')

print("\n" + "=" * 60)
print("🎯 DONE! Next steps:")
print(f"   1. Verify the model at: https://huggingface.co/{HF_USERNAME}/{MODEL_NAME}")
print(f"   2. Update app.py to use your fine-tuned model")
print("=" * 60)
