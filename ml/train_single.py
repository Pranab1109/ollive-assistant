"""
train_single.py — Train a single multiclass SetFit model (Arm B).

Produces a 5-class classifier:
    benign | prompt_injection | role_override | out_of_scope | sensitive_bias

The trained model is saved to models/guardrail_single/ by default.

Usage:
    python ml/train_single.py
"""

import os
import json
import time
from collections import Counter

import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["benign", "prompt_injection", "role_override", "out_of_scope", "sensitive_bias"]
OUTPUT_DIR = os.path.abspath(os.path.join(HERE, "..", "models", "guardrail_single"))

TRAINING_ARGS = dict(
    batch_size=64,
    num_epochs=1,             # SetFit converges fast; 1 epoch is sufficient
    num_iterations=10,        # contrastive pairs per example
    body_learning_rate=2e-5,
    head_learning_rate=1e-2,
    logging_steps=1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU detected: {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    print("  No CUDA GPU found — training on CPU (will be slower).")
    return "cpu"


class ProgressCallback(TrainerCallback):
    """Prints elapsed time after every training step."""

    def __init__(self):
        self.step = 0
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        elapsed = time.time() - self.t0
        print(f"  step {self.step:>4d}  |  elapsed {elapsed:6.1f}s", flush=True)


def load_jsonl_as_dataset(filename: str) -> Dataset:
    """Load a JSONL file from ml/data/ and return a HuggingFace Dataset."""
    path = os.path.join(HERE, "data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Run build_dataset.py first.")

    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Training Single Multiclass SetFit Model (Arm B)")
    print("=" * 60)

    device = detect_device()

    # Load datasets
    print("\n  Loading datasets...")
    train_ds = load_jsonl_as_dataset("train.jsonl")
    test_ds = load_jsonl_as_dataset("test.jsonl")

    dist = Counter(train_ds["label"])
    print(f"  train: {len(train_ds)} samples  |  test: {len(test_ds)} samples")
    for label in LABELS:
        print(f"    {label:>20s}: {dist.get(label, 0)}")

    # Load base model
    print(f"\n  Loading base model: {BASE_MODEL}")
    model = SetFitModel.from_pretrained(BASE_MODEL, labels=LABELS)
    model.to(device)

    # Train
    args = TrainingArguments(**TRAINING_ARGS)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
        callbacks=[ProgressCallback()],
    )

    print("\n  Starting training...")
    t0 = time.time()
    trainer.train()
    print(f"\n  ✓ Training complete in {time.time() - t0:.1f}s")

    # Evaluate
    print("\n  Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results.get('accuracy', 'N/A')}")

    try:
        preds = model.predict(test_ds["text"])
        report = classification_report(test_ds["label"], preds, labels=LABELS, zero_division=0)
        print(f"\n  Per-class classification report:\n{report}")
    except Exception as exc:
        print(f"  (per-class report skipped: {exc})")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    print(f"  Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
