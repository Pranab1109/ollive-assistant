"""
train_all.py — Unified training pipeline for Single + Ensemble SetFit models.

Trains both model architectures sequentially on GPU and saves them to
*_new directories so the currently active models are not overwritten:
  - models/guardrail_single_new/
  - models/guardrail_ensemble_new/<class_name>/

After validation with evaluate_local.py, the new models can be promoted
to the active directories (guardrail_single, guardrail_ensemble).

Usage:
    python ml/train_all.py
"""

import os
import json
import time
import random

import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["benign", "prompt_injection", "role_override", "out_of_scope", "sensitive_bias"]
POSITIVE_CLASSES = ["prompt_injection", "role_override", "out_of_scope", "sensitive_bias"]

SEED = 42
OTHER_ATTACK_SAMPLE_CAP = 60  # negative down-sample cap for ensemble binary datasets

SINGLE_OUTPUT_DIR = os.path.abspath(os.path.join(HERE, "..", "models", "guardrail_single_new"))
ENSEMBLE_OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "models", "guardrail_ensemble_new"))

SINGLE_TRAINING_ARGS = dict(
    batch_size=64,
    num_epochs=1,
    num_iterations=10,
    body_learning_rate=2e-5,
    head_learning_rate=1e-2,
    logging_steps=1,
)

ENSEMBLE_TRAINING_ARGS = dict(
    batch_size=64,
    num_epochs=1,
    num_iterations=20,
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

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.step = 0
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        elapsed = time.time() - self.t0
        tag = f"[{self.prefix}] " if self.prefix else ""
        print(f"  {tag}step {self.step:>4d}  |  elapsed {elapsed:6.1f}s", flush=True)


def load_jsonl(filename: str) -> list[dict]:
    """Load a JSONL file from ml/data/ and return a list of dicts."""
    path = os.path.join(HERE, "data", filename)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def free_gpu(device: str) -> None:
    """Delete cached tensors to reclaim VRAM between training runs."""
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 1 — Single multiclass model
# ---------------------------------------------------------------------------
def train_single_model(device: str, train_rows: list[dict], test_rows: list[dict]) -> None:
    """Train a single 5-class SetFit model and save to SINGLE_OUTPUT_DIR."""
    print("\n" + "=" * 70)
    print("  STAGE 1: Training Single Multiclass Model")
    print("=" * 70)

    train_ds = Dataset.from_list(train_rows)
    test_ds = Dataset.from_list(test_rows)

    print(f"  Loading base model: {BASE_MODEL}")
    model = SetFitModel.from_pretrained(BASE_MODEL, labels=LABELS)
    model.to(device)

    args = TrainingArguments(**SINGLE_TRAINING_ARGS)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
        callbacks=[ProgressCallback(prefix="Single")],
    )

    t0 = time.time()
    trainer.train()
    print(f"\n  ✓ Single model training completed in {time.time() - t0:.1f}s")

    os.makedirs(SINGLE_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(SINGLE_OUTPUT_DIR)
    print(f"  Saved to: {SINGLE_OUTPUT_DIR}")

    del model, trainer
    free_gpu(device)


# ---------------------------------------------------------------------------
# Stage 2 — Ensemble of binary models
# ---------------------------------------------------------------------------
def make_binary_dataset(all_rows: list[dict], target_class: str) -> Dataset:
    """Build a binary dataset: target_class → 1, benign + sampled others → 0."""
    positives = [r for r in all_rows if r["label"] == target_class]
    benign = [r for r in all_rows if r["label"] == "benign"]
    others = [r for r in all_rows if r["label"] not in (target_class, "benign")]

    rng = random.Random(SEED)
    sampled_others = rng.sample(others, min(len(others), OTHER_ATTACK_SAMPLE_CAP))

    out = []
    for r in positives:
        out.append({"text": r["text"], "label": 1})
    for r in benign + sampled_others:
        out.append({"text": r["text"], "label": 0})

    return Dataset.from_list(out)


def train_ensemble_models(device: str, train_rows: list[dict]) -> None:
    """Train one binary SetFit model per positive class and save each."""
    print("\n" + "=" * 70)
    print("  STAGE 2: Training Ensemble Binary Models")
    print("=" * 70)

    overall_start = time.time()

    for i, target in enumerate(POSITIVE_CLASSES, 1):
        print(f"\n  [{i}/{len(POSITIVE_CLASSES)}] Training binary classifier: {target}")
        print("  " + "-" * 40)

        ds = make_binary_dataset(train_rows, target)
        print(f"  Samples: {len(ds)} (Pos: {sum(ds['label'])}, Neg: {len(ds) - sum(ds['label'])})")

        model = SetFitModel.from_pretrained(BASE_MODEL, labels=[0, 1])
        model.to(device)

        args = TrainingArguments(**ENSEMBLE_TRAINING_ARGS)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            callbacks=[ProgressCallback(prefix=target)],
        )

        t0 = time.time()
        trainer.train()
        print(f"  ✓ Trained {target} in {time.time() - t0:.1f}s")

        out_dir = os.path.join(ENSEMBLE_OUTPUT_ROOT, target)
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        print(f"  Saved to: {out_dir}")

        del model, trainer
        free_gpu(device)

    print(f"\n  ✓ All ensemble models completed in {time.time() - overall_start:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = detect_device()

    print("\n  Loading train/test datasets...")
    train_rows = load_jsonl("train.jsonl")
    test_rows = load_jsonl("test.jsonl")
    print(f"  Loaded: {len(train_rows)} train rows | {len(test_rows)} test rows")

    train_single_model(device, train_rows, test_rows)
    train_ensemble_models(device, train_rows)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — new models are ready for validation.")
    print("=" * 70)
    print(f"  Single model:   {SINGLE_OUTPUT_DIR}")
    print(f"  Ensemble models: {ENSEMBLE_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
