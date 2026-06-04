"""
train_ensemble.py — Train an ensemble of binary SetFit classifiers (Arm C).

Creates one binary model per threat class (prompt_injection, role_override,
out_of_scope, sensitive_bias).  Each model is trained as "this class vs.
everything else", with other attack classes down-sampled to prevent the
negative set from dominating.

Models are saved to models/guardrail_ensemble/<class_name>/.

Usage:
    python ml/train_ensemble.py
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
POSITIVE_CLASSES = ["prompt_injection", "role_override", "out_of_scope", "sensitive_bias"]
OUTPUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "models", "guardrail_ensemble"))

SEED = 42
OTHER_ATTACK_SAMPLE_CAP = 60  # max negative samples from non-target attack classes

TRAINING_ARGS = dict(
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

    def __init__(self, class_name: str):
        self.class_name = class_name
        self.step = 0
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        elapsed = time.time() - self.t0
        print(f"  [{self.class_name}] step {self.step:>4d}  |  elapsed {elapsed:6.1f}s", flush=True)


def load_train_rows() -> list[dict]:
    """Load train.jsonl and return a list of {text, label} dicts."""
    path = os.path.join(HERE, "data", "train.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Run build_dataset.py first.")

    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_binary_dataset(all_rows: list[dict], target_class: str) -> Dataset:
    """Build a binary dataset: target_class → 1, everything else → 0.

    Other attack classes are deterministically down-sampled to
    OTHER_ATTACK_SAMPLE_CAP to keep the dataset small and balanced.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Training Ensemble Binary SetFit Models (Arm C)")
    print("=" * 60)

    device = detect_device()
    all_rows = load_train_rows()
    print(f"  Total training rows: {len(all_rows)}")

    overall_start = time.time()

    for i, target in enumerate(POSITIVE_CLASSES, 1):
        print(f"\n  [{i}/{len(POSITIVE_CLASSES)}] Training binary classifier: {target}")
        print("  " + "-" * 40)

        ds = make_binary_dataset(all_rows, target)
        pos_count = sum(ds["label"])
        neg_count = len(ds) - pos_count
        print(f"  Samples: {len(ds)}  |  Positive: {pos_count}  |  Negative: {neg_count}")

        model = SetFitModel.from_pretrained(BASE_MODEL, labels=[0, 1])
        model.to(device)

        args = TrainingArguments(**TRAINING_ARGS)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            callbacks=[ProgressCallback(class_name=target)],
        )

        t0 = time.time()
        trainer.train()
        print(f"  ✓ {target} trained in {time.time() - t0:.1f}s")

        out_dir = os.path.join(OUTPUT_ROOT, target)
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        print(f"  Saved to: {out_dir}")

        # Free GPU memory before loading the next model
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - overall_start
    print(f"\n  ✓ All {len(POSITIVE_CLASSES)} ensemble models trained in {total:.1f}s")


if __name__ == "__main__":
    main()
