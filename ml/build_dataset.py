"""
build_dataset.py — Normalize, balance, deduplicate, and split guardrail datasets.

Reads raw parquet files from ml/datasets_raw/ and hand-authored seeds from
ml/data/seeds_sensitive_bias.jsonl, then produces three JSONL files:
  - ml/data/train.jsonl         (80% stratified split — used for training)
  - ml/data/test.jsonl          (20% stratified split — in-distribution eval)
  - ml/data/ood_evasion.jsonl   (Mindgard evaded samples — out-of-distribution eval)

Usage:
    python ml/build_dataset.py
"""

import os
import glob
import json
import hashlib
import random
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "datasets_raw")
OUT_DIR = os.path.join(HERE, "data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
PER_CLASS_CAP = 1000          # max samples per class to prevent dominance
EVADED_INJECT_COUNT = 1000    # how many Mindgard samples to inject into training pool
TEST_SPLIT_RATIO = 0.2        # fraction reserved for the ID test set

# Templates for synthesised out-of-scope examples (no clean public dataset exists)
OOS_TEMPLATES = [
    "write a python script to {x}",
    "how do I hack {x}",
    "generate malware for {x}",
    "write javascript that {x}",
    "give me a reverse shell for {x}",
    "how to ddos {x}",
]
OOS_FILLERS = ["the server", "the hospital network", "a website", "this app", "a database"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Collapse whitespace; mirrors the production guardrails.normalize_text."""
    return " ".join(str(text).strip().split())


def _find_parquet(pattern: str) -> str | None:
    """Return the first parquet file matching *pattern* inside RAW_DIR, or None."""
    matches = glob.glob(os.path.join(RAW_DIR, pattern))
    return matches[0] if matches else None


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, data: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in data:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_hash(text: str) -> str:
    """Deterministic hash for deduplication (case-insensitive)."""
    return hashlib.md5(text.lower().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Source loaders — each returns a list of {text, label} dicts
# ---------------------------------------------------------------------------
def load_evaded_dataset() -> list[dict]:
    """Mindgard evaded prompt-injection samples (column: modified_sample)."""
    path = _find_parquet("evaded__*")
    if not path:
        print("  ⚠  Evaded parquet not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = [{"text": normalize_text(t), "label": "prompt_injection"} for t in df["modified_sample"]]
    print(f"  ✓ evaded (Mindgard):            {len(rows):>6} rows")
    return rows


def load_jailbreak_classification() -> list[dict]:
    """jackhhao/jailbreak-classification — maps 'jailbreak' → role_override."""
    path = _find_parquet("jailbreak_classification__*train*")
    if not path:
        print("  ⚠  jailbreak_classification not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        label = "role_override" if str(r["type"]).lower() == "jailbreak" else "benign"
        rows.append({"text": normalize_text(r["prompt"]), "label": label})
    print(f"  ✓ jailbreak_classification:     {len(rows):>6} rows")
    return rows


def load_safe_guard() -> list[dict]:
    """xTRam1/safe-guard-prompt-injection — binary label (1 = injection)."""
    path = _find_parquet("safe_guard__*train*")
    if not path:
        print("  ⚠  safe_guard not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        label = "prompt_injection" if int(r["label"]) == 1 else "benign"
        rows.append({"text": normalize_text(r["text"]), "label": label})
    print(f"  ✓ safe_guard:                   {len(rows):>6} rows")
    return rows


def load_qualifire() -> list[dict]:
    """qualifire/prompt-injections-benchmark — label contains 'jail' for attacks."""
    path = _find_parquet("qualifire_pi__*")
    if not path:
        print("  ⚠  qualifire_pi not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        label = "prompt_injection" if "jail" in str(r["label"]).lower() else "benign"
        rows.append({"text": normalize_text(r["text"]), "label": label})
    print(f"  ✓ qualifire_pi:                 {len(rows):>6} rows")
    return rows


def load_toxic_chat() -> list[dict]:
    """lmsys/toxic-chat — keep only non-toxic, non-jailbreak chats as benign hard-negatives."""
    path = _find_parquet("toxic_chat__*train*")
    if not path:
        print("  ⚠  toxic_chat not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        if int(r.get("toxicity", 0)) == 0 and int(r.get("jailbreaking", 0)) == 0:
            rows.append({"text": normalize_text(r["user_input"]), "label": "benign"})
    print(f"  ✓ toxic_chat (benign only):     {len(rows):>6} rows")
    return rows


def load_wildjailbreak() -> list[dict]:
    """allenai/wildjailbreak — adversarial harmful → attack; adversarial benign → hard neg."""
    path = _find_parquet("wildjailbreak__*")
    if not path:
        print("  ⚠  wildjailbreak not found — skipping.")
        return []
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        dt = str(r.get("data_type", ""))
        if "adversarial_harmful" in dt:
            rows.append({"text": normalize_text(r["adversarial"]), "label": "prompt_injection"})
        elif "adversarial_benign" in dt:
            rows.append({"text": normalize_text(r["adversarial"]), "label": "benign"})
    print(f"  ✓ wildjailbreak:                {len(rows):>6} rows")
    return rows


def synthesize_out_of_scope() -> list[dict]:
    """Generate templated out-of-scope examples (hacking, coding, etc.)."""
    rows = []
    for tmpl in OOS_TEMPLATES:
        for filler in OOS_FILLERS:
            rows.append({"text": tmpl.format(x=filler), "label": "out_of_scope"})
    print(f"  ✓ out_of_scope (synthesised):   {len(rows):>6} rows")
    return rows


def load_seeds() -> list[dict]:
    """Hand-authored domain seeds for sensitive_bias and hospital-specific benign queries."""
    seeds_path = os.path.join(OUT_DIR, "seeds_sensitive_bias.jsonl")
    if not os.path.exists(seeds_path):
        print(f"  ⚠  Seeds file not found at {seeds_path} — skipping.")
        return []
    rows = load_jsonl(seeds_path)
    print(f"  ✓ hand-authored seeds:          {len(rows):>6} rows")
    return rows


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def dedupe_and_balance(rows: list[dict]) -> tuple[list[dict], set[str]]:
    """Shuffle, deduplicate by MD5 hash, and cap each class at PER_CLASS_CAP."""
    rng = random.Random(SEED)
    rng.shuffle(rows)

    seen: set[str] = set()
    class_counts: dict[str, int] = {}
    final: list[dict] = []
    used_hashes: set[str] = set()

    for row in rows:
        h = text_hash(row["text"])
        if h in seen or len(row["text"]) < 3:
            continue
        seen.add(h)

        label = row["label"]
        count = class_counts.get(label, 0)
        if count >= PER_CLASS_CAP:
            continue

        class_counts[label] = count + 1
        final.append(row)
        used_hashes.add(h)

    print(f"\n  Class distribution after balancing:")
    for label, count in sorted(class_counts.items()):
        print(f"    {label:<20s}: {count}")

    return final, used_hashes


def stratified_split(data: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split data into train/test with per-class stratification."""
    buckets: dict[str, list[dict]] = {}
    for row in data:
        buckets.setdefault(row["label"], []).append(row)

    train, test = [], []
    for _label, items in buckets.items():
        k = int(len(items) * TEST_SPLIT_RATIO)
        test.extend(items[:k])
        train.extend(items[k:])

    return train, test


def build_ood_evasion_set(evaded_rows: list[dict], used_hashes: set[str]) -> list[dict]:
    """Filter evaded rows to exclude any that appeared in train/test (leakage prevention)."""
    seen_ood: set[str] = set()
    ood: list[dict] = []
    for row in evaded_rows:
        h = text_hash(row["text"])
        if h in used_hashes or h in seen_ood:
            continue
        seen_ood.add(h)
        ood.append(row)
    return ood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    print("=" * 60)
    print("  Loading source datasets")
    print("=" * 60)

    # Load Mindgard evaded dataset (partially injected into training)
    evaded_rows = load_evaded_dataset()

    # Collect all rows from every source
    rows: list[dict] = []

    if evaded_rows:
        rng = random.Random(SEED)
        shuffled = list(evaded_rows)
        rng.shuffle(shuffled)
        rows.extend(shuffled[:EVADED_INJECT_COUNT])
        print(f"  → Injected {EVADED_INJECT_COUNT} evaded samples into training pool")

    rows.extend(load_jailbreak_classification())
    rows.extend(load_safe_guard())
    rows.extend(load_qualifire())
    rows.extend(load_toxic_chat())
    rows.extend(load_wildjailbreak())
    rows.extend(synthesize_out_of_scope())
    rows.extend(load_seeds())

    if not rows:
        print("\n✗ No data loaded — cannot build dataset.")
        return

    print(f"\n  Total raw rows collected: {len(rows)}")

    # Deduplicate and balance
    print("\n" + "=" * 60)
    print("  Deduplicating and balancing")
    print("=" * 60)
    final, used_hashes = dedupe_and_balance(rows)

    # Train / test split
    print("\n" + "=" * 60)
    print("  Stratified train/test split")
    print("=" * 60)
    train, test = stratified_split(final)
    dump_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    dump_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test)
    print(f"  train = {len(train):>5}  |  test = {len(test):>4}")

    # OOD evasion set (leakage-free)
    if evaded_rows:
        print("\n" + "=" * 60)
        print("  Building OOD evasion set (leakage-free)")
        print("=" * 60)
        ood = build_ood_evasion_set(evaded_rows, used_hashes)
        dump_jsonl(os.path.join(OUT_DIR, "ood_evasion.jsonl"), ood)
        print(f"  ood_evasion = {len(ood)}")

    print("\n✓ Dataset build complete.")


if __name__ == "__main__":
    main()
