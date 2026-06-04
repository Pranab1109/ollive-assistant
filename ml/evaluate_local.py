"""
evaluate_local.py — Benchmark Regex vs. Single SetFit vs. Ensemble SetFit guardrails.

Evaluates all three guardrail engines on the held-out test set and the
Mindgard OOD evasion set.  Uses GPU-batched inference to process 11,500+
samples in under 3 minutes instead of hours.

Outputs:
  - Console summary tables (accuracy, precision, recall, F1, latency)
  - Seaborn PNG charts saved to ml/plots/
  - Predictions spreadsheet (Excel or CSV) saved to ml/plots/

Usage:
    python ml/evaluate_local.py                           # evaluate active models
    $env:EVAL_NEW_MODELS="true"; python ml/evaluate_local.py  # evaluate _new models
"""

import os
import sys
import json
import time
import asyncio

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Mock asyncio.sleep to bypass the 50 ms delay baked into GuardrailsService
# ---------------------------------------------------------------------------
_original_sleep = asyncio.sleep

async def _noop_sleep(_delay):
    pass

asyncio.sleep = _noop_sleep

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.app.services.guardrail_engines import (
    RegexEngine,
    SingleSetFitEngine,
    EnsembleSetFitEngine,
    _emergency_or_pii,
    CRITICAL,
    HIGH,
    ROUTE,
)
from backend.app.services.guardrails import decode_text

PLOTS_DIR = os.path.join(HERE, "plots")
BATCH_SIZE = 256               # GPU inference batch size
ATTACK_CLASSES = {"prompt_injection", "role_override", "out_of_scope"}
ENGINE_PALETTE = {
    "regex": "#e76f51",
    "setfit_single": "#2a9d8f",
    "setfit_ensemble": "#e9c46a",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Per-engine evaluation functions
# ═══════════════════════════════════════════════════════════════════════════
async def evaluate_regex(engine, test_data: list[dict]) -> tuple[dict, list[dict]]:
    """Evaluate the Regex engine one sample at a time."""
    print("  Evaluating Regex Engine...")
    y_true, y_pred, latencies, results = [], [], [], []

    for row in test_data:
        text = row["text"]
        gold_block = 1 if row["label"] in ATTACK_CLASSES else 0

        t0 = time.time()
        is_safe, reason = await engine.verify_input(text)
        latency_ms = (time.time() - t0) * 1000

        pred_block = 0 if is_safe else 1
        y_true.append(gold_block)
        y_pred.append(pred_block)
        latencies.append(latency_ms)
        results.append({
            "text": text,
            "gold_label": row["label"],
            "gold_block": gold_block,
            "pred_block": pred_block,
            "reason": reason,
            "latency_ms": latency_ms,
            "split": row.get("split", "test"),
        })

    return _calculate_metrics(y_true, y_pred, latencies), results


async def evaluate_single_setfit(engine, test_data: list[dict]) -> tuple[dict, list[dict]]:
    """Evaluate the Single SetFit engine using batched GPU inference."""
    print("  Evaluating Single SetFit Engine (batched)...")
    if not engine.active:
        print("    Warning: model not loaded — falling back to Regex.")
        return await evaluate_regex(engine.regex_fallback, test_data)

    y_true, latencies = [], []
    placeholders: list[dict] = []
    texts_for_gpu: list[str] = []
    gpu_indices: list[int] = []

    # Pass 1: run deterministic regex checks (emergency / PII)
    for idx, row in enumerate(test_data):
        text = row["text"]
        gold_block = 1 if row["label"] in ATTACK_CLASSES else 0
        y_true.append(gold_block)

        t0 = time.time()
        pre = _emergency_or_pii(text)

        if pre is not None:
            is_safe, reason = pre
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            placeholders.append({
                "text": text, "gold_label": row["label"], "gold_block": gold_block,
                "pred_block": 0 if is_safe else 1, "reason": reason,
                "latency_ms": latency_ms, "split": row.get("split", "test"),
            })
        else:
            decoded = decode_text(text)
            texts_for_gpu.append(text if not decoded else f"{text} {decoded}")
            gpu_indices.append(idx)
            latencies.append(0.0)
            placeholders.append({
                "text": text, "gold_label": row["label"], "gold_block": gold_block,
                "split": row.get("split", "test"),
            })

    # Pass 2: batched GPU inference
    if texts_for_gpu:
        t0 = time.time()
        print(f"    Running batch prediction for {len(texts_for_gpu)} samples...")
        probs_all = engine.model.predict_proba(texts_for_gpu, batch_size=BATCH_SIZE)
        per_sample_ms = ((time.time() - t0) * 1000) / len(texts_for_gpu)
        print(f"    Batch done in {time.time() - t0:.2f}s ({per_sample_ms:.2f} ms/sample)")

        labels = engine.model.labels
        for i, idx in enumerate(gpu_indices):
            scored = dict(zip(labels, [float(p) for p in probs_all[i]]))
            top = max(scored, key=scored.get)

            if top in (CRITICAL | HIGH) and scored[top] >= engine.threshold:
                is_safe, reason = False, f"Flagged Input: {top} (p={scored[top]:.2f})."
            elif top in ROUTE and scored[top] >= engine.threshold:
                is_safe, reason = True, f"sensitive_topic:{top}"
            else:
                is_safe, reason = True, "Passed: Input is safe."

            placeholders[idx].update(pred_block=0 if is_safe else 1,
                                     reason=reason, latency_ms=per_sample_ms)
            latencies[idx] = per_sample_ms

    y_pred = [r["pred_block"] for r in placeholders]
    return _calculate_metrics(y_true, y_pred, latencies), placeholders


async def evaluate_ensemble_setfit(engine, test_data: list[dict]) -> tuple[dict, list[dict]]:
    """Evaluate the Ensemble SetFit engine using batched GPU inference."""
    print("  Evaluating Ensemble SetFit Engine (batched)...")
    if not engine.active:
        print("    Warning: ensemble not loaded — falling back to Regex.")
        return await evaluate_regex(engine.regex_fallback, test_data)

    y_true, latencies = [], []
    placeholders: list[dict] = []
    texts_for_gpu: list[str] = []
    gpu_indices: list[int] = []

    # Pass 1: deterministic regex checks
    for idx, row in enumerate(test_data):
        text = row["text"]
        gold_block = 1 if row["label"] in ATTACK_CLASSES else 0
        y_true.append(gold_block)

        t0 = time.time()
        pre = _emergency_or_pii(text)

        if pre is not None:
            is_safe, reason = pre
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            placeholders.append({
                "text": text, "gold_label": row["label"], "gold_block": gold_block,
                "pred_block": 0 if is_safe else 1, "reason": reason,
                "latency_ms": latency_ms, "split": row.get("split", "test"),
            })
        else:
            decoded = decode_text(text)
            texts_for_gpu.append(text if not decoded else f"{text} {decoded}")
            gpu_indices.append(idx)
            latencies.append(0.0)
            placeholders.append({
                "text": text, "gold_label": row["label"], "gold_block": gold_block,
                "split": row.get("split", "test"),
            })

    # Pass 2: batched GPU inference across all 4 binary models
    if texts_for_gpu:
        t0 = time.time()
        print(f"    Running ensemble batch predictions for {len(texts_for_gpu)} samples...")

        probs_by_model: dict[str, list[float]] = {}
        for cls, model in engine.models.items():
            print(f"      Model: {cls}...")
            probs = model.predict_proba(texts_for_gpu, batch_size=BATCH_SIZE)
            probs_by_model[cls] = [float(p[1]) for p in probs]

        per_sample_ms = ((time.time() - t0) * 1000) / len(texts_for_gpu)
        print(f"    Ensemble batch done in {time.time() - t0:.2f}s ({per_sample_ms:.2f} ms/sample)")

        for i, idx in enumerate(gpu_indices):
            fired = {cls: probs_by_model[cls][i]
                     for cls in engine.THRESHOLDS
                     if probs_by_model[cls][i] >= engine.THRESHOLDS[cls]}

            if not fired:
                is_safe, reason = True, "Passed: Input is safe."
            else:
                top = max(fired, key=fired.get)
                if top in (CRITICAL | HIGH):
                    is_safe, reason = False, f"Flagged Input: {top} (p={fired[top]:.2f})."
                else:
                    is_safe, reason = True, f"sensitive_topic:{top}"

            placeholders[idx].update(pred_block=0 if is_safe else 1,
                                     reason=reason, latency_ms=per_sample_ms)
            latencies[idx] = per_sample_ms

    y_pred = [r["pred_block"] for r in placeholders]
    return _calculate_metrics(y_true, y_pred, latencies), placeholders


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def _calculate_metrics(y_true, y_pred, latencies) -> dict:
    """Compute accuracy, precision, recall, F1, confusion counts, and latency."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "avg_latency_ms": np.mean(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
    }


def _has_ood(all_results: dict[str, list[dict]]) -> bool:
    """Return True if any engine produced results with split == 'ood'."""
    return any(
        any(r["split"] == "ood" for r in results)
        for results in all_results.values()
    )


def _get_categories(has_ood: bool) -> list[str]:
    """Return the list of display categories, including OOD if present."""
    cats = ["benign", "prompt_injection (ID)", "role_override", "out_of_scope", "sensitive_bias"]
    if has_ood:
        cats.insert(2, "prompt_injection (OOD)")
    return cats


def _filter_by_category(results: list[dict], category: str) -> list[dict]:
    """Return the subset of results matching a display category."""
    if category == "prompt_injection (ID)":
        return [r for r in results if r["gold_label"] == "prompt_injection" and r["split"] == "test"]
    if category == "prompt_injection (OOD)":
        return [r for r in results if r["gold_label"] == "prompt_injection" and r["split"] == "ood"]
    return [r for r in results if r["gold_label"] == category]


# ═══════════════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════════════
def _annotate_bars(ax, fmt="{:.3f}", fontsize=9):
    """Add value labels above each bar in a Seaborn barplot."""
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.annotate(
                fmt.format(h),
                (patch.get_x() + patch.get_width() / 2.0, h),
                ha="center", va="bottom", xytext=(0, 3),
                textcoords="offset points", fontsize=fontsize, fontweight="semibold",
            )


def plot_comparison_metrics(all_metrics: dict) -> None:
    """Bar chart: Accuracy / Precision / Recall / F1 per engine."""
    sns.set_theme(style="whitegrid")
    rows = []
    for eng, m in all_metrics.items():
        for metric in ("accuracy", "precision", "recall", "f1"):
            rows.append({"Engine": eng, "Metric": metric.capitalize(), "Value": m[metric]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Value", hue="Engine", data=df,
                     palette=ENGINE_PALETTE, edgecolor="black", linewidth=1)
    plt.title("Guardrail Classification Performance", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(title="Engine", loc="lower left", frameon=True)
    _annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison.png"), dpi=200)
    plt.close()
    print("  Saved: metrics_comparison.png")


def plot_latency_comparison(all_metrics: dict) -> None:
    """Bar chart: average and p95 latency per engine."""
    sns.set_theme(style="whitegrid")
    rows = []
    for eng, m in all_metrics.items():
        rows.append({"Engine": eng, "Type": "Average", "Latency (ms)": m["avg_latency_ms"]})
        rows.append({"Engine": eng, "Type": "p95", "Latency (ms)": m["p95_latency_ms"]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Type", y="Latency (ms)", hue="Engine", data=df,
                     palette=ENGINE_PALETTE, edgecolor="black", linewidth=1)
    plt.title("Guardrail Latency Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.legend(title="Engine", loc="upper left", frameon=True)
    _annotate_bars(ax, fmt="{:.2f}ms")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"), dpi=200)
    plt.close()
    print("  Saved: latency_comparison.png")


def plot_confusion_matrices(all_metrics: dict) -> None:
    """Side-by-side heatmaps of each engine's confusion matrix."""
    sns.set_theme(style="white")
    engines = list(all_metrics.keys())
    cmaps = {"regex": "Oranges", "setfit_single": "Blues", "setfit_ensemble": "Purples"}
    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 5))
    fig.suptitle("Confusion Matrices (Allowed vs. Blocked)", fontsize=16, fontweight="bold", y=1.05)

    for idx, eng in enumerate(engines):
        m = all_metrics[eng]
        cm = [[m["true_negatives"], m["false_positives"]],
              [m["false_negatives"], m["true_positives"]]]
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps.get(eng, "Blues"),
                    cbar=False, ax=ax,
                    xticklabels=["Allowed", "Blocked"],
                    yticklabels=["Allowed", "Blocked"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(eng.upper(), fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrices.png")


def plot_category_block_rates(all_results: dict, engine_names: list[str]) -> None:
    """Bar chart: block rate (%) per category per engine."""
    categories = _get_categories(_has_ood(all_results))
    rows = []
    for eng in engine_names:
        results = all_results[eng]
        for cat in categories:
            subset = _filter_by_category(results, cat)
            blocked = sum(1 for r in subset if r["pred_block"] == 1)
            rate = (blocked / len(subset) * 100) if subset else 0.0
            rows.append({"Engine": eng, "Category": cat, "Block Rate (%)": rate})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Category", y="Block Rate (%)", hue="Engine", data=df,
                     palette=ENGINE_PALETTE, edgecolor="black", linewidth=1)
    plt.title("Block Rate by Category\n(Target: 100 % for attacks, 0 % for benign/bias)",
              fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0, 110)
    plt.legend(title="Engine", loc="upper right", frameon=True)
    _annotate_bars(ax, fmt="{:.1f}%", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "category_block_rates.png"), dpi=200)
    plt.close()
    print("  Saved: category_block_rates.png")


def plot_regex_bypass_performance(bypass_metrics: list[dict]) -> None:
    """Bar chart: performance on the subset of samples where Regex failed."""
    df = pd.DataFrame(bypass_metrics)
    df_melted = df.melt(id_vars="Engine", value_vars=["Precision", "Recall", "F1"],
                        var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Metric", y="Score", hue="Engine", data=df_melted,
                     palette=ENGINE_PALETTE, edgecolor="black", linewidth=1)
    plt.title("Performance on Regex-Failure Subset\n(Neural Generalization Test)",
              fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0, 1.05)
    plt.legend(title="Engine", loc="lower left", frameon=True)
    _annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "regex_bypass_performance.png"), dpi=200)
    plt.close()
    print("  Saved: regex_bypass_performance.png")


# ═══════════════════════════════════════════════════════════════════════════
# Console reports
# ═══════════════════════════════════════════════════════════════════════════
def print_summary_table(all_metrics: dict) -> None:
    """Print the top-level comparison table."""
    print("\n" + "=" * 80)
    print("                     GUARDRAIL ENGINE COMPARISON SUMMARY")
    print("=" * 80)
    hdr = f"{'Engine':<18} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FP':<4} | {'FN':<4} | {'Latency':>10}"
    print(hdr)
    print("-" * 80)
    for eng, m in all_metrics.items():
        print(f"{eng:<18} | {m['accuracy']:<6.3f} | {m['precision']:<6.3f} | "
              f"{m['recall']:<6.3f} | {m['f1']:<6.3f} | {m['false_positives']:<4d} | "
              f"{m['false_negatives']:<4d} | {m['avg_latency_ms']:>7.2f} ms")
    print("=" * 80)


def print_category_block_rates(all_results: dict, engine_names: list[str]) -> None:
    """Print per-category block rates."""
    categories = _get_categories(_has_ood(all_results))

    print("\n" + "=" * 80)
    print("                     CATEGORY-WISE BLOCK RATES (%)")
    print("=" * 80)
    header = f"{'Engine':<18} | " + " | ".join(f"{c:<22}" for c in categories)
    print(header)
    print("-" * len(header))

    for eng in engine_names:
        results = all_results[eng]
        cols = []
        for cat in categories:
            subset = _filter_by_category(results, cat)
            blocked = sum(1 for r in subset if r["pred_block"] == 1)
            rate = (blocked / len(subset) * 100) if subset else 0.0
            cols.append(f"{rate:>5.1f}% ({blocked}/{len(subset)})")
        print(f"{eng:<18} | " + " | ".join(f"{c:<22}" for c in cols))
    print("=" * 80)


def print_split_breakdown(all_results: dict) -> None:
    """Print ID vs. OOD metrics breakdown and the Regex-bypass generalization test."""
    print("\n" + "=" * 80)
    print("              BREAKDOWN: IN-DISTRIBUTION vs. OUT-OF-DISTRIBUTION")
    print("=" * 80)

    engines = list(all_results.keys())

    for split in ("test", "ood"):
        label = "In-Distribution (test.jsonl)" if split == "test" else "OOD (ood_evasion.jsonl)"
        print(f"\n  >>> {label}")
        print(f"  {'Engine':<18} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FP':<4} | {'FN':<4}")
        print("  " + "-" * 70)

        for eng in engines:
            subset = [r for r in all_results[eng] if r["split"] == split]
            if not subset:
                continue
            y_true = [r["gold_block"] for r in subset]
            y_pred = [r["pred_block"] for r in subset]
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            print(f"  {eng:<18} | {acc:<6.3f} | {p:<6.3f} | {r:<6.3f} | {f1:<6.3f} | {fp:<4d} | {fn:<4d}")
        print("  " + "=" * 70)

    # Regex-bypass generalization test
    print("\n" + "=" * 80)
    print("      GENERALIZATION TEST: PERFORMANCE ON REGEX-FAILURE SUBSET")
    print("=" * 80)
    print("  Isolates queries where Regex failed (FPs + FNs) to test neural generalization.\n")

    regex_res = all_results["regex"]
    failed_idx = [i for i, r in enumerate(regex_res)
                  if (r["gold_block"] != r["pred_block"])]

    if not failed_idx:
        print("  No Regex failures found — generalization test skipped.")
        return []

    print(f"  {len(failed_idx)} samples where Regex failed.")
    print(f"  {'Engine':<18} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FP':<4} | {'FN':<4}")
    print("  " + "-" * 70)

    bypass_metrics = []
    for eng in engines:
        res = all_results[eng]
        yt = [res[i]["gold_block"] for i in failed_idx]
        yp = [res[i]["pred_block"] for i in failed_idx]
        acc = accuracy_score(yt, yp)
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        print(f"  {eng:<18} | {acc:<6.3f} | {p:<6.3f} | {r:<6.3f} | {f1:<6.3f} | {fp:<4d} | {fn:<4d}")
        bypass_metrics.append({"Engine": eng, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1})
    print("  " + "=" * 70)

    return bypass_metrics


def print_failure_samples(all_results: dict, engine_names: list[str], max_show: int = 4) -> None:
    """Print a few example false positives and false negatives per engine."""
    print("\n" + "=" * 80)
    print("                      SAMPLE FAILURE ANALYSIS")
    print("=" * 80)

    for eng in engine_names:
        results = all_results[eng]
        fps = [r for r in results if r["gold_block"] == 0 and r["pred_block"] == 1]
        fns = [r for r in results if r["gold_block"] == 1 and r["pred_block"] == 0]

        print(f"\n  >>> {eng.upper()}")
        print(f"    False Positives (safe → blocked) [{len(fps)} total]:")
        for i, r in enumerate(fps[:max_show]):
            print(f"      {i+1}. \"{r['text'][:120]}...\"")
            print(f"         Gold: {r['gold_label']} | Reason: {r['reason']}")

        print(f"    False Negatives (attack → missed) [{len(fns)} total]:")
        for i, r in enumerate(fns[:max_show]):
            print(f"      {i+1}. \"{r['text'][:120]}...\"")
            print(f"         Gold: {r['gold_label']} | Reason: {r['reason']}")
        print("  " + "-" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════
def export_predictions(test_data: list[dict], all_results: dict) -> None:
    """Export a spreadsheet with prompt, actual label, and all three predictions."""
    rows = []
    for idx, row in enumerate(test_data):
        rows.append({
            "Prompt": row.get("text", ""),
            "Actual Class": row.get("label", ""),
            "Regex Predicted": all_results["regex"][idx]["reason"] if idx < len(all_results["regex"]) else "N/A",
            "Single SetFit Predicted": all_results["setfit_single"][idx]["reason"] if idx < len(all_results["setfit_single"]) else "N/A",
            "Ensemble SetFit Predicted": all_results["setfit_ensemble"][idx]["reason"] if idx < len(all_results["setfit_ensemble"]) else "N/A",
        })
    df = pd.DataFrame(rows)

    excel_path = os.path.join(PLOTS_DIR, "guardrail_predictions.xlsx")
    csv_path = os.path.join(PLOTS_DIR, "guardrail_predictions.csv")

    try:
        df.to_excel(excel_path, index=False)
        print(f"  Exported: {excel_path}")
    except ImportError:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  openpyxl not installed — exported CSV instead: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Load data ---
    test_path = os.path.join(HERE, "data", "test.jsonl")
    ood_path = os.path.join(HERE, "data", "ood_evasion.jsonl")

    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Run build_dataset.py first.")
        return

    test_data = load_jsonl(test_path)
    print(f"Loaded test set: {len(test_data)} samples")

    if os.path.exists(ood_path):
        ood_data = load_jsonl(ood_path)
        print(f"Loaded OOD evasion set: {len(ood_data)} samples")
        for r in ood_data:
            r["split"] = "ood"
        test_data += ood_data
        print(f"Combined evaluation set: {len(test_data)} samples")

    # --- Initialise engines ---
    use_new = os.environ.get("EVAL_NEW_MODELS") == "true"
    single_path = "models/guardrail_single_new" if use_new else "models/guardrail_single"
    ensemble_path = "models/guardrail_ensemble_new" if use_new else "models/guardrail_ensemble"
    tag = "newly trained" if use_new else "active"
    print(f"\nEvaluating {tag} models...\n")

    engines = {
        "regex": RegexEngine(),
        "setfit_single": SingleSetFitEngine(model_dir=single_path),
        "setfit_ensemble": EnsembleSetFitEngine(root=ensemble_path),
    }

    # --- Run evaluations ---
    all_metrics, all_results = {}, {}
    t_start = time.time()

    for name, eval_fn in [("regex", evaluate_regex),
                          ("setfit_single", evaluate_single_setfit),
                          ("setfit_ensemble", evaluate_ensemble_setfit)]:
        metrics, results = await eval_fn(engines[name], test_data)
        all_metrics[name] = metrics
        all_results[name] = results

    print(f"\nAll evaluations completed in {time.time() - t_start:.2f}s\n")

    # --- Reports ---
    engine_names = list(engines.keys())
    print_summary_table(all_metrics)

    print("\nGenerating charts...")
    plot_comparison_metrics(all_metrics)
    plot_latency_comparison(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_category_block_rates(all_results, engine_names)

    bypass_metrics = print_split_breakdown(all_results)
    if bypass_metrics:
        plot_regex_bypass_performance(bypass_metrics)

    print_category_block_rates(all_results, engine_names)
    print_failure_samples(all_results, engine_names)

    print("\nExporting predictions spreadsheet...")
    export_predictions(test_data, all_results)

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
