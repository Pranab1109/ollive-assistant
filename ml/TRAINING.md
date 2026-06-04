# Guardrail ML Training Documentation

This document describes how the **Single Multiclass** and **Ensemble Binary** SetFit models are trained, evaluated, and compared against the baseline Regex engine.

---

## Table of Contents

1. [Quick Start: Regenerating Results](#1-quick-start-regenerating-results)
2. [Architecture Overview](#2-architecture-overview)
3. [Dataset Pipeline](#3-dataset-pipeline)
4. [Model A — Single Multiclass SetFit](#4-model-a--single-multiclass-setfit)
5. [Model B — Ensemble Binary SetFit](#5-model-b--ensemble-binary-setfit)
6. [Unified Training Pipeline](#6-unified-training-pipeline)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [File Reference](#8-file-reference)

---

## 1. Quick Start: Regenerating Results

To train the models from scratch, generate evaluation metrics, and reproduce comparison plots/spreadsheets, execute the scripts in the following order:

### Step 1: Download Raw Datasets
If the raw dataset Parquet files are missing from `ml/datasets_raw/`, fetch them from Hugging Face:
```bash
python ml/download_datasets.py
```
> [!NOTE]
> Gated datasets may require a Hugging Face Hub token. Set it in a `.env` file at the project root as `HF_TOKEN=your_token_here`.

### Step 2: Prepare Datasets
Build, deduplicate, balance, and split the data into training, in-distribution test, and out-of-distribution (OOD) test sets:
```bash
python ml/build_dataset.py
```
This produces three JSONL files under `ml/data/`: `train.jsonl`, `test.jsonl`, and `ood_evasion.jsonl`.

### Step 3: Train Models
Run the unified training script to train both the **Single Multiclass** model and the **Ensemble Binary** models sequentially on GPU/CPU:
```bash
python ml/train_all.py
```
> [!TIP]
> To avoid overwriting currently active models, this script saves the models under `models/guardrail_single_new/` and `models/guardrail_ensemble_new/`. If you want to train only one model type, you can run `python ml/train_single.py` or `python ml/train_ensemble.py` instead.

### Step 4: Evaluate & Generate Plots
Benchmark the newly trained models against the active models and the baseline regex engine:
- **Windows (PowerShell)**:
  ```powershell
  $env:EVAL_NEW_MODELS="true"; python ml/evaluate_local.py
  ```
- **Linux / macOS / Git Bash**:
  ```bash
  EVAL_NEW_MODELS=true python ml/evaluate_local.py
  ```
- **Windows (CMD)**:
  ```cmd
  set EVAL_NEW_MODELS=true && python ml/evaluate_local.py
  ```
This computes all comparison metrics, generates plots in `ml/plots/`, and exports a detailed spreadsheet to `ml/plots/guardrail_predictions.xlsx` (or CSV).

### Step 5: Promote Models to Active
Once satisfied with the evaluation results, promote the new models to active status by replacing the active directory names:
- **Windows (PowerShell)**:
  ```powershell
  Remove-Item -Recurse -Force models/guardrail_single
  Rename-Item models/guardrail_single_new guardrail_single
  
  Remove-Item -Recurse -Force models/guardrail_ensemble
  Rename-Item models/guardrail_ensemble_new guardrail_ensemble
  ```

---

## 2. Architecture Overview

The guardrail system classifies user prompts into **five classes**:

| Class              | Disposition | Description                                    |
| :----------------- | :---------: | :--------------------------------------------- |
| `benign`           |   Allow     | Safe, on-topic queries                         |
| `prompt_injection` |   Block     | Attempts to override the system prompt         |
| `role_override`    |   Block     | Jailbreak-style persona hijacking              |
| `out_of_scope`     |   Block     | Off-topic requests (hacking, coding, malware)  |
| `sensitive_bias`   |   Route     | Bias/sensitive topics → routed, not blocked    |

Three engines are compared:

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────────┐
│   Regex Engine   │     │  Single SetFit (5-cls)│     │ Ensemble SetFit (4×bin) │
│   (baseline)     │     │  (Arm B)              │     │ (Arm C)                 │
└─────────────────┘     └─────────────────────┘     └─────────────────────────┘
```

All three engines share a common **pre-filter** (`_emergency_or_pii`) that catches PII and emergency/crisis language via deterministic regex before the neural models run.

---

## 3. Dataset Pipeline

**Script:** [`build_dataset.py`](./build_dataset.py)

### 3.1 Raw Sources

Seven data sources are normalised into `{text, label}` format:

| # | Source                              | Label mapping                                                    | Lines in code                |
|---|-------------------------------------|------------------------------------------------------------------|------------------------------|
| 0 | Mindgard evaded prompts             | All → `prompt_injection`                                         | `load_evaded_dataset()`      |
| 1 | `jackhhao/jailbreak-classification` | `jailbreak` → `role_override`, else → `benign`                  | `load_jailbreak_classification()` |
| 2 | `xTRam1/safe-guard-prompt-injection`| `label=1` → `prompt_injection`, `label=0` → `benign`            | `load_safe_guard()`          |
| 3 | `qualifire/prompt-injections`       | Contains "jail" → `prompt_injection`, else → `benign`           | `load_qualifire()`           |
| 4 | `lmsys/toxic-chat`                  | Non-toxic, non-jailbreak rows → `benign` (hard negatives)       | `load_toxic_chat()`          |
| 5 | `allenai/wildjailbreak`             | `adversarial_harmful` → `prompt_injection`, `adversarial_benign` → `benign` | `load_wildjailbreak()` |
| 6 | Synthesised out-of-scope templates  | All → `out_of_scope`                                            | `synthesize_out_of_scope()`  |
| 7 | Hand-authored seeds JSONL           | Pre-labelled `sensitive_bias` + domain benign                   | `load_seeds()`               |

### 3.2 Mindgard Injection Strategy

The Mindgard evaded dataset contains **10,000+** adversarially modified prompts designed to bypass regex-based guardrails.  We handle it in two phases:

1. **Inject 1,000 samples** into the training pool (shuffled with seed 42) so the model learns evasion patterns.
2. **Hold out the remaining 9,000+** as a strict **Out-of-Distribution (OOD)** test set.

Leakage is prevented via **MD5 hashing** on the lowercased text — any sample whose hash appears in the train/test split is excluded from the OOD set.

> See `build_ood_evasion_set()` and the `text_hash()` helper.

### 3.3 Balancing & Deduplication

- **Per-class cap:** `PER_CLASS_CAP = 1000` prevents any single source from dominating.
- **MD5 dedup:** exact duplicates (case-insensitive) are removed.
- **Min length filter:** texts shorter than 3 characters are dropped.

> See `dedupe_and_balance()`.

### 3.4 Train / Test Split

An **80 / 20 stratified split** ensures every class is proportionally represented in both sets:

```
train.jsonl  →  ~80% of each class  (used for training)
test.jsonl   →  ~20% of each class  (in-distribution evaluation)
```

> See `stratified_split()`.

### 3.5 Output Files

| File                  | Purpose                                |
| :-------------------- | :------------------------------------- |
| `ml/data/train.jsonl` | Training set                           |
| `ml/data/test.jsonl`  | In-distribution (ID) test set          |
| `ml/data/ood_evasion.jsonl` | Out-of-distribution evasion test set |

---

## 4. Model A — Single Multiclass SetFit

**Script:** [`train_single.py`](./train_single.py)

### 4.1 Architecture

A single [SetFit](https://huggingface.co/docs/setfit) model that produces probability distributions over all 5 classes.

- **Base model:** `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim embeddings)
- **Head:** Logistic regression over the 5 label classes

### 4.2 Training Hyperparameters

| Parameter            | Value   | Rationale                                               |
| :------------------- | :------ | :------------------------------------------------------ |
| `batch_size`         | 64      | Fits comfortably in 6 GB VRAM                           |
| `num_epochs`         | 1       | SetFit's contrastive learning converges in 1 epoch      |
| `num_iterations`     | 10      | Contrastive pairs per example                           |
| `body_learning_rate` | 2e-5    | Fine-tune the sentence-transformer body slowly          |
| `head_learning_rate` | 1e-2    | The classification head learns faster                   |

> See the `TRAINING_ARGS` dict in `train_single.py`.

### 4.3 Inference Logic

At inference time (inside `SingleSetFitEngine.verify_input()`):

1. Run `_emergency_or_pii()` regex pre-filter.
2. If no regex match, pass the text through the SetFit model → get class probabilities.
3. If the top class is in `CRITICAL ∪ HIGH` and probability ≥ threshold → **block**.
4. If the top class is in `ROUTE` and probability ≥ threshold → **route** (sensitive topic).
5. Otherwise → **allow**.

### 4.4 Output

```
models/guardrail_single/         ← active model (production)
models/guardrail_single_new/     ← newly trained model (validation)
```

---

## 5. Model B — Ensemble Binary SetFit

**Script:** [`train_ensemble.py`](./train_ensemble.py)

### 5.1 Architecture

Four independent binary SetFit models, one per threat class:

```
prompt_injection  →  Binary model 1 (attack vs. not)
role_override     →  Binary model 2
out_of_scope      →  Binary model 3
sensitive_bias    →  Binary model 4
```

Each model outputs `P(positive)` — the probability that the input belongs to its target class.

### 5.2 Binary Dataset Construction

For each target class, the training data is restructured:

| Component        | Label | Description                                                  |
| :--------------- | :---: | :----------------------------------------------------------- |
| Target class     |   1   | All samples of the target class                              |
| Benign           |   0   | All benign samples                                           |
| Other attacks    |   0   | Down-sampled to 60 (seed 42) to prevent overwhelming the negative set |

> See `make_binary_dataset()`.

### 5.3 Training Hyperparameters

| Parameter            | Value   | Rationale                                               |
| :------------------- | :------ | :------------------------------------------------------ |
| `batch_size`         | 64      | Same as single model                                    |
| `num_epochs`         | 1       | Contrastive convergence                                 |
| `num_iterations`     | 20      | More pairs than single (binary task is simpler, but we want finer boundaries) |
| `body_learning_rate` | 2e-5    | Same fine-tuning rate                                   |
| `head_learning_rate` | 1e-2    | Same head rate                                          |

### 5.4 Inference Logic

At inference time (inside `EnsembleSetFitEngine.verify_input()`):

1. Run `_emergency_or_pii()` regex pre-filter.
2. Run all 4 binary models in parallel → each returns `P(positive)`.
3. Collect all classes whose probability exceeds their per-class threshold.
4. If any `CRITICAL ∪ HIGH` class fired → **block** (using the highest probability class).
5. If only `ROUTE` classes fired → **route** (sensitive topic).
6. If nothing fired → **allow**.

### 5.5 Per-Class Thresholds

Each binary model has its own threshold (defined in `EnsembleSetFitEngine.THRESHOLDS`), allowing fine-grained control over sensitivity.

### 5.6 Output

```
models/guardrail_ensemble/
  ├── prompt_injection/
  ├── role_override/
  ├── out_of_scope/
  └── sensitive_bias/
```

---

## 6. Unified Training Pipeline

**Script:** [`train_all.py`](./train_all.py)

This script runs **both** training stages sequentially on a single GPU:

```
Stage 1: Train Single Multiclass Model  →  models/guardrail_single_new/
Stage 2: Train Ensemble Binary Models   →  models/guardrail_ensemble_new/
```

Key design decisions:

- **Sequential execution** to avoid GPU OOM — each model is trained, saved, then deleted from VRAM before the next.
- **`_new` suffix** on output directories prevents overwriting production models during validation.
- **`torch.cuda.empty_cache()`** is called between stages and between each ensemble model.

### Running

```bash
python ml/train_all.py
```

Typical training time: **~2 minutes** on an RTX 4060 (6 GB VRAM).

---

## 7. Evaluation Methodology

**Script:** [`evaluate_local.py`](./evaluate_local.py)

### 7.1 Test Sets

| Set              | File                       | Samples | Purpose                                  |
| :--------------- | :------------------------- | :-----: | :--------------------------------------- |
| In-Distribution  | `ml/data/test.jsonl`       | ~900    | Standard held-out test (20% split)       |
| Out-of-Distribution | `ml/data/ood_evasion.jsonl` | ~9,000+ | Mindgard evasion prompts (never trained on) |

### 7.2 Batched GPU Inference

Instead of calling `verify_input()` one sample at a time (which includes a 50 ms sleep per call), the evaluation script:

1. **Mocks `asyncio.sleep`** to eliminate the production rate-limiting delay.
2. **Separates regex pre-filter hits** from neural inference candidates.
3. **Batches all neural candidates** into a single `predict_proba()` call with `batch_size=256`.

This reduces evaluation time from **hours** to **seconds**.

### 7.3 Metrics Computed

| Metric              | Scope              | Description                                       |
| :------------------ | :----------------- | :------------------------------------------------ |
| Accuracy            | Overall            | Correct predictions / total                       |
| Precision           | Binary (block/allow) | TP / (TP + FP) — measures false alarm rate      |
| Recall              | Binary             | TP / (TP + FN) — measures attack catch rate       |
| F1                  | Binary             | Harmonic mean of precision and recall             |
| Confusion Matrix    | Per engine         | TN / FP / FN / TP counts                         |
| Block Rate          | Per category       | Percentage of each class that was blocked         |
| Latency (avg / p95) | Per engine         | Inference time in milliseconds                    |

### 7.4 Generalization Test

The evaluation isolates **samples where Regex failed** (both false positives and false negatives) and measures how well the neural models perform on this hard subset.  This directly demonstrates the value of ML over pattern matching.

### 7.5 Outputs

| Output                                | Format | Description                                    |
| :------------------------------------ | :----- | :--------------------------------------------- |
| `ml/plots/metrics_comparison.png`     | PNG    | Bar chart — Acc/Prec/Rec/F1 across engines    |
| `ml/plots/latency_comparison.png`     | PNG    | Bar chart — avg + p95 latency                  |
| `ml/plots/confusion_matrices.png`     | PNG    | Side-by-side confusion matrices                |
| `ml/plots/category_block_rates.png`   | PNG    | Block rate per category per engine             |
| `ml/plots/regex_bypass_performance.png` | PNG  | Generalization on regex-failure subset          |
| `ml/plots/guardrail_predictions.xlsx` | Excel  | Per-sample predictions from all three engines  |

### Running

```bash
# Evaluate production (active) models
python ml/evaluate_local.py

# Evaluate newly trained models (after train_all.py)
$env:EVAL_NEW_MODELS="true"; python ml/evaluate_local.py
```

---

## 8. File Reference

| File                                            | Lines | Purpose                                                      |
| :---------------------------------------------- | ----: | :----------------------------------------------------------- |
| [`build_dataset.py`](./build_dataset.py)        | ~230  | Normalize → balance → dedup → split datasets                |
| [`train_single.py`](./train_single.py)          | ~120  | Train single 5-class SetFit model (standalone)              |
| [`train_ensemble.py`](./train_ensemble.py)      | ~130  | Train 4 binary SetFit models (standalone)                   |
| [`train_all.py`](./train_all.py)                | ~185  | Unified pipeline: single + ensemble in sequence             |
| [`evaluate_local.py`](./evaluate_local.py)      | ~420  | Benchmark all 3 engines with charts + spreadsheet export    |

### Key Functions

| Function                          | File                  | Description                                        |
| :-------------------------------- | :-------------------- | :------------------------------------------------- |
| `dedupe_and_balance()`            | `build_dataset.py`    | MD5-based dedup with per-class cap                 |
| `build_ood_evasion_set()`         | `build_dataset.py`    | Leakage-free OOD set from Mindgard data            |
| `make_binary_dataset()`           | `train_ensemble.py`   | Create binary (target vs. rest) training set       |
| `evaluate_single_setfit()`        | `evaluate_local.py`   | Batched GPU inference for single model             |
| `evaluate_ensemble_setfit()`      | `evaluate_local.py`   | Batched GPU inference across 4 ensemble models     |
| `print_split_breakdown()`         | `evaluate_local.py`   | ID vs. OOD metrics + regex-bypass generalization   |
