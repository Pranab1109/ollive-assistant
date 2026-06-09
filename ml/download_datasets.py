"""Download all source datasets into ml/datasets_raw/ as parquet/csv."""
import os
from datasets import load_dataset

# Load .env file from the root directory to populate environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

hf_token = os.getenv("HF_TOKEN")

OUT = os.path.join(os.path.dirname(__file__), "datasets_raw")
os.makedirs(OUT, exist_ok=True)

SOURCES = {
    "jailbreak_classification": ("jackhhao/jailbreak-classification", None),
    "safe_guard":               ("xTRam1/safe-guard-prompt-injection", None),
    "qualifire_pi":             ("qualifire/prompt-injections-benchmark", None),
    "toxic_chat":               ("lmsys/toxic-chat", "toxicchat0124"),
    "wildjailbreak":            ("allenai/wildjailbreak", "eval"),   # 'train' is huge
    "evaded":                   ("Mindgard/evaded-prompt-injection-and-jailbreak-samples", None),
}

for name, (repo, config) in SOURCES.items():
    try:
        print(f"Loading dataset: {repo}...")
        ds = load_dataset(repo, config, token=hf_token) if config else load_dataset(repo, token=hf_token)
        for split in ds:
            path = os.path.join(OUT, f"{name}__{split}.parquet")
            ds[split].to_parquet(path)
            print(f"Saved {path} ({len(ds[split])} rows)")
    except Exception as e:
        print(f"[skip] {name}: {e}")

# TrustAIRLab/JailbreakLLMs is on GitHub:
#   git clone https://github.com/TrustAIRLab/JailbreakLLMs ml/datasets_raw/JailbreakLLMs
print("Now: git clone https://github.com/TrustAIRLab/JailbreakLLMs into datasets_raw/")
