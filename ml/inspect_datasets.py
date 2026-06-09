import pandas as pd
import glob
import os

RAW = os.path.join(os.path.dirname(__file__), "datasets_raw")
OUT = os.path.join(os.path.dirname(__file__), "data")

SOURCES = {
    "jailbreak_classification": "jailbreak_classification__*",
    "safe_guard": "safe_guard__*",
    "qualifire_pi": "qualifire_pi__*",
    "toxic_chat": "toxic_chat__*",
    "wildjailbreak": "wildjailbreak__*",
    "evaded": "evaded__*"
}

with open(os.path.join(OUT, "inspect_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("============================================================\n")
    fh.write("                RAW DATASETS SCHEMA & SAMPLE INSPECTION\n")
    fh.write("============================================================\n\n")

    for name, pattern in SOURCES.items():
        files = glob.glob(os.path.join(RAW, pattern))
        if not files:
            fh.write(f"Source: {name}\n  Status: ❌ No files found matching {pattern}\n\n")
            continue
            
        fh.write(f"Source: {name}\n")
        fh.write(f"  File path: {files[0]}\n")
        try:
            df = pd.read_parquet(files[0])
            fh.write(f"  Shape: {df.shape}\n")
            fh.write(f"  Columns: {list(df.columns)}\n")
            fh.write("  Sample Rows:\n")
            for idx in range(min(2, len(df))):
                row = df.iloc[idx].to_dict()
                fh.write(f"    Sample {idx + 1}:\n")
                for col, val in row.items():
                    val_str = str(val)
                    if len(val_str) > 120:
                        val_str = val_str[:120] + "... [truncated]"
                    fh.write(f"      {col}: {repr(val_str)}\n")
            fh.write("\n")
        except Exception as e:
            fh.write(f"  Status: ❌ Error loading: {e}\n\n")

print("Inspection complete. Saved to ml/data/inspect_report.txt")
