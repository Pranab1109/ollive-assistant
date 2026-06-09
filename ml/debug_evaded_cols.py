import pandas as pd
import glob
import os

RAW = os.path.join(os.path.dirname(__file__), "datasets_raw")
OUT = os.path.join(os.path.dirname(__file__), "data")

files = glob.glob(os.path.join(RAW, "evaded__*"))
if files:
    df = pd.read_parquet(files[0])
    with open(os.path.join(OUT, "debug_cols.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Columns: {list(df.columns)}\n")
        fh.write(f"Shape: {df.shape}\n")
        fh.write(f"Head:\n{df.head().to_string()}\n")
else:
    print("No evaded files found")
