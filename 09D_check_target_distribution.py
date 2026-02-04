# scripts/09D_check_target_distribution.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATASET = Path(r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet")  # 改成你的
SPLITS = Path(r"E:\RAW_DATA\outputs\09_seq\splits.json")                        # 改成你的
TARGET = "SoH_trend"

def describe(y, name):
    q = np.quantile(y, [0.01, 0.05, 0.10, 0.25, 0.50])
    print(f"\n[{name}] n={len(y)}")
    print("q01,q05,q10,q25,q50 =", [float(v) for v in q])
    print("pct(y<0.90) =", float(np.mean(y < 0.90)))
    print("pct(y<0.92) =", float(np.mean(y < 0.92)))
    print("pct(y<0.95) =", float(np.mean(y < 0.95)))

def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    df = pd.read_parquet(DATASET, engine="pyarrow")
    df["vin"] = df["vin"].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()

    tr = df[df["vin"].isin(splits["train"])][TARGET].to_numpy()
    va = df[df["vin"].isin(splits["val"])][TARGET].to_numpy()
    te = df[df["vin"].isin(splits["test"])][TARGET].to_numpy()

    describe(tr, "train")
    describe(va, "val")
    describe(te, "test")

if __name__ == "__main__":
    main()
