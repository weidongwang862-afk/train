# scripts/08B_merge_qc_into_dataset.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

def main():
    DATASET = Path(os.environ.get("DATASET", r"E:\RAW_DATA\outputs\04_features\dataset_all_step7plus_relax.parquet"))
    QC_CSV  = Path(os.environ.get("QC_CSV",  r"E:\RAW_DATA\outputs\04_features\event_qc_report_qc_v2.csv"))
    OUT     = Path(os.environ.get("OUT",     r"E:\RAW_DATA\outputs\04_features\dataset_all_step7plus_relax_qc.parquet"))

    df = pd.read_parquet(DATASET, engine="pyarrow")
    qc = pd.read_csv(QC_CSV)

    key = ["vin", "t_start", "t_end"]
    for c in key:
        if c not in df.columns:
            raise ValueError(f"dataset missing key col: {c}")
        if c not in qc.columns:
            raise ValueError(f"qc csv missing key col: {c}")

    keep = key + [c for c in ["qc_weight", "qc_good"] if c in qc.columns]
    qc = qc[keep].copy()

    out = df.merge(qc, on=key, how="left")
    out["qc_weight"] = pd.to_numeric(out["qc_weight"], errors="coerce").fillna(1.0)
    out["qc_good"] = pd.to_numeric(out.get("qc_good", 1), errors="coerce").fillna(1).astype(int)

    out.to_parquet(OUT, index=False, engine="pyarrow")
    print("Saved:", OUT, "rows:", len(out), "vins:", out["vin"].nunique())

if __name__ == "__main__":
    main()
