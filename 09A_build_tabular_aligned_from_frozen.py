# scripts/09A_build_tabular_aligned_from_frozen.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

SEQ_DIR = OUT_ROOT / "09_seq_featcore"
SEQ_INDEX_CSV = SEQ_DIR / "seq_index_with_tend.csv"   # 关键：用你刚生成的
FROZEN_PARQUET = Path("E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet")

# 特征清单：只用来拿列名（feature 列）
FEAT_LIST_CSV = OUT_ROOT / "04_features" / "features_FINAL_core.csv"

OUT_NPZ = SEQ_DIR / "tabular_aligned_corefeat.npz"
OUT_REPORT = SEQ_DIR / "tabular_aligned_corefeat_report.txt"


def main():
    idx = pd.read_csv(SEQ_INDEX_CSV)
    idx["vin"] = idx["vin"].astype(str)
    idx["t_end"] = idx["t_end"].astype(np.int64)

    feat_list = pd.read_csv(FEAT_LIST_CSV)
    feat_cols = feat_list["feature"].astype(str).tolist()

    df = pd.read_parquet(FROZEN_PARQUET)
    df["vin"] = df["vin"].astype(str)
    df["t_end"] = df["t_end"].astype(np.int64)

    # 冻结数据里必须含 vin,t_end 以及特征列（有缺失列会被记录）
    need = ["vin", "t_end"] + [c for c in feat_cols if c in df.columns]
    missing_feat = [c for c in feat_cols if c not in df.columns]
    sub = df[need].copy()

    key = idx[["vin", "t_end"]].copy()
    merged = key.merge(sub, on=["vin", "t_end"], how="left", validate="many_to_one")

    used_feat_cols = [c for c in feat_cols if c in merged.columns]
    X_tab = merged[used_feat_cols].to_numpy(dtype=np.float32)

    n = len(merged)
    n_miss_row = int(np.any(np.isnan(X_tab), axis=1).sum())
    nan_total = int(np.isnan(X_tab).sum())

    X_tab = np.nan_to_num(X_tab, nan=0.0)

    np.savez_compressed(
        OUT_NPZ,
        X_tab=X_tab,
        feat_cols=np.array(used_feat_cols, dtype=object),
        vin=merged["vin"].to_numpy(dtype=object),
        t_end=merged["t_end"].to_numpy(dtype=np.int64),
    )

    report = []
    report.append(f"SEQ_INDEX rows = {n}")
    report.append(f"Frozen rows = {len(df)}")
    report.append(f"Requested feat_cols = {len(feat_cols)}")
    report.append(f"Used feat_cols      = {len(used_feat_cols)}")
    report.append(f"Missing feat_cols   = {len(missing_feat)}")
    report.append(f"Rows with any NaN before fill = {n_miss_row}")
    report.append(f"Total NaN cells before fill   = {nan_total}")
    report.append("First 30 missing feat cols:")
    report.append(", ".join(missing_feat[:30]))
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("Saved:", OUT_NPZ)
    print("Saved:", OUT_REPORT)
    print("\n".join(report))


if __name__ == "__main__":
    main()
