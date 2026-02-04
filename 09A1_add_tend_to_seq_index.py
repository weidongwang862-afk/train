# scripts/09A1_add_tend_to_seq_index.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

IN_CSV = SEQ_DIR / "seq_index.csv"
OUT_CSV = SEQ_DIR / "seq_index_with_tend.csv"

def main():
    idx = pd.read_csv(IN_CSV)
    if "vin" not in idx.columns or "t_idx" not in idx.columns:
        raise RuntimeError("seq_index.csv 必须包含 vin 和 t_idx 两列。")
    idx["vin"] = idx["vin"].astype(str)
    idx["t_idx"] = idx["t_idx"].astype(int)

    tend_col = np.empty(len(idx), dtype=np.int64)

    # 按 vin 分组，一次性加载该 vin 的 npz，再批量取 t_end[t_idx]
    for vin, g in idx.groupby("vin", sort=False):
        fp = VIN_DIR / f"{vin}.npz"
        if not fp.exists():
            raise FileNotFoundError(f"Missing npz: {fp}")
        obj = np.load(fp)
        t_end = obj["t_end"].astype(np.int64)  # (T,)

        gi = g.index.to_numpy()
        ti = g["t_idx"].to_numpy(dtype=np.int64)

        # 越界检查
        if ti.min() < 0 or ti.max() >= len(t_end):
            raise RuntimeError(f"vin={vin} 的 t_idx 越界：min={ti.min()} max={ti.max()} len(t_end)={len(t_end)}")

        tend_col[gi] = t_end[ti]

    idx["t_end"] = tend_col
    idx.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_CSV)
    print("rows =", len(idx))
    print("unique vins =", idx["vin"].nunique())
    print("t_end min/max =", int(idx["t_end"].min()), int(idx["t_end"].max()))
    print(idx.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
