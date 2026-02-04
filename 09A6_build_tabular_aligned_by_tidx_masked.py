# scripts/09A6_build_tabular_aligned_by_tidx_masked.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

SEQ_DIR = OUT_ROOT / "09_seq_featcore"
SEQ_INDEX_CSV = SEQ_DIR / "seq_index.csv"  # vin,t_idx
FROZEN_PARQUET = OUT_ROOT / "E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet"
FEAT_LIST_CSV = OUT_ROOT / "04_features" / "features_FINAL_core.csv"

OUT_NPZ = SEQ_DIR / "tabular_aligned_corefeat_by_tidx_masked.npz"
OUT_REPORT = SEQ_DIR / "tabular_aligned_corefeat_by_tidx_masked_report.txt"


def main():
    idx = pd.read_csv(SEQ_INDEX_CSV)
    idx["vin"] = idx["vin"].astype(str)
    idx["t_idx"] = idx["t_idx"].astype(int)

    feat_list = pd.read_csv(FEAT_LIST_CSV)
    feat_cols = feat_list["feature"].astype(str).tolist()

    df = pd.read_parquet(FROZEN_PARQUET)
    df["vin"] = df["vin"].astype(str)

    sub = df[["vin", "t_end"] + feat_cols].copy()

    N = len(idx)
    D = len(feat_cols)

    X_val = np.full((N, D), np.nan, dtype=np.float32)
    X_msk = np.zeros((N, D), dtype=np.float32)  # 1 表示缺失
    tend_frozen = np.full(N, np.nan, dtype=np.float64)

    # 逐 vin 对齐：按 t_end 排序后用 t_idx 取行
    for vin, g in idx.groupby("vin", sort=False):
        g_idx = g.index.to_numpy()
        tpos = g["t_idx"].to_numpy(dtype=np.int64)

        sub_v = sub[sub["vin"] == vin].sort_values("t_end", kind="mergesort").reset_index(drop=True)
        rows = sub_v.loc[tpos, feat_cols]
        X_val[g_idx, :] = rows.to_numpy(dtype=np.float32)
        tend_frozen[g_idx] = sub_v.loc[tpos, "t_end"].to_numpy()

    # 统计缺失率
    miss = np.isnan(X_val)
    X_msk[miss] = 1.0
    nan_total = int(miss.sum())
    row_any_nan = int(miss.any(axis=1).sum())

    # 填充策略：按“全体样本的列中位数”填充（鲁棒），并保留缺失标记
    col_med = np.nanmedian(X_val, axis=0)
    # 若某列全 NaN，nanmedian 会是 NaN，这种列直接置 0
    col_med = np.where(np.isnan(col_med), 0.0, col_med).astype(np.float32)

    X_filled = X_val.copy()
    for j in range(D):
        mj = col_med[j]
        mcol = miss[:, j]
        if mcol.any():
            X_filled[mcol, j] = mj

    # 拼接：value + mask => 2D 维输入
    X_tab = np.concatenate([X_filled, X_msk], axis=1).astype(np.float32)

    # 输出报告
    nan_rate = miss.mean(axis=0)
    top = np.argsort(-nan_rate)[:15]
    lines = []
    lines.append(f"rows={N} feat_cols={D} X_tab_dim={X_tab.shape[1]}")
    lines.append(f"rows with any NaN(before fill) = {row_any_nan}")
    lines.append(f"total NaN cells(before fill)   = {nan_total}")
    lines.append("Top15 NaN-rate features:")
    for j in top:
        lines.append(f"{feat_cols[j]}\t{nan_rate[j]:.6f}")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    np.savez_compressed(
        OUT_NPZ,
        X_tab=X_tab,  # (N, 2D)
        feat_cols=np.array(feat_cols, dtype=object),
        vin=idx["vin"].to_numpy(dtype=object),
        t_idx=idx["t_idx"].to_numpy(dtype=np.int64),
        t_end_frozen=tend_frozen,
        col_median=col_med,
    )

    print("Saved:", OUT_NPZ)
    print("Saved:", OUT_REPORT)
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
