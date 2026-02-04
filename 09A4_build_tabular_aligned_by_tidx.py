# scripts/09A2_build_tabular_aligned_by_tidx.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

SEQ_DIR = OUT_ROOT / "09_seq_featcore"
SEQ_INDEX_CSV = SEQ_DIR / "seq_index.csv"  # 用原始 index（vin,t_idx），行顺序就是训练样本顺序
FROZEN_PARQUET = OUT_ROOT / "E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet"

# 特征清单：只取 feature 列做列名列表
FEAT_LIST_CSV = OUT_ROOT / "04_features" / "features_FINAL_core.csv"

OUT_NPZ = SEQ_DIR / "tabular_aligned_corefeat_by_tidx.npz"
OUT_REPORT = SEQ_DIR / "tabular_aligned_corefeat_by_tidx_report.txt"


def main():
    idx = pd.read_csv(SEQ_INDEX_CSV)
    if "vin" not in idx.columns or "t_idx" not in idx.columns:
        raise RuntimeError("seq_index.csv 必须包含 vin 和 t_idx")
    idx["vin"] = idx["vin"].astype(str)
    idx["t_idx"] = idx["t_idx"].astype(int)

    feat_list = pd.read_csv(FEAT_LIST_CSV)
    if "feature" not in feat_list.columns:
        raise RuntimeError("features_FINAL_core.csv 缺少 feature 列")
    feat_cols = feat_list["feature"].astype(str).tolist()

    # 读取冻结数据
    df = pd.read_parquet(FROZEN_PARQUET)
    if "vin" not in df.columns:
        raise RuntimeError("冻结数据缺少 vin 列")
    if "t_end" not in df.columns:
        raise RuntimeError("冻结数据缺少 t_end 列（用于排序）")

    df["vin"] = df["vin"].astype(str)

    missing_feat = [c for c in feat_cols if c not in df.columns]
    if missing_feat:
        raise RuntimeError(f"冻结数据缺少特征列: {missing_feat[:10]} ... 共 {len(missing_feat)} 个")

    # 只保留必要列，减少内存
    sub = df[["vin", "t_end"] + feat_cols].copy()

    N = len(idx)
    D = len(feat_cols)
    X_tab = np.full((N, D), np.nan, dtype=np.float32)

    # 额外输出：对齐到的冻结 t_end，便于核对
    tend_frozen = np.full(N, np.nan, dtype=np.float64)

    bad_vins = []
    for vin, g in idx.groupby("vin", sort=False):
        g_idx = g.index.to_numpy()
        tpos = g["t_idx"].to_numpy(dtype=np.int64)
        max_pos = int(tpos.max())

        sub_v = sub[sub["vin"] == vin].sort_values("t_end", kind="mergesort").reset_index(drop=True)
        if len(sub_v) <= max_pos:
            bad_vins.append((vin, len(sub_v), max_pos))
            continue

        rows = sub_v.loc[tpos, feat_cols]
        X_tab[g_idx, :] = rows.to_numpy(dtype=np.float32)
        tend_frozen[g_idx] = sub_v.loc[tpos, "t_end"].to_numpy()

    nan_total = int(np.isnan(X_tab).sum())
    n_row_any_nan = int(np.any(np.isnan(X_tab), axis=1).sum())

    # 不做 fillna，先把对齐质量写报告
    report = []
    report.append(f"SEQ_INDEX rows = {N}")
    report.append(f"Frozen rows = {len(df)}")
    report.append(f"feat_cols = {D}")
    report.append(f"rows with any NaN = {n_row_any_nan}")
    report.append(f"total NaN cells   = {nan_total}")
    report.append(f"bad_vins (len(sub_v) <= max_t_idx) = {len(bad_vins)}")
    if bad_vins:
        report.append("first 10 bad_vins: " + ", ".join([f"{v}(len={l},max={m})" for v, l, m in bad_vins[:10]]))

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    # 只有在完全对齐时才写 npz，避免把错误文件继续传下去
    if n_row_any_nan > 0:
        print("\n".join(report))
        raise RuntimeError("对齐仍存在缺失（出现 NaN）。先解决 bad_vins / 冻结数据缺行问题，再进入融合训练。")

    np.savez_compressed(
        OUT_NPZ,
        X_tab=X_tab,
        feat_cols=np.array(feat_cols, dtype=object),
        vin=idx["vin"].to_numpy(dtype=object),
        t_idx=idx["t_idx"].to_numpy(dtype=np.int64),
        t_end_frozen=tend_frozen,
    )

    print("Saved:", OUT_NPZ)
    print("Saved:", OUT_REPORT)
    print("\n".join(report))


if __name__ == "__main__":
    main()
