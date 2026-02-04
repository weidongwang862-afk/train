# scripts/09A4_diag_alignment_nan.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

SEQ_DIR = OUT_ROOT / "09_seq_featcore"
SEQ_INDEX_CSV = SEQ_DIR / "seq_index.csv"
FROZEN_PARQUET = Path("E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet")
FEAT_LIST_CSV = OUT_ROOT / "04_features" / "features_FINAL_core.csv"

def main():
    idx = pd.read_csv(SEQ_INDEX_CSV)
    idx["vin"] = idx["vin"].astype(str)
    idx["t_idx"] = idx["t_idx"].astype(int)

    feat_list = pd.read_csv(FEAT_LIST_CSV)
    feat_cols = feat_list["feature"].astype(str).tolist()

    df = pd.read_parquet(FROZEN_PARQUET)
    df["vin"] = df["vin"].astype(str)

    # 只保留必要列
    sub = df[["vin", "t_end"] + feat_cols].copy()

    # 1) 统计每个 vin：seq_index 需要的最大 t_idx vs 冻结数据可用长度
    need = idx.groupby("vin")["t_idx"].max().rename("max_t_idx").to_frame()
    avail = sub.groupby("vin").size().rename("n_frozen").to_frame()
    stat = need.join(avail, how="left")
    stat["n_frozen"] = stat["n_frozen"].fillna(0).astype(int)
    stat["need_len"] = stat["max_t_idx"] + 1
    stat["short_by"] = stat["need_len"] - stat["n_frozen"]

    bad = stat[stat["short_by"] > 0].sort_values("short_by", ascending=False)

    print("=== VIN length check ===")
    print("n_vins seq_index =", stat.shape[0])
    print("n_vins frozen    =", avail.shape[0])
    print("bad_vins(short_by>0) =", bad.shape[0])
    if bad.shape[0] > 0:
        print("\nTop 20 bad_vins:")
        print(bad.head(20).to_string())

    # 2) 抽样检查：冻结数据特征列是否本身大量 NaN
    nan_rate = sub[feat_cols].isna().mean().sort_values(ascending=False)
    print("\n=== Frozen feature NaN rate (top 15) ===")
    print(nan_rate.head(15).to_string())

    # 3) 对齐抽样：随机挑 3 个 vin（优先挑 bad_vins，否则挑正常 vin），看实际取出的行是否 NaN
    pick = []
    if bad.shape[0] > 0:
        pick += list(bad.index[:3])
    else:
        pick += list(stat.sample(n=min(3, len(stat)), random_state=0).index)

    print("\n=== Sample alignment check ===")
    for vin in pick:
        g = idx[idx["vin"] == vin]
        max_t = int(g["t_idx"].max())
        sub_v = sub[sub["vin"] == vin].sort_values("t_end", kind="mergesort").reset_index(drop=True)
        print(f"\nVIN={vin} | seq_index rows={len(g)} | max_t_idx={max_t} | frozen_len={len(sub_v)}")

        # 取 5 个位置：最早/中间/末尾
        tpos = [int(g["t_idx"].min()), int(np.median(g["t_idx"])), max_t]
        tpos = [p for p in tpos if p >= 0]
        for p in tpos:
            if p >= len(sub_v):
                print(f"  t_idx={p}  -> out of range")
                continue
            row = sub_v.loc[p, feat_cols]
            n_nan = int(row.isna().sum())
            print(f"  t_idx={p}  -> nan_cells={n_nan}/{len(feat_cols)} | t_end={sub_v.loc[p,'t_end']}")

if __name__ == "__main__":
    main()
