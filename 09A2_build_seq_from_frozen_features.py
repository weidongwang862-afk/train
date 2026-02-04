from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# === 路径改这里 ===
FROZEN_PARQUET = Path(r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet")
FEATURE_LIST   = Path(r"E:\RAW_DATA\outputs\04_features\features_FINAL_core.csv")  # 你的特征列名单
SEQ_DIR        = Path(r"E:\RAW_DATA\outputs\09_seq_featcore")          # 新输出目录，别覆盖旧09_seq
VIN_DIR        = SEQ_DIR / "vin_npz"
SEQ_LEN        = int(os.environ.get("SEQ_LEN", "64"))

# === 你需要把这三个列名改成你冻结数据里真实存在的列名 ===
COL_VIN   = "vin"
COL_TIME  = "odo_end"          # 或 terminaltime / totalodometer，选一个“单调递增排序键”
COL_Y     = "SoH_trend"              # 你的SoH/容量标签列名（例如 soh、soh3、cap_norm 等）

def main():
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    VIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 读数据
    df = pd.read_parquet(FROZEN_PARQUET)
    df[COL_VIN] = df[COL_VIN].astype(str)

    # 2) 读特征列名单
    feat_df = pd.read_csv(FEATURE_LIST)
    # 兼容两种格式：一列名叫 feature / 或直接第一列就是特征名
    if "feature" in feat_df.columns:
        feats = feat_df["feature"].astype(str).tolist()
    else:
        feats = feat_df.iloc[:, 0].astype(str).tolist()

    # 3) 保证列存在
    need = [COL_VIN, COL_TIME, COL_Y] + feats
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in frozen parquet: {miss}")

    # 4) 基础清理：按排序键升序，去掉y为空
    df = df[need].dropna(subset=[COL_Y]).copy()
    df = df.sort_values([COL_VIN, COL_TIME]).reset_index(drop=True)

    # 5) 简单处理缺失特征（先用全局中位数占位；真正严谨做法是用train统计量，这一步先把数据集构出来）
    med = df[feats].median(numeric_only=True)
    df[feats] = df[feats].fillna(med)

    # 6) 生成每个vin的npz + 索引
    index_rows = []
    vin_list = df[COL_VIN].unique().tolist()

    # 用全局统计量先做一版（下一步训练前再用train-only统计替换更严谨）
    mu = df[feats].mean().astype(float).values
    sd = df[feats].std().replace(0, 1.0).astype(float).values

    for vin in vin_list:
        g = df[df[COL_VIN] == vin]
        X = g[feats].to_numpy(dtype=np.float32)
        y = g[COL_Y].to_numpy(dtype=np.float32)
        t = g[COL_TIME].to_numpy()

        # 标准化
        X = (X - mu) / sd

        T = len(g)
        if T < SEQ_LEN:
            continue

        # 保存 npz
        np.savez_compressed(VIN_DIR / f"{vin}.npz", X=X, y=y, t_end=t)

        # 可作为样本的 t_idx：从 SEQ_LEN-1 到 T-1
        for t_idx in range(SEQ_LEN - 1, T):
            index_rows.append((vin, t_idx))

    index_df = pd.DataFrame(index_rows, columns=["vin", "t_idx"])
    index_df.to_csv(SEQ_DIR / "seq_index.csv", index=False, encoding="utf-8-sig")

    stats = {
        "seq_len": SEQ_LEN,
        "features": feats,
        "norm_mu": mu.tolist(),
        "norm_sd": sd.tolist(),
        "col_time": COL_TIME,
        "col_y": COL_Y,
    }
    (SEQ_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", SEQ_DIR)
    print("n_vins_npz =", len(list(VIN_DIR.glob("*.npz"))))
    print("n_samples  =", len(index_df))

if __name__ == "__main__":
    main()
