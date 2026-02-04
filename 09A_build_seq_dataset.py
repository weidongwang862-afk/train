# scripts/09A_build_seq_dataset.py
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import config

OUT_ROOT = Path(config.OUT_DIR)
OUT_DIR = OUT_ROOT / "09_seq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 输入数据（冻结版本，只读）=====
DEFAULT_FROZEN = r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"
DATASET = Path(os.environ.get("DATASET", DEFAULT_FROZEN)).expanduser()

# 你想预测的目标
TARGET = os.environ.get("TARGET", "SoH_trend").strip()

# 事件主键（你项目里一直用这三列）
KEYS = ["vin", "t_start", "t_end"]

# 序列窗口长度
SEQ_LEN = int(os.environ.get("SEQ_LEN", "20"))

# VIN 切分比例
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.20"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.10"))
SEED = int(os.environ.get("SEED", "42"))

# 特征列表：沿用你现有 outputs/04_features 里的最终特征表
# 如果你要用 06D 的输出，就把这个环境变量指向对应 csv
DEFAULT_FEATS = str(OUT_ROOT / "04_features" / "features_FINAL_core.csv")
FEATURE_LIST = Path(os.environ.get("FEATURE_LIST", DEFAULT_FEATS)).expanduser()

# 是否把进度特征加入（你 06D 里有同款逻辑）
ADD_PROGRESS = os.environ.get("ADD_PROGRESS", "1").strip() == "1"


def load_feature_list(fp: Path) -> list[str]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing feature list: {fp}")
    df = pd.read_csv(fp)
    if "feature" in df.columns:
        feats = df["feature"].astype(str).tolist()
    else:
        feats = df.iloc[:, 0].astype(str).tolist()
    feats = [f for f in feats if f and f.lower() != "nan"]
    # 去重但保序
    return list(dict.fromkeys(feats))


def add_progress_features(df: pd.DataFrame) -> pd.DataFrame:
    # 与 06D 的思路一致：按 vin 排序后给出相对时间与事件序号
    if not {"vin", "t_start", "t_end"}.issubset(df.columns):
        return df
    df = df.sort_values(["vin", "t_start"]).copy()
    t0 = df.groupby("vin")["t_start"].transform("min")
    df["t_rel_days"] = (df["t_end"] - t0) / 86400.0
    df["event_idx"] = df.groupby("vin").cumcount().astype("int32")
    return df


def split_vins(vins: np.ndarray, seed: int, test_frac: float, val_frac: float):
    rng = np.random.default_rng(seed)
    vins = np.array(sorted(vins.astype(str)))
    rng.shuffle(vins)

    n = len(vins)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * (n - n_test))))

    test_v = vins[:n_test]
    remain = vins[n_test:]
    val_v = remain[:n_val]
    train_v = remain[n_val:]
    return train_v.tolist(), val_v.tolist(), test_v.tolist()


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    if not FEATURE_LIST.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_LIST}")

    print("Read parquet:", DATASET)
    df = pd.read_parquet(DATASET, engine="pyarrow")

    # 关键列检查
    for k in KEYS:
        if k not in df.columns:
            raise ValueError(f"Dataset missing key column: {k}")
    if TARGET not in df.columns:
        raise ValueError(f"Dataset missing target column: {TARGET}")

    # 目标转数值，过滤掉目标为空
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()

    # 加进度特征
    if ADD_PROGRESS:
        df = add_progress_features(df)

    # 读取特征列并取交集
    feats = load_feature_list(FEATURE_LIST)
    feats = [c for c in feats if c in df.columns]

    # 如果你开启了进度特征，这里强制补进去（不依赖 feature_list）
    if ADD_PROGRESS:
        for c in ["t_rel_days", "event_idx"]:
            if c in df.columns and c not in feats:
                feats.append(c)

    if len(feats) < 5:
        raise ValueError(f"Too few usable features after intersection: {len(feats)}")

    # 只保留必要列，减少内存
    keep_cols = [c for c in KEYS if c in df.columns] + feats + [TARGET]
    df = df[keep_cols].copy()

    # 排序，保证每辆车是时间序
    df["vin"] = df["vin"].astype(str)
    df = df.sort_values(["vin", "t_start"]).reset_index(drop=True)

    vins = df["vin"].unique()
    train_vins, val_vins, test_vins = split_vins(vins, SEED, TEST_FRAC, VAL_FRAC)

    splits = {"train": train_vins, "val": val_vins, "test": test_vins}
    (OUT_DIR / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", OUT_DIR / "splits.json")

    # ===== 训练集拟合缺失填充与标准化参数（只用训练 VIN）=====
    df_tr = df[df["vin"].isin(train_vins)].copy()
    Xtr = df_tr[feats].apply(pd.to_numeric, errors="coerce")

    med = Xtr.median(numeric_only=True)
    Xtr_f = Xtr.fillna(med)

    mean = Xtr_f.mean()
    std = Xtr_f.std().replace(0.0, 1.0)

    stats = {
        "features": feats,
        "median": med.to_dict(),
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "seq_len": SEQ_LEN,
        "target": TARGET,
        "keys": KEYS,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", OUT_DIR / "stats.json")

    # ===== 把每个 VIN 存成一个 npz：X 标准化后的特征序列 + y + keys =====
    vin_dir = OUT_DIR / "vin_npz"
    vin_dir.mkdir(parents=True, exist_ok=True)

    # 为了训练脚本快速采样，生成一个样本索引表：每行一个可用窗口的结尾位置 t
    rows = []
    for vin, g in df.groupby("vin", sort=False):
        g = g.sort_values("t_start").reset_index(drop=True)
        X = g[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
        X = (X - mean) / std
        X = X.to_numpy(dtype=np.float32)

        y = g[TARGET].to_numpy(dtype=np.float32)
        t_start = g["t_start"].to_numpy(dtype=np.int64)
        t_end = g["t_end"].to_numpy(dtype=np.int64)

        out_fp = vin_dir / f"{vin}.npz"
        np.savez_compressed(out_fp, X=X, y=y, t_start=t_start, t_end=t_end)

        # 从 SEQ_LEN-1 开始才能形成完整窗口
        T = len(g)
        for t in range(SEQ_LEN - 1, T):
            rows.append((vin, int(t)))

    index_df = pd.DataFrame(rows, columns=["vin", "t_idx"])
    index_fp = OUT_DIR / "seq_index.csv"
    index_df.to_csv(index_fp, index=False, encoding="utf-8-sig")
    print("Saved:", index_fp)
    print("Total seq samples:", len(index_df))


if __name__ == "__main__":
    main()
