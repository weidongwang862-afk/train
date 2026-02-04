# scripts/09C_run_baseline_on_splits.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

OUT_DIR = Path("outputs")
DATASET = Path(r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet")  # 改成你的冻结表
SPLITS = OUT_DIR / "09_seq" / "splits.json"
FEAT_LIST = OUT_DIR / "04_features" / "features_FINAL_core.csv"               # 改成你最终用的特征表
TARGET = "SoH_trend"

SAVE_DIR = OUT_DIR / "09_seq"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_features(fp: Path) -> list[str]:
    df = pd.read_csv(fp)
    if "feature" in df.columns:
        feats = df["feature"].astype(str).tolist()
    else:
        feats = df.iloc[:, 0].astype(str).tolist()
    feats = [f for f in feats if f and f.lower() != "nan"]
    return list(dict.fromkeys(feats))

def vin_mae_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = (pred_df.assign(abs_err=(pred_df["y_true"] - pred_df["y_pred"]).abs())
                 .groupby("vin", as_index=False)["abs_err"].mean()
                 .rename(columns={"abs_err":"mae"}))
    return out

def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    feats = load_features(FEAT_LIST)

    df = pd.read_parquet(DATASET, engine="pyarrow")
    df["vin"] = df["vin"].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()

    feats = [c for c in feats if c in df.columns]
    keep = ["vin", "t_end", TARGET] + feats
    df = df[keep].copy()

    # 训练集拟合缺失填充（只用 train VIN）
    tr = df[df["vin"].isin(splits["train"])].copy()
    va = df[df["vin"].isin(splits["val"])].copy()
    te = df[df["vin"].isin(splits["test"])].copy()

    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median(numeric_only=True)

    def prep(d: pd.DataFrame):
        X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
        y = d[TARGET].to_numpy(dtype=np.float32)
        return X.to_numpy(dtype=np.float32), y

    Xtr, ytr = prep(tr)
    Xva, yva = prep(va)
    Xte, yte = prep(te)

    # baseline：HGBR（和你原基线一致的树系）
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        random_state=42,
    )
    model.fit(Xtr, ytr)

    # 可选：用 val 做一次早停/调参，你现在先对齐口径即可
    pte = model.predict(Xte).astype(np.float32)

    pred_df = pd.DataFrame({
        "vin": te["vin"].to_numpy(),
        "t_end": te["t_end"].to_numpy(),
        "y_true": yte,
        "y_pred": pte,
    })
    vin_mae = vin_mae_table(pred_df)

    pred_path = SAVE_DIR / "predictions_test_BASELINE.csv"
    vin_path = SAVE_DIR / "vin_mae_test_BASELINE.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    vin_mae.to_csv(vin_path, index=False, encoding="utf-8-sig")
    print("Saved:", pred_path)
    print("Saved:", vin_path)
    print("[TEST] MAE =", float(np.mean(np.abs(pred_df["y_true"] - pred_df["y_pred"]))))

if __name__ == "__main__":
    main()
