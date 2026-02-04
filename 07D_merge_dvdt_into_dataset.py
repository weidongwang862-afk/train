# scripts/07D_merge_dvdt_into_dataset.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config

BASE_OUT = Path(config.OUT_DIR)
FEAT_DIR = BASE_OUT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 简洁命名：all 单数据集 ======
ALL_IN   = FEAT_DIR / "dataset_all.parquet"
ALL_OUT  = FEAT_DIR / "dataset_all_step7.parquet"

# 兼容旧结构：如果没有 dataset_all.parquet，就把 train/test 拼成 all
TRAIN_PATH = FEAT_DIR / "dataset_train.parquet"
TEST_PATH  = FEAT_DIR / "dataset_test.parquet"

KEY = ["vin", "t_start", "t_end"]

def load_all_dataset() -> pd.DataFrame:
    if ALL_IN.exists():
        return pd.read_parquet(ALL_IN, engine="pyarrow")

    if TRAIN_PATH.exists() and TEST_PATH.exists():
        tr = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
        te = pd.read_parquet(TEST_PATH, engine="pyarrow")
        df = pd.concat([tr, te], ignore_index=True)
        df.to_parquet(ALL_IN, index=False, engine="pyarrow")
        print("Built:", ALL_IN, "rows:", len(df))
        return df

    raise FileNotFoundError("Missing dataset_all.parquet AND missing dataset_train/test.parquet")

def force_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vin"] = df["vin"].astype(str)
    df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce").astype("Int64")
    df["t_end"]   = pd.to_numeric(df["t_end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["t_start", "t_end"])
    df["t_start"] = df["t_start"].astype("int64")
    df["t_end"]   = df["t_end"].astype("int64")
    return df

def dedup_by_key(aux: pd.DataFrame) -> pd.DataFrame:
    num_cols = aux.select_dtypes(include=["number"]).columns.tolist()
    agg = {c: "mean" for c in num_cols if c not in KEY}
    for c in aux.columns:
        if c in KEY or c in agg:
            continue
        agg[c] = "first"
    return aux.groupby(KEY, as_index=False).agg(agg)

def main():
    base = load_all_dataset()
    base = force_key_types(base)

    dvdt_files = sorted(FEAT_DIR.glob("features_dvdt_*.parquet"))
    if not dvdt_files:
        print("Warning: no features_dvdt_*.parquet found, saving base only.")
        base.to_parquet(ALL_OUT, index=False, engine="pyarrow")
        print("Saved:", ALL_OUT, "shape:", base.shape, "vins:", base["vin"].nunique())
        return

    aux = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in dvdt_files], ignore_index=True)
    aux = force_key_types(aux)
    aux = dedup_by_key(aux)

    out = base.merge(aux, on=KEY, how="left")
    assert len(out) == len(base), "Row count changed after dvdt merge!"
    out.to_parquet(ALL_OUT, index=False, engine="pyarrow")
    print("Saved:", ALL_OUT, "shape:", out.shape, "vins:", out["vin"].nunique())

if __name__ == "__main__":
    main()
