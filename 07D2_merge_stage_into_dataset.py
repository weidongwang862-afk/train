# scripts/07D2_merge_stage_into_dataset.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config

BASE_OUT = Path(config.OUT_DIR)
FEAT_DIR = BASE_OUT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

INP  = FEAT_DIR / "dataset_all_step7.parquet"
OUTP = FEAT_DIR / "dataset_all_step7plus.parquet"

KEY = ["vin", "t_start", "t_end"]

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
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}. Run 07D_merge_dvdt_into_dataset.py first.")

    base = pd.read_parquet(INP, engine="pyarrow")
    base = force_key_types(base)

    stage_files = sorted(FEAT_DIR.glob("features_stage_*.parquet"))
    if not stage_files:
        print("No features_stage_*.parquet found, saving base only.")
        base.to_parquet(OUTP, index=False, engine="pyarrow")
        print("Saved:", OUTP, "shape:", base.shape, "vins:", base["vin"].nunique())
        return

    stage = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in stage_files], ignore_index=True)
    stage = force_key_types(stage)
    stage = dedup_by_key(stage)

    out = base.merge(stage, on=KEY, how="left")
    assert len(out) == len(base), "Row count changed after stage merge!"
    out.to_parquet(OUTP, index=False, engine="pyarrow")
    print("Saved:", OUTP, "shape:", out.shape, "vins:", out["vin"].nunique())

if __name__ == "__main__":
    main()
