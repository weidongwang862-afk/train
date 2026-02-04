# scripts/07D3_merge_relax_into_dataset.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)
FEAT_DIR = OUT_ROOT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

INP = FEAT_DIR / "dataset_all_step7plus.parquet"

RELAX_PATH = FEAT_DIR / "features_relax_L1.parquet"

OUTP   = FEAT_DIR / "dataset_all_step7plus_relax.parquet"
REPORT = FEAT_DIR / "relax_cov_all.csv"

KEY = ["vin", "t_start", "t_end"]

def _force_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vin"] = df["vin"].astype(str)
    df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce").astype("Int64")
    df["t_end"]   = pd.to_numeric(df["t_end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["t_start", "t_end"])
    df["t_start"] = df["t_start"].astype("int64")
    df["t_end"]   = df["t_end"].astype("int64")
    return df

def _dedup_relax(relax: pd.DataFrame) -> pd.DataFrame:
    num_cols = relax.select_dtypes(include=["number"]).columns.tolist()
    agg = {}
    for c in relax.columns:
        if c in KEY:
            continue
        agg[c] = "mean" if c in num_cols else "first"
    return relax.groupby(KEY, as_index=False).agg(agg)

def add_any_relax(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns

    pre_n  = "pre_relax_n"  if "pre_relax_n" in cols else None
    post_n = "post_relax_n" if "post_relax_n" in cols else None

    any_hit = pd.Series(False, index=df.index)
    any_src = pd.Series(0, index=df.index, dtype="int8")

    if pre_n:
        mpre = df[pre_n].notna()
        any_hit |= mpre
        any_src = any_src.mask(mpre, 2)

    if post_n:
        mpost_src = df[post_n].notna() & (~any_hit)
        any_hit |= df[post_n].notna()
        any_src = any_src.mask(mpost_src, 1)

    df["any_hit"] = any_hit.astype("int8")
    df["any_relax_src"] = any_src

    if pre_n and post_n:
        df["any_relax_n"] = df[pre_n].where(df[pre_n].notna(), df[post_n])
    elif pre_n:
        df["any_relax_n"] = df[pre_n]
    elif post_n:
        df["any_relax_n"] = df[post_n]

    pre_cols  = [c for c in cols if c.startswith("pre_relax_")]
    post_cols = [c for c in cols if c.startswith("post_relax_")]
    pre_map  = {c[len("pre_relax_"):]: c for c in pre_cols}
    post_map = {c[len("post_relax_"):]: c for c in post_cols}

    for suf in sorted(set(pre_map) | set(post_map)):
        pc = pre_map.get(suf)
        qc = post_map.get(suf)
        outc = f"any_relax_{suf}"
        if pc and qc:
            df[outc] = df[pc].where(df[pc].notna(), df[qc])
        elif pc:
            df[outc] = df[pc]
        elif qc:
            df[outc] = df[qc]
    return df

def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    post_n = "post_relax_n" if "post_relax_n" in cols else None
    pre_n  = "pre_relax_n"  if "pre_relax_n" in cols else None

    any_mask = (df["any_hit"] > 0) if "any_hit" in cols else pd.Series(False, index=df.index)

    rows = []
    for vin, g in df.groupby("vin"):
        rows.append({
            "vin": vin,
            "n_rows": int(len(g)),
            "cov_post": float(g[post_n].notna().mean()) if post_n else 0.0,
            "cov_pre":  float(g[pre_n].notna().mean()) if pre_n else 0.0,
            "cov_any":  float(any_mask.loc[g.index].mean()) if len(g) else 0.0,
        })
    return pd.DataFrame(rows)

def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing {INP}. Run 07D2_merge_stage_into_dataset.py first.")
    if not RELAX_PATH.exists():
        raise FileNotFoundError(f"Missing {RELAX_PATH}. Run 05B_extract_relax_features.py first.")

    base = pd.read_parquet(INP, engine="pyarrow")
    relax = pd.read_parquet(RELAX_PATH, engine="pyarrow")

    if not set(KEY).issubset(base.columns):
        raise ValueError("Base dataset missing (vin,t_start,t_end), cannot merge safely.")
    if not set(KEY).issubset(relax.columns):
        raise ValueError(f"Relax features missing keys {KEY}")

    base = _force_key_types(base)
    relax = _force_key_types(relax)
    relax = relax.drop(columns=[c for c in ["chg_seg_id", "rst_seg_id"] if c in relax.columns], errors="ignore")
    relax_agg = _dedup_relax(relax)

    out = base.merge(relax_agg, on=KEY, how="left")
    assert len(out) == len(base), "Row count changed after relax merge!"

    out = add_any_relax(out)
    out.to_parquet(OUTP, index=False, engine="pyarrow")

    rep = coverage_report(out)
    rep.to_csv(REPORT, index=False, encoding="utf-8-sig")

    print("Saved:", OUTP, "shape:", out.shape, "vins:", out["vin"].nunique())
    print("Saved report:", REPORT)

if __name__ == "__main__":
    main()
