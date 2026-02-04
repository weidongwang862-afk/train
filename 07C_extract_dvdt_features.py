# scripts/07C_extract_dvdt_features.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import config

BASE_OUT = Path(config.OUT_DIR)
AUX_DIR  = BASE_OUT / "05_aux_dvdt"
LABEL_DIR= BASE_OUT / "03_labels"
FEAT_DIR = BASE_OUT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

VINS = [Path(f).stem for f in config.ALL_FILES]

TAIL_SECONDS = 600  # 尾段10分钟，可改 300/900
COL_T = "terminaltime"

def _fix_path(vin: str, p: str) -> Path:
    pp = Path(p)
    if pp.exists():
        return pp
    # fallback：按当前环境重拼
    return AUX_DIR / vin / pp.name

def load_aux_window(vin, t0, t1):
    idx_path = AUX_DIR / vin / "aux_part_index.parquet"
    if not idx_path.exists():
        return pd.DataFrame(columns=[COL_T, "dv_cell", "dT_probe"])

    idx = pd.read_parquet(idx_path)
    hit = idx[(idx["tmax"] >= t0) & (idx["tmin"] <= t1)].sort_values("part_id")
    if len(hit) == 0:
        return pd.DataFrame(columns=[COL_T, "dv_cell", "dT_probe"])

    frames = []
    for p in hit["path"].tolist():
        fp = _fix_path(vin, p)
        if fp.exists():
            frames.append(pd.read_parquet(fp))
    if not frames:
        return pd.DataFrame(columns=[COL_T, "dv_cell", "dT_probe"])

    df = pd.concat(frames, ignore_index=True)
    df = df[(df[COL_T] >= t0) & (df[COL_T] <= t1)].sort_values(COL_T)
    return df

def _robust_stats(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, p95=np.nan, p50=np.nan, iqr=np.nan)
    q25, q50, q95 = np.percentile(x, [25, 50, 95])
    return dict(mean=float(np.mean(x)), p95=float(q95), p50=float(q50), iqr=float(q50 - q25))

def _slope(t: np.ndarray, y: np.ndarray):
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    tt = (t[m] - t[m][0]).astype("float64")
    yy = y[m].astype("float64")
    return float(np.polyfit(tt, yy, 1)[0])

def agg_tail(df: pd.DataFrame):
    if df.empty:
        return {
            "tail_n": 0,
            "dv_valid_frac": np.nan, "dT_valid_frac": np.nan,
            "dv_end": np.nan, "dv_slope": np.nan, "dv_mean": np.nan, "dv_p95": np.nan, "dv_p50": np.nan, "dv_iqr": np.nan,
            "dT_end": np.nan, "dT_slope": np.nan, "dT_mean": np.nan, "dT_p95": np.nan, "dT_p50": np.nan, "dT_iqr": np.nan,
        }

    t = df[COL_T].to_numpy(dtype="int64")
    dv = df["dv_cell"].to_numpy(dtype="float64")
    dT = df["dT_probe"].to_numpy(dtype="float64")

    res = {"tail_n": int(len(df))}
    res["dv_valid_frac"] = float(np.isfinite(dv).mean())
    res["dT_valid_frac"] = float(np.isfinite(dT).mean())

    res["dv_end"] = float(dv[np.isfinite(dv)][-1]) if np.isfinite(dv).any() else np.nan
    res["dT_end"] = float(dT[np.isfinite(dT)][-1]) if np.isfinite(dT).any() else np.nan

    dv_s = _robust_stats(dv); dT_s = _robust_stats(dT)
    res.update({f"dv_{k}": v for k, v in dv_s.items()})
    res.update({f"dT_{k}": v for k, v in dT_s.items()})

    res["dv_slope"] = _slope(t, dv)
    res["dT_slope"] = _slope(t, dT)
    return res

def main():
    for vin in VINS:
        labels_path = LABEL_DIR / f"labels_post_{vin}.parquet"
        if not labels_path.exists():
            continue

        lab = pd.read_parquet(labels_path).sort_values("t_start").reset_index(drop=True)
        rows = []
        for _, e in lab.iterrows():
            t_start = int(e["t_start"])
            t_end = int(e["t_end"])
            t0 = max(t_start, t_end - TAIL_SECONDS)

            w = load_aux_window(vin, t0, t_end)
            f = agg_tail(w)
            f.update({"vin": vin, "t_start": t_start, "t_end": t_end})
            rows.append(f)

        if rows:
            df = pd.DataFrame(rows)
            outp = FEAT_DIR / f"features_dvdt_{vin}.parquet"
            df.to_parquet(outp, index=False)
            print(vin, "rows=", len(df), "saved:", outp)

if __name__ == "__main__":
    main()
