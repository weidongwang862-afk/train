# scripts/07C2_extract_stage_features.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import config

BASE_OUT = Path(config.OUT_DIR)
AUX_DIR  = BASE_OUT / "05_aux_stage"
LABEL_DIR= BASE_OUT / "03_labels"
FEAT_DIR = BASE_OUT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

VINS = [Path(f).stem for f in config.ALL_FILES]

COL_T="terminaltime"; COL_SOC="soc"; COL_IABS="I_abs"

I_CC_THR = 20.0
SOC_HI_THR = 90.0
EPS = 1e-3

def _fix_path(vin: str, p: str) -> Path:
    pp = Path(p)
    if pp.exists():
        return pp
    return AUX_DIR / vin / pp.name

def load_aux_window(vin, t0, t1):
    idx_path = AUX_DIR / vin / "aux_part_index.parquet"
    if not idx_path.exists():
        return pd.DataFrame()

    idx = pd.read_parquet(idx_path)
    hit = idx[(idx["tmax"] >= t0) & (idx["tmin"] <= t1)].sort_values("part_id")
    if len(hit) == 0:
        return pd.DataFrame()

    frames = []
    for p in hit["path"].tolist():
        fp = _fix_path(vin, p)
        if fp.exists():
            frames.append(pd.read_parquet(fp))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[(df[COL_T] >= t0) & (df[COL_T] <= t1)].sort_values(COL_T)
    return df

def _stats(arr: np.ndarray):
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return dict(mean=np.nan, p95=np.nan, p50=np.nan, iqr=np.nan)
    q25, q50, q95 = np.percentile(x, [25, 50, 95])
    return dict(mean=float(np.mean(x)), p95=float(q95), p50=float(q50), iqr=float(q50 - q25))

def main():
    for vin in VINS:
        labels_path = LABEL_DIR / f"labels_post_{vin}.parquet"
        if not labels_path.exists():
            continue

        lab = pd.read_parquet(labels_path).sort_values("t_start")
        rows=[]

        for _, e in lab.iterrows():
            t0 = int(e["t_start"]); t1 = int(e["t_end"])
            w = load_aux_window(vin, t0, t1)

            rec = {"vin": vin, "t_start": t0, "t_end": t1}
            if w.empty:
                rows.append(rec)
                continue

            # 有效占比
            rec["aux_n"] = int(len(w))
            rec["dv_valid_frac"] = float(np.isfinite(w["dv_cell"]).mean()) if "dv_cell" in w.columns else np.nan
            rec["dT_valid_frac"] = float(np.isfinite(w["dT_probe"]).mean()) if "dT_probe" in w.columns else np.nan

            # 阶段切片
            cc = w[w[COL_IABS] >= I_CC_THR] if (COL_IABS in w.columns) else w.iloc[0:0]
            hi = w[w[COL_SOC] >= SOC_HI_THR] if (COL_SOC in w.columns) else w.iloc[0:0]

            rec["cc_frac"] = float(len(cc) / len(w)) if len(w) else np.nan
            rec["hi_soc_frac"] = float(len(hi) / len(w)) if len(w) else np.nan

            # dv/dT 统计
            if len(cc):
                dv_cc = _stats(cc["dv_cell"].to_numpy(dtype="float64"))
                dT_cc = _stats(cc["dT_probe"].to_numpy(dtype="float64"))
                rec.update({f"dv_cc_{k}": v for k, v in dv_cc.items()})
                rec.update({f"dT_cc_{k}": v for k, v in dT_cc.items()})

                ratio = (cc["dv_cell"] / (cc[COL_IABS] + EPS)).to_numpy(dtype="float64")
                dvI = _stats(ratio)
                rec.update({f"dvI_cc_{k}": v for k, v in dvI.items()})

            if len(hi):
                dv_hi = _stats(hi["dv_cell"].to_numpy(dtype="float64"))
                dT_hi = _stats(hi["dT_probe"].to_numpy(dtype="float64"))
                rec.update({f"dv_hi_{k}": v for k, v in dv_hi.items()})
                rec.update({f"dT_hi_{k}": v for k, v in dT_hi.items()})

            # 温度上升（控制热管理差异）
            if "maxtemperaturevalue" in w.columns:
                rec["Tmax_rise"] = float(w["maxtemperaturevalue"].iloc[-1] - w["maxtemperaturevalue"].iloc[0])
            else:
                rec["Tmax_rise"] = np.nan

            rows.append(rec)

        outp = FEAT_DIR / f"features_stage_{vin}.parquet"
        pd.DataFrame(rows).to_parquet(outp, index=False)
        print(vin, "saved:", outp)

if __name__ == "__main__":
    main()
