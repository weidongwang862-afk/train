# scripts/04_compute_capacity_labels2.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import config

VINS = [Path(x).stem for x in config.ALL_FILES]

OUT_ROOT = Path(config.OUT_DIR)
EVENT_DIR = OUT_ROOT / "02_events"
CHG_DIR = OUT_ROOT / "02_segments" / "charge_points"

OUT_DIR = OUT_ROOT / "03_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- 参数（按10s采样与实车噪声设定） ----------------
MIN_DELTA_SOC = 8.0          # 充入SOC至少8%
MAX_DT = 120                 # dt超过120s视为断裂（charge_points已带dt，仍加保险）
MIN_Q_AH = 0.5               # 充入电量至少0.5Ah
C_EST_MIN_AH = 20.0
C_EST_MAX_AH = 500.0

# 静置质量标记：>=2h为高质量
REST_GOOD_SEC = 2 * 3600
REST_MATCH_WIN_SEC = 24 * 3600  # 在24h内找最近一次静置（可写论文时说明）

# charge_points 需要的列
PT_COLS = ["terminaltime", "soc", "totalcurrent", "totalodometer", "totalvoltage", "speed", "dt", "seg_id"]


def load_charge_points_one_vin(vin: str) -> pd.DataFrame:
    parts = sorted((CHG_DIR / vin).glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=PT_COLS)

    dfs = []
    for p in parts:
        df = pd.read_parquet(p, columns=[c for c in PT_COLS if c in pd.read_parquet(p, engine="pyarrow").columns])
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=PT_COLS)

    out = pd.concat(dfs, ignore_index=True)
    # 只保留 seg_id 有效点
    if "seg_id" in out.columns:
        out = out.dropna(subset=["seg_id"])
    out["seg_id"] = pd.to_numeric(out["seg_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["seg_id"])
    out["seg_id"] = out["seg_id"].astype("int64")
    out = out.sort_values(["seg_id", "terminaltime"]).reset_index(drop=True)
    return out


def integrate_one_segment(seg_df: pd.DataFrame):
    # dt 保险：负值/超大间隔不计入
    t = pd.to_numeric(seg_df["terminaltime"], errors="coerce").astype("int64")
    if "dt" in seg_df.columns:
        dt = pd.to_numeric(seg_df["dt"], errors="coerce").fillna(0).astype("int64")
    else:
        dt = t.diff().fillna(0).astype("int64")
    dt = dt.where((dt > 0) & (dt <= MAX_DT), 0)

    I = pd.to_numeric(seg_df["totalcurrent"], errors="coerce").astype("float64")
    I_chg = (-I).clip(lower=0)  # 充电电流转正

    Q_ah = float((I_chg * dt).sum() / 3600.0)

    soc = pd.to_numeric(seg_df["soc"], errors="coerce").astype("float64")
    soc_s = float(soc.iloc[0])
    soc_e = float(soc.iloc[-1])
    d_soc = float(soc_e - soc_s)

    odo = pd.to_numeric(seg_df["totalodometer"], errors="coerce").astype("float64")
    odo_end = float(odo.iloc[-1])

    v = pd.to_numeric(seg_df["totalvoltage"], errors="coerce").astype("float64")
    v_end = float(v.iloc[-1])

    t0 = int(t.iloc[0])
    t1 = int(t.iloc[-1])
    dur = float(t1 - t0)

    return Q_ah, soc_s, soc_e, d_soc, odo_end, v_end, t0, t1, dur


def match_rest(rest_df: pd.DataFrame, t_start: int):
    if rest_df is None or rest_df.empty:
        return None
    cand = rest_df[rest_df["t_end"] <= t_start].copy()
    if cand.empty:
        return None
    cand["gap"] = t_start - cand["t_end"]
    cand = cand[(cand["gap"] >= 0) & (cand["gap"] <= REST_MATCH_WIN_SEC)]
    if cand.empty:
        return None
    return cand.sort_values("gap").iloc[0]


def compute_one_vin(vin: str):
    # 基础事件（由03B从 segment_summary 派生）
    f_charge_ev = EVENT_DIR / f"charge_events_{vin}.parquet"
    f_rest_ev = EVENT_DIR / f"rest_events_{vin}.parquet"
    if not f_charge_ev.exists():
        print(vin, "MISSING charge_events")
        return pd.DataFrame()

    charge_ev = pd.read_parquet(f_charge_ev)
    rest_ev = pd.read_parquet(f_rest_ev) if f_rest_ev.exists() else pd.DataFrame()

    # 读该车所有 charge_points（仅充电点，量显著小于全量）
    pts = load_charge_points_one_vin(vin)
    if pts.empty:
        print(vin, "NO charge_points")
        return pd.DataFrame()

    rows = []
    # charge_events 里有 seg_id（来自segment_summary），我们用 seg_id 精确对齐
    if "seg_id" not in charge_ev.columns:
        print(vin, "charge_events has no seg_id -> SKIP")
        return pd.DataFrame()

    for ev in tqdm(charge_ev.itertuples(index=False), total=len(charge_ev), desc=f"Capacity {vin}"):
        sid = int(getattr(ev, "seg_id"))
        seg_df = pts[pts["seg_id"] == sid]
        if len(seg_df) < 5:
            continue

        # 基础有效性：电压应为正、速度应接近0（若存在速度列）
        if "totalvoltage" in seg_df.columns:
            seg_df = seg_df[seg_df["totalvoltage"] > 0]
            if len(seg_df) < 5:
                continue

        Q_ah, soc_s, soc_e, d_soc, odo_end, v_end, t0, t1, dur = integrate_one_segment(seg_df)
        if d_soc < MIN_DELTA_SOC or Q_ah < MIN_Q_AH:
            continue

        C_est = Q_ah / (d_soc / 100.0)
        if (C_est < C_EST_MIN_AH) or (C_est > C_EST_MAX_AH):
            continue

        # 匹配静置事件（用于质量标记）
        r = match_rest(rest_ev, t0)
        if r is None:
            quality = "C_no_rest"
            rest_gap = np.nan
            rest_dur = np.nan
        else:
            rest_gap = float(r["gap"])
            rest_dur = float(r["duration_s"]) if "duration_s" in r.index else float(r.get("duration_s", np.nan))
            quality = "A_rest>=2h" if (np.isfinite(rest_dur) and rest_dur >= REST_GOOD_SEC) else "B_rest<2h"

        rows.append({
            "vin": vin,
            "seg_id": sid,
            "t_start": int(t0),
            "t_end": int(t1),
            "duration_s": float(dur),
            "soc_start": float(soc_s),
            "soc_end": float(soc_e),
            "delta_soc": float(d_soc),
            "Q_ah": float(Q_ah),
            "C_est_ah": float(C_est),
            "odo_end": float(odo_end),
            "v_end": float(v_end),
            "quality_flag": quality,
            "rest_gap_s": rest_gap,
            "rest_duration_s": rest_dur,
        })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"capacity_labels_{vin}.parquet"
    out.to_parquet(out_path, index=False)
    print(vin, "labels:", len(out), "saved:", out_path)
    return out


def main():
    all_rows = []
    errs = []

    for vin in VINS:
        try:
            df = compute_one_vin(vin)
            if df is not None and len(df):
                all_rows.append(df)
        except Exception as e:
            errs.append({"vin": vin, "error": repr(e)})
            print(f"[ERROR] {vin}: {e}")

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_parquet(OUT_DIR / "all_capacity_labels.parquet", index=False)
        print("ALL labels:", len(all_df))

    if errs:
        pd.DataFrame(errs).to_csv(OUT_DIR / "capacity_label_errors.csv", index=False)
        print("Errors:", OUT_DIR / "capacity_label_errors.csv")


if __name__ == "__main__":
    main()
