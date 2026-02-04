# -*- coding: utf-8 -*-
"""
Step 4.5  Postprocess capacity labels -> produce smoothed/physical SoH trend.

适配 30车（后续 300车也一样）：
- 从 scripts/config.py 读取 VIN 列表（config.ALL_FILES）
- C0_trend 采用“早期窗口 + 上分位数/TopK”稳健估计，减少 C0 低估导致的 SoH>1 偏置
- C_trend 对 odo（或时间）施加单调不增约束：容量随里程不应上升
- 可选：若车辆数据起点里程较低，但早期 SoH 中位数仍明显 >1，则上调 C0 使早期 SoH≈1

输出：
  outputs/03_labels/labels_post_<vin>.parquet
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config


OUT_DIR = Path(config.OUT_DIR)
LABEL_DIR = OUT_DIR / "03_labels"

VINS = [Path(f).stem for f in config.ALL_FILES]

# ------------------- 参数（后续可再调） -------------------
MIN_DELTA_SOC = 15.0
MIN_DURATION_S = 10 * 60
MAX_DURATION_S = 12 * 3600
C_EST_MIN_AH = 20.0
C_EST_MAX_AH = 500.0

# C0 估计策略（关键）
BASE_N = 200          # 取最早（按里程）前 N 条事件作为“早期窗口”
TOPK = 30             # 早期窗口内取 TopK 的中位数
C0_Q = 0.90           # 早期窗口内取 90% 分位数
PREFER_QUALITY = True # 优先使用 quality_flag 含 "A_" 的事件（若有该列）

# 趋势平滑
LOWESS_FRAC = 0.12
HAMP_WIN = 9
HAMP_NSIG = 3.0

# “新车段”归一化修正（主要针对 vin25 这种 SoH 整体偏高）
NEW_CAR_ODO_P05_MAX = 50_000.0
EARLY_MED_SOHR_MAX = 1.02


def _get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def hampel_filter(x: pd.Series, window: int = 9, n_sig: float = 3.0) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if len(s) < max(5, window):
        return s

    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    thr = n_sig * 1.4826 * mad

    out = s.copy()
    mask = (s - med).abs() > thr
    out[mask] = med[mask]
    return out


def smooth_trend_lowess(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    y = pd.to_numeric(pd.Series(y), errors="coerce").astype(float).values
    if len(y) < 10:
        return y

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
        return np.asarray(lowess(y, x, frac=frac, return_sorted=False), dtype=float)
    except Exception:
        return pd.Series(y).rolling(21, center=True, min_periods=1).median().astype(float).values


def enforce_monotone_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    物理约束：容量趋势随里程（或时间）单调不增。
    优先 isotonic regression；失败则用 cumulative minimum 兜底。
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce").astype(float).values
    if len(y) < 3:
        return y

    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        return np.asarray(ir.fit_transform(x, y), dtype=float)
    except Exception:
        return np.minimum.accumulate(y)


def estimate_c0(
    df: pd.DataFrame,
    cap_col: str,
    key_col: str,
    base_n: int = 200,
    topk: int = 30,
    q: float = 0.90,
    prefer_quality: bool = True,
) -> float:
    """
    C0（基准容量）稳健估计：
    在“早期窗口（最早 base_n 条）”里取 max(quantile(q), median(top-k))，
    这能明显降低 C0 被“早期不完整充电”拉低的问题。
    """
    if cap_col not in df.columns:
        return float("nan")

    base = df.copy()

    if prefer_quality and ("quality_flag" in base.columns):
        good = base[base["quality_flag"].astype(str).str.contains("A_", na=False)]
        if len(good) >= 20:
            base = good

    base = base.sort_values(key_col).head(min(base_n, len(base)))
    vals = pd.to_numeric(base[cap_col], errors="coerce").dropna().values.astype(float)
    if len(vals) == 0:
        return float("nan")

    vals.sort()
    qv = float(np.quantile(vals, q))
    top = vals[-min(topk, len(vals)):]
    top_med = float(np.median(top))
    return float(max(qv, top_med))


def main() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for vin in VINS:
        src = LABEL_DIR / f"capacity_labels_{vin}.parquet"
        if not src.exists():
            continue

        df = pd.read_parquet(src)

        ds_col = _get_col(df, ["delta_soc", "delta_soc_feat"])
        dur_col = _get_col(df, ["duration_s", "duration_s_feat"])
        odo_col = _get_col(df, ["odo_end", "totalodometer_end", "totalodometer", "odo"])
        key_col = odo_col if odo_col else "t_end"

        # --- basic filters ---
        mask = pd.Series(True, index=df.index)

        if ds_col:
            mask &= pd.to_numeric(df[ds_col], errors="coerce").between(MIN_DELTA_SOC, 100.0)
        if dur_col:
            mask &= pd.to_numeric(df[dur_col], errors="coerce").between(MIN_DURATION_S, MAX_DURATION_S)
        if "C_est_ah" in df.columns:
            mask &= pd.to_numeric(df["C_est_ah"], errors="coerce").between(C_EST_MIN_AH, C_EST_MAX_AH)

        df = df.loc[mask].copy()
        if len(df) == 0:
            print(f"{vin}: 0 rows after filters, skip")
            continue

        # numeric columns
        for c in ["t_start", "t_end"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        if odo_col and odo_col in df.columns:
            df[odo_col] = pd.to_numeric(df[odo_col], errors="coerce").astype(float)

        # sort by key (odo or time)
        df = df.sort_values(key_col).reset_index(drop=True)

        # --- Hampel on capacity ---
        if "C_est_ah" in df.columns:
            df["C_hampel"] = hampel_filter(df["C_est_ah"], window=HAMP_WIN, n_sig=HAMP_NSIG)
        else:
            df["C_hampel"] = np.nan

        # x-axis for smoothing / monotone
        if odo_col and odo_col in df.columns:
            x = df[odo_col].astype(float).values
        else:
            x = np.arange(len(df), dtype=float)

        # --- smooth + monotone ---
        df["C_smooth"] = smooth_trend_lowess(x, df["C_hampel"].values, frac=LOWESS_FRAC)
        df["C_trend"] = enforce_monotone_decreasing(x, df["C_smooth"].values)

        # --- robust C0_trend ---
        C0_trend = estimate_c0(
            df=df,
            cap_col="C_trend",
            key_col=key_col,
            base_n=BASE_N,
            topk=TOPK,
            q=C0_Q,
            prefer_quality=PREFER_QUALITY,
        )
        if (not np.isfinite(C0_trend)) or (C0_trend <= 0):
            print(f"{vin}: bad C0_trend={C0_trend}, skip")
            continue

        df["C0_trend"] = float(C0_trend)
        df["SoH_trend"] = df["C_trend"] / float(C0_trend)

        # --- optional renormalization (fix vin25-like global offset) ---
        try:
            odo_p05 = float(df[key_col].quantile(0.05)) if key_col in df.columns else 0.0
        except Exception:
            odo_p05 = 0.0

        early_n = min(100, len(df))
        early_med = float(pd.to_numeric(df.loc[:early_n - 1, "SoH_trend"], errors="coerce").median())

        if (odo_p05 < NEW_CAR_ODO_P05_MAX) and np.isfinite(early_med) and (early_med > EARLY_MED_SOHR_MAX):
            C0_trend2 = float(C0_trend) * float(early_med)
            df["C0_trend"] = C0_trend2
            df["SoH_trend"] = df["C_trend"] / C0_trend2
            C0_trend = C0_trend2

        # flags for diagnosis
        df["flag_soh_outlier"] = (df["SoH_trend"] > 1.08) | (df["SoH_trend"] < 0.60)

        df["vin"] = vin
        outp = LABEL_DIR / f"labels_post_{vin}.parquet"
        df.to_parquet(outp, index=False)

        print(
            f"{vin} post rows= {len(df)} "
            f"C0_trend= {float(C0_trend):.3f} early_SoH_med= {early_med:.3f} saved: {outp}"
        )
        ok += 1

    print(f"DONE. ok={ok} / total={len(VINS)}")


if __name__ == "__main__":
    main()
