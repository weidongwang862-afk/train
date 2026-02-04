# scripts/05_extract_core_features.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import config
except Exception:
    config = None


OUT_ROOT = Path(config.OUT_DIR) if config is not None else Path("outputs")

IDX_DIR   = OUT_ROOT / "03_labels"
LABEL_DIR = OUT_ROOT / "03_labels"
CHG_DIR   = OUT_ROOT / "02_segments" / "charge_points"
OUT_DIR   = OUT_ROOT / "04_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

READ_COLS = [
    "terminaltime", "soc", "speed", "totalodometer",
    "totalvoltage", "totalcurrent",
    "mintemperaturevalue", "maxtemperaturevalue",
    "minvoltagebattery", "maxvoltagebattery",
    "dt", "seg_id"
]
MAX_DT = 120


def _read_vin_txt(path: str | Path) -> list[str]:
    p = Path(path)
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _get_vins_from_env_or_config():
    vin_list = os.environ.get("VIN_LIST", "").strip()
    train_list = os.environ.get("TRAIN_LIST", "").strip()
    test_list  = os.environ.get("TEST_LIST", "").strip()

    if vin_list:
        all_vins = _read_vin_txt(vin_list)
        if train_list and test_list:
            train_vins = _read_vin_txt(train_list)
            test_vins  = _read_vin_txt(test_list)
        else:
            train_vins, test_vins = [], []
        return all_vins, train_vins, test_vins

    if config is None:
        raise RuntimeError("未找到 VIN_LIST 环境变量，也无法 import config.py。")

    def _stem(x: str) -> str:
        return Path(x).stem

    all_vins = [_stem(x) for x in getattr(config, "ALL_FILES", [])]
    train_vins = [_stem(x) for x in getattr(config, "TRAIN_FILES", [])]
    test_vins  = [_stem(x) for x in getattr(config, "TEST_FILES", [])]
    return all_vins, train_vins, test_vins


ALL_VINS, TRAIN_VINS, TEST_VINS = _get_vins_from_env_or_config()

@lru_cache(maxsize=12)
def read_part(part_path: str) -> pd.DataFrame:
    pf = pq.ParquetFile(part_path)
    cols = [c for c in READ_COLS if c in pf.schema.names]
    return pd.read_parquet(part_path, columns=cols)


def load_event_points(parts_idx: pd.DataFrame, t0: int, t1: int) -> pd.DataFrame:
    hit = parts_idx[(parts_idx["t_max"] >= t0) & (parts_idx["t_min"] <= t1)]
    chunks = []
    for p in hit["part_path"].tolist():
        df = read_part(p)
        sub = df[(df["terminaltime"] >= t0) & (df["terminaltime"] <= t1)]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=READ_COLS)

    ev = pd.concat(chunks, ignore_index=True).sort_values("terminaltime")
    return ev


def safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def extract_ic_features(t: np.ndarray, I: np.ndarray, V_total: np.ndarray, V_cell: np.ndarray | None, max_dt: int = 120) -> dict:
    out = {}
    if len(t) < 10:
        return out

    dt = np.diff(t)
    ok = np.isfinite(dt) & (dt > 0) & (dt <= max_dt)
    if ok.sum() < 5:
        return out

    t2 = t[1:][ok]
    I2 = I[1:][ok]
    V2 = V_total[1:][ok]
    if len(t2) < 10:
        return out

    Q = np.cumsum(I2 * np.diff(np.r_[t[0], t2])) / 3600.0
    dQ = np.gradient(Q)
    dV = np.gradient(V2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dQdV = np.where(np.abs(dV) > 1e-6, dQ / dV, np.nan)

    out["ic_mean"] = float(np.nanmean(dQdV))
    out["ic_std"] = float(np.nanstd(dQdV))
    out["ic_p95"] = float(np.nanpercentile(dQdV, 95)) if np.isfinite(dQdV).any() else float("nan")

    if V_cell is not None:
        Vc2 = V_cell[1:][ok]
        dVc = np.gradient(Vc2)
        with np.errstate(divide="ignore", invalid="ignore"):
            dQdVc = np.where(np.abs(dVc) > 1e-6, dQ / dVc, np.nan)
        out["ic_cell_mean"] = float(np.nanmean(dQdVc))
        out["ic_cell_std"] = float(np.nanstd(dQdVc))
    return out


def extract_features(ev: pd.DataFrame) -> dict:
    t = ev["terminaltime"].astype("int64").to_numpy()
    I = ev["totalcurrent"].astype("float64").to_numpy()
    V = ev["totalvoltage"].astype("float64").to_numpy()

    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(V)
    if ok.sum() < 10:
        return {}

    t = t[ok]; I = I[ok]; V = V[ok]
    if len(t) < 10:
        return {}

    dt = np.diff(t)
    good = np.isfinite(dt) & (dt > 0) & (dt <= MAX_DT)
    if good.sum() < 5:
        return {}

    dur = float(t[-1] - t[0])
    I_mean = float(np.nanmean(I))
    V_mean = float(np.nanmean(V))
    I_std = float(np.nanstd(I))
    V_std = float(np.nanstd(V))

    end_I_ratio = float(np.nanmean(I[-max(5, len(I)//20):]) / (np.nanmean(I[:max(5, len(I)//20)]) + 1e-6))

    V_slope = float(np.polyfit(t.astype("float64"), V, 1)[0]) if len(t) >= 20 else float("nan")
    I_slope = float(np.polyfit(t.astype("float64"), I, 1)[0]) if len(t) >= 20 else float("nan")

    cc_frac = float(np.mean(np.abs(np.diff(I)) < 1.0)) if len(I) > 2 else float("nan")
    cv_frac = float(np.mean(np.abs(np.diff(V)) < 0.2)) if len(V) > 2 else float("nan")

    out = {
        "dur_s": dur,
        "I_mean": I_mean,
        "V_mean": V_mean,
        "I_std": I_std,
        "V_std": V_std,
        "cc_frac": cc_frac,
        "cv_frac": cv_frac,
        "end_I_ratio": end_I_ratio,
        "V_slope": V_slope,
        "I_slope": I_slope,
    }

    V_cell = None
    if "minvoltagebattery" in ev.columns and "maxvoltagebattery" in ev.columns:
        minv = ev["minvoltagebattery"].astype("float64").to_numpy()
        maxv = ev["maxvoltagebattery"].astype("float64").to_numpy()
        if np.isfinite(minv).any() and np.isfinite(maxv).any():
            V_cell = 0.5 * (minv + maxv)

    out.update(extract_ic_features(t, I, V_total=V, V_cell=V_cell, max_dt=MAX_DT))
    return out


def main():
    split_mode = os.environ.get("SPLIT_MODE", "all").strip().lower()
    # split_mode:
    # - "all": 不切分，输出 dataset_all.parquet（同时 dataset_train=all, dataset_test=empty 兼容旧脚本）
    # - "split": 若 TRAIN_LIST/TEST_LIST 不给，则按 80/20 自动分 VIN
    # - "nosplit": 只输出 dataset_all.parquet，不写 dataset_train/test

    all_feat = []

    for vin in ALL_VINS:
        f_labels = LABEL_DIR / f"labels_post_{vin}.parquet"
        f_idx = IDX_DIR / f"part_index_core_{vin}.parquet"

        if (not f_labels.exists()) or (not f_idx.exists()):
            print(vin, "SKIP missing labels_post or part_index")
            continue

        labels = pd.read_parquet(f_labels)
        if labels.empty:
            print(vin, "labels_post EMPTY")
            continue

        parts_idx = pd.read_parquet(f_idx)

        rows = []
        for r in tqdm(labels.itertuples(index=False), total=len(labels), desc=f"Features {vin}"):
            try:
                t0 = int(getattr(r, "t_start"))
                t1 = int(getattr(r, "t_end"))
            except Exception:
                continue

            ev = load_event_points(parts_idx, t0, t1)
            if ev.empty:
                continue

            feats = extract_features(ev)

            rec = {"vin": str(vin), "t_start": t0, "t_end": t1}
            for col in ["odo_end", "SoH_trend", "C_trend", "C0_trend", "quality_flag"]:
                if hasattr(r, col):
                    rec[col] = getattr(r, col)
            rec.update(feats)
            rows.append(rec)

        if not rows:
            print(vin, "NO feature rows")
            continue

        data = pd.DataFrame(rows)
        out_vin = OUT_DIR / f"core_{vin}.parquet"   # 文件名更简洁
        data.to_parquet(out_vin, index=False)
        print(vin, "rows=", len(data), "saved:", out_vin.name)

        all_feat.append(data)

    if not all_feat:
        raise RuntimeError("No features generated. Check labels_post and part_index paths.")

    all_df = pd.concat(all_feat, ignore_index=True)

    # ===== 永远写全量 =====
    (OUT_DIR / "features_core_all.parquet").write_bytes(b"")  # 占位，避免误读旧文件
    all_df.to_parquet(OUT_DIR / "features_core_all.parquet", index=False)
    all_df.to_parquet(OUT_DIR / "dataset_all.parquet", index=False)

    # ===== 是否写 train/test =====
    if split_mode == "nosplit":
        print("Saved: dataset_all.parquet only")
        print("ALL rows:", len(all_df), "vins:", all_df["vin"].nunique(), "IC enabled SciPy:", _HAS_SCIPY)
        return

    if split_mode == "split":
        if (not TRAIN_VINS) and (not TEST_VINS):
            vins = sorted(all_df["vin"].astype(str).unique().tolist())
            cut = int(len(vins) * 0.8)
            train_vins = set(vins[:cut])
            test_vins  = set(vins[cut:])
        else:
            train_vins = set([str(x) for x in TRAIN_VINS])
            test_vins  = set([str(x) for x in TEST_VINS])

        train_df = all_df[all_df["vin"].astype(str).isin(train_vins)].copy()
        test_df  = all_df[all_df["vin"].astype(str).isin(test_vins)].copy()

        train_df.to_parquet(OUT_DIR / "dataset_train.parquet", index=False)
        test_df.to_parquet(OUT_DIR / "dataset_test.parquet", index=False)

        print("ALL:", len(all_df), "TRAIN:", len(train_df), "TEST:", len(test_df))
        print("Train vins:", len(train_vins), "Test vins:", len(test_vins), "IC enabled SciPy:", _HAS_SCIPY)
        return

    # split_mode == "all"
    all_df.to_parquet(OUT_DIR / "dataset_train.parquet", index=False)
    pd.DataFrame(columns=all_df.columns).to_parquet(OUT_DIR / "dataset_test.parquet", index=False)

    print("Saved: dataset_all.parquet + dataset_train(par=all) + dataset_test(empty)")
    print("ALL rows:", len(all_df), "vins:", all_df["vin"].nunique(), "IC enabled SciPy:", _HAS_SCIPY)


if __name__ == "__main__":
    main()
