# scripts/05B_extract_relax_features.py
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import config

OUT = Path(config.OUT_DIR)

PAIR_DIR = OUT / "02_events"        # charge_rest_pairs_<vin>.parquet
SEG_DIR  = OUT / "02_segments"      # segment_summary_<vin>.parquet
IDX_DIR  = OUT / "03_labels"        # part_index_core_<vin>.parquet
OUT_DIR  = OUT / "04_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# core columns
TCOL = "terminaltime"
VCOL = "totalvoltage"
TMIN = "mintemperaturevalue"
TMAX = "maxtemperaturevalue"
READ_COLS = [TCOL, VCOL, TMIN, TMAX]

# Level-1 windows (s)
WIN_SEC = [30, 120, 600, 1800]
MAX_WIN = int(max(WIN_SEC))

MIN_PTS_MAP = {30: 3, 120: 6, 600: 6, 1800: 6}

PRE_GAP_MAX_S = int(os.environ.get("PRE_GAP_MAX_S", str(2 * 3600)))
PRE_MIN_REST_S = 30 * 60  # rest duration >= 30min


def _read_vin_txt(path: str | Path) -> list[str]:
    p = Path(path)
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def vins() -> list[str]:
    """
    优先使用环境变量 VIN_LIST；否则退回 config.ALL_FILES。
    这样 05B 和 05/07D 系列跑同一批 VIN，不会“特征提了 150 但合并还是 50”。
    """
    vin_list = os.environ.get("VIN_LIST", "").strip()
    if vin_list:
        # 允许相对路径：相对 E:\RAW_DATA 来解析最稳
        p = Path(vin_list)
        if not p.is_absolute():
            p2 = Path(r"E:\RAW_DATA") / p
            p = p2 if p2.exists() else p
        if not p.exists():
            raise FileNotFoundError(f"VIN_LIST not found: {p}")
        return _read_vin_txt(p)

    return [Path(f).stem for f in config.ALL_FILES]


def load_pairs(vin: str) -> pd.DataFrame:
    fp = PAIR_DIR / f"charge_rest_pairs_{vin}.parquet"
    df = pd.read_parquet(fp, engine="pyarrow")
    need = ["chg_seg_id", "chg_t_start", "chg_t_end", "rst_seg_id", "rst_t_start", "rst_t_end", "rst_duration_s"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{fp.name} missing columns: {miss}")
    for c in ["chg_seg_id", "rst_seg_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["chg_t_start", "chg_t_end", "rst_t_start", "rst_t_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["rst_duration_s"] = pd.to_numeric(df["rst_duration_s"], errors="coerce")
    return df.dropna(subset=["chg_t_start", "chg_t_end", "rst_t_start", "rst_t_end"]).reset_index(drop=True)


def load_index_core(vin: str) -> pd.DataFrame:
    fp = IDX_DIR / f"part_index_core_{vin}.parquet"
    idx = pd.read_parquet(fp, engine="pyarrow")
    for need in ["part_path", "t_min", "t_max"]:
        if need not in idx.columns:
            raise ValueError(f"{fp.name} missing {need}, cols={idx.columns.tolist()}")
    idx["t_min"] = pd.to_numeric(idx["t_min"], errors="coerce").astype("int64")
    idx["t_max"] = pd.to_numeric(idx["t_max"], errors="coerce").astype("int64")
    idx["part_path"] = idx["part_path"].astype(str)
    return idx.sort_values("t_min").reset_index(drop=True)


def load_segment_summary(vin: str) -> pd.DataFrame:
    fp = SEG_DIR / f"segment_summary_{vin}.parquet"
    seg = pd.read_parquet(fp, engine="pyarrow")
    need = ["seg_id", "state", "t_start", "t_end", "duration_s"]
    miss = [c for c in need if c not in seg.columns]
    if miss:
        raise ValueError(f"{fp.name} missing columns: {miss}")
    seg["seg_id"] = pd.to_numeric(seg["seg_id"], errors="coerce").astype("int64")
    seg["t_start"] = pd.to_numeric(seg["t_start"], errors="coerce").astype("int64")
    seg["t_end"] = pd.to_numeric(seg["t_end"], errors="coerce").astype("int64")
    seg["duration_s"] = pd.to_numeric(seg["duration_s"], errors="coerce")
    seg["state"] = seg["state"].astype(str).str.lower()
    return seg


def _hit_parts(idx: pd.DataFrame, t0: int, t1: int) -> pd.DataFrame:
    return idx[(idx["t_max"] >= t0) & (idx["t_min"] <= t1)]


def read_window(idx: pd.DataFrame, t0: int, t1: int) -> pd.DataFrame:
    hit = _hit_parts(idx, t0, t1)
    if hit.empty:
        return pd.DataFrame(columns=READ_COLS)

    chunks = []
    for p in hit["part_path"].tolist():
        pf = pq.ParquetFile(p)
        df = pf.read(columns=READ_COLS).to_pandas()
        df[TCOL] = pd.to_numeric(df[TCOL], errors="coerce")
        df[VCOL] = pd.to_numeric(df[VCOL], errors="coerce")
        df = df.dropna(subset=[TCOL, VCOL])
        sub = df[(df[TCOL] >= t0) & (df[TCOL] <= t1)]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=READ_COLS)
    return pd.concat(chunks, ignore_index=True).sort_values(TCOL)


def relax_L1(ev: pd.DataFrame, anchor_t0: int, prefix: str) -> dict:
    if ev.empty or len(ev) < 3:
        return {}

    t = pd.to_numeric(ev[TCOL], errors="coerce").to_numpy(dtype="float64")
    V = pd.to_numeric(ev[VCOL], errors="coerce").to_numpy(dtype="float64")
    m = np.isfinite(V)
    if m.sum() < 3:
        return {}

    i0 = int(np.argmax(m))
    V0 = float(V[i0])
    t_first = float(t[i0])

    out = {f"{prefix}relax_n": int(len(ev))}

    if TMIN in ev.columns and TMAX in ev.columns:
        Tm = pd.to_numeric(ev[TMIN], errors="coerce").to_numpy(dtype="float64")
        Tx = pd.to_numeric(ev[TMAX], errors="coerce").to_numpy(dtype="float64")
        out[f"{prefix}relax_T0"] = float(0.5 * (Tm[i0] + Tx[i0])) if np.isfinite(Tm[i0]) and np.isfinite(Tx[i0]) else np.nan
        out[f"{prefix}relax_dT"] = float(np.nanmax(Tx) - np.nanmin(Tm)) if (np.isfinite(Tx).any() and np.isfinite(Tm).any()) else np.nan
    else:
        out[f"{prefix}relax_T0"] = np.nan
        out[f"{prefix}relax_dT"] = np.nan

    for w in WIN_SEC:
        mask = (t >= t_first) & (t <= (anchor_t0 + w)) & np.isfinite(V)
        min_pts = int(MIN_PTS_MAP.get(int(w), 6))
        if mask.sum() < min_pts:
            out[f"{prefix}relax_dV_{w}s"] = np.nan
            out[f"{prefix}relax_slope_{w}s"] = np.nan
            out[f"{prefix}relax_std_{w}s"] = np.nan
            continue
        tw = (t[mask] - t[mask][0]).astype("float64")
        Vw = V[mask].astype("float64")
        out[f"{prefix}relax_dV_{w}s"] = float(Vw[-1] - V0)
        out[f"{prefix}relax_slope_{w}s"] = float(np.polyfit(tw, Vw, 1)[0])
        out[f"{prefix}relax_std_{w}s"] = float(np.std(Vw))
    return out


def match_pre_rest(rest_df: pd.DataFrame, chg_t_start: int) -> dict | None:
    ends = rest_df["t_end"].to_numpy(dtype="int64")
    i = np.searchsorted(ends, chg_t_start, side="right") - 1
    if i < 0:
        return None

    while i >= 0:
        t_end = int(rest_df.iloc[i]["t_end"])
        gap = int(chg_t_start - t_end)
        if gap < 0:
            i -= 1
            continue
        if gap > PRE_GAP_MAX_S:
            return None
        dur = float(rest_df.iloc[i]["duration_s"])
        if dur >= PRE_MIN_REST_S:
            t_start = int(rest_df.iloc[i]["t_start"])
            return {"t_start": t_start, "t_end": t_end, "duration_s": dur, "gap_s": float(gap)}
        i -= 1
    return None


def main():
    rows = []
    stats = []

    for vin in tqdm(vins(), desc="relax_L1+ (post+pre)"):
        pairs = load_pairs(vin)
        idx = load_index_core(vin)
        seg = load_segment_summary(vin)

        rest = seg[seg["state"].eq("rest")].copy()
        rest = rest.dropna(subset=["t_start", "t_end", "duration_s"])
        rest = rest.sort_values("t_end").reset_index(drop=True)

        total = 0
        post_hit = 0
        pre_match = 0
        pre_hit = 0
        any_hit = 0

        for r in pairs.itertuples(index=False):
            total += 1
            chg_t_start = int(r.chg_t_start)
            chg_t_end = int(r.chg_t_end)

            post_ok = False
            rt0 = int(r.rst_t_start)
            rt1 = int(min(int(r.rst_t_end), rt0 + MAX_WIN))
            ev_post = read_window(idx, rt0, rt1)
            f_post = relax_L1(ev_post, rt0, prefix="post_")
            if f_post:
                post_ok = True
                post_hit += 1

            pre_ok = False
            pre = match_pre_rest(rest, chg_t_start)
            f_pre = {}
            pre_meta = {}
            if pre is not None:
                pre_match += 1
                pr1 = int(pre["t_end"])
                pr0 = int(max(int(pre["t_start"]), pr1 - MAX_WIN))
                ev_pre = read_window(idx, pr0, pr1)
                f_pre = relax_L1(ev_pre, pr0, prefix="pre_")
                pre_meta = {
                    "pre_rest_t_start": int(pre["t_start"]),
                    "pre_rest_t_end": int(pre["t_end"]),
                    "pre_rest_dur_s": float(pre["duration_s"]),
                    "pre_gap_s": float(pre["gap_s"]),
                    "pre_win_t0": int(pr0),
                    "pre_win_t1": int(pr1),
                }
                if f_pre:
                    pre_ok = True
                    pre_hit += 1

            if not (post_ok or pre_ok):
                continue

            any_hit += 1
            rec = {
                "vin": str(vin),
                "t_start": chg_t_start,
                "t_end": chg_t_end,
                "chg_seg_id": int(r.chg_seg_id) if pd.notna(r.chg_seg_id) else -1,
                "rst_seg_id": int(r.rst_seg_id) if pd.notna(r.rst_seg_id) else -1,
                "rst_t_start": rt0,
                "rst_t_end": int(r.rst_t_end),
                "rst_duration_s": float(r.rst_duration_s) if pd.notna(r.rst_duration_s) else float(int(r.rst_t_end) - rt0),
            }
            rec.update(f_post)
            rec.update(pre_meta)
            rec.update(f_pre)
            rows.append(rec)

        stats.append((vin, total, post_hit, pre_match, pre_hit, any_hit, "ok"))

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "features_relax_L1.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")

    stat_df = pd.DataFrame(
        stats,
        columns=["vin", "pairs_total", "post_hit", "pre_match", "pre_hit", "any_hit", "status"]
    )
    stat_path = OUT_DIR / "features_relax_L1_stats.csv"
    stat_df.to_csv(stat_path, index=False, encoding="utf-8-sig")

    print("Saved:", out_path, "rows:", len(out), "cols:", len(out.columns))
    print("Saved stats:", stat_path)

    denom = max(1, int(stat_df["pairs_total"].sum()))
    print("Post hit-rate(avg):", float(stat_df["post_hit"].sum() / denom))
    print("Pre  hit-rate(avg):", float(stat_df["pre_hit"].sum() / denom))
    print("Any  hit-rate(avg):", float(stat_df["any_hit"].sum() / denom))
    print("PRE_GAP_MAX_S used:", PRE_GAP_MAX_S, "seconds")


if __name__ == "__main__":
    main()
