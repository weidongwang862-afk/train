from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import config

VINS = [Path(x).stem for x in config.ALL_FILES]

# 统一口径：只从 03 的 segment_summary 派生事件
SEG_DIR = config.OUT_DIR / "02_segments"
OUT_DIR = config.OUT_DIR / "02_events"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 保留这些变量名（你原脚本里有），但在“统一定义”后它们不再用于逐点识别
SPEED_STOP = 1.0
I_CHARGE = -5.0
I_REST = 2.0

# 采样 10s/帧：用“时长阈值（秒）”替代点数阈值，更稳健
MIN_CHARGE_DUR_S = 10 * 60     # 10分钟
MIN_REST_DUR_S   = 60 * 60     # 1小时（你后续若要2小时，可在下游再筛）
PAIR_GAP_MAX_S   = 120         # charge结束到rest开始允许的最大间隔（秒）


def run_extract(vin: str):
    seg_path = SEG_DIR / f"segment_summary_{vin}.parquet"
    if not seg_path.exists():
        print(f"[WARN] missing segment summary: {seg_path}")
        return

    df = pd.read_parquet(seg_path)
    if df.empty:
        # 兜底写空
        pd.DataFrame().to_parquet(OUT_DIR / f"charge_events_{vin}.parquet", index=False)
        pd.DataFrame().to_parquet(OUT_DIR / f"rest_events_{vin}.parquet", index=False)
        pd.DataFrame().to_parquet(OUT_DIR / f"charge_rest_pairs_{vin}.parquet", index=False)
        print(vin, "segment_summary empty")
        return

    # 统一排序（seg_id 是分段主键；若你后面改成按时间更稳，也可同时排序 t_start）
    df = df.sort_values(["seg_id", "t_start"]).reset_index(drop=True)

    # 基础事件：直接按 state 过滤（统一口径）
    df_c = df[df["state"] == "charge"].copy()
    df_r = df[df["state"] == "rest"].copy()

    # 时长门槛（稳健）
    if "duration_s" in df_c.columns:
        df_c = df_c[df_c["duration_s"] >= MIN_CHARGE_DUR_S]
    if "duration_s" in df_r.columns:
        df_r = df_r[df_r["duration_s"] >= MIN_REST_DUR_S]

    # 写出基础事件（保持你原文件命名）
    df_c.to_parquet(OUT_DIR / f"charge_events_{vin}.parquet", index=False, compression="snappy")
    df_r.to_parquet(OUT_DIR / f"rest_events_{vin}.parquet", index=False, compression="snappy")

    # charge→rest 配对：看“下一段”是否为 rest，且时间间隔足够小
    nxt_state = df["state"].shift(-1)
    nxt_t_start = df["t_start"].shift(-1)
    nxt_t_end = df["t_end"].shift(-1)
    nxt_duration = df["duration_s"].shift(-1) if "duration_s" in df.columns else None

    gap_s = (nxt_t_start - df["t_end"]).astype("float64")
    ok = (
        (df["state"] == "charge") &
        (nxt_state == "rest") &
        (gap_s >= 0) &
        (gap_s <= PAIR_GAP_MAX_S)
    )

        # charge→rest 配对：看“下一段”是否为 rest，且时间间隔足够小
    nxt_state = df["state"].shift(-1)
    nxt_t_start = df["t_start"].shift(-1)

    gap_s = (nxt_t_start - df["t_end"]).astype("float64")
    ok = (
        (df["state"] == "charge") &
        (nxt_state == "rest") &
        (gap_s >= 0) &
        (gap_s <= PAIR_GAP_MAX_S)
    )

    # 注意：segment_summary 里通常没有 vin 列，所以这里不再强行索引 "vin"
    base_cols = [
        "seg_id", "t_start", "t_end", "duration_s",
        "soc_start", "soc_end", "delta_soc",
        "odo_start", "odo_end", "delta_odo",
        "I_mean", "I_min", "I_max",
        "V_mean", "V_min", "V_max",
        "T_min", "T_max",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    pairs = df.loc[ok, base_cols].copy()
    pairs.insert(0, "vin", vin)  # 补上 vin 列，保证后续一致


    # 追加 rest 段信息（来自下一行）
    pairs.rename(columns={
        "seg_id": "chg_seg_id",
        "t_start": "chg_t_start",
        "t_end": "chg_t_end",
        "duration_s": "chg_duration_s",
        "soc_start": "chg_soc_start",
        "soc_end": "chg_soc_end",
        "delta_soc": "chg_delta_soc",
        "odo_start": "chg_odo_start",
        "odo_end": "chg_odo_end",
        "delta_odo": "chg_delta_odo",
        "I_mean": "chg_I_mean",
        "I_min": "chg_I_min",
        "I_max": "chg_I_max",
        "V_mean": "chg_V_mean",
        "V_min": "chg_V_min",
        "V_max": "chg_V_max",
        "T_min": "chg_T_min",
        "T_max": "chg_T_max",
    }, inplace=True)

    # 用 shift(-1) 取下一段（rest）字段
    pairs["rst_seg_id"] = (df["seg_id"].shift(-1)).loc[ok].astype("int64")
    pairs["rst_t_start"] = (df["t_start"].shift(-1)).loc[ok].astype("int64")
    pairs["rst_t_end"] = (df["t_end"].shift(-1)).loc[ok].astype("int64")
    if "duration_s" in df.columns:
        pairs["rst_duration_s"] = (df["duration_s"].shift(-1)).loc[ok].astype("int64")

    # rest 的统计量
    for col in ["soc_start", "soc_end", "delta_soc",
                "odo_start", "odo_end", "delta_odo",
                "I_mean", "I_min", "I_max",
                "V_mean", "V_min", "V_max",
                "T_min", "T_max"]:
        if col in df.columns:
            pairs[f"rst_{col}"] = (df[col].shift(-1)).loc[ok].to_numpy()

    pairs["gap_s"] = gap_s.loc[ok].astype("int64").to_numpy()

    # 可选：给个常用 QC 标记（例如 rest>=2h）
    if "rst_duration_s" in pairs.columns:
        pairs["rst_ge_2h"] = pairs["rst_duration_s"] >= (2 * 3600)

    pairs.to_parquet(OUT_DIR / f"charge_rest_pairs_{vin}.parquet", index=False, compression="snappy")

    print(vin, "charge_events:", len(df_c), "rest_events:", len(df_r), "pairs:", len(pairs))


def main():
    for vin in VINS:
        run_extract(vin)


if __name__ == "__main__":
    main()
