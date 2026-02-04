# scripts/07B_aux_dvdt_to_parquet.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)
CORE_DIR = OUT_ROOT / "01_clean_core"                # 关键：改为读 clean_core
OUT_DIR  = OUT_ROOT / "05_aux_dvdt"                  # 与你原目录一致
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_T="terminaltime"
COL_VMIN="minvoltagebattery"
COL_VMAX="maxvoltagebattery"
COL_TMIN="mintemperaturevalue"
COL_TMAX="maxtemperaturevalue"
USECOLS=[COL_T,COL_VMIN,COL_VMAX,COL_TMIN,COL_TMAX]

# 物理过滤阈值（宽松）
DV_MAX = 2.0      # 单体极差上限（V）
DT_MAX = 80.0     # 温差上限（°C）

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    vins = [Path(f).stem for f in config.ALL_FILES]
    print(f"Aux dvdt from clean_core, VINs={len(vins)}")

    for vin in vins:
        vin_core = CORE_DIR / vin
        if not vin_core.exists():
            print(f"[WARN] {vin} missing clean_core dir: {vin_core}")
            continue

        parts = sorted(vin_core.glob("part_*.parquet"))
        if not parts:
            print(f"[WARN] {vin} no clean_core parts")
            continue

        vin_out = OUT_DIR / vin
        ensure_dir(vin_out)

        idx_rows = []
        part_id = 0

        for p in parts:
            df = pd.read_parquet(p, columns=[c for c in USECOLS if c in pd.read_parquet(p, engine="pyarrow").columns])
            if COL_T not in df.columns or df.empty:
                continue

            df[COL_T] = pd.to_numeric(df[COL_T], errors="coerce").astype("Int64")
            df = df.dropna(subset=[COL_T])
            if df.empty:
                continue

            for c in [COL_VMIN, COL_VMAX, COL_TMIN, COL_TMAX]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if (COL_VMIN in df.columns) and (COL_VMAX in df.columns):
                dv = (df[COL_VMAX] - df[COL_VMIN]).astype("float32")
                dv = dv.where((dv >= 0) & (dv <= DV_MAX), np.nan)
            else:
                dv = pd.Series(np.nan, index=df.index, dtype="float32")

            if (COL_TMIN in df.columns) and (COL_TMAX in df.columns):
                dT = (df[COL_TMAX] - df[COL_TMIN]).astype("float32")
                dT = dT.where((dT >= 0) & (dT <= DT_MAX), np.nan)
            else:
                dT = pd.Series(np.nan, index=df.index, dtype="float32")

            out = pd.DataFrame({
                COL_T: df[COL_T].astype("int64"),
                "dv_cell": dv,
                "dT_probe": dT,
            }).sort_values(COL_T)

            if out.empty:
                continue

            tmin = int(out[COL_T].min())
            tmax = int(out[COL_T].max())
            nrow = int(len(out))

            out_path = vin_out / f"part_{part_id:05d}.parquet"
            out.to_parquet(out_path, index=False, compression="snappy")

            idx_rows.append({
                "vin": vin,
                "part_id": part_id,
                "path": str(out_path),   # 仍存绝对路径，但后续会兼容修复
                "tmin": tmin,
                "tmax": tmax,
                "nrow": nrow,
            })
            part_id += 1

        if idx_rows:
            idx = pd.DataFrame(idx_rows)
            idx_path = vin_out / "aux_part_index.parquet"
            idx.to_parquet(idx_path, index=False)
            print(f"{vin}: parts={len(idx_rows)} saved={idx_path}")
        else:
            print(f"{vin}: no valid aux data")

if __name__ == "__main__":
    main()
