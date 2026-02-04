# scripts/07B2_aux_stage_to_parquet.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)
CORE_DIR = OUT_ROOT / "01_clean_core"
OUT_DIR  = OUT_ROOT / "05_aux_stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_T="terminaltime"; COL_SOC="soc"; COL_I="totalcurrent"
COL_VMIN="minvoltagebattery"; COL_VMAX="maxvoltagebattery"
COL_TMIN="mintemperaturevalue"; COL_TMAX="maxtemperaturevalue"
USECOLS=[COL_T,COL_SOC,COL_I,COL_VMIN,COL_VMAX,COL_TMIN,COL_TMAX]

DV_MAX=2.0
DT_MAX=80.0

def main():
    vins = [Path(f).stem for f in config.ALL_FILES]
    print(f"Aux stage from clean_core, VINs={len(vins)}")

    for vin in vins:
        vin_core = CORE_DIR / vin
        parts = sorted(vin_core.glob("part_*.parquet"))
        if not parts:
            print(f"[WARN] {vin} no clean_core parts")
            continue

        vin_out = OUT_DIR / vin
        vin_out.mkdir(parents=True, exist_ok=True)

        idx_rows=[]; part_id=0
        for p in parts:
            df = pd.read_parquet(p, columns=[c for c in USECOLS if c in pd.read_parquet(p, engine="pyarrow").columns])
            if COL_T not in df.columns or df.empty:
                continue

            df[COL_T] = pd.to_numeric(df[COL_T], errors="coerce").astype("Int64")
            df = df.dropna(subset=[COL_T])
            if df.empty:
                continue

            for c in [COL_SOC, COL_I, COL_VMIN, COL_VMAX, COL_TMIN, COL_TMAX]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # dv / dT
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

            I = df[COL_I].astype("float32") if COL_I in df.columns else pd.Series(np.nan, index=df.index, dtype="float32")
            I_abs = I.abs()

            out = pd.DataFrame({
                COL_T: df[COL_T].astype("int64"),
                COL_SOC: df[COL_SOC].astype("float32") if COL_SOC in df.columns else np.nan,
                COL_I: I,
                "I_abs": I_abs.astype("float32"),
                "dv_cell": dv,
                "dT_probe": dT,
                COL_TMIN: df[COL_TMIN].astype("float32") if COL_TMIN in df.columns else np.nan,
                COL_TMAX: df[COL_TMAX].astype("float32") if COL_TMAX in df.columns else np.nan,
            }).sort_values(COL_T)

            if out.empty:
                continue

            tmin = int(out[COL_T].min()); tmax = int(out[COL_T].max()); nrow = int(len(out))
            out_path = vin_out / f"part_{part_id:05d}.parquet"
            out.to_parquet(out_path, index=False, compression="snappy")

            idx_rows.append({"vin":vin,"part_id":part_id,"path":str(out_path),"tmin":tmin,"tmax":tmax,"nrow":nrow})
            part_id += 1

        if idx_rows:
            idx = pd.DataFrame(idx_rows)
            idx.to_parquet(vin_out / "aux_part_index.parquet", index=False)
            print(vin, "parts=", len(idx_rows))
        else:
            print(vin, "no aux_stage data")

if __name__ == "__main__":
    main()
