# scripts/04_build_part_index1.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import config

OUT = Path(config.OUT_DIR)
CORE_DIR = OUT / "01_clean_core"
OUT_DIR = OUT / "03_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCOL = "terminaltime"

def build_one_vin(vin: str):
    vdir = CORE_DIR / vin
    if not vdir.exists():
        raise FileNotFoundError(f"missing clean_core dir: {vdir}")

    parts = sorted(vdir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parts under: {vdir}")

    rows = []
    for fp in parts:
        pf = pq.ParquetFile(fp)
        # 用 metadata 统计 terminaltime 的 min/max（无需读全表）
        schema_names = pf.schema_arrow.names
        if TCOL not in schema_names:
            raise ValueError(f"{fp} has no {TCOL}, cols={schema_names}")

        t_idx = schema_names.index(TCOL)
        md = pf.metadata

        tmin = None
        tmax = None
        nrows = 0

        for i in range(pf.num_row_groups):
            rg = md.row_group(i)
            nrows += rg.num_rows
            col = rg.column(t_idx)
            st = col.statistics
            if st is None:
                # 没统计信息就跳过（一般不会）
                continue
            try:
                mn = float(st.min)
                mx = float(st.max)
            except Exception:
                continue
            tmin = mn if tmin is None else min(tmin, mn)
            tmax = mx if tmax is None else max(tmax, mx)

        if tmin is None or tmax is None:
            # 兜底：读一小列求 min/max
            df = pq.read_table(fp, columns=[TCOL]).to_pandas()
            tmin = float(df[TCOL].min())
            tmax = float(df[TCOL].max())
            nrows = len(df)

        rows.append({
            "vin": vin,
            "part_path": str(fp),
            "t_min": int(tmin),
            "t_max": int(tmax),
            "rows": int(nrows),
        })

    out = pd.DataFrame(rows).sort_values("t_min").reset_index(drop=True)
    out_fp = OUT_DIR / f"part_index_core_{vin}.parquet"
    out.to_parquet(out_fp, index=False, engine="pyarrow")
    print("[OK]", vin, "->", out_fp, "n_parts:", len(out))

def main():
    vins = [Path(f).stem for f in config.ALL_FILES]
    for vin in tqdm(vins, desc="build_part_index_core"):
        try:
            build_one_vin(vin)
        except Exception as e:
            print("[ERROR]", vin, e)

if __name__ == "__main__":
    main()
