# scripts/04_compute_soh3.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import config

VINS = [Path(x).stem for x in config.ALL_FILES]

LABEL_DIR = Path(config.OUT_DIR) / "03_labels"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

def compute_one(vin: str):
    fp = LABEL_DIR / f"capacity_labels_{vin}.parquet"
    if not fp.exists():
        print(vin, "MISSING capacity_labels")
        return

    df = pd.read_parquet(fp)
    if df.empty:
        print(vin, "capacity_labels EMPTY")
        return

    # 按里程排序
    key = "odo_end" if "odo_end" in df.columns else "t_end"
    df = df.sort_values(key).reset_index(drop=True)

    # ========================== 核心修改 ==========================
    # 错误逻辑：自动推算 C0 (会导致老车 SoH 虚高)
    # base_pool = good.head(10) if len(good) >= 5 else df.head(10)
    # C0 = float(np.median(pd.to_numeric(base_pool["C_est_ah"], errors="coerce").dropna().values))

    # 正确逻辑：使用论文 (Liu et al. 2025) 指定的标称容量
    C0 = 155.0  
    # ============================================================

    if not np.isfinite(C0) or C0 <= 0:
        print(vin, "BAD C0, skip")
        return

    # 计算 SoH
    # 过滤掉计算异常导致的超大值 (比如 > 180Ah 的噪声)
    df = df[df["C_est_ah"] < (C0 * 1.2)] 
    
    df["SoH"] = pd.to_numeric(df["C_est_ah"], errors="coerce").astype(float) / C0
    
    outp = LABEL_DIR / f"soh_{vin}.parquet"
    df.to_parquet(outp, index=False)
    print(vin, "C0=", C0, "rows=", len(df), "saved:", outp)


def main():
    for vin in VINS:
        compute_one(vin)

if __name__ == "__main__":
    main()
