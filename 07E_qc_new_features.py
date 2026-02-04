import pandas as pd
import numpy as np
import os
import config

BASE_OUT = config.OUT_DIR
# 这里我们检查最终产物 step7plus
path = os.path.join(BASE_OUT, "04_features", "dataset_train_step7plus.parquet")

if not os.path.exists(path):
    print(f"File not found: {path}")
    print("Checking step7 instead...")
    path = os.path.join(BASE_OUT, "04_features", "dataset_train_step7.parquet")

if not os.path.exists(path):
    print("No step7 dataset found.")
    exit(1)

df = pd.read_parquet(path)

# 新增特征列列表（包含 dvdt 和 stage 的特征）
new_cols = [
    "dv_end","dv_p95","dv_mean","dT_end","dT_p95","dT_mean","tail_n",
    "dv_cc_mean", "dv_cc_p95", "dv_hi_mean", "dT_cc_mean", "dvI_cc_mean", "Tmax_rise"
]
cols = [c for c in new_cols if c in df.columns]

print(f"QC File: {path}")
print("Rows:", len(df))
print("\n缺失率 (NaN ratio):")
print(df[cols].isna().mean().sort_values(ascending=False))

print("\n方差 (Variance, nan ignored):")
print(df[cols].var(numeric_only=True).sort_values(ascending=True))

print("\n分位数 (Quantiles):")
qs = df[cols].quantile([0.01,0.05,0.5,0.95,0.99], numeric_only=True)
print(qs)