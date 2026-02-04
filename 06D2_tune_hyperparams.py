# scripts/06H_tune_hyperparams.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# ================= 配置 =================
# 指向你的冻结数据集
DATASET_PATH = r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"
FEATURE_LIST_CSV = r"E:\RAW_DATA\outputs\04_features\features_FINAL_core.csv"  # 用你真实基线特征口径
SPLITS_VINS_CSV  = r"E:\RAW_DATA\outputs\09_seq_featcore\splits_vins.csv"  # 或你实际路径
USE_VINS = "trainval"   # "train" 或 "trainval"

# 或者是 dataset_all_C_frozen.parquet，看你哪个是最新的

# 结果保存路径
OUT_CSV = "hyperparam_search_results.csv"

# 固定的特征列表 (如果没有 features_FINAL.csv，就用排除法自动选)
# 这里为了省事，我们用排除法自动选所有数值特征，和 06D 逻辑一致
DROP_COLS = ["vin", "t_start", "t_end", "terminaltime", "datetime", 
             "C_trend", "SoH_trend", "C_est_ah", "SoH_raw", "quality_flag"]

# 要搜索的网格
PARAM_GRID = {
    # 1. 学习率扫描 (固定 depth=6)
    "learning_rate": [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20],
    
    # 2. 树深扫描 (固定 lr=0.06)
    "max_depth": [3, 4, 5, 6, 7, 8, 10, 15]
}

# 默认基准参数
BASE_PARAMS = {
    "loss": "absolute_error",
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": 42
}

def main():
    print(f"Reading {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    df = pd.read_parquet(DATASET_PATH)
    spl = pd.read_csv(SPLITS_VINS_CSV)
    if USE_VINS == "train":
        keep_vins = set(spl.loc[spl["split"]=="train", "vin"].astype(str))
    else:
        keep_vins = set(spl.loc[spl["split"].isin(["train","val"]), "vin"].astype(str))

    df["vin"] = df["vin"].astype(str)
    df = df[df["vin"].isin(keep_vins)].copy()

    
    # 简单的特征筛选
    feat_df = pd.read_csv(FEATURE_LIST_CSV)
    feat_cols = feat_df["feature"].astype(str).tolist()
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"Using {len(feat_cols)} features from FEATURE_LIST.")
    X = df[feat_cols]  # 不要 fillna(0)
    y = df["SoH_trend"]
    groups = df["vin"]
    
    results = []

    # --- 实验 1: 扫描 Learning Rate ---
    print("\nStarting Learning Rate Sweep...")
    base_depth = 6
    for lr in tqdm(PARAM_GRID["learning_rate"]):
        # 5-Fold CV
        gkf = GroupKFold(n_splits=3) # 用 3-Fold 快一点
        maes = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = HistGradientBoostingRegressor(
                learning_rate=lr,
                max_depth=base_depth,
                **BASE_PARAMS
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_va)
            maes.append(mean_absolute_error(y_va, p))
        
        avg_mae = float(np.mean(maes))
        std_mae = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
        results.append({"param_type":"learning_rate","value":lr,"mae_mean":avg_mae,"mae_std":std_mae})
        print(f"LR={lr} -> MAE={avg_mae:.6f}")

    # --- 实验 2: 扫描 Max Depth ---
    print("\nStarting Max Depth Sweep...")
    base_lr = 0.06
    for d in tqdm(PARAM_GRID["max_depth"]):
        gkf = GroupKFold(n_splits=3)
        maes = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = HistGradientBoostingRegressor(
                learning_rate=base_lr,
                max_depth=d,
                **BASE_PARAMS
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_va)
            maes.append(mean_absolute_error(y_va, p))
        
        avg_mae = float(np.mean(maes))
        std_mae = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
        results.append({"param_type":"max_depth","value":d,"mae_mean":avg_mae,"mae_std":std_mae})

        print(f"Depth={d} -> MAE={avg_mae:.6f}")

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")

if __name__ == "__main__":
    main()