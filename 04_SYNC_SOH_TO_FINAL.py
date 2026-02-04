# 04_SYNC_SOH_TO_FINAL.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
# 1. 刚刚生成的正确标签目录
LABEL_DIR = Path(r"E:\RAW_DATA\outputs\03_labels")

# 2. 最终特征文件 (将被修改)
DATASET_PATH = Path(r"E:\RAW_DATA\outputs\04_features\dataset_all_D_frozen.parquet")
# ========================================

def main():
    print(f"1. 正在扫描正确标签: {LABEL_DIR} ...")
    soh_files = list(LABEL_DIR.glob("soh_*.parquet"))
    
    if not soh_files:
        print("❌ 错误：没找到 soh_*.parquet 文件！请确认您是否真的跑了 04_compute_soh3.py")
        return

    # === 第一步：从新标签中提取每辆车的 C0 ===
    print("   正在提取每辆车的 C0 基准...")
    vin_c0_map = {}
    
    for fp in tqdm(soh_files):
        try:
            # 只需要读取一行就能知道这辆车的 C0 是多少
            # SoH = C_est / C0  ==> C0 = C_est / SoH
            df_label = pd.read_parquet(fp, columns=["C_est_ah", "SoH"])
            df_label = df_label[df_label["SoH"] > 0] # 过滤异常
            
            if not df_label.empty:
                # 取中位数反推 C0，保证稳健
                c0_est = (df_label["C_est_ah"] / df_label["SoH"]).median()
                vin = fp.stem.replace("soh_", "")
                vin_c0_map[vin] = c0_est
        except Exception:
            pass
            
    print(f"   提取完成，共获取 {len(vin_c0_map)} 辆车的正确 C0。")
    print(f"   示例: {list(vin_c0_map.items())[:3]}")

    # === 第二步：注入最终数据集 ===
    print(f"\n2. 正在加载最终数据集: {DATASET_PATH} ...")
    if not DATASET_PATH.exists():
         print("❌ 错误：找不到 dataset_all_C_frozen.parquet！")
         return
         
    df_frozen = pd.read_parquet(DATASET_PATH)
    print(f"   行数: {len(df_frozen)}")
    
    if "C_trend" not in df_frozen.columns:
        print("❌ 错误：数据集中没有 C_trend 列，无法计算！")
        return

    print("3. 正在覆盖 SoH_trend ...")
    # 确保 vin 是字符串
    df_frozen["vin"] = df_frozen["vin"].astype(str)
    
    # 映射 C0
    df_frozen["C0_new"] = df_frozen["vin"].map(vin_c0_map)
    
    # 填补可能匹配不到的车（用默认 155.0）
    missing_count = df_frozen["C0_new"].isna().sum()
    if missing_count > 0:
        print(f"⚠️ 警告: 有 {missing_count} 行数据没匹配到 C0，使用默认值 155.0")
        df_frozen["C0_new"] = df_frozen["C0_new"].fillna(155.0)
    
    # ★★★ 核心修正公式 ★★★
    # 用特征表里的容量(C_trend) 除以 正确的C0
    df_frozen["SoH_trend"] = df_frozen["C_trend"] / df_frozen["C0_new"]
    
    # 简单清洗
    df_frozen["SoH_trend"] = df_frozen["SoH_trend"].clip(0.1, 1.5)
    
    print(f"4. 保存修正后的数据集...")
    df_frozen.to_parquet(DATASET_PATH, index=False)
    print("✅ 同步完成！现在 dataset_all_C_frozen.parquet 里已经是正确的 SoH 了。")

if __name__ == "__main__":
    main()