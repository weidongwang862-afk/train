# 99_plot_fig2_1_journal_v3_multi_vin.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import config

# ================= 配置区域 =================
LABEL_DIR = Path(config.OUT_DIR) / "03_labels"
OUT_FILE = Path(config.OUT_DIR).parent / "Fig2-1_Journal_Robustness_MultiVIN.png"

# 1. 获取所有可用的标签文件
all_files = list(LABEL_DIR.glob("labels_post_*.parquet"))
if not all_files:
    print(f"Error: No labels_post_*.parquet found in {LABEL_DIR}")
    exit()

# 2. 选择要展示的车辆数量 (建议 3-4 辆，太多画面会乱)
NUM_VINS_TO_SHOW = 4

# 为了每次画出的图固定，设置随机种子；您也可以手动指定几个典型的 VIN (如 vin125, vin089)
random.seed(42) 
selected_files = random.sample(all_files, min(NUM_VINS_TO_SHOW, len(all_files)))

# ================= 绘图风格设置 (Nature/Science 风格) =================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

def main():
    # 创建 (2, 1) 子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 定义一组顶刊常用配色 (蓝、红、绿、紫、橙)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    
    for i, file_path in enumerate(selected_files):
        df = pd.read_parquet(file_path)
        vin_name = file_path.stem.split('_')[-1] # 提取文件名中的 vin (例如 vin125)
        color = colors[i % len(colors)]
        
        # 准备数据
        x = df["odo_end"] / 1000.0  # 转换为 '千公里'
        y_raw = df["C_est_ah"]
        y_smooth = df["C_smooth"]
        y_final = df["C_trend"]
        
        # --- Subplot (a): Raw Data & LOWESS Smoothing ---
        # 绘制原始散点：多辆车叠加时，为了防止太乱，散点透明度调低 (alpha=0.15)，并且不加边框
        ax1.scatter(x, y_raw, c=color, s=15, alpha=0.15, edgecolors='none')
        # 绘制 LOWESS：加标签以供图例显示
        ax1.plot(x, y_smooth, c=color, linestyle='--', linewidth=2, label=f'LOWESS ({vin_name})')
        
        # --- Subplot (b): Monotonic Isotonic Regression ---
        # 背景噪音进一步弱化 (alpha=0.08)
        ax2.scatter(x, y_raw, c=color, s=10, alpha=0.08, edgecolors='none')
        # 强调最终强制单调的趋势线
        ax2.plot(x, y_final, c=color, linewidth=2.5, label=f'C_trend ({vin_name})')
        
    # ================= 装饰与图例调整 =================
    # (a) 图装饰
    ax1.set_ylabel("Capacity (Ah)", fontweight='bold')
    ax1.set_title("(a) Raw Reconstruction & Initial Smoothing (LOWESS)", loc='left', fontweight='bold')
    # 图例分列显示，防止遮挡曲线
    ax1.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=0.9, ncol=2)
    ax1.grid(False)
    
    # (b) 图装饰
    ax2.set_xlabel(r"Accumulated Mileage ($\times 10^3$ km)", fontweight='bold')
    ax2.set_ylabel("Capacity (Ah)", fontweight='bold')
    ax2.set_title("(b) Physical Consistency Constraint (Isotonic Regression)", loc='left', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=0.9, ncol=2)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(OUT_FILE, bbox_inches='tight')
    print(f"Figure saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()