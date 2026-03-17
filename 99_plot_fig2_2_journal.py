# 99_plot_fig2_2_journal_v2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

# ================= 配置区域 =================
OUT_DIR = Path(config.OUT_DIR)
RELAX_FILE = OUT_DIR / "04_features" / "features_relax_L1.parquet"
# 自动寻找标签文件

LABEL_FILE = Path("E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet") # 需要带 SOH 标签的文件

OUT_PNG = OUT_DIR.parent / "Fig2-2_Relaxation_Physical_Evolution.png"

# 核心物理特征：充电后1800s的电压变化
TARGET_COL = 'post_relax_dV_1800s'
LABEL_COL = 'SoH_trend'

# 绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "mathtext.fontset": "stix"
})

def main():
    print(">>> Loading Data...")
    if not RELAX_FILE.exists():
        print(f"Error: {RELAX_FILE} not found.")
        return

    # 1. 读取数据
    df_relax = pd.read_parquet(RELAX_FILE, columns=["vin", "t_start", TARGET_COL])
    
    # 自动兼容标签列名
    try:
        df_label = pd.read_parquet(LABEL_FILE, columns=["vin", "t_start", LABEL_COL])
    except:
        print(f"Warning: '{LABEL_COL}' not found, trying 'y_true'...")
        LABEL_COL_ACTUAL = "y_true"
        df_label = pd.read_parquet(LABEL_FILE, columns=["vin", "t_start", "y_true"])
        df_label = df_label.rename(columns={"y_true": LABEL_COL})

    # 确保合并键格式一致
    df_relax["vin"] = df_relax["vin"].astype(str)
    df_label["vin"] = df_label["vin"].astype(str)
    
    merged = pd.merge(df_label, df_relax, on=["vin", "t_start"], how="inner")
    print(f"Total Merged Rows: {len(merged)}")

    # 2. 关键修正：取绝对值！(把负的电压回落变成正的幅度)
    merged[TARGET_COL] = merged[TARGET_COL].abs()

    # 3. 过滤 (现在 >0.001 是安全的，因为都是正数了)
    valid = merged.dropna(subset=[LABEL_COL, TARGET_COL])
    valid = valid[
        (valid[TARGET_COL] > 0.001) &  # 幅度 > 1mV
        (valid[TARGET_COL] < 0.5) &    # 幅度 < 0.5V (排除极端异常值)
        (valid[LABEL_COL] > 0.6) &     
        (valid[LABEL_COL] < 1.1)
    ]
    
    n_count = len(valid)
    # 计算 Spearman 相关性
    # 预期：SOH 越低(Battery Old)，极化越大，电压回落幅度越大 -> 负相关
    spearman = valid[[LABEL_COL, TARGET_COL]].corr(method='spearman').iloc[0, 1]
    
    print(f"Valid Samples (After Abs Fix): {n_count} (Expect ~55k)")
    print(f"Spearman Correlation: {spearman:.3f}")

    if n_count < 100:
        print("Error: Samples are still too few. Check value range.")
        return

    # 4. 绘图
    print(">>> Plotting...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # (A) Hexbin 热力图 (显示数据密度)
    hb = ax.hexbin(
        valid[LABEL_COL], valid[TARGET_COL], 
        gridsize=45, cmap='Blues', mincnt=1, linewidths=0, alpha=0.9
    )
    cb = plt.colorbar(hb, ax=ax, label='Sample Density')
    
    # (B) Lowess 趋势线 (红色)
    # 降采样加速拟合
    plot_data = valid.sample(5000, random_state=42) if n_count > 5000 else valid
    
    sns.regplot(
        data=plot_data, x=LABEL_COL, y=TARGET_COL, 
        scatter=False, lowess=True, 
        color='#d62728', 
        line_kws={'linewidth': 3, 'label': 'Non-linear Trend (LOWESS)'},
        ax=ax
    )

    # (C) 装饰
    ax.set_xlabel(r"State of Health (SOH)")
    ax.set_ylabel(r"Relaxation Voltage Drop $|\Delta V_{relax}|$ (V)")
    ax.set_title("Physical Evolution of Sparse Relaxation Features", fontweight='bold')
    
    # X轴反转：从新电池(1.0) -> 老电池(0.7)
    ax.set_xlim(1.05, 0.65)
    # Y轴范围自适应
    ax.set_ylim(0, valid[TARGET_COL].quantile(0.99) * 1.2)

    # 统计框
    textstr = '\n'.join((
        r'$N_{valid} \approx %.1fk$' % (n_count/1000,),
        r'Coverage $\approx 54.7\%$',
        r'Spearman $\rho = %.3f$' % (spearman, )
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches='tight')
    print(f"Figure saved to: {OUT_PNG}")

if __name__ == "__main__":
    main()