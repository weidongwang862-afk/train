# 99_plot_fig2_2_journal_v4_1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from scipy import stats

try:
    import config
    OUT_DIR = Path(config.OUT_DIR)
except ImportError:
    OUT_DIR = Path('outputs')

# ================= 配置区域 =================
RELAX_FILE = OUT_DIR / "04_features" / "features_relax_L1.parquet"
LABEL_FILE = Path("E:/RAW_DATA/outputs/04_features/dataset_all_C_frozen.parquet") # 需要带 SOH 标签的文件
if not LABEL_FILE.exists():
    LABEL_FILE = OUT_DIR / "04_features" / "features_core_all.parquet"

OUT_PNG = OUT_DIR.parent / "Fig2-2_Relaxation_Panel_Final_v4.1.png"

# 字段映射
TARGET_COL = 'post_relax_dV_1800s'
LABEL_COL = 'SoH_trend'
TEMP_COL = 'post_relax_T0' # 弛豫时刻温度
SOC_COL = 'soc_end'
DURATION_COL = 'rst_duration_s'

# 绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "mathtext.fontset": "stix"
})

def get_stats_str(x, y):
    """计算 Spearman 相关性并返回格式化字符串 (含 p 值)"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    corr, pval = stats.spearmanr(x[mask], y[mask])
    p_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    return f"$\\rho = {corr:.3f}$\n({p_str})"

def main():
    print(">>> Loading Data...")
    if not RELAX_FILE.exists():
        print("Error: Relax file missing.")
        return

    # 1. 数据对齐
    df_relax = pd.read_parquet(RELAX_FILE)
    df_main = pd.read_parquet(LABEL_FILE)
    
    # 统一 Label 列名
    l_col = LABEL_COL if LABEL_COL in df_main.columns else ('y_true' if 'y_true' in df_main.columns else 'SoH')
    
    df_relax["vin"] = df_relax["vin"].astype(str)
    df_main["vin"] = df_main["vin"].astype(str)
    
    # Left join 以计算总分母
    merged = pd.merge(df_main, df_relax, on=["vin", "t_start"], how="left", suffixes=('', '_rel'))
    merged['dV_abs'] = merged[TARGET_COL].abs()
    merged['has_relax'] = merged[TARGET_COL].notna() & (merged['dV_abs'] > 0.001)

    # 筛选分析集
    df_plot = merged[(merged[l_col] >= 0.6) & (merged[l_col] <= 1.05)].copy()

    # 2. 准备子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- (a) Conditional Availability ---
    ax = axes[0, 0]
    df_plot['bin'] = pd.cut(df_plot[l_col], bins=12)
    stats_df = df_plot.groupby('bin', observed=True).agg(
        total=('has_relax', 'count'),
        valid=('has_relax', 'sum')
    )
    stats_df['avail'] = stats_df['valid'] / stats_df['total'] * 100
    mids = [i.mid for i in stats_df.index]
    # 修正图例对应关系
    bar = ax.bar(mids, stats_df['avail'], width=0.025, color='gray', alpha=0.5, label='Availability $P(m=1|SOH)$')
    ax_r = ax.twinx()
    line, = ax_r.plot(mids, stats_df['valid'], 'b-o', markersize=4, label='Valid Sample Count ($N$)')

    ax.set_title('(a) Conditional Availability $P(m=1|SOH)$', fontweight='bold', loc='left')
    ax.set_ylabel('Availability (%)')
    ax_r.set_ylabel('Valid Count (N)')
    ax.invert_xaxis()

        # 合并两个坐标轴的图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- (b) Raw Distribution ---
    ax = axes[0, 1]
    # 严格清洗绘图集
    valid_data = df_plot[df_plot['has_relax'] & (df_plot['dV_abs'] < 0.5)].copy()
    ax.hexbin(valid_data[l_col], valid_data['dV_abs'], gridsize=35, cmap='Blues', mincnt=1, linewidths=0)
    sns.regplot(data=valid_data, x=l_col, y='dV_abs', scatter=False, lowess=True, 
                color='red', ax=ax, line_kws={'label': 'Binned Median + LOWESS'})
    ax.text(0.05, 0.9, get_stats_str(valid_data[l_col], valid_data['dV_abs']), 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_title('(b) Raw Relationship (Global)', fontweight='bold', loc='left')
    ax.set_ylabel(r'$|\Delta V_{relax}|$ (V)')
    ax.legend(loc='upper right')
    ax.invert_xaxis()

    # --- (c) Temperature Stratification ---
    ax = axes[1, 0]
    t_col = TEMP_COL if TEMP_COL in valid_data.columns else 'maxtemperaturevalue'
    if t_col in valid_data.columns:
        valid_data['t_bin'] = pd.qcut(valid_data[t_col], 3, labels=['Low T', 'Mid T', 'High T'])
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        for i, label in enumerate(['Low T', 'Mid T', 'High T']):
            sub = valid_data[valid_data['t_bin'] == label]
            if len(sub) > 50:
                # 粗分箱画带误差棒的线
                sub = sub.copy()
                sub['s_bin'] = pd.cut(sub[l_col], bins=8)
                g = sub.groupby('s_bin', observed=True)['dV_abs'].agg(['median', 'sem']).dropna()
                ax.errorbar([b.mid for b in g.index], g['median'], yerr=g['sem'], fmt='-o', 
                            capsize=3, color=colors[i], label=f'{label} (N={len(sub)})')
    ax.set_title('(c) Stratified by Temperature', fontweight='bold', loc='left')
    ax.set_ylabel(r'Median $|\Delta V_{relax}|$ (V)')
    ax.legend(loc='best', fontsize=8)
    ax.invert_xaxis()

    # --- (d) Residualized Analysis ---
    ax = axes[1, 1]
    # 选取混杂因子进行线性回归
    confounders = [c for c in [t_col, SOC_COL, DURATION_COL] if c in valid_data.columns]
    if len(confounders) >= 2:
        reg_df = valid_data.dropna(subset=confounders + ['dV_abs']).copy()
        X = reg_df[confounders]
        y = reg_df['dV_abs']
        model = Ridge(alpha=1.0).fit(X, y)
        reg_df['resid'] = y - model.predict(X)
        
        ax.hexbin(reg_df[l_col], reg_df['resid'], gridsize=35, cmap='Blues', mincnt=1, linewidths=0)
        sns.regplot(data=reg_df, x=l_col, y='resid', scatter=False, lowess=True, 
                    color='red', ax=ax, line_kws={'label': 'Binned Median + LOWESS'})
        ax.legend(loc='upper right')
        ax.text(0.05, 0.9, get_stats_str(reg_df[l_col], reg_df['resid']), 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_title('(d) Residuals (Confounders Removed)', fontweight='bold', loc='left')
    ax.set_ylabel(r'Residual $\Delta V^*$ (V)')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f">>> Success: Figure saved to {OUT_PNG}")

if __name__ == "__main__":
    main()