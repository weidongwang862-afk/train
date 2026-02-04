import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib import rcParams

# ============== 1. 全局配置 ==============
DATA_PATH = r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"
OUT_PNG   = r"E:\RAW_DATA\Fig4-6_HI_Smooth_Trend.png"

# 特征配置 (文案优化)
FEATURE_CONFIG = {
    "ic_p95": {
        "label": "IC Peak Intensity ($IC_{p95}$)",
        "cn": "电化学衰退：峰值强度下降",
        "unit": "Ah/V",
        "color": "#1f77b4", # 蓝
    },
    "dv_hi_p95": {
        "label": "High-SoC Voltage Drop ($dV_{hi}$)",
        "cn": "内阻增长：电压回弹增大",
        "unit": "V",
        "color": "#d62728", # 红
    },
    "end_I_ratio": {
        "label": "End-Current Ratio ($I_{end}/I_{cc}$)",
        "cn": "动力学恶化：截止电流比升高",
        "unit": "Ratio",
        "color": "#2ca02c", # 绿
    },
    "dT_hi_mean": {
        "label": "Temp. Rise ($\Delta T_{mean}$)",
        "cn": "热失效风险：产热增加",
        "unit": "°C",
        "color": "#ff7f0e", # 橙
    },
}

# 风格设置
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks", {"axes.grid": True})
rcParams['font.family'] = ['Times New Roman', 'SimSun']
rcParams['axes.unicode_minus'] = False

def remove_outliers(df, col):
    """温和的去噪：只剔除 1% - 99% 分位之外的极端值"""
    q_low = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    return df[(df[col] >= q_low) & (df[col] <= q_high)]

def plot_smooth_evolution():
    print(f"正在加载数据: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print("错误：找不到数据文件")
        return
    
    df = pd.read_parquet(DATA_PATH)
    
    # 锁定 SOH 列
    cand_soh = ["SoH_trend", "soh", "y", "y_true"]
    soh_col = next((c for c in cand_soh if c in df.columns), None)
    if soh_col is None:
        print("错误：找不到 SOH 列")
        return

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.flatten()
    
    features = list(FEATURE_CONFIG.keys())
    
    # 过滤 SOH 范围 (0.7-1.05)，保证X轴不过宽
    base_df = df[(df[soh_col] > 0.70) & (df[soh_col] < 1.05)].copy()

    for i, feature in enumerate(features):
        ax = axes[i]
        conf = FEATURE_CONFIG[feature]
        
        if feature not in base_df.columns:
            continue
        
        # 1. 数据清洗 (去噪)
        sub = base_df[[soh_col, feature]].dropna()
        sub = remove_outliers(sub, feature) # 关键：去掉干扰视线的噪点
        
        # 转换 SOH 为百分比
        sub['SOH_pct'] = sub[soh_col] * 100
        x = sub['SOH_pct']
        y = sub[feature]
        
        # 计算相关性 (用于标注)
        corr, _ = spearmanr(x, y)
        
        # 2. 绘制背景云图 (Hexbin 或 Scatter)
        # 这里用 Hexbin (六边形热力图) 效果最好，能展示数据密度，还没噪点
        # mincnt=1 表示不画空白区域，gridsize 控制颗粒度
        hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=5, 
                       alpha=0.4, linewidths=0)
        
        # 3. 绘制平滑趋势线 (核心)
        # 使用 lowess (局部加权回归) 拟合出一条最平滑的线
        sns.regplot(x=x, y=y, ax=ax, scatter=False, 
                    lowess=True, # 【神器】开启局部平滑，不强求直线，拟合真实物理曲线
                    line_kws={'color': conf['color'], 'linewidth': 3, 'alpha': 0.9},
                    label='Trend')
        
        # 装饰
        title_str = f"{conf['label']}\n{conf['cn']}"
        ax.set_title(title_str, fontsize=14, fontweight='bold', pad=12)
        
        if i >= 2:
            ax.set_xlabel("State of Health (SOH) [%]", fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel("")
            
        ax.set_ylabel(f"Feature Value [{conf['unit']}]", fontsize=12)
        
        # 标注 Spearman
        # 根据相关性正负调整位置，避免遮挡线条
        text_y = 0.92 if corr < 0 else 0.08 
        va = 'top' if corr < 0 else 'bottom'
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=conf['color'], alpha=0.9, lw=1.5)
        ax.text(0.05, text_y, f"Spearman $\\rho = {corr:.2f}$", transform=ax.transAxes,
                ha='left', va=va, fontsize=13, fontweight='bold', color=conf['color'], bbox=bbox_props)
        
        # X 轴反转 (100 -> 70)
        ax.set_xlim(102, 68)
        
        # 简化网格
        ax.grid(True, linestyle='--', alpha=0.4)

    print(f"正在保存平滑趋势图至: {OUT_PNG}")
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_smooth_evolution()