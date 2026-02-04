import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib import rcParams
from matplotlib import font_manager as fm

# ================= 配置区域 =================
INPUT_FILE = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_RESCTX_OOF_pred_core_tfmr.csv"
OUTPUT_FIG = "Fig5-8_Shrinkage_Correction_Stats.png"

# ================= 数据处理 =================
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit()

df = pd.read_csv(INPUT_FILE)

# 1. 计算校正幅度 (Correction Delta)
# Delta = Final - Base (代表 Deep Model 对基线的修正量)
df['delta'] = df['final_pred'] - df['base_pred']
df['abs_delta'] = df['delta'].abs()

# 2. 按老化阶段分箱 (Binning by SoH)
# 创建 bins: 0.70 到 1.02，步长 0.02
bins = np.arange(0.70, 1.03, 0.02)
df['soh_bin'] = pd.cut(df['y_true'], bins=bins)

# 计算分箱统计量
bin_stats = df.groupby('soh_bin', observed=True).agg({
    'delta': 'mean',
    'abs_delta': 'mean',
    'y_true': 'count'
}).rename(columns={'y_true': 'count'})

# 获取 bin 的中心点用于绘图
bin_stats['soh_center'] = [i.mid for i in bin_stats.index]
# 过滤掉样本极少的空箱
bin_stats = bin_stats[bin_stats['count'] > 10]
# 按 SoH 降序排列 (从新到旧)
bin_stats = bin_stats.sort_values('soh_center', ascending=False)

# ================= 绘图逻辑 =================
sns.set_context("paper", font_scale=1.6)
sns.set_style("ticks")
    
   # 可选：检查字体是否存在（不存在就打印出来方便你装/改名）
available = {f.name for f in fm.fontManager.ttflist}
need = ["Times New Roman", "SimSun"]
missing = [x for x in need if x not in available]
if missing:
        print("警告：系统未发现字体：", missing)
        # 如果缺少 SimSun，可把 'SimSun' 改成 'NSimSun' 或 'Songti SC'(Mac) 等

    # 核心：按顺序设置字体家族（英文优先 TNR，中文回退宋体）
rcParams["font.family"] = ["Times New Roman", "SimSun"]  # 顺序很重要
rcParams["axes.unicode_minus"] = False                   # 负号正常显示
# 将 SimSun 加入到等宽字体序列中（放在英文等宽字体后面）
rcParams['font.monospace'] = ['SimHei'] 
# 或者更标准的等宽组合：

    # 数学公式也用 Times New Roman 风格更协调（可选）
rcParams["mathtext.fontset"] = "stix"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["mathtext.bf"] = "Times New Roman:bold"

    # 如果你想更“硬”一点：强制 sans-serif/serif 各自的首选
rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.sans-serif"] = ["SimSun"]

fig = plt.figure(figsize=(14, 6), dpi=300)
# 右图稍微宽一点，为了放双坐标轴
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.2)

ax_dist = fig.add_subplot(gs[0])
ax_segment = fig.add_subplot(gs[1])

# --- (a) 左图：Delta 分布 (Distribution) ---
# 主直方图
sns.histplot(df['delta'], bins=60, kde=True, color='#5C6BC0', ax=ax_dist, stat='density', alpha=0.6, linewidth=0)
ax_dist.axvline(0, color='k', linestyle='--', linewidth=1)

# Inset: 长尾特写 (Log Scale)
ax_ins = inset_axes(ax_dist, width="35%", height="25%", loc='upper left', borderpad=2)
sns.histplot(df['abs_delta'], bins=30, color='#EF5350', ax=ax_ins, element='step', fill=True, linewidth=0)
ax_ins.set_yscale('log')
ax_ins.set_xlabel("|Δ| (对数刻度)", fontsize=8, fontweight='bold')
ax_ins.set_ylabel("计数", fontsize=8,fontweight='bold',labelpad=-2)
ax_ins.set_title("长尾影响", fontsize=9, fontweight='bold')
ax_ins.tick_params(labelsize=8)

ax_dist.set_xlabel("校正幅度 Δ ", fontweight='bold', fontsize=11)
ax_dist.set_ylabel("密度", fontweight='bold', fontsize=11)
ax_dist.set_title("(a) 校正强度分布", fontweight='bold', fontsize=13, pad=12)
ax_dist.grid(True, linestyle=':', alpha=0.4)

# --- (b) 右图：分段特性 (Segmented Characteristics) ---
# 双坐标轴：左轴=修正量，右轴=样本数
ax_seg_count = ax_segment.twinx()


# 2. 绘制修正强度 (Mean |Delta|) - 红色实线
l1, = ax_segment.plot(bin_stats['soh_center'], bin_stats['abs_delta'], color='#D32F2F', marker='o', markersize=6, linewidth=2.5, label='平均 Δ (强度)')

# 3. 绘制修正方向 (Mean Delta) - 蓝色虚线
l2, = ax_segment.plot(bin_stats['soh_center'], bin_stats['delta'], color='#1976D2', marker='x', markersize=6, linestyle='--', linewidth=2, label='平均 Δ (方向)')

# 1. 绘制样本量背景 (Bars)
ax_seg_count.bar(bin_stats['soh_center'], bin_stats['count'], width=0.015, color="#95F1C3", label='样本量', alpha=0.4)
ax_seg_count.set_ylabel("样本量", color='#78909C', fontweight='bold', fontsize=11)
ax_seg_count.tick_params(axis='y', labelcolor='#78909C')
ax_seg_count.grid(False) 
# 假设样本量最大值是 max_count
max_count = bin_stats['count'].max()
ax_seg_count.set_ylim(0, max_count * 1.2) # 扩大 y 轴上限，使柱子看起来变矮了

# 装饰
ax_segment.set_xlabel("真实 SoH (老化阶段)", fontweight='bold', fontsize=11)
ax_segment.set_ylabel("校正幅度", fontweight='bold', fontsize=11)
ax_segment.set_title("(b) 按老化阶段的校正特性", fontweight='bold', fontsize=13, pad=12)
# 反转 X 轴：从 1.0 (新) 到 0.7 (旧)
ax_segment.set_xlim(1.02, 0.70)
ax_segment.grid(True, linestyle='--', alpha=0.5)
ax_segment.axhline(0, color='k', linestyle=':', linewidth=1)

# 图例合并
lines = [l1, l2, Patch(color='#ECEFF1', label='样本量')]
labels = [l.get_label() for l in lines]
ax_segment.legend(lines, labels, loc='upper right', fontsize=10, frameon=True, framealpha=0.9)
# 打印一些统计信息以供核对
print("-" * 30)
print(f"平均修正量 (Mean Delta): {df['delta'].mean():.6f}")
print(f"平均修正强度 (Mean |Delta|): {df['abs_delta'].mean():.6f}")
print("-" * 30)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches='tight')
print(f"图表已生成: {OUTPUT_FIG}")
plt.show()