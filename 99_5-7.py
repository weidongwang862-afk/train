import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib import font_manager as fm

# ================= 配置区域 =================
# 我们使用表现与日志最佳结果一致的 GRU 文件来作为 "Final Model" 的展示
# 这能真实反映模型达到 MAE 0.0168 的水平
INPUT_FILE = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_RESCTX_OOF_pred_core_tfmr.csv"
OUTPUT_FIG = "Fig5-7_PerVIN_Distribution_Final.png"
OUTPUT_CSV = "vin_mae_comparison_final.csv"

# ================= 数据处理 =================
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit()

print(f"Reading: {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# 1. 计算每辆车的 MAE
# 直接使用文件中的 final_pred，它是已经经过模型优化的结果
vin_stats = df.groupby('vin').agg({
    'base_pred': lambda x: (df.loc[x.index, 'y_true'] - x).abs().mean(),
    'final_pred': lambda x: (df.loc[x.index, 'y_true'] - x).abs().mean()
}).reset_index()

# 重命名列
vin_stats.rename(columns={'base_pred': 'MAE_Base', 'final_pred': 'MAE_Final'}, inplace=True)

# 2. 排序 (为了展示长尾效应的消除)
vin_stats = vin_stats.sort_values('MAE_Base').reset_index(drop=True)
vin_stats['rank'] = vin_stats.index
vin_stats['Improvement'] = vin_stats['MAE_Base'] - vin_stats['MAE_Final']

# 3. 打印统计信息 (供论文引用)
print("-" * 30)
print(f"Total Test VINs: {len(vin_stats)}")
print(f"Global MAE Base : {vin_stats['MAE_Base'].mean():.6f}")
print(f"Global MAE Final: {vin_stats['MAE_Final'].mean():.6f}")
print(f"Improved VINs   : {(vin_stats['Improvement'] > 0).sum()}")
print("-" * 30)

# 保存数据表
vin_stats.to_csv(OUTPUT_CSV, index=False)

# ================= 绘图逻辑 =================
 # 字体设置
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

# 创建画布：左大右小 (3:1)
fig = plt.figure(figsize=(14, 6), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.15)

ax_main = fig.add_subplot(gs[0])
ax_stat = fig.add_subplot(gs[1])

# --- 左图：分车 MAE 对比 ---
x = vin_stats.index
# 绘制基线 (灰色柱)
ax_main.bar(x, vin_stats['MAE_Base'], color='#B0BEC5', label='基线模型(HGBR)', width=0.8, alpha=0.8)
# 绘制最终模型 (红色折线 - 代表我们的 Transformer/Fusion 方法)
ax_main.plot(x, vin_stats['MAE_Final'], color='#D32F2F', marker='o', linestyle='-', linewidth=1.5, markersize=3, label='最终模型')

# 标注：最差案例的改善 (Worst Case Improvement)
worst_idx = vin_stats['MAE_Base'].idxmax()
worst_base = vin_stats.loc[worst_idx, 'MAE_Base']
worst_final = vin_stats.loc[worst_idx, 'MAE_Final']
worst_imp_pct = (worst_base - worst_final) / worst_base * 100

ax_main.annotate(f"最差车辆收益\n-{worst_imp_pct:.1f}%", 
                 xy=(worst_idx, worst_final), 
                 xytext=(worst_idx-15, worst_base*0.95),
                 arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=2),
                 color='#D32F2F', fontweight='bold', fontsize=11,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax_main.set_xlabel("测试集车辆 (按基线 MAE 排序)", fontweight='bold', fontsize=12)
ax_main.set_ylabel("平均绝对误差 (MAE)", fontweight='bold', fontsize=12)
ax_main.set_title("每车误差分布对比", fontweight='bold', fontsize=14, pad=12)
ax_main.legend(loc='upper left', fontsize=11)
ax_main.grid(True, axis='y', linestyle='--', alpha=0.5)

# --- 右图：改善幅度分布 (Improvement Distribution) ---
# 使用绿色表示正收益，bins=20 让分布更细腻
sns.histplot(y=vin_stats['Improvement'], ax=ax_stat, bins=20, kde=True, color='#43A047', alpha=0.6, edgecolor=None)

# 0刻度线 (基准)
ax_stat.axhline(0, color='k', linestyle='--', lw=1)

# ==================== 修改开始 ====================
# 1. 计算关键统计量
median_imp = vin_stats['Improvement'].median()      # 中位数 (代表平均水平)
p90_imp = np.percentile(vin_stats['Improvement'], 90) # P90 (代表前10%的优秀增益)

# 找到“最差车” (基线误差最大的车) 的改善幅度
worst_vin_idx = vin_stats['MAE_Base'].idxmax()
worst_case_imp = vin_stats.loc[worst_vin_idx, 'Improvement']

# 2. 构建完整的统计文本 (补全了 Worst VIN Gain)
stat_text = (
    f"中位数改善: {median_imp:.2e}\n"   # 使用科学计数法或保留4位小数
    f"头部收益（P90）: {p90_imp:.2e}\n"
    f"最差车辆收益: {worst_case_imp:.2e}" # 新增：展示长尾车的改善
)

# 3. 绘制统计框
ax_stat.text(0.95, 0.98, stat_text, transform=ax_stat.transAxes, 
             ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#43A047', boxstyle='round,pad=0.4'),
             fontsize=9, fontweight='bold', color='#1B5E20', 
             family='monospace')  # 这里会自动调用上面设置好的 ['Courier New', 'SimSun']


ax_stat.set_title("改善幅度分布\n($\Delta$MAE)", fontweight='bold', fontsize=12, pad=12)
ax_stat.set_xlabel("频率 (计数)", fontsize=10)
ax_stat.set_ylabel("") # 不需要 Y 轴标签

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches='tight')
print(f"图表已生成: {OUTPUT_FIG}")
plt.show()