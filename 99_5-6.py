import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
import platform
import numpy as np
from matplotlib import rcParams
from matplotlib import font_manager as fm

# ====== 1. 设置中文字体 ======
# 优先寻找系统中存在的字体，避免豆腐块
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

    # 数学公式也用 Times New Roman 风格更协调（可选）
rcParams["mathtext.fontset"] = "stix"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["mathtext.bf"] = "Times New Roman:bold"

    # 如果你想更“硬”一点：强制 sans-serif/serif 各自的首选
rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False 

# ====== 2. 数据准备 ======
maes = np.array([0.021863, 0.017273, 0.016600, 0.016559, 0.016460])
labels_en = ["Baseline (HGBR)", "+ OOF Residual", "+ Seq Context", "+ Attention", "+ Opt Shrinkage"]
labels_cn = ["基线", "残差校正", "序列化", "注意力机制", "最优收缩"]

x = np.arange(len(maes))
deltas = np.r_[0.0, np.diff(maes)] 
starts = np.r_[maes[0], maes[:-1]] 
ends = maes.copy()

# ====== 3. 绘图设置 (清爽专业风) ======
plt.rcParams["figure.dpi"] = 150
fig, ax = plt.subplots(figsize=(10, 5.8))

# --- 定义新配色 ---
c_base  = "#3949AB"  # 基线：深靛蓝 (Indigo)
c_impr  = "#00897B"  # 改善：蓝绿色/鸭翅绿 (Teal) - 清新且护眼
c_line  = "#263238"  # 线条：深炭灰
c_text_val = "#212121" # MAE数值颜色
c_text_pct = "#00695C" # 百分比颜色 (与柱子同色系但更深)

bar_w = 0.5

# --- A. 绘制柱子 ---
# 1. 基线柱
ax.bar(x[0], maes[0], width=bar_w, color=c_base, edgecolor=c_line, 
       linewidth=0.8, zorder=3, alpha=0.85, label="Baseline")

# 2. 变化柱 (全是下降/改善，统一用一种颜色)
ax.bar(x[1:], deltas[1:], bottom=starts[1:], width=bar_w,
       color=c_impr, edgecolor=c_line, linewidth=0.8, zorder=3, alpha=0.85)

# --- B. 辅助虚线 (Waterfall Connector) ---
# 连接前一根柱顶和当前柱顶
for i in range(len(maes) - 1):
    ax.plot([x[i], x[i+1]], [maes[i], maes[i]], 
            color="gray", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

# --- C. 趋势折线 ---
ax.plot(x, maes, marker="o", markersize=6, linewidth=1.5, color=c_line, 
        markerfacecolor="white", markeredgewidth=1.5, zorder=4)

# --- D. 智能标注 (文字不重叠的关键) ---
# 计算Y轴跨度，用于动态调整偏移量
yr = maes.max() - maes.min()
offset_up = 0.12 * yr    # 向上偏移量 (放MAE)
offset_down = 0.12 * yr  # 向下偏移量 (放百分比)

for i, y in enumerate(maes):
    # 1. 上方：绝对数值 (MAE)
    # 将数值放在黑点上方，避开柱子区域
    ax.text(i, y + offset_up, f"{y:.6f}", 
            ha="center", va="bottom", fontsize=10, 
            fontweight='bold', color=c_text_val)
    
    # 2. 下方：变化百分比 (跳过基线)
    if i > 0:
        pct = (maes[i] / maes[i-1] - 1.0) * 100.0
        # 将百分比放在黑点下方，这样就完全移出了柱子(Box)内部
        ax.text(i, y - offset_down, f"{pct:.2f}%", 
                ha="center", va="top", fontsize=9.5, 
                fontweight='normal', color=c_text_pct, style='italic')

# --- E. 总改善标注 (箭头与总结) ---
total_pct = (maes[-1] / maes[0] - 1.0) * 100.0
# 使用圆角框框住结论，放在右上角空白处
ax.annotate(
    f"总体改善\n{total_pct:.2f}%",
    xy=(len(maes)-1, maes[-1]), xycoords='data', # 箭头指向最后一个点
    xytext=(len(maes)-0.5, maes.max() + 0.4*yr), textcoords='data', # 文本位置
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                    color="#EF5350", lw=2), # 红色箭头突出重点
    ha="center", va="center", fontsize=11, color="#C62828", fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#EF9A9A", alpha=0.9) # 浅红背景框
)

# --- F. 坐标轴美化 ---
ax.set_title("各模块逐步贡献分析", 
             fontsize=15, fontweight='bold', pad=25, color="#37474F")
ax.set_ylabel("MAE (测试集)", fontsize=12, labelpad=10, color="#37474F")

# X轴标签
xt = [f"{en}\n{cn}" for en, cn in zip(labels_en, labels_cn)]
ax.set_xticks(x)
ax.set_xticklabels(xt, fontsize=10)

# 动态调整Y轴范围，确保文字不被切掉
ax.set_ylim(maes.min() - 0.5*yr, maes.max() + 0.8*yr)

# 背景网格
ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CFD8DC")
ax.spines["bottom"].set_color("#CFD8DC")

plt.tight_layout()
plt.show()
fig.savefig("Fig5-6stepwise_contribution.png", dpi=300, facecolor='white', pad_inches=0.1)