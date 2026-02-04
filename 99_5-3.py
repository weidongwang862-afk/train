import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib import font_manager as fm
# =========================
# 1) 配置区域
# =========================
# 修改为你的 CSV 所在目录
BASE_DIR = r"E:\RAW_DATA\outputs" 
OUT_DIR  = r"E:\RAW_DATA"
os.makedirs(OUT_DIR, exist_ok=True)
TRUE_COL = "y_true"     # 按你的文件实际改
PRED_COL_CANDIDATES = ["pred_core", "y_pred", "final_pred", "pred"]
VIN_COL  = "vin"        # 后面做 macro 用
AGG_MODE = "macro"   # "micro" 或 "macro"
MIN_COUNT_PER_BIN = 15



# 模型列表：(显示名称, 文件名匹配模式)
# 建议顺序：基线 -> 普通深度 -> 高级深度 (画图时图例会按这个顺序)
MODEL_FILES = [
    ("HGBR",        "predictions_test_FULL_pred_core_HGBR_OOF.csv"),
    ("GRU", "predictions_test_SEQ_GRU.csv"),
    ("BiGRU",    "predictions_test_SEQ_BIGRU.csv"),
    ("ConvGRU",  "predictions_test_SEQ_CONVGRU.csv"),
    ("Transformer",     "predictions_test_SEQ_TFMR.csv"),
    ("Fusion",   "predictions_test_SEQ_FUSION.csv"),
]

# 视觉样式配置 (Visual Hierarchy)
# 格式: {模型名关键词: {颜色, 线型, 标记, 线宽, zorder}}
STYLE_MAP = {
    "HGBR":      {"c": "#455A64", "ls": "--", "mk": "o", "lw": 2.0, "z": 1}, # 基线：深灰虚线
    "GRU":       {"c": "#42A5F5", "ls": ":",  "mk": "^", "lw": 1.5, "z": 2}, # 蓝色点线
    "BiGRU":     {"c": "#1E88E5", "ls": ":",  "mk": "v", "lw": 1.5, "z": 2}, # 深蓝点线
    "ConvGRU":   {"c": "#00ACC1", "ls": ":",  "mk": "d", "lw": 1.5, "z": 2}, # 青色点线
    "Transformer": {"c": "#FB8C00", "ls": "-",  "mk": "s", "lw": 2.2, "z": 3}, # 橙色实线
    "Fusion":    {"c": "#D32F2F", "ls": "-",  "mk": "*", "lw": 2.5, "z": 4}, # 红色实线 (主角)
}

# SOH 分箱设置
BIN_STEP = 0.02
SOH_RANGE = (0.86, 1.08) 

# =========================
# 2) 核心处理函数
# =========================
def find_file(pattern):
    # 递归查找
    hits = glob.glob(os.path.join(BASE_DIR, "**", pattern), recursive=True)
    if not hits:
        hits = glob.glob(pattern) # 尝试当前目录
    if not hits:
        return None
    # 取最新的
    hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return hits[0]

def get_binned_stats(name, pattern):
    # 1. --- 查找文件 ---
    path = find_file(pattern)
    if not path:
        print(f"Warning: 找不到模型 {name} 的文件")
        return None
    
    print(f"[{name}] 读取: {os.path.basename(path)}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[{name}] 读取失败: {e}")
        return None
        
    # 2. --- 动态匹配列名 ---
    # 找预测列：遍历候选列表
    pred_col = next((c for c in PRED_COL_CANDIDATES if c in df.columns), None)
    # 找真实列
    true_col = TRUE_COL if TRUE_COL in df.columns else None

    # 检查是否缺列
    if not true_col or not pred_col:
        print(f"[{name}] 缺列报错！")
        print(f"   - 需要真实列: {TRUE_COL} (状态: {'√' if true_col else '×'})")
        print(f"   - 需要预测列 (任一): {PRED_COL_CANDIDATES} (状态: 全部缺失)")
        print(f"   - 文件现有列: {list(df.columns)}")
        return None

    print(f"   -> 使用列: 真值='{true_col}', 预测='{pred_col}'")

    # 3. --- 核心计算逻辑 ---
    # 计算误差
    df["error"] = df[pred_col] - df[true_col]
    df["abs_error"] = df["error"].abs()
    
    # 分箱 (只需执行一次)
    bins = np.arange(SOH_RANGE[0], SOH_RANGE[1] + 1e-9, BIN_STEP)
    df["bin"] = pd.cut(df[true_col], bins=bins)
    
    # 4. --- 聚合计算 (Micro/Macro) ---
    if AGG_MODE == "micro":
        stats = df.groupby("bin", observed=True).agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            count=(true_col, "count"),
        ).reset_index()

    else:  # macro
        if VIN_COL not in df.columns:
            # 如果是 Macro 模式但缺 VIN 列，降级为 Micro 或报错
            print(f"[{name}] 警告: 缺少 VIN 列 '{VIN_COL}'，无法进行 Macro 聚合，降级为 Micro。")
            stats = df.groupby("bin", observed=True).agg(
                mae=("abs_error", "mean"),
                bias=("error", "mean"),
                count=(true_col, "count"),
            ).reset_index()
        else:
            # 先按 车+箱 聚合
            per = df.groupby([VIN_COL, "bin"], observed=True).agg(
                mae=("abs_error", "mean"),
                bias=("error", "mean"),
                count=(true_col, "count"),
            ).reset_index()
            # 再按 箱 聚合
            stats = per.groupby("bin", observed=True).agg(
                mae=("mae", "mean"),
                bias=("bias", "mean"),
                count=("count", "sum"),
                mae_std=("mae", "std"),      
                bias_std=("bias", "std"),
            ).reset_index()

    # 5. --- 后处理与坐标生成 ---
    # 过滤稀疏箱
    stats = stats[stats["count"] > MIN_COUNT_PER_BIN].reset_index(drop=True)
    
    # 确定分箱列名 (通常是 'bin')
    bin_col = "bin" if "bin" in stats.columns else ("bin_range" if "bin_range" in stats.columns else None)
    if bin_col is None:
        print(f"[{name}] 错误: 聚合后找不到分箱列。")
        return None

    # 计算 x 轴坐标 (箱中点)
    stats["x"] = stats[bin_col].apply(lambda r: r.mid).astype(float)
    stats["bin_label"] = stats[bin_col].astype(str)
    
    # 6. --- 关键修复：只返回 DataFrame ---
    # 你的 main 函数期望得到 stats DataFrame，而不是元组
    return stats

    

def main():
    # 1. 读取所有模型数据
    data_map = {}
    sample_counts = None # 用于画背景
    
    for name, pat in MODEL_FILES:
        stats = get_binned_stats(name, pat)
        if stats is not None:
            data_map[name] = stats
            # 用第一个读到的模型(通常是HGBR)的样本量作为背景
            if sample_counts is None:
                sample_counts = stats[["x", "count"]]

    if not data_map:
        print("错误：没有读取到任何有效数据。")
        return

    # =========================
    # 3) 绘图 (High-Level Journal Style)
    # =========================
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True, dpi=300)
    
    # --- 公共背景：样本量 ---
    if sample_counts is not None:
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(sample_counts["x"], 0, sample_counts["count"], 
                              color='#B0BEC5', alpha=0.15, zorder=0, label='Sample Count')
        ax1_twin.set_ylabel("样本分布 (Sample Distribution)", color='#78909C', fontsize=11)
        ax1_twin.tick_params(axis='y', colors='#78909C')
        ax1_twin.grid(False)
        # 将 twinx 放到最底层
        ax1.set_zorder(ax1_twin.get_zorder()+1)
        ax1.patch.set_visible(False)

    # --- 循环绘制曲线 ---
    # 为了图例顺序，我们按 MODEL_FILES 的顺序画
    
    for name, _ in MODEL_FILES:
        if name not in data_map: continue
        
        df = data_map[name]
        st = STYLE_MAP.get(name, {"c": "gray", "ls": "-", "mk": "", "lw": 1, "z": 1})
        
        # Subplot 1: MAE
        ax1.plot(df["x"], df["mae"], 
                 color=st["c"], linestyle=st["ls"], marker=st["mk"], 
                 linewidth=st["lw"], markersize=5, alpha=0.9, zorder=st["z"]+10,
                 label=name)
        
        # Subplot 2: Bias
        ax2.plot(df["x"], df["bias"], 
                 color=st["c"], linestyle=st["ls"], marker=st["mk"], 
                 linewidth=st["lw"], markersize=5, alpha=0.9, zorder=st["z"]+10,
                 label=name)

    # --- 装饰 Subplot 1 (MAE) ---
    ax1.set_ylabel("MAE (平均绝对误差)", fontweight='bold')
    ax1.set_title("(a) 误差分析:不同老化阶段的平均绝对误差", fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 图例 (两列显示，放在顶部)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), 
               ncol=3, frameon=True, fancybox=False, edgecolor='gray', fontsize=10)

    # --- 装饰 Subplot 2 (Bias) ---
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.8) # 零线
    ax2.set_ylabel("Bias (预测偏差)", fontweight='bold')
    ax2.set_xlabel("真实 SOH (老化阶段)", fontweight='bold')
    ax2.set_title("(b) 校准分析:老化阶段的偏差", fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 标注长尾改进 (找出 Fusion 和 HGBR 在最左侧的差距)
    if "Fusion" in data_map and "HGBR" in data_map:
        df_base = data_map["HGBR"]
        df_best = data_map["Fusion"]
        # 找最小的 x (Low SOH)
        min_idx = df_base["x"].idxmin()
        x_pos = df_base.loc[min_idx, "x"]
        y_base = df_base.loc[min_idx, "bias"]
        
        # 对应 Fusion 的值
        # 注意：需要对齐 index 或插值，这里简单取最左侧近似
        y_best = df_best.loc[df_best["x"].idxmin(), "bias"]
        
        ax2.annotate("Bias Corrected", 
                     xy=(x_pos, y_best), 
                     xytext=(x_pos + 0.004, y_best + (0.001 if y_best>0 else 0.00008)),
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#D32F2F', lw=1.5),
                     color='#D32F2F', fontweight='bold', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Fig5-3_MultiModel_SoH_Comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"图表已生成: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()