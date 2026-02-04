import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
import platform
import numpy as np
from matplotlib import rcParams
from matplotlib import font_manager as fm

# ================= 配置区域 =================
INPUT_FILE = r"E:\RAW_DATA\outputs\03_labels\labels_post_vin1.parquet" # <--- 修改为你的文件路径

def plot_label_restoration_journal():
    print(f"正在读取 {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    try:
        # 尝试读取 Parquet
        df = pd.read_parquet(INPUT_FILE)
        print("成功读取文件。")
    except Exception as e:
        print(f"读取失败 (请确保安装 pyarrow): {e}")
        return

    # ================= 1. 智能查找 X 轴列名 =================
    x_col = None
    xlabel = "索引"
    
    candidate_cols = {
    "totalodometer": "总行驶里程 (10⁴ km)",
    "odo_end": "总行驶里程 (10⁴ km)",
    "odo": "总行驶里程 (10⁴ km)",
    "t_end": "时间 (Days)",
}



    for col, label in candidate_cols.items():
        if col in df.columns:
            x_col = col
            # 必须排序，否则连线会乱
            df = df.sort_values(x_col)
            
            if "odo" in col:
                x = df[col] / 10000.0
                xlabel = label
            elif "t" in col: # 时间列
                # 将时间戳转换为相对天数，更直观
                t_min = df[col].min()
                x = (df[col] - t_min) / (3600 * 24)
                xlabel = "时间 (Days)"
            else:
                x = df[col]
                xlabel = label
            break
            
    if x_col is None:
        x = np.arange(len(df))
        xlabel = "充电事件序号 (Event Index)"

    # ================= 2. 绘图风格设置 =================
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

    # 数学公式也用 Times New Roman 风格更协调（可选）
    rcParams["mathtext.fontset"] = "stix"
    rcParams["mathtext.rm"] = "Times New Roman"
    rcParams["mathtext.it"] = "Times New Roman:italic"
    rcParams["mathtext.bf"] = "Times New Roman:bold"

    # 如果你想更“硬”一点：强制 sans-serif/serif 各自的首选
    rcParams["font.serif"] = ["Times New Roman"]
    rcParams["font.sans-serif"] = ["SimSun"]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # ================= 3. 绘制核心图层 (Layered Plotting) =================
    
    # Layer 1: 原始噪声数据 (Raw) - 作为背景
    if "C_est_ah" in df.columns:
        # 极低透明度 + 小点，防止遮挡
        ax.scatter(x, df["C_est_ah"], color='#90A4AE', alpha=0.15, s=12, 
                   edgecolors='none', label='原始容量 (Raw data)')
    
    # Layer 2: Hampel 滤波后 (Step 1) - 散点展示
    if "C_hampel" in df.columns:
        # 关键修改：用 scatter 而不是 plot，消除竖向连线尖刺
        # 颜色用清爽的蓝色
        ax.scatter(x, df["C_hampel"], color='#29B6F6', alpha=0.5, s=15, 
                   marker='o', label='步骤1: 离群点剔除 (Outlier removal)')

    # Layer 3: LOWESS 平滑后 (Step 2) - 实线
    if "C_smooth" in df.columns:
        # 橙色实线，代表局部趋势
        ax.plot(x, df["C_smooth"], color='#FF9800', linewidth=2.5, alpha=0.9, 
                label='步骤2: 平滑处理 (Smoothing)')

    # Layer 4: 单调趋势 (Step 3) - 顶层红线
    if "C_trend" in df.columns:
        y_trend = df["C_trend"].values
        # 关键修改：强制施加 visual strict monotony
        # 这样即使原始数据有微小波动，画出来也是严格平滑向下的
        y_trend_strict = np.minimum.accumulate(y_trend)
        
        ax.plot(x, y_trend_strict, color='#D32F2F', linewidth=3.5, 
                zorder=10, # 保证压在所有图层最上面
                label='步骤3: 单调趋势 (Isotonic/Monotone)')

    # ================= 4. 装饰与保存 =================
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    ax.set_ylabel("电池容量 (Ah)", fontweight='bold', fontsize=14)
    ax.set_title("容量标签处理流程", fontweight='bold', fontsize=16, pad=15)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例优化：去掉边框，放在右上角
    leg = ax.legend(frameon=True, loc='upper right', fontsize=11)
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_alpha(0.95)

    
    # 紧凑布局
    plt.tight_layout()
    
    save_path = 'Fig3-3_Journal_Final.png'
    plt.savefig(save_path, dpi=300)
    print(f"成功生成期刊级高清图表: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_label_restoration_journal()