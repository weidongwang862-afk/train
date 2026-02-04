import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
import platform
import glob
from matplotlib import rcParams
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.patches as mpatches
# ================= 配置区域 =================
# 1. 分段摘要文件 (用于画背景)
SEG_FILE = r"E:\RAW_DATA\outputs\02_segments\segment_summary_vin1.parquet"

# 2. 原始清洗数据的路径
# 选项 A: 指向文件夹 (自动读取文件夹下所有 part，适合内存大)
# RAW_PATH = r"E:\RAW_DATA\outputs\01_clean_core\vin1"

# 选项 B: 指向具体的某一个 part 文件 (速度快，画图足够了)
# 你可以直接用你刚上传的这个 part_00000.parquet 的绝对路径
RAW_PATH = r"E:\RAW_DATA\outputs\01_clean_core\vin1"

def merge_and_filter_segments(df_seg, t_min_win, t_max_win, min_dur_sec=60):
    # 先裁剪到窗口，再按state合并相邻段，并过滤短段
    seg = df_seg.copy()
    seg = seg[(seg["t_end"] >= t_min_win) & (seg["t_start"] <= t_max_win)].copy()
    if len(seg) == 0:
        return seg

    seg["t_start"] = seg["t_start"].clip(lower=t_min_win, upper=t_max_win)
    seg["t_end"]   = seg["t_end"].clip(lower=t_min_win, upper=t_max_win)
    seg = seg.sort_values("t_start").reset_index(drop=True)

    # state 规范化
    seg["state_norm"] = seg["state"].astype(str).str.lower()
    seg.loc[seg["state_norm"].str.contains("charge"), "state_norm"] = "charge"
    seg.loc[seg["state_norm"].str.contains("drive"),  "state_norm"] = "drive"
    seg.loc[seg["state_norm"].str.contains("rest"),   "state_norm"] = "rest"
    seg.loc[~seg["state_norm"].isin(["charge","drive","rest"]), "state_norm"] = "other"

    merged = []
    cur = seg.loc[0].to_dict()
    for i in range(1, len(seg)):
        r = seg.loc[i]
        if r["state_norm"] == cur["state_norm"] and r["t_start"] <= cur["t_end"]:
            cur["t_end"] = max(cur["t_end"], r["t_end"])
        else:
            merged.append(cur)
            cur = r.to_dict()
    merged.append(cur)

    out = pd.DataFrame(merged)
    out["dur"] = out["t_end"] - out["t_start"]
    out = out[out["dur"] >= min_dur_sec].reset_index(drop=True)
    return out
def break_on_gaps(x_minutes, y, gap_min=5.0):
    x = x_minutes.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    dx = np.diff(x)
    mask = np.ones_like(y, dtype=bool)
    # 断档后一段的第一个点设为断开：通过把该点置为 NaN 打断线段
    cut_idx = np.where(dx > gap_min)[0] + 1
    y2 = y.copy()
    y2[cut_idx] = np.nan
    return y2
def pick_continuous_window(df_raw, center_iloc, left_pts=1500, right_pts=3000, gap_sec=300):
    # 从center附近向两侧扩展，遇到大断档就停止，保证窗口连续
    t = df_raw["terminaltime"].to_numpy(dtype=float)
    n = len(df_raw)

    L = center_iloc
    while L > 0 and (t[L] - t[L-1]) <= gap_sec and (center_iloc - L) < left_pts:
        L -= 1

    R = center_iloc
    while R < n-1 and (t[R+1] - t[R]) <= gap_sec and (R - center_iloc) < right_pts:
        R += 1

    if R - L < 500:  # 连续段太短时，退回原策略
        L = max(0, center_iloc - left_pts)
        R = min(n-1, center_iloc + right_pts)

    return L, R

def plot_segmentation_ecg():
    print("Step 1: 读取分段摘要 (背景)...")
    if not os.path.exists(SEG_FILE):
        print(f"错误：找不到 {SEG_FILE}")
        return
    try:
        df_seg = pd.read_parquet(SEG_FILE)
        # 确保按时间排序
        df_seg = df_seg.sort_values("t_start")
    except Exception as e:
        print(f"读取分段失败: {e}")
        return

    print(f"Step 2: 读取原始波形数据 from {RAW_PATH} ...")
    try:
        # 如果是文件夹，Pandas 会尝试读取所有 parquet
        # 如果是单文件，就读单文件
        # 只读需要的列，飞快
        cols = ["terminaltime", "totalcurrent", "speed"]
        df_raw = pd.read_parquet(RAW_PATH, columns=cols)
        
        # 确保按时间排序 (因为 part 之间可能乱序)
        df_raw = df_raw.sort_values("terminaltime")
        
        print(f"成功读取 {len(df_raw)} 条原始数据。")
        
    except Exception as e:
        print(f"读取原始波形失败: {e}")
        print("建议：请检查路径是否正确，或者缺少 pyarrow 库 (pip install pyarrow)")
        return

    # ================= Step 3: 截取一段“精彩片段” =================
    # 我们不需要画几百万行，只需要截取一段包含 [行驶 -> 充电 -> 静置] 的混合工况
    # 策略：在 df_raw 里找一段电流变化丰富的时间窗
    
    # 找一段有充电 (I < -10) 的时刻
    charge_indices = df_raw[df_raw["totalcurrent"] < -10].index
    if len(charge_indices) > 0:
        # 以前第一个充电点为中心，前后各取 1 小时
        center_idx = charge_indices[0]
        # 找对应的行号位置
        iloc_idx = df_raw.index.get_loc(center_idx)
        
       # 优先截取“连续记录段”，避免跨断档导致长直线
        start_i, end_i = pick_continuous_window(df_raw, iloc_idx, left_pts=1500, right_pts=3500, gap_sec=300)
        df_plot = df_raw.iloc[start_i : end_i + 1].copy()
    else:
        # 如果没充电，就随便取中间一段
        mid = len(df_raw) // 2
        df_plot = df_raw.iloc[mid : mid+4000].copy()
        
    if len(df_plot) == 0:
        print("数据为空，无法绘图")
        return

    # 准备 X 轴 (分钟)
    t0 = df_plot["terminaltime"].min()
    x_time = (df_plot["terminaltime"] - t0) / 60.0
    y_current = df_plot["totalcurrent"]
    y_speed = df_plot["speed"]
    # 断档处打断曲线，避免长直线误导
    y_current_plot = break_on_gaps(x_time, y_current, gap_min=5.0)
    y_speed_plot   = break_on_gaps(x_time, y_speed,   gap_min=5.0)


    # ================= Step 4: 绘图设置 (期刊风格) =================
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
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 4.1 绘制背景工况色块
    # 筛选出落在当前绘图时间窗内的 segments
    t_min_win = df_plot["terminaltime"].min()
    t_max_win = df_plot["terminaltime"].max()
    
    segs_in_window = merge_and_filter_segments(df_seg, t_min_win, t_max_win, min_dur_sec=60)


    
    color_map = {
        "charge": "#E8F5E9", # 浅绿 (充电)
        "drive":  "#FFF3E0", # 浅橙 (行驶)
        "rest":   "#ECEFF1", # 浅灰 (静置)
        "other":  "#FAFAFA"
    }
    
    print(f"正在绘制背景，当前窗口包含 {len(segs_in_window)} 个工况段...")
    
    for _, row in segs_in_window.iterrows():
        # 确定状态颜色
        st = str(row["state"]).lower()
        c = "#FFFFFF"
        if "charge" in st: c = color_map["charge"]
        elif "drive" in st: c = color_map["drive"]
        elif "rest" in st: c = color_map["rest"]
        
        # 计算起止坐标 (分钟)
        t_s = max(row["t_start"], t_min_win)
        t_e = min(row["t_end"], t_max_win)
        x0 = (t_s - t0) / 60.0
        x1 = (t_e - t0) / 60.0
        
        # 画矩形
        ax1.axvspan(x0, x1, facecolor=c, alpha=1.0, edgecolor='none', zorder=0)
        # 选一个示例充电事件窗口：窗口内最长 charge 段
        
    if len(segs_in_window) > 0:
        chg = segs_in_window[segs_in_window["state_norm"] == "charge"].copy()
        if len(chg) > 0:
            row = chg.sort_values("dur", ascending=False).iloc[0]
            xs = (row["t_start"] - t0) / 60.0
            xe = (row["t_end"] - t0) / 60.0
            ax1.axvline(xs, color="0.25", linestyle="--", linewidth=0.9, zorder=4)
            ax1.axvline(xe, color="0.25", linestyle="--", linewidth=0.9, zorder=4)
            ax1.axvspan(xs, xe, color="#90CAF9", alpha=0.12, edgecolor="none", zorder=1)
            ax1.text((xs + xe)/2, ax1.get_ylim()[1], "充电事件窗口（用于容量标签计算）",
                    ha="center", va="bottom", fontsize=10, color="0.25", zorder=4)
            # ymin, ymax = ax1.get_ylim()
            # ax1.text(
            #     (xs + xe) / 2,
            #     ymax - 0.06 * (ymax - ymin),
            #     "充电事件窗口（用于容量标签计算）",
            #     ha="center", va="center_baseline", fontsize=10, color="0.25",
            #     bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", alpha=0.85),
            #     zorder=6
            # )
            

    # 4.2 绘制波形 (心电图)
    # 左轴：电流 (蓝色)
    ln1 = ax1.plot(x_time, y_current_plot, color='#1565C0', linewidth=1.2, label='总电流 (A)', zorder=2)
    ax1.set_xlabel("时间 (Minutes)", fontweight='bold')
    ax1.set_ylabel("电流 (A)", fontweight='bold', color='#1565C0')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    cur = df_plot["totalcurrent"].to_numpy()
    cur = cur[np.isfinite(cur)]
    if len(cur) > 50:
        lo = np.percentile(cur, 1)
        hi = np.percentile(cur, 99)
        pad = 0.10 * (hi - lo + 1e-6)
        ax1.set_ylim(lo - pad, hi + pad)

    
    # 右轴：车速 (灰色虚线)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(x_time, y_speed_plot, color='#78909C', linewidth=0.9, linestyle='--', alpha=0.55, label='车速 (km/h)', zorder=1)
    ax2.set_ylabel("车速 (km/h)", fontweight='bold', color='#78909C')
    ax2.tick_params(axis='y', labelcolor='#78909C',colors="0.45")

    # 4.3 装饰
    ax1.set_title("工况分段与事件窗口定义示意图", fontweight='bold', fontsize=16, pad=15)
    
    # 构建图例
    # 色块图例
    patches = [
        mpatches.Patch(color=color_map['charge'], label='充电 (Charge)'),
        mpatches.Patch(color=color_map['drive'], label='行驶 (Drive)'),
        mpatches.Patch(color=color_map['rest'], label='静置 (Rest)'),
    ]
    # 线条图例 (合并 ln1 和 ln2)
    lines = [
        plt.Line2D([0], [0], color='#1565C0', lw=1.5, label='电流 (Current)'),
        plt.Line2D([0], [0], color='#78909C', lw=1.5, ls='--', label='车速 (Speed)')
    ]
    
    ax1.legend(handles=patches + lines, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               frameon=False, ncol=5, fontsize=11)
    win_patch = mpatches.Patch(color="#90CAF9", alpha=0.12, label="事件窗口 (Label window)")
    handles = patches + [win_patch] + lines
    
    plt.tight_layout()
    save_path = 'Fig3-2_工况分段心电图.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"成功生成图表: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_segmentation_ecg()