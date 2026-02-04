import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import platform

# ================= 配置区域 =================
# 指向你生成的最终冻结数据集
INPUT_FILE = "E:\\RAW_DATA\\outputs\\04_features\\dataset_all_C_frozen.parquet"
ODO_UNIT = "km"   # "km" 或 "m"


# 如果车辆太多，是否只画前 N 辆？
# 建议：False (画所有车)，这样能看出整体分布的“长尾”效应，更有说服力
SHOW_TOP_N = False 
TOP_N = 50

def plot_data_heterogeneity_corrected():
    print(f"正在读取数据集: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    try:
        # 智能读取列
        import pyarrow.parquet as pq
        file_cols = pq.read_schema(INPUT_FILE).names
        
        cols_needed = ["vin"]
        # 找里程列
        odo_col = next((c for c in ["totalodometer", "odo_end", "odo"] if c in file_cols), None)
        if odo_col: cols_needed.append(odo_col)
        # 找 SOH 列
        soh_col = next((c for c in ["SoH_trend", "SoH", "C_trend", "capacity_ah"] if c in file_cols), None)
        if soh_col: cols_needed.append(soh_col)
        
        df = pd.read_parquet(INPUT_FILE, columns=cols_needed)
        print(f"成功读取 {len(df)} 行数据。")
        
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # ================= 数据聚合 =================
    print("正在计算每辆车的统计指标...")
    
    agg_list = []
    grouped = df.groupby("vin")
    
    for vin, group in grouped:
        n_samples = len(group)
        
        # 计算里程跨度 (万公里)
        if odo_col:
            odo_span = (group[odo_col].max() - group[odo_col].min())
            if ODO_UNIT == "km":
                odo_span = odo_span / 10000.0          # -> 10^4 km
            else:
                odo_span = odo_span / 1e7              # m -> 10^4 km（因为 10^4 km = 10^7 m）
        else:
            odo_span = 0
            
        # 提取 SOH 数据用于画箱线图
        if soh_col:
            soh_values = group[soh_col].dropna().values
        else:
            soh_values = []
            
        agg_list.append({
            "vin": vin,
            "count": n_samples,
            "odo_span": odo_span,
            "soh_values": soh_values
        })
    
    df_agg = pd.DataFrame(agg_list)
    
    # 按样本数量降序排列 (符合长尾分布规律)
    df_agg = df_agg.sort_values("count", ascending=False).reset_index(drop=True)
    
    if SHOW_TOP_N:
        df_agg = df_agg.head(TOP_N)

    # 准备绘图数据
    x = range(len(df_agg))
    bar_width = 0.8 if len(df_agg) < 50 else 1.0

    # ================= 绘图风格 (期刊标准) =================
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    # 字体混排设置
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建 3 行 1 列的三联图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # --- 子图 A: 样本规模 (Scale) ---
    ax1.bar(x, df_agg["count"], color='#3F51B5', width=bar_width, alpha=0.9, label='样本量')
    ax1.set_ylabel("样本数量 (个)", fontweight='bold')
    # 修正标题为中文，符合图 3-1 的定位
    ax1.set_title("(a) 各车辆样本规模分布 (Sample Size Distribution)", fontweight='bold', pad=10, loc='left')
    ax1.grid(axis='x', visible=False)
    
    # 标注最大值
    max_y = df_agg["count"].max()
    ax1.text(0, max_y, f"Max: {max_y}", ha='left', va='bottom', color='#3F51B5', fontsize=10)

    # --- 子图 B: 里程跨度 (Coverage) ---
    ax2.bar(x, df_agg["odo_span"], color='#009688', width=bar_width, alpha=0.9, label='里程跨度')
    ax2.set_ylabel("里程跨度 (10⁴ km)", fontweight='bold')
    ax2.set_title("(b) 各车辆行驶里程覆盖范围 (Mileage Coverage)", fontweight='bold', pad=10, loc='left')
    ax2.grid(axis='x', visible=False)


    # --- 子图 C: SOH 异质性 (Heterogeneity) ---
    # 箱线图
    box_data = df_agg["soh_values"].tolist()

    # 只保留有标签的车辆，避免用0伪造数据
    has_label = [len(v) > 0 for v in box_data]
    box_data = [v for v, ok in zip(box_data, has_label) if ok]
    pos = [i for i, ok in enumerate(has_label) if ok]

    flierprops = dict(marker='o', markerfacecolor='gray', markersize=1,
                    linestyle='none', alpha=0.2, markeredgewidth=0)

    bp = ax3.boxplot(
        box_data, positions=pos, widths=bar_width*0.8,
        patch_artist=True,
        boxprops=dict(facecolor='#F48FB1', color='#C2185B', linewidth=0.8),
        whiskerprops=dict(color='#C2185B', linewidth=0.8),
        capprops=dict(color='#C2185B', linewidth=0.8),
        flierprops=flierprops,
        medianprops=dict(linewidth=1.0, color='black')
    )
    
    
    # 自动判断 Y 轴标签
    if soh_col and ("SoH" in soh_col or "trend" in soh_col):
        ylab = "SoH (归一化)"
    else:
        ylab = "容量 (Ah)"
        
    ax3.set_ylabel(ylab, fontweight='bold')
    ax3.set_title("(c) 各车辆老化状态分布异质性 (Aging Heterogeneity)", fontweight='bold', pad=10, loc='left')
    ax3.grid(axis='x', visible=False)
    
    # X 轴标签
    ax3.set_xlabel("车辆编号 (按样本数量排序)", fontweight='bold')
    # 让y轴更可读：用整体分位数确定显示范围
    all_y = np.concatenate([np.asarray(v) for v in box_data if len(v) > 0])
    all_y = all_y[np.isfinite(all_y)]
    if len(all_y) > 100:
        lo = np.percentile(all_y, 1)
        hi = np.percentile(all_y, 99)
        pad = 0.08 * (hi - lo + 1e-6)
        ax3.set_ylim(lo - pad, hi + pad)

    
    # 如果车辆很多，不显示具体 ID，只显示 1, 10, 20...
    if len(df_agg) > 30:
        ticks = np.arange(0, len(df_agg), 10)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(ticks + 1, rotation=0)
    else:
        ax3.set_xticks(x)
        # 简化 VIN 显示 (后4位)
        vins_short = [v[-4:] for v in df_agg["vin"].astype(str)]
        ax3.set_xticklabels(vins_short, rotation=90)

    plt.tight_layout()
    
    # 保存文件名改为 Fig 3-1
    save_path = 'Fig3-1_Dataset_Heterogeneity.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"成功生成图表: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_data_heterogeneity_corrected()