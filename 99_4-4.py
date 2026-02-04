import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
# 请指向你的 HGBR 测试集预测文件
# 通常在 outputs/04_features/ 下，名字类似 predictions_test_HGBR_...
INPUT_CSV = r"E:\RAW_DATA\outputs\04_features\predictions_test_HGBR_BASELINE_SoH_trend.csv"

# 如果你找不到 HGBR 的，也可以先用 Transformer 的 (predictions_test_RESCTX_...csv) 来看效果
# INPUT_CSV = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_RESCTX_OOF_pred_gated_any_tfmr.csv"

def plot_vin_error_distribution():
    print(f"正在读取预测文件: {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        print(f"错误：找不到文件 {INPUT_CSV}")
        print("请修改脚本中的 INPUT_CSV 路径，指向你的 predictions_test_*.csv 文件")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 1. 智能识别列名
    # 找真实值列
    col_true = next((c for c in ["y_true", "SoH_trend", "target"] if c in df.columns), None)
    # 找预测值列 (优先找 final_pred, 然后 pred_core, 然后 pred)
    col_pred = next((c for c in ["final_pred", "pred_core", "pred", "y_pred"] if c in df.columns), None)
    
    if not col_true or not col_pred:
        print(f"错误：无法识别真实值或预测值列。现有列名: {df.columns.tolist()}")
        return

    print(f"使用列: 真实值=[{col_true}], 预测值=[{col_pred}]")

    # 2. 计算每辆车的 MAE
    df["abs_error"] = (df[col_true] - df[col_pred]).abs()
    
    vin_stats = df.groupby("vin")["abs_error"].agg(["mean", "std", "count"]).reset_index()
    vin_stats = vin_stats.rename(columns={"mean": "mae", "std": "std_ae"})
    
    # 按 MAE 从低到高排序 (表现好的在左边，差的在右边)
    vin_stats = vin_stats.sort_values("mae")
    worst_5 = vin_stats.tail(5).iloc[::-1] # 取最后5个并反转
    print("\n【重点关注】误差最大的 5 辆车 (论文 Case Study 候选):")
    print(worst_5.to_string(index=False))
    
    # 3. 准备绘图
    # 如果车辆太多 (>50)，X轴就不显示具体 VIN，只显示排名索引
    n_vins = len(vin_stats)
    x = range(n_vins)
    
    # ================= 绘图风格 =================
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False 

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色映射：MAE 越低越绿，越高越红
    # 使用 matplotlib 的 colormap
    norm = plt.Normalize(vin_stats["mae"].min(), vin_stats["mae"].max())
    colors = plt.cm.RdYlGn_r(norm(vin_stats["mae"].values)) # _r 表示反转，低(绿)-高(红)

    # 4. 绘制条形图
    bars = ax.bar(x, vin_stats["mae"], color=colors, alpha=0.9, width=0.8, label='Vehicle MAE')
    
    # 可选：添加一条全集平均线
    global_mae_micro = df["abs_error"].mean()          # 按样本加权
    global_mae_macro = vin_stats["mae"].mean()         # 按车辆平均（与柱子一致）

    ax.axhline(y=global_mae_macro, color='#1A237E', linestyle='--', linewidth=2,
            label=f'Global MAE (macro): {global_mae_macro:.4f}')


    # 5. 装饰
    ax.set_ylabel("平均绝对误差 (MAE)", fontweight='bold')
    ax.set_xlabel(f"测试集车辆 (按误差排序, Total={n_vins})", fontweight='bold')
    ax.set_title("分车辆误差分布 (Per-Vehicle Error Distribution)", fontweight='bold', pad=15)
    
    # X 轴标签处理
    if n_vins <= 30:
        ax.set_xticks(x)
        # 简化 VIN 显示
        short_vins = [v[-4:] for v in vin_stats["vin"].astype(str)]
        ax.set_xticklabels(short_vins, rotation=45, ha='right', fontsize=10)
    else:
        # 车辆太多时，只标记关键位置 (Min, Median, Max)
        # 或者每隔 10 个标记一下
        ticks = np.arange(0, n_vins, max(1, n_vins//15))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"#{i+1}" for i in ticks], rotation=0)

    # 添加文本标注：最好和最差
    best_vin = vin_stats.iloc[0]
    worst_vin = vin_stats.iloc[-1]
    
    # 标注 Best
    ax.annotate(f"Best: {best_vin['mae']:.4f}\n(VIN{str(best_vin['vin'])[-4:]})", 
                xy=(0, best_vin['mae']), 
                xytext=(n_vins*0.1, best_vin['mae'] + global_mae_macro*0.3),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold')
    
    # 标注 Worst
    ax.annotate(f"Worst: {worst_vin['mae']:.4f}\n(VINi{str(worst_vin['vin'])[-4:]})", 
                xy=(n_vins-1, worst_vin['mae']), 
                xytext=(n_vins*0.75, worst_vin['mae'] + global_mae_macro*0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')
    # ===== inset 放大：显示主体区间（去掉最差长尾压扁）=====
    p95 = float(np.percentile(vin_stats["mae"].values, 95))
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width="38%", height="38%", loc="upper center", borderpad=1.2)
    axins.bar(x, vin_stats["mae"], color=colors, alpha=0.9, width=0.8)
    axins.axhline(y=global_mae_macro, color='#1A237E', linestyle='--', linewidth=1.4)
    axins.set_ylim(0, p95)
    axins.set_xticks([])
    axins.set_yticks([0, round(p95, 3)])
    axins.set_title("Zoom: ≤P95", fontsize=11)
    axins.grid(True, alpha=0.25)

    ax.legend(frameon=True, fancybox=False, loc='upper left')
    
    plt.tight_layout()
    save_path = 'Fig4-4_VIN_MAE_Distribution.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_vin_error_distribution()