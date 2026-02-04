import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
# 指向你刚才跑出来的结果文件
INPUT_CSV = r"E:\RAW_DATA\outputs\04_features\hyperparam_search_results.csv"

def plot_fig4_3_final():
    print(f"正在读取: {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        print(f"错误：找不到文件 {INPUT_CSV}")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
        print("成功读取数据，列名:", df.columns.tolist())
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 智能列名映射：不管你的 CSV 里叫 mae 还是 mae_mean，都能识别
    mae_col = "mae"
    if "mae_mean" in df.columns:
        mae_col = "mae_mean"
    elif "mae" in df.columns:
        mae_col = "mae"
    else:
        print("错误：CSV 中找不到 'mae' 或 'mae_mean' 列")
        return

    # ================= 绘图风格设置 =================
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    # 字体设置 (中西文混排)
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 子图 1: Learning Rate ---
    d1 = df[df["param_type"] == "learning_rate"]
    if not d1.empty:
        # 绘制折线
        ax1.plot(d1["value"], d1[mae_col], marker='o', color='#1565C0', lw=2.5, label='验证集 MAE')
        
        # 自动寻找并标记最低点
        best_idx = d1[mae_col].idxmin()
        best_val = d1.loc[best_idx, "value"]
        best_mae = d1.loc[best_idx, mae_col]
        
        ax1.scatter([best_val], [best_mae], s=150, color='#D32F2F', zorder=5, 
                    edgecolors='white', linewidth=1.5,
                    label=f"最优点 ({best_val}, {best_mae:.4f})")
        
        ax1.set_xlabel("学习率 (Learning Rate)", fontweight='bold')
        ax1.set_ylabel("验证集 MAE ", fontweight='bold')
        ax1.set_title("(a) 学习率敏感性分析", fontweight='bold', pad=12)
        ax1.legend(frameon=True, fancybox=False)
        ax1.grid(True, linestyle='--', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "无 Learning Rate 数据", ha='center', va='center')

    # --- 子图 2: Max Depth ---
    d2 = df[df["param_type"] == "max_depth"]
    if not d2.empty:
        # 绘制折线
        ax2.plot(d2["value"], d2[mae_col], marker='s', color='#2E7D32', lw=2.5, label='验证集 MAE')
        
        # 自动寻找并标记最低点
        best_idx = d2[mae_col].idxmin()
        best_val = int(d2.loc[best_idx, "value"])
        best_mae = d2.loc[best_idx, mae_col]
        
        ax2.scatter([best_val], [best_mae], s=150, color='#D32F2F', zorder=5, 
                    edgecolors='white', linewidth=1.5,
                    label=f"最优点 ({best_val}, {best_mae:.4f})")
        
        ax2.set_xlabel("最大树深 (Max Depth)", fontweight='bold')
        # ax2.set_ylabel("验证集 MAE", fontweight='bold') # 共用Y轴标签可省略
        ax2.set_title("(b) 树深敏感性分析", fontweight='bold', pad=12)
        ax2.legend(frameon=True, fancybox=False)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "无 Max Depth 数据", ha='center', va='center')

    plt.tight_layout()
    save_path = "Fig4-3_Hyperparam_Search_Final.png"
    plt.savefig(save_path, dpi=300)
    print(f"高清图表已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_fig4_3_final()