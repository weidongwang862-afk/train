import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib import font_manager as fm



def plot_gamma_comparison():
    ## ================= 1. 数据准备 (Updated from core_result.txt) =================
    # X轴: Gamma 值从 0.00 到 1.00
    gammas = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 
                       0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    
    # 基线 HGBR (Gamma=0)
    baseline_mae = 0.021760

    # 模型数据数组 (直接提取自您的日志)
    # MLP (Model=mlp)
    mae_mlp = np.array([0.021760, 0.021290, 0.020836, 0.020402, 0.019989, 0.019597, 
                        0.019228, 0.018886, 0.018571, 0.018286, 0.018032, 0.017812, 
                        0.017629, 0.017480, 0.017370, 0.017300, 0.017268, 0.017273, 
                        0.017313, 0.017393, 0.017520])

    # GRU (Model=gru)
    mae_gru = np.array([0.021760, 0.021216, 0.020694, 0.020196, 0.019723, 0.019275, 
                        0.018857, 0.018467, 0.018107, 0.017778, 0.017481, 0.017221, 
                        0.017005, 0.016834, 0.016709, 0.016630, 0.016600, 0.016623, 
                        0.016700, 0.016832, 0.017020])

    # TCN (Model=tcn)
    mae_tcn = np.array([0.021760, 0.021605, 0.021460, 0.021325, 0.021199, 0.021085, 
                        0.020984, 0.020895, 0.020815, 0.020744, 0.020683, 0.020633, 
                        0.020593, 0.020564, 0.020547, 0.020541, 0.020547, 0.020565, 
                        0.020595, 0.020637, 0.020693])

    # Transformer (Model=tfmr) - The Winner
    mae_tfmr = np.array([0.021760, 0.021248, 0.020756, 0.020285, 0.019837, 0.019409, 
                         0.019002, 0.018617, 0.018257, 0.017923, 0.017620, 0.017350, 
                         0.017115, 0.016913, 0.016746, 0.016613, 0.016513, 0.016449, 
                         0.016427, 0.016455, 0.016541])

    # SOTA 标注信息
    best_gamma = 0.90
    min_mae = 0.016427
    best_model_name = "Transformer"

     # ================= 绘图风格设置 =================
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

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # ================= 3. 绘制四条曲线 =================
    # 颜色策略：主角(TFMR)用显眼的红色，其他用冷色调
    
    # (1) Transformer (红线，最粗，带实心点)
    ax.plot(gammas, mae_tfmr, color='#D32F2F', linewidth=2.5, marker='o', markersize=7, 
            label='Transformer', zorder=10)
    
    # (2) GRU (蓝线，虚线)
    ax.plot(gammas, mae_gru, color='#1976D2', linewidth=2.0, marker='s', markersize=6, linestyle='--',
            label='GRU', zorder=5)

    # (3) MLP (绿线，点划线)
    ax.plot(gammas, mae_mlp, color='#388E3C', linewidth=2.0, marker='^', markersize=6, linestyle='-.',
            label='MLP', zorder=4)

    # (4) TCN (橙线，点线)
    ax.plot(gammas, mae_tcn, color='#FF9800', linewidth=2.0, marker='d', markersize=6, linestyle=':',
            label='TCN', zorder=3)
    
    # (5) Baseline (灰线)
    ax.axhline(y=baseline_mae, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
               label=f'基线模型(HGBR): {baseline_mae:.6f}')
    
    # ================= 4. 关键点标注 =================
    # 标记 SOTA 最优点
    ax.scatter([best_gamma], [min_mae], color='#D32F2F', s=150, zorder=15, edgecolors='white', linewidth=2)
    
    # 箭头标注
    ax.annotate(f'最佳模型 ({best_model_name})\nMAE: {min_mae:.6f}', 
                xy=(best_gamma, min_mae), 
                xytext=(best_gamma - 0.02, min_mae - 0.001),
                arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5),
                fontsize=12, fontweight='bold', color='#D32F2F',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", alpha=0.9)) # 加个框更清晰

    # ================= 5. 标签与图例 =================
    ax.set_xlabel(r"残差收缩系数 (γ)", fontweight='bold', fontsize=14)
    ax.set_ylabel("测试集 MAE ", fontweight='bold', fontsize=14)
    ax.set_title("不同残差修正模型的 γ 敏感性对比", fontweight='bold', fontsize=16, pad=15)
    
    # 优化网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 图例 (放在合适的位置，两列显示)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower left', ncol=2, fontsize=11)
    
    plt.tight_layout()
    save_path = 'Fig5-9_Model_Comparison_CN.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_gamma_comparison()