import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# 0) 路径与参数（按你真实情况改）
# =========================
DATASET_PATH  = r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"
IMP_CSV_PATH  = r"E:\RAW_DATA\outputs\04_features\features_FINAL_any.csv"   # 置换重要性文件，用于选Top-K特征
OUT_DIR       = r"E:\RAW_DATA"

FIG_PNG = os.path.join(OUT_DIR, "Fig4-2b_corr_heatmap_spearman_any.png")

TOPK = 18                 # Top-K特征数量（再加目标变量一行一列，矩阵大小=TOPK+1）
SAMPLE_N = 60000          # 抽样行数（数据很大时建议抽样，保证速度与稳健性）
RANDOM_SEED = 42
TARGET_COL = "SoH_trend"  # 也可以改成 "C_trend"（容量趋势标签）

# =========================
# 1) 字体：中文宋体 + 英文Times New Roman
# =========================
rcParams["font.family"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = False
rcParams["text.usetex"] = False

def read_parquet_safe(path):
    # 本地Windows环境通常直接 pd.read_parquet 即可；这里保留最常规写法
    return pd.read_parquet(path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读取置换重要性，选Top-K特征名
    imp = pd.read_csv(IMP_CSV_PATH)
    need = {"feature", "perm_mean_avg"}
    miss = sorted(list(need - set(imp.columns)))
    if miss:
        raise ValueError(f"重要性CSV缺少列: {miss}")

    top_feats = (
        imp.sort_values("perm_mean_avg", ascending=False)["feature"]
        .astype(str).tolist()
    )
    top_feats = top_feats[:TOPK]

    # 2) 读取数据集
    df = read_parquet_safe(DATASET_PATH)

    # 3) 列检查：目标列 + 特征列必须存在
    cols = [TARGET_COL] + top_feats
    miss2 = [c for c in cols if c not in df.columns]
    if miss2:
        raise ValueError(
            "数据集中缺少以下列：\n"
            + "\n".join(miss2)
            + "\n请检查 TARGET_COL、features_FINAL_core.csv 与数据集列名是否一致。"
        )

    sub = df[cols].copy()

    # 4) 抽样（避免28万行以上时计算慢）
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(sub) > SAMPLE_N:
        sub = sub.sample(n=SAMPLE_N, random_state=RANDOM_SEED)

    # 5) Spearman相关系数
    corr = sub.corr(method="spearman")

    # =========================
    # 2) 绘图：相关性热力图 + 数值标注
    # =========================
    labels = corr.columns.tolist()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(11.0, 8.5), dpi=220)

    # 颜色：正相关偏绿，负相关偏红
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", interpolation="nearest")

    # 刻度
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # 网格线（细分隔，打印更清晰）
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 数值标注：矩阵不大时可读
    # 字体颜色按背景亮度切换，保证可读性
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            txt_color = "black" if abs(v) < 0.55 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)

    # 标题与口径说明
    ax.set_title("候选特征相关性矩阵（Spearman）", fontsize=13)
    ax.text(
        0.99, -0.10,
        f"口径：{TARGET_COL} + Top-{TOPK} 特征（来自置换重要性排序）；抽样 n={len(sub)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.35"
    )

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", rotation=90)

    plt.tight_layout()
    fig.savefig(FIG_PNG, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", FIG_PNG)
if __name__ == "__main__":
    main()
