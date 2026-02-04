import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# =========================
# 0) 配置
# =========================
CSV_PATH = r"E:\RAW_DATA\outputs\04_features\features_FINAL_any.csv"   # 改成你的实际位置
OUT_PNG  = r"E:\RAW_DATA\Fig4-2_perm_importance_topk_any.png"
TOPK = 20

# 字体：中文宋体 + 英文Times New Roman
rcParams["font.family"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = False
rcParams["text.usetex"] = False

# 组名映射（按你 CSV 里的 group）
GROUP_NAME_ZH = {
    "voltage": "电压类",
    "current": "电流类",
    "ic": "IC类",
    "stage": "阶段类",
    "other": "其他/差分/弛豫",
}

def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    need = {"feature", "group", "perm_mean_avg", "perm_std_avg"}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise ValueError(f"CSV缺少必要列: {miss}，请检查文件: {CSV_PATH}")

    # 取TopK并排序（小数值也照样能画）
    df = df.sort_values("perm_mean_avg", ascending=False).head(TOPK).copy()
    df = df.sort_values("perm_mean_avg", ascending=True).reset_index(drop=True)

    # 统一组名显示
    df["group_zh"] = df["group"].map(GROUP_NAME_ZH).fillna(df["group"].astype(str))

    # 颜色：按组分配（使用matplotlib内置tab10，不依赖外部库）
    groups = list(dict.fromkeys(df["group_zh"].tolist()))
    cmap = plt.cm.get_cmap("tab10", max(3, len(groups)))
    color_map = {g: cmap(i) for i, g in enumerate(groups)}
    colors = [color_map[g] for g in df["group_zh"].tolist()]

    # 横轴尺度：重要性很小，统一放大到 1e3 量级更可读（不改变相对关系）
    scale = 1e3
    x = df["perm_mean_avg"].to_numpy() * scale
    xerr = df["perm_std_avg"].to_numpy() * scale
    y = np.arange(len(df))

    # =========================
    # 1) 作图
    # =========================
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=220)

    ax.barh(
        y, x,
        xerr=xerr,
        height=0.72,
        linewidth=0,
        alpha=0.92,
        error_kw=dict(ecolor="0.25", elinewidth=0.9, capsize=2, capthick=0.9),
        color=colors
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].tolist(), fontsize=10)
    ax.set_xlabel(f"置换重要性（MAE 增量）×10^-3", fontsize=12)
    ax.set_title("特征置换重要性 Top-K（按特征组归类）", fontsize=13)

    # 细网格，增强可读性
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # 图例：按组
    handles = [Patch(facecolor=color_map[g], edgecolor="none", label=g) for g in groups]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    leg.get_frame().set_edgecolor("0.85")

    

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PNG)
if __name__ == "__main__":
    main()
