import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# =========================
# 1) 路径：按你的真实目录改
# =========================
BASE_DIR = r"E:\RAW_DATA\outputs"
OUT_DIR  = r"E:\RAW_DATA"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 2) 模型文件模式（三联：GRU(opt) / ATTENTION / FUSION）
# =========================
MODEL_A = ("HGBR (Baseline)", "predictions_test_FULL_pred_core_HGBR_OOF.csv")
MODEL_B_LIST = [
    ("GRU", "predictions_test_SEQ_GRU.csv"),
    ("Transformer", "predictions_test_SEQ_TFMR.csv"),
    ("Fusion", "predictions_test_SEQ_FUSION.csv"),
]

# 近似持平阈值：SoH量纲下建议 0.001；你也可改成 0.0005/0.002
EPS = 0.001

# 统一坐标：用P95视窗避免极端VIN把主群压扁
USE_P95_VIEW = True
P_VIEW = 0.95

# =========================
# 3) 列名适配（兼容你多种预测csv）
# =========================
VIN_CANDS  = ["vin", "VIN"]
TRUE_CANDS = ["y_true", "true", "soh_true", "label", "soh_trend"]
PRED_CANDS = ["y_pred", "pred", "pred_core", "final_pred", "yhat"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def find_file(pattern):
    hits = glob.glob(os.path.join(BASE_DIR, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"找不到文件: {pattern}（搜索目录：{BASE_DIR}）")
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def calc_vin_mae(model_name, pattern):
    path = find_file(pattern)
    df = pd.read_csv(path)

    vin_col  = pick_col(df, VIN_CANDS)
    true_col = pick_col(df, TRUE_CANDS)
    pred_col = pick_col(df, PRED_CANDS)

    if vin_col is None:
        raise ValueError(f"[{model_name}] 缺少 vin 列，无法计算 per-VIN MAE。")
    if true_col is None or pred_col is None:
        raise ValueError(f"[{model_name}] 缺少 y_true/pred 列。现有列：{df.columns.tolist()}")

    df = df[[vin_col, true_col, pred_col]].copy()
    df[vin_col] = df[vin_col].astype(str)
    df["abs_err"] = (df[true_col] - df[pred_col]).abs()

    vin_mae = df.groupby(vin_col)["abs_err"].mean().reset_index()
    vin_mae.columns = ["vin", f"mae_{model_name}"]
    return vin_mae, os.path.basename(path)

def vin_tag(v):  # 显示末4位
    v = str(v)
    return v[-4:] if len(v) >= 4 else v

def main():
    # 字体：中文宋体 + 英文Times（用字体文件锁定，避免方框）
    fp_cn = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
    fp_en = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
    plt.rcParams["axes.unicode_minus"] = False

    # 读基线
    df_a, file_a = calc_vin_mae(MODEL_A[0], MODEL_A[1])
    col_a = f"mae_{MODEL_A[0]}"

    # 预读所有B，统一坐标lim
    merged_list = []
    files_b = []
    all_vals = []

    for name_b, pat_b in MODEL_B_LIST:
        df_b, file_b = calc_vin_mae(name_b, pat_b)
        files_b.append((name_b, file_b))
        col_b = f"mae_{name_b}"
        merged = pd.merge(df_a, df_b, on="vin", how="inner")
        merged["diff"] = merged[col_b] - merged[col_a]
        merged_list.append((name_b, merged, col_b))
        all_vals.append(merged[col_a].values)
        all_vals.append(merged[col_b].values)

    all_vals = np.concatenate(all_vals)
    if USE_P95_VIEW:
        lim = float(np.quantile(all_vals, P_VIEW) * 1.10)
    else:
        lim = float(all_vals.max() * 1.05)
    lim = max(lim, 1e-6)

    # 画布：1x3
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), dpi=300, sharex=True, sharey=True)
    panel_tags = ["(a)", "(b)", "(c)"]

    # 统一细节风格（减少“廉价感”）
    for ax in axes:
        ax.spines["top"].set_alpha(0.6)
        ax.spines["right"].set_alpha(0.6)
        ax.spines["left"].set_alpha(0.8)
        ax.spines["bottom"].set_alpha(0.8)


    # 统一背景/对角线/epsilon带
    xs = np.linspace(0, lim, 200)

    for ax, (name_b, merged, col_b) in zip(axes, merged_list):
        better  = merged[merged["diff"] < -EPS]
        worse   = merged[merged["diff"] >  EPS]
        neutral = merged[(merged["diff"] >= -EPS) & (merged["diff"] <= EPS)]
        n = len(merged)

        # 两类区域底色（更直观：谁在 y<x / y>x）
        ax.fill_between(xs, 0, xs, color="#FFF3E0", alpha=0.22, zorder=0)      # Deep better (y<x)
        ax.fill_between(xs, xs, lim, color="#E3F2FD", alpha=0.18, zorder=0)    # Baseline better (y>x)

        # 持平带：|y-x|<=EPS（核心信息）
        y1 = np.clip(xs - EPS, 0, lim)
        y2 = np.clip(xs + EPS, 0, lim)
        ax.fill_between(xs, y1, y2, color="#ECEFF1", alpha=0.55, zorder=1)

        # 基准线：y=x 与持平带边界
        ax.plot(xs, xs, ls="--", c="0.45", lw=1.6, zorder=2)
        ax.plot(xs, y2, ls=":", c="0.55", lw=1.1, zorder=2)
        ax.plot(xs, y1, ls=":", c="0.55", lw=1.1, zorder=2)

        # 散点：持平灰 / B更好红 / A更好蓝
        ax.scatter(neutral[col_a], neutral[col_b], s=44, c="#B0BEC5", alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(better[col_a], better[col_b], s=52, c="#E64A19", alpha=0.80,
                   edgecolors="white", linewidths=0.6, zorder=4)
        ax.scatter(worse[col_a], worse[col_b], s=52, c="#1565C0", alpha=0.80,
                   edgecolors="white", linewidths=0.6, zorder=4)

        # 极端标注：HGBR最差1个；diff最差1个；diff最好1个
        worst_base = merged.sort_values(col_a, ascending=False).head(1)
        for _, r in worst_base.iterrows():
            ax.annotate(vin_tag(r["vin"]),
                        (min(float(r[col_a]), lim*0.995), min(float(r[col_b]), lim*0.995)),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, fontproperties=fp_en, color="black", fontweight="bold",clip_on=False)

        worst_det = merged.sort_values("diff", ascending=False).head(1)
        for _, r in worst_det.iterrows():
            if float(r["diff"]) > EPS:
                ax.annotate(f"Worst-gap:{vin_tag(r['vin'])}",
                            (min(float(r[col_a]), lim*0.995), min(float(r[col_b]), lim*0.995)),
                            textcoords="offset points", xytext=(-4, 10),
                            ha="right", fontsize=9, fontproperties=fp_en, color="#1565C0", fontweight="bold",clip_on=False)

        best_imp = merged.sort_values("diff", ascending=True).head(1)
        for _, r in best_imp.iterrows():
            if float(r["diff"]) < -EPS:
                ax.annotate(f"Best-gain:{vin_tag(r['vin'])}",
                            (min(float(r[col_a]), lim*0.995), min(float(r[col_b]), lim*0.995)),
                            textcoords="offset points", xytext=(6, -12),
                            ha="left", fontsize=9, fontproperties=fp_en, color="#E64A19", fontweight="bold",clip_on=False)

        # 子图标题 + 统计
        nb, nw, nn = len(better), len(worse), len(neutral)
        idx = list(axes).index(ax)
        ax.set_title(f"{panel_tags[idx]}  {name_b}", fontproperties=fp_en, fontsize=13, fontweight="bold", pad=8)
        ax.text(0.02, 0.02,
                f"Better {nb}/{n} ({nb/n:.0%}) | Worse {nw}/{n} ({nw/n:.0%}) | Equal {nn}/{n} ({nn/n:.0%})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontproperties=fp_en, fontsize=9.5, color="0.35")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(True, alpha=0.28)

    # 统一轴标签
    axes[0].set_ylabel("Model per-VIN MAE", fontproperties=fp_en, fontsize=13, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("HGBR (Baseline) per-VIN MAE", fontproperties=fp_en, fontsize=13, fontweight="bold")

    # 总标题（中文宋体 + 英文行）
    fig.suptitle("分VIN误差对照诊断：强基线与端到端模型的长尾差异",
                 fontproperties=fp_cn, fontsize=16, fontweight="bold", y=1.03)
    fig.text(0.5, 0.95, "HGBR (Baseline) vs {GRU / Attention / Fusion}",
             ha="center", va="top", fontproperties=fp_en, fontsize=12, fontweight="bold")

    # 图例（放在最右侧子图上，避免重复）
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color="0.45", lw=1.6, ls="--", label="y = x"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#E64A19", markeredgecolor="white",
               markersize=8, label="Deep Better (y<x)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0", markeredgecolor="white",
               markersize=8, label="Baseline Better (y>x)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#B0BEC5", markeredgecolor="white",
               markersize=8, label=f"~Equal (|Δ|≤{EPS:g})"),
    ]
    axes[-1].legend(handles=legend_elems, loc="upper left", frameon=True,
                    fancybox=False, prop=fp_en, fontsize=10)

    # 文件信息


    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "Fig5-2_Triplet_HGBR_vs_Deep.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()

    print("Saved:", out_png)

if __name__ == "__main__":
    main()

