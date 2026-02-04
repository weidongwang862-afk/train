import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置区域 =================
INPUT_CSV = r"E:\RAW_DATA\outputs\04_features\predictions_test_HGBR_BASELINE_SoH_trend.csv"

BIN_STEP = 0.02
MIN_COUNT = 30          # 每个bin最少样本数，避免噪点
USE_MACRO = True        # 额外输出“按VIN均匀加权”的macro曲线（更贴合跨VIN泛化叙事）
SHOW_CI95 = False       # 期刊风格可开：MAE的95%置信区间误差棒

# 可选：如果文件里有vin列，macro会更稳；没有vin列也能跑，只是macro不可用
VIN_COL_CAND = ["vin", "VIN"]

TRUE_COL_CAND = ["y_true", "true", "label", "soh_true", "soh_trend"]
PRED_COL_CAND = ["pred_core", "pred", "y_pred", "soh_pred"]

def _pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _align_floor(x, step):
    return np.floor(x / step) * step

def _align_ceil(x, step):
    return np.ceil(x / step) * step

def plot_soh_bin_analysis():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    true_col = _pick_col(df, TRUE_COL_CAND)
    pred_col = _pick_col(df, PRED_COL_CAND)
    vin_col  = _pick_col(df, VIN_COL_CAND)

    if true_col is None or pred_col is None:
        raise ValueError(f"缺少必要列，现有列: {df.columns.tolist()}")

    # 误差定义：Bias = Pred - True
    df = df[[c for c in [vin_col, true_col, pred_col] if c is not None]].copy()
    df["error"] = df[pred_col] - df[true_col]
    df["abs_error"] = df["error"].abs()

    # ===== 分箱边界严格按 BIN_STEP 对齐 =====
    min_soh = float(df[true_col].min())
    max_soh = float(df[true_col].max())
    left = _align_floor(min_soh, BIN_STEP)
    right = _align_ceil(max_soh, BIN_STEP)
    bins = np.arange(left, right + BIN_STEP, BIN_STEP)

    df["soh_bin"] = pd.cut(df[true_col], bins=bins, include_lowest=True)

    # ===== micro统计：箱内所有样本直接求均值（sample-weighted）=====
    micro = df.groupby("soh_bin", observed=True).agg(
        mae=("abs_error", "mean"),
        mae_std=("abs_error", "std"),
        bias=("error", "mean"),
        bias_std=("error", "std"),
        count=(true_col, "count"),
    ).reset_index()

    micro = micro[micro["count"] >= MIN_COUNT].reset_index(drop=True)
    if micro.empty:
        raise ValueError(f"所有bin样本数都 < MIN_COUNT={MIN_COUNT}，请调小MIN_COUNT或调大BIN_STEP")

    # x轴用bin中点
    micro["x_mid"] = micro["soh_bin"].apply(lambda x: float(x.mid))
    micro["x_label"] = micro["x_mid"].map(lambda v: f"{v:.2f}")

    # ===== macro统计：先按VIN-箱求MAE/Bias，再对VIN平均（vehicle-weighted）=====
    macro = None
    if USE_MACRO and vin_col is not None:
        vin_bin = df.groupby([vin_col, "soh_bin"], observed=True).agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            count=(true_col, "count")
        ).reset_index()

        # VIN-箱内也做一次最小样本过滤，避免单车在某箱里只有几条点造成极端值
        vin_bin = vin_bin[vin_bin["count"] >= 5].copy()

        macro = vin_bin.groupby("soh_bin", observed=True).agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            bias=("bias", "mean"),
            bias_std=("bias", "std"),
            n_vins=(vin_col, "nunique"),
        ).reset_index()

        macro = macro.merge(micro[["soh_bin", "x_mid", "x_label"]], on="soh_bin", how="inner")
        macro = macro.sort_values("x_mid").reset_index(drop=True)

    # ===== 画图：中文宋体 + 英文Times =====
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, dpi=300)

    x = np.arange(len(micro))

    # ---------- (a) MAE vs SoH ----------
    ax1.bar(x, micro["mae"].values, width=0.62, alpha=0.85,
            label="MAE（micro, 样本加权）")

    if SHOW_CI95:
        # 95%CI 近似：1.96 * std / sqrt(n)
        ci95 = 1.96 * (micro["mae_std"].fillna(0).values / np.sqrt(micro["count"].values))
        ax1.errorbar(x, micro["mae"].values, yerr=ci95, fmt="none", capsize=3,
                     ecolor="0.35", linewidth=1.0, alpha=0.9)

    if macro is not None:
        ax1.plot(x, macro["mae"].values, marker="o", linewidth=2.0,
                 label="MAE（macro, VIN均匀加权）")

    # 右轴样本量：浅灰折线，不遮挡主体
    ax1r = ax1.twinx()
    ax1r.plot(x, micro["count"].values, color="0.70", linewidth=1.6, marker=".",
              label="样本数量（Count）")
    ax1r.set_ylabel("样本数量（Count）", color="0.45")
    ax1r.tick_params(axis="y", labelcolor="0.45")
    ax1r.grid(False)

    ax1.set_ylabel("MAE", fontweight="bold")
    ax1.set_title("(a) 不同老化阶段的预测误差（MAE）", fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.35)

    # 标注最大MAE（micro）
    imax = int(np.argmax(micro["mae"].values))
    ax1.annotate(f"Max: {micro['mae'].iloc[imax]:.4f}",
                 xy=(imax, micro["mae"].iloc[imax]),
                 xytext=(imax, micro["mae"].iloc[imax] * 1.12),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 合并图例：左轴 + 右轴
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fancybox=False)

    # ---------- (b) Bias ----------
    ax2.plot(x, micro["bias"].values, marker="o", linewidth=2.2,
             label="Bias（micro, 样本加权）")
    if macro is not None:
        ax2.plot(x, macro["bias"].values, marker="s", linewidth=2.0,
                 label="Bias（macro, VIN均匀加权）")

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax2.set_ylabel("Bias（Pred−True）", fontweight="bold")
    ax2.set_title("(b) 不同老化阶段的系统偏差（Bias）", fontweight="bold", pad=8)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, fancybox=False)

    # x轴刻度：太密时隔一个显示
    ax2.set_xticks(x)
    labels = micro["x_label"].tolist()
    if len(labels) > 12:
        labels = [lab if (i % 2 == 0) else "" for i, lab in enumerate(labels)]
    ax2.set_xticklabels(labels, rotation=0)
    ax2.set_xlabel("真实 SoH 分箱中点（True SoH Bin Midpoint）", fontweight="bold")

    plt.tight_layout()
    out_png = "Fig4-5_SOH_Bin_Analysis.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_png)

if __name__ == "__main__":
    plot_soh_bin_analysis()
