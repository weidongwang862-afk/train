import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ================= Configuration =================
INPUT_CSV = r"E:\RAW_DATA\outputs\04_features\predictions_test_HGBR_BASELINE_SoH_trend.csv"
rcParams['axes.grid'] = False
BIN_STEP = 0.02
MIN_COUNT = 30          # Minimum samples per bin
USE_MACRO = True        # Weighted by VIN (Required for this plot version)
SHOW_CI95 = False       

# Column Candidates
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
    # --- Check File ---
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File not found at {INPUT_CSV}")
        return 

    df = pd.read_csv(INPUT_CSV)

    true_col = _pick_col(df, TRUE_COL_CAND)
    pred_col = _pick_col(df, PRED_COL_CAND)
    vin_col  = _pick_col(df, VIN_COL_CAND)

    if true_col is None or pred_col is None:
        raise ValueError(f"Missing necessary columns. Available: {df.columns.tolist()}")

    # Definition: Bias = Pred - True
    df = df[[c for c in [vin_col, true_col, pred_col] if c is not None]].copy()
    df["error"] = df[pred_col] - df[true_col]
    df["abs_error"] = df["error"].abs()

    # ===== Binning (Strict Alignment) =====
    min_soh = float(df[true_col].min())
    max_soh = float(df[true_col].max())
    left = _align_floor(min_soh, BIN_STEP)
    right = _align_ceil(max_soh, BIN_STEP)
    bins = np.arange(left, right + BIN_STEP, BIN_STEP)

    df["soh_bin"] = pd.cut(df[true_col], bins=bins, include_lowest=True)

    # ===== Micro Stats (Used for Sample Count) =====
    micro = df.groupby("soh_bin", observed=True).agg(
        count=(true_col, "count"),
    ).reset_index()

    micro = micro[micro["count"] >= MIN_COUNT].reset_index(drop=True)
    if micro.empty:
        print("No bins with enough samples.")
        return

    micro["x_mid"] = micro["soh_bin"].apply(lambda x: float(x.mid))
    micro["x_label"] = micro["x_mid"].map(lambda v: f"{v:.2f}")

    # ===== Macro Stats (Primary Metric) =====
    if USE_MACRO and vin_col is not None:
        # 1. Aggregate per VIN per Bin
        vin_bin = df.groupby([vin_col, "soh_bin"], observed=True).agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            count=(true_col, "count")
        ).reset_index()

        # Filter sparse VIN-bins to ensure robust macro stats
        vin_bin = vin_bin[vin_bin["count"] >= 5].copy()

        # 2. Average across VINs
        macro = vin_bin.groupby("soh_bin", observed=True).agg(
            mae=("mae", "mean"),
            bias=("bias", "mean"),
            n_vins=(vin_col, "nunique"),
        ).reset_index()

        # Merge with x_mid from micro for alignment
        macro = macro.merge(micro[["soh_bin", "x_mid", "x_label"]], on="soh_bin", how="inner")
        
        # Sort ascending: 0.75 -> 0.91 (Low SOH -> High SOH)
        macro = macro.sort_values("x_mid").reset_index(drop=True)
        
        # Filter micro to match macro bins
        micro = micro[micro["soh_bin"].isin(macro["soh_bin"])].sort_values("x_mid").reset_index(drop=True)
    else:
        # Fallback if no VIN col
        macro = micro.copy()

    # ================= Plotting =================
    # Global Font Settings: Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 12
    rcParams['axes.linewidth'] = 1.2
    rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)
    
    x = np.arange(len(macro))

    # ---------- Subplot (a): MAE (Macro Only) ----------
    # Plot Macro MAE as Bars
    ax1.bar(x, macro["mae"].values, width=0.6, alpha=0.85, 
            color='#4c72b0', edgecolor='black', linewidth=0.8,
            label="MAE (Macro, VIN-balanced)")

    # Right Axis: Sample Count ("Light gray thin line")
    ax1r = ax1.twinx()
    ax1r.plot(x, micro["count"].values, color="#cccccc", linewidth=1.0, 
              linestyle='-', marker='.', markersize=4,
              label="Count")
    
    # "Right axis label shorter"
    ax1r.set_ylabel("Count", color="gray", fontsize=11)
    ax1r.tick_params(axis="y", labelcolor="gray", labelsize=10)
    ax1r.grid(False) 

    # Left Axis config
    ax1.set_ylabel("MAE", fontweight="bold", fontsize=12)
    ax1.set_title("(a) Prediction Error vs Aging Stage", fontweight="bold", loc='left', fontsize=13)
    ax1.grid(False)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, edgecolor='gray', fancybox=False)

    # ---------- Subplot (b): Systematic Bias ----------
    # Narrative Focus: Systematic Bias vs SOH
    # Plot Macro Bias (Prominent)
    ax2.plot(x, macro["bias"].values, marker="s", markersize=6, linewidth=2.0, 
             color='#c44e52', label="Bias (Macro)")
    
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Bias (Pred - True)", fontweight="bold", fontsize=12)
    
    # Title with definition
    ax2.set_title("(b) Systematic Bias (Bias = Mean(Pred - True))", fontweight="bold", loc='left', fontsize=13)
    ax2.grid(False)
    ax2.legend(loc="upper right", frameon=True, edgecolor='gray', fancybox=False)

    # X-axis formatting
    ax2.set_xticks(x)
    
    # Sparse labels if too many bins
    labels = macro["x_label"].tolist()
    if len(labels) > 12:
        labels = [lab if (i % 2 == 0) else "" for i, lab in enumerate(labels)]
        
    ax2.set_xticklabels(labels, rotation=0, fontsize=11)
    ax2.set_xlabel("True SOH (Bin Midpoint)", fontweight="bold", fontsize=12)

    plt.tight_layout()
    out_png = "Fig4-5_SOH_Bin_Analysis_Revised.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_png}")
    plt.show()

if __name__ == "__main__":
    plot_soh_bin_analysis()