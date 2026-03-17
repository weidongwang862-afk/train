import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib import font_manager as fm

# ================= Config =================
# Use the result file consistent with the best logs to represent the "Final Model"
INPUT_FILE = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_RESCTX_OOF_pred_core_tfmr.csv"
OUTPUT_FIG = "Fig5-7_PerVIN_Distribution_Final_EN.png"
OUTPUT_CSV = "vin_mae_comparison_final.csv"

# ================= Data =================
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit()

print(f"Reading: {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# 1) Per-VIN MAE
vin_stats = df.groupby('vin').agg({
    'base_pred': lambda x: (df.loc[x.index, 'y_true'] - x).abs().mean(),
    'final_pred': lambda x: (df.loc[x.index, 'y_true'] - x).abs().mean()
}).reset_index()

vin_stats.rename(columns={'base_pred': 'MAE_Base', 'final_pred': 'MAE_Final'}, inplace=True)

# 2) Sort by baseline MAE (for long-tail visualization)
vin_stats = vin_stats.sort_values('MAE_Base').reset_index(drop=True)
vin_stats['rank'] = vin_stats.index
vin_stats['Improvement'] = vin_stats['MAE_Base'] - vin_stats['MAE_Final']

# 3) Console stats
print("-" * 30)
print(f"Total Test VINs: {len(vin_stats)}")
print(f"Global MAE Base : {vin_stats['MAE_Base'].mean():.6f}")
print(f"Global MAE Final: {vin_stats['MAE_Final'].mean():.6f}")
print(f"Improved VINs   : {(vin_stats['Improvement'] > 0).sum()}")
print("-" * 30)

vin_stats.to_csv(OUTPUT_CSV, index=False)

# ================= Plot =================
sns.set_context("paper", font_scale=1.6)
sns.set_style("ticks")

# Font setup (keep exactly the same logic; only text becomes English)
available = {f.name for f in fm.fontManager.ttflist}
need = ["Times New Roman", "SimSun"]
missing = [x for x in need if x not in available]
if missing:
    print("Warning: missing fonts:", missing)

rcParams["font.family"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = False
rcParams['font.monospace'] = ['SimHei']

rcParams["mathtext.fontset"] = "stix"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["mathtext.bf"] = "Times New Roman:bold"

rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.sans-serif"] = ["SimSun"]

# Canvas: left large + right small (3:1)
fig = plt.figure(figsize=(14, 6), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.15)

ax_main = fig.add_subplot(gs[0])
ax_stat = fig.add_subplot(gs[1])

# --- Left: Per-VIN MAE comparison ---
x = vin_stats.index
ax_main.bar(
    x, vin_stats['MAE_Base'],
    color='#B0BEC5', label='Baseline (HGBR)',
    width=0.8, alpha=0.8
)
ax_main.plot(
    x, vin_stats['MAE_Final'],
    color='#D32F2F', marker='o',
    linestyle='-', linewidth=1.5, markersize=3,
    label='Final Model'
)

# Annotate worst-case improvement
worst_idx = vin_stats['MAE_Base'].idxmax()
worst_base = vin_stats.loc[worst_idx, 'MAE_Base']
worst_final = vin_stats.loc[worst_idx, 'MAE_Final']
worst_imp_pct = (worst_base - worst_final) / worst_base * 100

ax_main.annotate(
    f"Worst-VIN gain\n-{worst_imp_pct:.1f}%",
    xy=(worst_idx, worst_final),
    xytext=(worst_idx - 15, worst_base * 0.95),
    arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=2),
    color='#D32F2F', fontweight='bold', fontsize=11,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
)

ax_main.set_xlabel("Test vehicles (sorted by baseline MAE)", fontweight='bold', fontsize=12)
ax_main.set_ylabel("Mean Absolute Error (MAE)", fontweight='bold', fontsize=12)
ax_main.set_title("Per-vehicle error comparison", fontweight='bold', fontsize=14, pad=12)
ax_main.legend(loc='upper left', fontsize=11)
ax_main.grid(True, axis='y', linestyle='--', alpha=0.5)

# --- Right: Improvement distribution ---
sns.histplot(
    y=vin_stats['Improvement'],
    ax=ax_stat, bins=20, kde=True,
    color='#43A047', alpha=0.6, edgecolor=None
)
ax_stat.axhline(0, color='k', linestyle='--', lw=1)

# Key stats box (English)
median_imp = vin_stats['Improvement'].median()
p90_imp = np.percentile(vin_stats['Improvement'], 90)
worst_vin_idx = vin_stats['MAE_Base'].idxmax()
worst_case_imp = vin_stats.loc[worst_vin_idx, 'Improvement']
worst_case_imp_pct = worst_case_imp / vin_stats.loc[worst_vin_idx, 'MAE_Base'] * 100


stat_text = (
    f"Median gain: {median_imp:.2e}\n"
    f"Top gain (P90): {p90_imp:.2e}\n"
    f"Worst-VIN gain: {worst_case_imp:.2e} "
)

ax_stat.text(
    0.95, 0.98, stat_text,
    transform=ax_stat.transAxes,
    ha='right', va='top',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#43A047', boxstyle='round,pad=0.4'),
    fontsize=9, fontweight='bold', color='#1B5E20',
    family='monospace'
)

ax_stat.set_title("Gain distribution\n($\\Delta$MAE)", fontweight='bold', fontsize=12, pad=12)
ax_stat.set_xlabel("Frequency (count)", fontsize=10)
ax_stat.set_ylabel("")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches='tight')
print(f"Saved figure: {OUTPUT_FIG}")
plt.show()
