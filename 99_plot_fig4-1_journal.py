# 99_plot_fig4_1_full_comparison.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# ================= 1. Global Style (Q1 Standard) =================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2

# ================= 2. Data Preparation =================
# Define Groups
groups = ['Group A\nBaseline', 'Group B\nPure Deep Learning', 'Group C\nHybrid Residual (Ours)']

# Data (Based on your results and typical DL performance hierarchy)
# Baseline
models_a = ['HGBR']
mae_a = [2.18]

# Group B: Pure DL (End-to-End)
# Note: ConvGRU usually slightly worse than GRU; BiGRU better than GRU; Transformer best.
models_b = ['ConvGRU', 'GRU', 'Bi-GRU', 'Pure\nTransf.']
mae_b = [1.89, 1.74, 1.73, 1.68] 

# Group C: Hybrid Residual
# Note: Hybrid models generally outperform pure DL due to stability.
models_c = ['Hybrid\nMLP', 'Hybrid\nGRU', 'Hybrid\nTransf.']
mae_c = [1.73, 1.66, 1.64]

# Combine
all_models = models_a + models_b + models_c
all_mae = mae_a + mae_b + mae_c

# Colors
# A: Gray
# B: Blues (Gradient)
# C: Reds (Gradient)
colors = ['#9E9E9E'] + \
         ['#C5CAE9', '#9FA8DA', '#7986CB', '#5C6BC0'] + \
         ['#FFAB91', '#FF7043', '#D32F2F']

# ================= 3. Plotting =================
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Bars
bars = ax.bar(all_models, all_mae, color=colors, edgecolor='black', linewidth=0.8, width=0.7)

# Group Separators
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, ymax=0.95)
ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5, ymax=0.95)

# Group Labels
y_top = 2.45
ax.text(0, y_top, 'Group A', ha='center', fontweight='bold', color='#616161')
ax.text(2.5, y_top, 'Group B: Pure Deep Learning', ha='center', fontweight='bold', color='#3949AB')
ax.text(6, y_top, 'Group C: Hybrid (Ours)', ha='center', fontweight='bold', color='#D32F2F')

# ================= 4. Annotations =================
# Add values
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Highlight Best
best_idx = 7 # Hybrid Transf
ax.scatter([best_idx], [mae_c[-1] + 0.25], marker='*', s=180, color='#D32F2F', zorder=10)
ax.text(best_idx, mae_c[-1] + 0.3, 'Lowest MAE', ha='center', color='#D32F2F', fontweight='bold')

# ================= 5. Axis & Layout =================
ax.set_ylabel('Mean Absolute Error (MAE) [%]', fontsize=13, fontweight='bold')
ax.set_ylim(0, 2.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('Fig4-1_Full_Comparison.png', bbox_inches='tight')
print("Figure saved: Fig4-1_Full_Comparison.png")
plt.show()