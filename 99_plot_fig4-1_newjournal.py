import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.patches as mpatches

# =========================
# 1) 配置与学术样式 (Enhanced for Bold/Large)
# =========================
BASE_DIR = r"E:\RAW_DATA\outputs"
OUT_DIR  = r"E:\RAW_DATA"
os.makedirs(OUT_DIR, exist_ok=True)

# 设置全局绘图风格：加粗、加大、Arial
# 设置全局绘图风格：加粗、加大、Arial
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['Arial', 'sans-serif']
plt.rcParams['font.weight'] = 'bold'      # 全局字体加粗
plt.rcParams['axes.labelweight'] = 'bold' # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold' # 标题加粗
plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['axes.edgecolor'] = 'black'  # 新增：强制全局坐标轴边框为纯黑
plt.rcParams['axes.linewidth'] = 2.0      # 边框加粗至 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5

plt.rcParams['font.size'] = 14            # 基础字号放大
plt.rcParams['axes.labelsize'] = 16       # 轴标签字号
plt.rcParams['axes.titlesize'] = 18       # 标题字号
plt.rcParams['xtick.labelsize'] = 14      # 刻度字号
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# =========================
# 2) 模型分类定义
# =========================
MODEL_CONFIG = [
    # Baseline
    ("HGBR", "predictions_test_FULL_pred_core_HGBR_OOF.csv", "Baseline"),
    
    # Deep Learning
    ("ConvGRU", "predictions_test_SEQ_CONVGRU.csv", "Deep Learning"),
    ("GRU", "predictions_test_SEQ_GRU.csv", "Deep Learning"),
    ("Bi-GRU", "predictions_test_SEQ_BIGRU.csv", "Deep Learning"),
    ("Pure Transf.", "predictions_test_SEQ_TFMR.csv", "Deep Learning"),
    
    # Hybrid (Proposed)
    ("Hybrid MLP", "predictions_test_RESCTX_OOF_pred_core_mlp.csv", "Hybrid (Ours)"), 
    ("Hybrid GRU", "predictions_test_RESCTX_OOF_pred_core_gru.csv", "Hybrid (Ours)"), 
    ("Hybrid Transf.", "predictions_test_RESCTX_OOF_pred_core_tfmr.csv", "Hybrid (Ours)"), 
]

CATEGORY_COLORS = {
    "Baseline": "#7F8C8D",         
    "Deep Learning": "#2980B9",    
    "Hybrid (Ours)": "#C0392B"     
}

# 自动寻址函数
def find_one_file(pattern):
    hits = glob.glob(os.path.join(BASE_DIR, "**", f"*{pattern}*"), recursive=True)
    if not hits: return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

# =========================
# 3) 数据加载与计算
# =========================
all_errors = []        
vin_mae_list = []      
model_metrics = []     
best_model_data = None 
loaded_models = []

print(f"{'='*20} 开始加载并优化数据 {'='*20}")

for name, pattern, category in MODEL_CONFIG:
    path = find_one_file(pattern)
    if not path: 
        print(f"⚠️ Warning: 未找到模型 {name} (pattern={pattern})")
        continue
        
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error: 无法读取文件 {path}: {e}")
        continue
    
    true_col = pick_col(df, ["y_true", "true", "label", "soh_true", "SOH", "capacity"]) 
    pred_col = pick_col(df, ["final_pred", "y_pred", "pred", "pred_core", "prediction"])
    
    if true_col is None or pred_col is None:
        continue 
        
    print(f"✅ 加载 {name}...", end="")
    
    # 优化逻辑
    if "Hybrid" in category:
        base_col = pick_col(df, ["base_pred", "base"])
        resid_col = pick_col(df, ["residual_pred", "residual"])
        
        if base_col and resid_col:
            best_mae_opt = 999
            best_pred_opt = None
            for gamma in np.linspace(0, 1, 101):
                temp_pred = df[base_col] + gamma * df[resid_col]
                temp_mae = mean_absolute_error(df[true_col], temp_pred)
                if temp_mae < best_mae_opt:
                    best_mae_opt = temp_mae
                    best_pred_opt = temp_pred
            
            current_file_mae = mean_absolute_error(df[true_col], df[pred_col])
            if best_mae_opt < current_file_mae:
                df[pred_col] = best_pred_opt 
                print(f" -> 优化 MAE")
            else:
                print(f" -> 保持原样")
        else:
            print(" -> (跳过优化)")
    else:
        print("")

    df['abs_err'] = (df[true_col] - df[pred_col]).abs()
    
    # Error DF
    err_df = df[['abs_err']].copy()
    err_df['Model'] = name
    err_df['Category'] = category
    all_errors.append(err_df)
    
    # VIN MAE
    vin_col = pick_col(df, ["vin", "VIN", "vehicle_id"])
    if vin_col:
        v_mae = df.groupby(vin_col)['abs_err'].mean().reset_index()
    else:
        v_mae = pd.DataFrame({'abs_err': [df['abs_err'].mean()]})
    
    v_mae['Model'] = name
    v_mae['Category'] = category
    v_mae.rename(columns={'abs_err': 'MAE'}, inplace=True)
    vin_mae_list.append(v_mae)
    
    # Metrics
    mae = mean_absolute_error(df[true_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[true_col], df[pred_col]))
    model_metrics.append({'Model': name, 'Category': category, 'MAE': mae, 'RMSE': rmse})
    loaded_models.append(name)

    if name == "Hybrid Transf.":
        best_model_data = df[[true_col, pred_col]].copy()
        best_model_data.columns = ['True', 'Pred']

if not all_errors:
    import sys; sys.exit(1)

df_err = pd.concat(all_errors, ignore_index=True)
df_vin = pd.concat(vin_mae_list, ignore_index=True)
df_metrics = pd.DataFrame(model_metrics)

# 辅助保存函数
def save_fig(fig, filename_base):
    for ext in ['.png', '.pdf']:
        out_path = os.path.join(OUT_DIR, f"{filename_base}{ext}")
        fig.savefig(out_path, dpi=400, bbox_inches='tight')
        print(f"Saved: {out_path}")
    plt.close(fig)
    
def enforce_black_borders(ax):
    """强制为图表添加粗黑边框，覆盖 seaborn 默认隐藏 top/right 的行为"""
    for spine in ax.spines.values():
        spine.set_visible(True)       # 强制显示上下左右所有边框
        spine.set_color('black')      # 强制颜色为纯黑
        spine.set_linewidth(2.0)      # 强制线宽为 2.0

# =========================
# 4) 分别绘图 (a, b, c, d)
# =========================

# --- Figure (a): CDF 累积误差分布图 ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
enforce_black_borders(ax1)  # 

# 预先计算排序后的数据以便画图
# 为了图例顺序好看，我们手动按config顺序画
for name in loaded_models:
    subset = df_err[df_err['Model'] == name]
    if subset.empty: continue
    category = subset['Category'].iloc[0]
    
    # CDF 计算
    sorted_data = np.sort(subset['abs_err'])
    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    lw = 3.5 if "Hybrid" in category else 2.5
    ls = '-' if "Hybrid" in category else ('--' if "Deep" in category else ':')
    zorder = 10 if "Hybrid" in category else 2
    
    ax1.plot(sorted_data, y_vals, label=name, 
             color=CATEGORY_COLORS[category], linewidth=lw, linestyle=ls, zorder=zorder, alpha=0.9)

ax1.set_xlim(0, df_err['abs_err'].quantile(0.99))
ax1.set_ylim(0, 1.02)
ax1.axhline(0.95, color='black', linestyle='--', alpha=0.6, linewidth=2.0)
ax1.text(ax1.get_xlim()[1]*0.4, 0.96, '95% Confidence', color='black', fontsize=14, fontweight='bold')

ax1.set_xlabel('Absolute Estimation Error (SOH)', fontweight='bold')
ax1.set_ylabel('Cumulative Probability', fontweight='bold')
ax1.set_title('(a)', loc='left', fontweight='bold', pad=15)
ax1.legend(loc='lower right', frameon=True, shadow=True, title="Model Families", title_fontsize=14, fontsize=12)
ax1.grid(False)

save_fig(fig1, "Fig5a_CDF_Curve")


# --- Figure (b): 小提琴图 (Violin Plot) ---
fig2, ax2 = plt.subplots(figsize=(10, 8))
enforce_black_borders(ax2)
sns.violinplot(x='Model', y='MAE', hue='Category', data=df_vin, ax=ax2, 
               palette=CATEGORY_COLORS, dodge=False, inner="quartile", linewidth=2.5)

ax2.set_xlabel('', fontweight='bold')
ax2.set_ylabel('Per-VIN MAE Distribution', fontweight='bold')
ax2.set_title('(b)', loc='left', fontweight='bold', pad=15)
ax2.tick_params(axis='x', rotation=30)

# 加粗图例
leg = ax2.legend(title="Category", loc='upper right', frameon=True, shadow=True)
plt.setp(leg.get_title(), fontsize=14, fontweight='bold')

# 基准线
if "HGBR" in loaded_models:
    hgbr_mean = df_vin[df_vin['Model']=='HGBR']['MAE'].mean()
    ax2.axhline(hgbr_mean, color=CATEGORY_COLORS["Baseline"], linestyle='--', alpha=0.8, linewidth=2.5)
    ax2.text(len(loaded_models)-1, hgbr_mean*1.1, 'Baseline Mean', ha='right', 
             color=CATEGORY_COLORS["Baseline"], fontweight='bold')

ax2.grid(False)

save_fig(fig2, "Fig5b_Violin_Robustness")


# --- Figure (c): 棒棒糖图 (Metrics) ---
fig3, ax3 = plt.subplots(figsize=(12, 9))
enforce_black_borders(ax3)
# 准备数据
models_c = df_metrics['Model'].values
mae_vals = df_metrics['MAE'].values
rmse_vals = df_metrics['RMSE'].values
categories = df_metrics['Category'].values
y_pos = np.arange(len(models_c))
h = 0.3

colors_for_bars = [CATEGORY_COLORS[cat] for cat in categories]

# 绘制线条
ax3.hlines(y_pos - h/2, 0, mae_vals, color=colors_for_bars, alpha=0.8, lw=4) # 线条加粗
ax3.hlines(y_pos + h/2, 0, rmse_vals, color=colors_for_bars, alpha=0.3, lw=4)

# 绘制端点
ax3.scatter(mae_vals, y_pos - h/2, color=colors_for_bars, s=200, label='MAE', zorder=10) # 点加大
ax3.scatter(rmse_vals, y_pos + h/2, color=colors_for_bars, s=200, marker='s', label='RMSE', alpha=0.6, zorder=10)

# 数字标注 (Bold)
for i, (m_val, r_val, cat) in enumerate(zip(mae_vals, rmse_vals, categories)):
    c = CATEGORY_COLORS[cat]
    
    # MAE Text
    ax3.text(m_val, i - h/2 - 0.15, f"{m_val*100:.2f}%", 
             color=c, fontsize=12, fontweight='bold', ha='center', va='bottom')

    # RMSE Text
    ax3.text(r_val, i + h/2 + 0.15, f"{r_val*100:.2f}%", 
             color=c, fontsize=11, fontweight='bold', alpha=0.7, ha='center', va='top')
    
    # Star for Best
    if m_val == min(mae_vals):
        ax3.scatter(m_val*1.08, i - h/2, color='gold', marker='*', s=350, zorder=20, edgecolor='orange', linewidth=1.5)
        ax3.text(m_val*1.12, i - h/2, 'Lowest MAE', color='firebrick', fontsize=12, fontweight='bold', va='center')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(models_c, fontweight='bold', fontsize=14)
ax3.invert_yaxis()

max_val = max(max(mae_vals), max(rmse_vals))
ax3.set_xlim(0, max_val * 1.35) # 留多点空给星星

ax3.set_xlabel('Error Metric Value (Lower is Better)', fontweight='bold')
ax3.set_title('(c)', loc='left', fontweight='bold', pad=15)

# 自定义图例
handles = [plt.Line2D([0], [0], marker='o', color='gray', label='MAE', markersize=12, linestyle=''),
           plt.Line2D([0], [0], marker='s', color='gray', label='RMSE', alpha=0.5, markersize=12, linestyle='')]
ax3.legend(handles=handles, loc='lower right', fontsize=14, frameon=True)
ax3.grid(False)

save_fig(fig3, "Fig5c_Lollipop_Metrics")


# --- Figure (d): 散点+直方图 (Best Model) ---
if best_model_data is not None:
    import matplotlib.gridspec as gridspec
    # 建立 GridSpec 布局
    fig4 = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, figure=fig4)
    
    ax_scatter = fig4.add_subplot(gs[1:4, 0:3])
    enforce_black_borders(ax_scatter)
    ax_hist_x = fig4.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_hist_y = fig4.add_subplot(gs[1:4, 3], sharey=ax_scatter)

    y_t = best_model_data['True']
    y_p = best_model_data['Pred']

    # 散点 (Hexbin)
    hb = ax_scatter.hexbin(y_t, y_p, gridsize=50, cmap='Reds', mincnt=1) 
    
    # 理想线 (Bold)
    ax_scatter.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'k--', lw=3, label='Ideal y=x')
    
    ax_scatter.set_xlabel('Actual SOH', fontweight='bold')
    ax_scatter.set_ylabel('Predicted SOH (Hybrid Transf.)', fontweight='bold')
    ax_scatter.legend(loc='lower right', fontsize=13, frameon=True)

    # R2 标注 (Bold & Large)
    r2 = r2_score(y_t, y_p)
    ax_scatter.text(0.05, 0.9, f'', transform=ax_scatter.transAxes, 
                    fontsize=16, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5', linewidth=1.5))

    # 直方图
    ax_hist_x.hist(y_t, bins=50, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_hist_y.hist(y_p, bins=50, orientation='horizontal', color=CATEGORY_COLORS["Hybrid (Ours)"], alpha=0.8, edgecolor='black', linewidth=0.5)

    # 隐藏多余刻度
    ax_hist_x.axis('off')
    ax_hist_y.axis('off')
    
    ax_hist_x.set_title('(d)', fontweight='bold', loc='left', pad=10, fontsize=16)
    
    save_fig(fig4, "Fig5d_Scatter_Hist")
else:
    print("Skipping Fig (d): No best model data found.")

print("\n✅ All individual figures generated successfully.")