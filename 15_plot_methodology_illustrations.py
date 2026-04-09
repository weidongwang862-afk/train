# 15_plot_methodology_illustrations.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure config is importable (this script lives next to config.py in repo root)
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局学术字体设置 (Arial 无衬线, 加粗边框)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['axes.linewidth'] = 2.0

# ==========================================
# 1. 路径配置
# ==========================================
PLOT_DIR = config.OUT_DIR / "10_plots" / "Methodology"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CORE_DIR = config.OUT_DIR / "01_clean_core"
LABEL_DIR = config.OUT_DIR / "03_labels"

# Window parameters for segment extraction
MIN_CHARGE_CURRENT_A = 20.0   # minimum current (A) to classify a sample as charging
WINDOW_BEFORE_CHARGE = 100    # rows to include before the first charge event
WINDOW_AFTER_CHARGE = 700     # rows to include after the first charge event

# 顶刊高级配色 (低饱和度，沉稳)
COLOR_VALID_BG = '#E2F0D9'    # 极淡的灰绿色 (背景)
COLOR_INVALID_BG = '#FADBD8'  # 极淡的灰红色 (背景)
COLOR_SOC = '#2F4F4F'         # 深岩灰
COLOR_CURRENT = '#4682B4'     # 钢蓝
COLOR_RAW_SCATTER = '#B0BEC5' # 银灰色
COLOR_OUTLIER = '#D9534F'     # 砖红色
COLOR_LOWESS = '#5DADE2'      # 矢车菊蓝
COLOR_ISOTONIC = '#800000'    # 栗色 (深红)

# ==========================================
# 辅助函数 (从 04_postprocess_labels5.py 提取)
# ==========================================

def hampel_filter(x: pd.Series, window: int = 9, n_sig: float = 3.0) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if len(s) < max(5, window):
        return s
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    thr = n_sig * 1.4826 * mad
    out = s.copy()
    mask = (s - med).abs() > thr
    out[mask] = med[mask]
    return out


def smooth_trend_lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.12) -> np.ndarray:
    y = pd.to_numeric(pd.Series(y), errors="coerce").astype(float).values
    if len(y) < 10:
        return y
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
        return np.asarray(lowess(y, x, frac=frac, return_sorted=False), dtype=float)
    except Exception:
        return pd.Series(y).rolling(21, center=True, min_periods=1).median().astype(float).values


def enforce_monotone_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = pd.to_numeric(pd.Series(y), errors="coerce").astype(float).values
    if len(y) < 3:
        return y
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        return np.asarray(ir.fit_transform(x, y), dtype=float)
    except Exception:
        return np.minimum.accumulate(y)


def _find_contiguous_regions(mask_arr: np.ndarray, min_len: int = 10) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for contiguous True runs."""
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for i, v in enumerate(mask_arr):
        if v and not in_region:
            in_region = True
            start = i
        elif not v and in_region:
            in_region = False
            if i - start >= min_len:
                regions.append((start, i - 1))
    if in_region and len(mask_arr) - start >= min_len:
        regions.append((start, len(mask_arr) - 1))
    return regions


# ==========================================
# 2. 绘制 2.2 节：有效充电片段筛选 (Data Segmentation)
# ==========================================
def plot_segment_filtering() -> None:
    """图1：展示如何从杂乱的时序中提取有效充电片段，并剔除劣质数据"""

    t_minutes: np.ndarray | None = None
    soc: np.ndarray | None = None
    current: np.ndarray | None = None

    # --- 尝试加载真实数据 ---
    if CLEAN_CORE_DIR.exists():
        for pf in sorted(CLEAN_CORE_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                needed = {'soc', 'totalcurrent', 'terminaltime'}
                if not needed.issubset(df.columns):
                    continue
                df = df.dropna(subset=list(needed))
                df = df.sort_values('terminaltime').reset_index(drop=True)
                # Find a window that contains both driving and charging
                charge_mask = df['totalcurrent'].values > MIN_CHARGE_CURRENT_A
                charge_indices = np.where(charge_mask)[0]
                if len(charge_indices) < 50:
                    continue
                # Window: start a bit before the first charge event
                start_i = max(0, charge_indices[0] - WINDOW_BEFORE_CHARGE)
                end_i = min(len(df), charge_indices[0] + WINDOW_AFTER_CHARGE)
                win = df.iloc[start_i:end_i].copy()
                if len(win) < 200:
                    continue
                t0 = float(win['terminaltime'].iloc[0])
                t_minutes = (win['terminaltime'].values.astype(float) - t0) / 60.0
                soc = win['soc'].values.astype(float)
                current = win['totalcurrent'].values.astype(float)
                print(f"✅ 加载真实数据 (segment): {pf.name}, {len(win)} rows")
                break
            except Exception as exc:
                print(f"  跳过 {pf.name}: {exc}")

    # --- 如无真实数据，回退到仿真数据 ---
    if soc is None:
        print("⚠️  未找到真实数据，使用仿真数据")
        rng = np.random.default_rng(0)
        t_minutes = np.linspace(0, 180, 1000)
        soc = np.ones(len(t_minutes)) * 30.0
        current = rng.normal(0, 5, len(t_minutes))

        # 区域A (0-20min): 正常行驶放电
        current[0:110] = rng.normal(-40, 10, 110)
        soc[0:110] = np.linspace(30, 20, 110)

        # 区域B (30-40min): 瞬间回馈充电 (应被剔除: Delta SOC < 15%)
        current[160:220] = rng.normal(60, 5, 60)
        soc[160:220] = np.linspace(20, 23, 60)
        soc[220:400] = 23

        # 区域C (70-130min): 完美快充 (应被保留)
        current[380:720] = rng.normal(120, 3, 340)
        soc[380:720] = np.linspace(23, 85, 340)
        soc[720:] = 85

        # 区域D (140-160min): 数据丢失的断点充电 (应被剔除: Delta t > 120s)
        current[770:880] = rng.normal(50, 3, 110)
        soc[770:880] = np.linspace(85, 95, 110)
        current[800:840] = np.nan
        soc[800:840] = np.nan

    t_max = float(t_minutes[-1])

    # Identify regions for annotations
    cur_arr = np.where(np.isnan(current), 0.0, current)
    charge_regions = _find_contiguous_regions(cur_arr > MIN_CHARGE_CURRENT_A, min_len=20)
    drive_regions = _find_contiguous_regions(cur_arr < -10.0, min_len=20)

    # --- 绘图 (双Y轴堆叠) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

    ax1.plot(t_minutes, soc, color=COLOR_SOC, linewidth=2.5)
    ax1.set_ylabel('SOC (%)', fontsize=15, fontweight='bold', color=COLOR_SOC)
    soc_min = float(np.nanmin(soc))
    soc_max = float(np.nanmax(soc))
    ax1.set_ylim(max(0.0, soc_min - 10), min(105.0, soc_max + 10))

    ax2.plot(t_minutes, current, color=COLOR_CURRENT, linewidth=2.0)
    ax2.set_ylabel('Current (A)', fontsize=15, fontweight='bold', color=COLOR_CURRENT)
    ax2.set_xlabel('Time (Minutes)', fontsize=15, fontweight='bold')
    c_finite = current[np.isfinite(current)]
    if len(c_finite) > 0:
        c_lo, c_hi = float(c_finite.min()), float(c_finite.max())
        margin = max((c_hi - c_lo) * 0.15, 5.0)
        ax2.set_ylim(c_lo - margin, c_hi + margin)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # --- 标注行驶/放电区段（剔除示例） ---
    if drive_regions:
        dr = drive_regions[0]
        t_s, t_e = t_minutes[dr[0]], t_minutes[dr[1]]
        ax1.axvspan(t_s, t_e, color=COLOR_INVALID_BG, alpha=0.8, zorder=0)
        ax2.axvspan(t_s, t_e, color=COLOR_INVALID_BG, alpha=0.8, zorder=0)
        mid_t = (t_s + t_e) / 2.0
        mid_soc = float(np.nanmean(soc[dr[0]:dr[1] + 1]))
        ax1.annotate(
            'Driving\n(Discharge)',
            xy=(mid_t, mid_soc),
            xytext=(mid_t + t_max * 0.08, mid_soc + 10),
            fontsize=12, fontweight='bold', color=COLOR_OUTLIER,
            arrowprops=dict(facecolor=COLOR_OUTLIER, shrink=0.05, width=2, headwidth=8),
        )

    # --- 标注有效充电区段（保留）和短充电（剔除示例） ---
    if charge_regions:
        longest_cr = max(charge_regions, key=lambda r: r[1] - r[0])
        t_s, t_e = t_minutes[longest_cr[0]], t_minutes[longest_cr[1]]
        ax1.axvspan(t_s, t_e, color=COLOR_VALID_BG, alpha=0.8, zorder=0)
        ax2.axvspan(t_s, t_e, color=COLOR_VALID_BG, alpha=0.8, zorder=0)
        mid_t = (t_s + t_e) / 2.0
        soc_top = float(np.nanmax(soc[longest_cr[0]:longest_cr[1] + 1]))
        ax1.text(
            mid_t, soc_top - 5,
            'Valid Charging Segment\n(Kept for extraction)',
            fontsize=12, fontweight='bold', color='#2E7D32', ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#2E7D32', boxstyle='round,pad=0.3'),
        )

        # Mark the shortest (rejected) charging region if there is more than one
        if len(charge_regions) > 1:
            short_cr = min(
                (r for r in charge_regions if r != longest_cr),
                key=lambda r: r[1] - r[0],
            )
            t_s2, t_e2 = t_minutes[short_cr[0]], t_minutes[short_cr[1]]
            ax1.axvspan(t_s2, t_e2, color=COLOR_INVALID_BG, alpha=0.8, zorder=0)
            ax2.axvspan(t_s2, t_e2, color=COLOR_INVALID_BG, alpha=0.8, zorder=0)
            mid_t2 = (t_s2 + t_e2) / 2.0
            mid_soc2 = float(np.nanmean(soc[short_cr[0]:short_cr[1] + 1]))
            ax1.annotate(
                'Rejected:\n$\\Delta$SOC < 15%',
                xy=(mid_t2, mid_soc2),
                xytext=(mid_t2 - t_max * 0.1, mid_soc2 + 15),
                fontsize=12, fontweight='bold', color=COLOR_OUTLIER,
                arrowprops=dict(facecolor=COLOR_OUTLIER, shrink=0.05, width=2, headwidth=8),
            )

    # --- 样式美化 ---
    ax1.set_title(
        "Rule-Based Extraction of Valid Charging Segments",
        loc='left', fontsize=16, fontweight='bold', pad=12,
    )
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=13)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    save_name = PLOT_DIR / "Fig_2_2_Segment_Filtering"
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_name}.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 生成完毕: {save_name.name}")


# ==========================================
# 3. 绘制 2.3 节：伪 SOH 标签生成过程 (Pseudo-label Pipeline)
# ==========================================
def plot_pseudo_label_generation() -> None:
    """图2：展示单车容量数据从散点 -> 去噪 -> 平滑 -> 单调物理约束的过程"""

    c_est: np.ndarray | None = None
    x_axis: np.ndarray | None = None
    vin_used: str | None = None

    # --- 尝试加载真实标签数据 ---
    if LABEL_DIR.exists():
        for lf in sorted(LABEL_DIR.glob("capacity_labels_*.parquet")):
            try:
                df = pd.read_parquet(lf)
                if 'C_est_ah' not in df.columns:
                    continue
                df = df.dropna(subset=['C_est_ah'])
                if len(df) < 30:
                    continue
                # Sort by odometer or time
                odo_col = next(
                    (c for c in ['odo_end', 'totalodometer_end', 'totalodometer', 'odo']
                     if c in df.columns),
                    None,
                )
                key_col = odo_col if odo_col else 't_end'
                if key_col in df.columns:
                    df = df.sort_values(key_col).reset_index(drop=True)
                    x_vals = df[key_col].astype(float).values
                    # Normalise to a 0-based sequence index for the x-axis label
                    x_axis = np.arange(len(df), dtype=float)
                else:
                    x_axis = np.arange(len(df), dtype=float)
                c_est = df['C_est_ah'].values.astype(float)
                vin_used = lf.stem.replace("capacity_labels_", "")
                print(f"✅ 加载真实标签数据: {lf.name}, {len(df)} rows")
                break
            except Exception as exc:
                print(f"  跳过 {lf.name}: {exc}")

    # --- 回退到仿真数据 ---
    if c_est is None:
        print("⚠️  未找到真实标签数据，使用仿真数据")
        rng = np.random.default_rng(42)
        n = 200
        x_axis = np.linspace(0, 300, n)
        true_cap = 100.0 - 0.02 * x_axis - 0.00015 * (x_axis ** 2)
        c_est = true_cap + rng.normal(0, 1.5, n)
        outlier_idx = rng.choice(range(20, 180), 8, replace=False)
        c_est[outlier_idx] += rng.choice([-1, 1], 8) * rng.uniform(5, 10, 8)

    assert x_axis is not None  # guaranteed above

    # --- 应用处理流水线 ---
    # Step 2: Hampel Filter
    c_hampel = hampel_filter(pd.Series(c_est), window=9, n_sig=3.0)
    outlier_mask = np.abs(c_est - c_hampel.values) > 0.01

    # Step 3: LOWESS Smoothing
    c_smooth = smooth_trend_lowess(x_axis, c_hampel.values, frac=0.12)

    # Step 4: Isotonic Regression (monotone decreasing)
    c_trend = enforce_monotone_decreasing(x_axis, c_smooth)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.scatter(x_axis, c_est, color=COLOR_RAW_SCATTER, s=20, alpha=0.6,
               label='Step 1: Raw Capacity via Ah-Integration')

    if outlier_mask.any():
        ax.scatter(x_axis[outlier_mask], c_est[outlier_mask],
                   color=COLOR_OUTLIER, marker='x', s=80, linewidths=2.5,
                   label='Step 2: Hampel Filter Outliers')

    ax.plot(x_axis, c_smooth, color=COLOR_LOWESS, linestyle='--', linewidth=2.5,
            label='Step 3: LOWESS Smoothing (Trend extraction)')

    ax.plot(x_axis, c_trend, color=COLOR_ISOTONIC, linestyle='-', linewidth=3.5,
            label='Step 4: Isotonic Regression (Monotonic Physical Constraint)')

    # --- 样式美化 ---
    title_suffix = f" ({vin_used})" if vin_used else ""
    ax.set_title(
        f"SOH Pseudo-label Generation Pipeline{title_suffix}",
        loc='left', fontsize=16, fontweight='bold', pad=12,
    )
    ax.set_xlabel('Charging Event Sequence', fontsize=15, fontweight='bold')
    ax.set_ylabel('Estimated Capacity (Ah)', fontsize=15, fontweight='bold')

    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.legend(loc='lower left', fontsize=11, frameon=True, edgecolor='black', fancybox=False)

    save_name = PLOT_DIR / "Fig_2_3_Pseudo_Label_Generation"
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_name}.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 生成完毕: {save_name.name}")


# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    print("\n开始生成方法论插图...")
    plot_segment_filtering()
    plot_pseudo_label_generation()
    print(f"\n🎉 全部方法论图表已存入 {PLOT_DIR} 目录中！")
