# scripts/fig3.py
"""
Publication-quality feature distribution plots and ICA curve.

Generates one PNG + PDF per feature for three feature categories:
  a) Geometric  : dur_s (charging duration), Q_ah (Ah throughput), odo_end
  b) Statistical : I_mean, I_std, V_mean, V_std, cc_frac, cv_frac,
                   end_I_ratio, V_slope, I_slope
  c) ICA         : ic_mean, ic_std, ic_p95, ic_cell_mean, ic_cell_std
                   + real dQ/dV curve from a charging segment

Styling requirements (顶刊级别):
  - Font  : Arial (fallback DejaVu Sans)
  - All text/tick labels : bold
  - Axes spine linewidth : 2.0
  - Tick width           : 2.0
  - No grid
  - Output              : 600 dpi PNG + PDF, one figure per feature
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

try:
    import config
    _OUT_ROOT = Path(config.OUT_DIR)
except Exception:
    _OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

# ─────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────
FEATURE_DIR = _OUT_ROOT / "04_features"
LABEL_DIR   = _OUT_ROOT / "03_labels"
CHG_DIR     = _OUT_ROOT / "02_segments" / "charge_points"
PLOT_DIR    = _OUT_ROOT / "10_plots" / "Fig3_Features"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2.  Global matplotlib style  (applied once)
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "DejaVu Sans"],
    "mathtext.fontset":    "stixsans",
    "axes.linewidth":      2.0,
    "xtick.major.width":   2.0,
    "ytick.major.width":   2.0,
    "xtick.minor.width":   1.2,
    "ytick.minor.width":   1.2,
    "xtick.major.size":    5.0,
    "ytick.major.size":    5.0,
    "axes.unicode_minus":  False,
    "pdf.fonttype":        42,   # embed fonts in PDF
    "ps.fonttype":         42,
})

# ─────────────────────────────────────────────
# 3.  Colour palettes per category
# ─────────────────────────────────────────────
_GREEN  = "#4EA660"   # Geometric
_BLUE   = "#5292F7"   # Statistical
_RED    = "#D6404E"   # ICA / dQ/dV curve
_ORANGE = "#E8814B"   # ICA distribution features (subtle differentiation)

# ─────────────────────────────────────────────
# 4.  Feature catalogue
#     (col_name, x-axis label, category, color, panel_letter)
# ─────────────────────────────────────────────
FEATURE_CATALOGUE: list[tuple[str, str, str, str, str]] = [
    # --- Geometric ---
    ("dur_s",        "Charging Duration (h)",         "Geometric",    _GREEN,  "c1"),
    ("Q_ah",         "Ah Throughput (Ah)",             "Geometric",    _GREEN,  "c2"),
    ("odo_end",      r"Odometer ($\times10^4$ km)",    "Geometric",    _GREEN,  "c3"),
    # --- Statistical ---
    ("I_mean",       "Mean Current (A)",               "Statistical",  _BLUE,   "a1"),
    ("I_std",        "Current Std. Dev. (A)",          "Statistical",  _BLUE,   "a2"),
    ("V_mean",       "Mean Voltage (V)",               "Statistical",  _BLUE,   "a3"),
    ("V_std",        "Voltage Std. Dev. (V)",          "Statistical",  _BLUE,   "a4"),
    ("cc_frac",      "CC-Phase Fraction",              "Statistical",  _BLUE,   "a5"),
    ("cv_frac",      "CV-Phase Fraction",              "Statistical",  _BLUE,   "a6"),
    ("end_I_ratio",  r"End-to-Start Current Ratio ($I_{end}/I_{start}$)",
                                                       "Statistical",  _BLUE,   "a7"),
    ("V_slope",      r"Voltage Slope (V s$^{-1}$)",   "Statistical",  _BLUE,   "a8"),
    ("I_slope",      r"Current Slope (A s$^{-1}$)",   "Statistical",  _BLUE,   "a9"),
    # --- ICA (distribution) ---
    ("ic_mean",      "dQ/dV Mean (Ah/V)",              "ICA",          _ORANGE, "b1"),
    ("ic_std",       "dQ/dV Std. Dev. (Ah/V)",         "ICA",          _ORANGE, "b2"),
    ("ic_p95",       "dQ/dV 95th Percentile (Ah/V)",   "ICA",          _ORANGE, "b3"),
    ("ic_cell_mean", "dQ/dV Cell Mean (Ah/V)",         "ICA",          _ORANGE, "b4"),
    ("ic_cell_std",  "dQ/dV Cell Std. Dev. (Ah/V)",    "ICA",          _ORANGE, "b5"),
]

# ─────────────────────────────────────────────
# 5.  Data loading
# ─────────────────────────────────────────────

def load_feature_data() -> pd.DataFrame:
    """Read dataset_all.parquet; optionally supplement Q_ah from labels."""
    feat_path = FEATURE_DIR / "dataset_all.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feat_path}\n"
            "Please run 05_extract_core_features.py first."
        )
    df = pd.read_parquet(feat_path)
    print(f"Loaded dataset_all: {len(df):,} rows, {len(df.columns)} columns")

    # Supplement Q_ah from label files if not already present
    if "Q_ah" not in df.columns:
        print("  Q_ah not in dataset_all – reading from labels_post_*.parquet …")
        q_frames: list[pd.DataFrame] = []
        for lf in sorted(LABEL_DIR.glob("labels_post_*.parquet")):
            try:
                tmp = pd.read_parquet(lf, columns=["vin", "t_start", "t_end", "Q_ah"])
                q_frames.append(tmp)
            except Exception as exc:
                print(f"  WARNING: could not read {lf.name}: {exc}")
        if q_frames:
            q_df = pd.concat(q_frames, ignore_index=True)
            df = df.merge(q_df[["vin", "t_start", "t_end", "Q_ah"]],
                          on=["vin", "t_start", "t_end"], how="left")
            print(f"  Q_ah merged: {df['Q_ah'].notna().sum():,} non-null rows")

    # Convert odo from raw units to 10^4 km if it looks like metres/cm
    if "odo_end" in df.columns:
        med_odo = df["odo_end"].median()
        if med_odo > 1_000_000:          # likely in metres → convert to 10^4 km
            df["odo_end"] = df["odo_end"] / 1e7
        elif med_odo > 10_000:           # likely in km → convert to 10^4 km
            df["odo_end"] = df["odo_end"] / 1e4

    # Convert dur_s to hours for human readability
    if "dur_s" in df.columns:
        df["dur_s"] = df["dur_s"] / 3600.0

    return df


def get_sample_charging_segment() -> pd.DataFrame | None:
    """
    Scan CHG_DIR for a VIN sub-directory with parquet parts.
    Return a DataFrame for a single segment with ≥ 100 data points.
    Only real data – no simulation fallback.
    """
    if not CHG_DIR.exists():
        print(f"  WARNING: charge_points directory not found: {CHG_DIR}")
        return None

    needed_cols = {"terminaltime", "totalvoltage", "totalcurrent"}

    for vin_dir in sorted(CHG_DIR.iterdir()):
        if not vin_dir.is_dir():
            continue
        parts = sorted(vin_dir.glob("part_*.parquet"))
        if not parts:
            parts = sorted(vin_dir.glob("*.parquet"))
        if not parts:
            continue

        for pf in parts:
            try:
                df_raw = pd.read_parquet(pf)
            except Exception as exc:
                print(f"  WARNING: could not read {pf}: {exc}")
                continue

            missing = needed_cols - set(df_raw.columns)
            if missing:
                continue

            # Find a segment with enough points for a smooth ICA curve
            if "seg_id" in df_raw.columns:
                df_raw["seg_id"] = pd.to_numeric(df_raw["seg_id"],
                                                  errors="coerce").astype("Int64")
                df_raw = df_raw.dropna(subset=["seg_id"])
                counts = df_raw["seg_id"].value_counts()
                valid = counts[counts >= 100]
                if valid.empty:
                    continue
                # Pick the segment with the most data points
                best_seg = valid.index[0]
                seg = (df_raw[df_raw["seg_id"] == best_seg]
                       .sort_values("terminaltime")
                       .reset_index(drop=True))
            else:
                if len(df_raw) < 100:
                    continue
                seg = df_raw.sort_values("terminaltime").reset_index(drop=True)

            print(f"  Charging segment: vin={vin_dir.name}, "
                  f"file={pf.name}, n_pts={len(seg)}")
            return seg

    print("  WARNING: No valid charging segment found in charge_points.")
    return None


# ─────────────────────────────────────────────
# 6.  Single-feature distribution plot
# ─────────────────────────────────────────────

def _apply_bold_ticks(ax: plt.Axes) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _save_fig(fig: plt.Figure, name: str) -> None:
    png_path = PLOT_DIR / f"{name}.png"
    pdf_path = PLOT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {png_path.name} / {pdf_path.name}")


def plot_single_dist(
    data: pd.Series,
    panel: str,
    category: str,
    xlabel: str,
    color: str,
    save_name: str,
) -> None:
    """Plot histogram + KDE for one feature."""
    data = pd.to_numeric(data, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        print(f"  SKIP (empty): {save_name}")
        return

    # Clip to [0.5%, 99.5%] to remove extreme outliers
    lo, hi = np.percentile(data, [0.5, 99.5])
    data_clean = data[(data >= lo) & (data <= hi)]
    if len(data_clean) < 10:
        print(f"  SKIP (too few after clip): {save_name}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    sns.histplot(
        data_clean,
        bins=40,
        kde=True,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        line_kws={"color": "#333333", "linewidth": 2.5},
        alpha=0.85,
        stat="density",
        ax=ax,
    )

    # Panel label + category subtitle
    title_str = f"({panel})  {category} Feature"
    ax.set_title(title_str, fontsize=15, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel("Density", fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", which="major", labelsize=12, width=2.0, length=5)
    ax.grid(False)
    _apply_bold_ticks(ax)

    # Statistics annotation
    stats_text = (
        f"$\\mathbf{{n}}$ = {len(data_clean):,}\n"
        f"$\\mathbf{{\\mu}}$ = {data_clean.mean():.3g}\n"
        f"$\\mathbf{{\\sigma}}$ = {data_clean.std():.3g}"
    )
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        ha="right", va="top", color="#333333",
        bbox=dict(
            facecolor="white", alpha=0.88,
            edgecolor="#cccccc", linewidth=1.2,
            boxstyle="round,pad=0.4"
        ),
    )

    _save_fig(fig, save_name)


# ─────────────────────────────────────────────
# 7.  ICA  dQ/dV curve plot
# ─────────────────────────────────────────────

def _compute_dqdv(seg: pd.DataFrame, max_dt: int = 120
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smoothed dQ/dV curve from a charging segment DataFrame.

    Returns (voltage_array, dqdv_array) clipped to a sensible range.
    """
    t = pd.to_numeric(seg["terminaltime"], errors="coerce").astype("float64").values
    I = pd.to_numeric(seg["totalcurrent"], errors="coerce").astype("float64").values
    V = pd.to_numeric(seg["totalvoltage"], errors="coerce").astype("float64").values

    # Mask out bad rows
    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(V) & (V > 10)
    t, I, V = t[ok], I[ok], V[ok]
    if len(t) < 30:
        raise ValueError("Not enough valid points after masking")

    # Time steps
    dt = np.diff(t, prepend=t[0])
    dt = np.where((dt > 0) & (dt <= max_dt), dt, 0.0)

    # Charging current is negative in this dataset → flip to positive Ah
    I_chg = np.abs(I)
    Q = np.cumsum(I_chg * dt) / 3600.0   # Ah

    # Sort by voltage (required for dQ/dV)
    order = np.argsort(V)
    V_sorted = V[order]
    Q_sorted = Q[order]

    # Smooth before differentiating to reduce noise
    sigma = max(5, len(V_sorted) // 60)
    V_sm = gaussian_filter1d(V_sorted.astype(float), sigma=sigma)
    Q_sm = gaussian_filter1d(Q_sorted.astype(float), sigma=sigma)

    dV = np.gradient(V_sm)
    dQ = np.gradient(Q_sm)

    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dV) > 1e-5, dQ / dV, np.nan)

    # Second smoothing pass on dQ/dV itself
    finite_mask = np.isfinite(dqdv)
    if finite_mask.sum() < 10:
        raise ValueError("dQ/dV has too few finite points")
    dqdv_sm = dqdv.copy()
    dqdv_sm[~finite_mask] = 0.0
    dqdv_sm = gaussian_filter1d(dqdv_sm, sigma=sigma)
    dqdv_sm[~finite_mask] = np.nan

    # Restrict to a physically meaningful voltage window and non-negative dQ/dV
    v_lo = np.nanpercentile(V_sm, 2)
    v_hi = np.nanpercentile(V_sm, 98)
    mask = (V_sm >= v_lo) & (V_sm <= v_hi) & np.isfinite(dqdv_sm) & (dqdv_sm >= 0)

    if mask.sum() < 10:
        raise ValueError("No valid dQ/dV points in voltage window")

    return V_sm[mask], dqdv_sm[mask]


def plot_ica_curve(seg: pd.DataFrame, save_name: str) -> None:
    """Plot a smoothed dQ/dV (ICA) curve from a real charging segment."""
    try:
        V, dqdv = _compute_dqdv(seg)
    except Exception as exc:
        print(f"  ERROR computing dQ/dV: {exc}")
        return

    # Remove extreme dQ/dV outliers
    q_lo, q_hi = np.nanpercentile(dqdv, [1, 99])
    mask = (dqdv >= q_lo) & (dqdv <= q_hi)
    V, dqdv = V[mask], dqdv[mask]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    ax.plot(V, dqdv, color=_RED, linewidth=2.5, zorder=3, label="Smoothed dQ/dV")
    ax.fill_between(V, 0, dqdv, color=_RED, alpha=0.12, zorder=2)

    ax.set_title("(b)  ICA Feature  (Electrochemical)",
                 fontsize=15, fontweight="bold", pad=10, loc="left")
    ax.set_xlabel("Voltage (V)", fontsize=14, fontweight="bold")
    ax.set_ylabel("dQ/dV  (Ah V$^{-1}$)", fontsize=14, fontweight="bold")

    ax.set_xlim(V.min(), V.max())
    ax.set_ylim(0, dqdv.max() * 1.12)

    ax.tick_params(axis="both", which="major", labelsize=12, width=2.0, length=5)
    ax.grid(False)
    _apply_bold_ticks(ax)

    # Note on method
    ax.text(
        0.03, 0.97,
        "Numerical discrete differentiation\n(no interpolation artefacts)",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        ha="left", va="top", color="#555555",
        bbox=dict(facecolor="white", alpha=0.80,
                  edgecolor="none", boxstyle="round,pad=0.35"),
    )

    _save_fig(fig, save_name)


# ─────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  fig3.py – Feature Visualisation")
    print(f"  Output directory: {PLOT_DIR}")
    print("=" * 60)

    # ── Load feature table ─────────────────────────────────────
    df = load_feature_data()
    print(f"  Columns available: {sorted(df.columns.tolist())}\n")

    # ── Distribution plots for every feature in the catalogue ──
    generated = 0
    skipped   = 0
    for col, xlabel, category, color, panel in FEATURE_CATALOGUE:
        if col not in df.columns:
            print(f"  SKIP (column absent): {col}")
            skipped += 1
            continue
        save_name = f"Fig3_{panel}_{col}"
        print(f"  Plotting [{category}] {col} …")
        plot_single_dist(df[col], panel, category, xlabel, color, save_name)
        generated += 1

    # ── ICA dQ/dV curve from a real charging segment ───────────
    print("\n  Looking for a real charging segment for ICA curve …")
    seg = get_sample_charging_segment()
    if seg is not None:
        print("  Plotting ICA dQ/dV curve …")
        plot_ica_curve(seg, "Fig3_b0_ICA_Curve")
        generated += 1
    else:
        print("  SKIP ICA curve: no charging segment data available.")
        skipped += 1

    print()
    print("=" * 60)
    print(f"  Done.  Generated: {generated}   Skipped: {skipped}")
    print(f"  All figures saved to: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
