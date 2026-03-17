# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# ===== 1) Font & style (English only) =====
EN_FONT = "Times New Roman"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [EN_FONT]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 300

FP_EN = FontProperties(family=EN_FONT)

# ===== 2) Palette (paper-friendly, low-saturation) =====
COL = {
    "input":    ("#EEF6FF", "#2B6CB0"),
    "stage1":   ("#F7F7F7", "#4A5568"),
    "oof":      ("#FFF7E6", "#B7791F"),
    "residual": ("#FFEDEE", "#C53030"),
    "stage2":   ("#EAFBF6", "#2F855A"),
    "context":  ("#F3F9ED", "#4C7A2F"),
    "shrink":   ("#F6EEFF", "#6B46C1"),
    "output":   ("#EEFBF0", "#2F855A"),
    "arrow":    "#2D3748",
    "soft":     "#718096",
}

# ===== 3) Primitives =====
def add_box(ax, center, w, h, text, fc="#fff", ec="#333",
            lw=1.6, fs=11, rounding=0.02, shadow=False):
    x = center[0] - w/2
    y = center[1] - h/2

    if shadow:
        sh = FancyBboxPatch(
            (x + 0.003, y - 0.003), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            fc="#000000", ec="none", alpha=0.10, zorder=1
        )
        ax.add_patch(sh)

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=5
    )
    ax.add_patch(box)

    ax.text(center[0], center[1], text,
            ha="center", va="center",
            fontsize=fs, fontproperties=FP_EN,
            color="#111827", zorder=10, linespacing=1.15)

    return {
        "cx": center[0], "cy": center[1],
        "lx": x, "rx": x + w, "ly": center[1], "ry": center[1],
        "tx": center[0], "ty": y + h, "bx": center[0], "by": y,
        "w": w, "h": h
    }

def add_arrow(ax, start, end, text=None, color=None, lw=1.5,
              rad=0.0, dashed=False, ms=12, text_fs=10, text_offset=(0, 0.015)):
    if color is None:
        color = COL["arrow"]
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="->",              # stable, no huge triangles
        connectionstyle=conn,
        mutation_scale=ms,
        lw=lw,
        linestyle="--" if dashed else "-",
        color=color,
        fill=False,
        zorder=3
    )
    ax.add_patch(arr)

    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mx, my, text,
            ha="center", va="center",
            fontsize=text_fs, fontproperties=FP_EN,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95),
            zorder=20
        )

def add_frame(ax, x, y, w, h, title, color, ls="--"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc="none", ec=color, lw=1.2, linestyle=ls, alpha=0.55, zorder=0
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.015, title,
            ha="center", va="center",
            fontsize=11, fontproperties=FP_EN, color=color, zorder=2)

def main():
    fig = plt.figure(figsize=(15.2, 8.0))
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "Fig. 3-1  Physics-Guided Two-Stage State-Aware Residual Learning Framework",
        ha="center", va="center",
        fontsize=16, fontproperties=FP_EN, weight="bold", color="#111827"
    )

    # Layout
    Y_TOP, Y_MID, Y_BOT = 0.70, 0.50, 0.30
    X_INPUT, X_S1, X_RES, X_S2, X_SHR, X_OUT = 0.12, 0.33, 0.52, 0.70, 0.70, 0.90
    X_OOF, X_CTX = 0.33, 0.52

    W_BOX, H_BOX = 0.17, 0.11
    W_WIDE = 0.20

    # Frames
    add_frame(ax, 0.23, 0.18, 0.20, 0.64, "Stage-I: Physics-informed baseline", COL["soft"], ls="--")
    add_frame(ax, 0.44, 0.18, 0.47, 0.64, "Stage-II: State-aware residual correction", COL["soft"], ls="--")

    # Nodes (English only)
    n_input = add_box(
        ax, (X_INPUT, Y_MID), W_BOX, 0.18,
        "Input: charge events\n(preprocessed & aligned)",
        fc=COL["input"][0], ec=COL["input"][1], lw=1.8
    )

    n_s1 = add_box(
        ax, (X_S1, Y_TOP), W_WIDE, H_BOX,
        "Stage-I: HGBR baseline\n(missing-aware $x_{phy}$)",
        fc=COL["stage1"][0], ec=COL["stage1"][1], lw=1.5
    )

    n_oof = add_box(
        ax, (X_OOF, Y_BOT), W_WIDE, H_BOX,
        "OOF prediction\n(within train VINs)",
        fc=COL["oof"][0], ec=COL["oof"][1], lw=1.5
    )

    n_res = add_box(
        ax, (X_RES, Y_BOT), W_BOX, H_BOX,
        "Residual target\n$ r = y - Z_{prior}$",
        fc=COL["residual"][0], ec=COL["residual"][1], lw=1.5
    )

    n_ctx = add_box(
        ax, (X_CTX, Y_MID), W_BOX, H_BOX*0.92,
        "State context\n($Z_{prior}$ as condition)",
        fc=COL["context"][0], ec=COL["context"][1], lw=1.5
    )

    n_s2 = add_box(
        ax, (X_S2, Y_BOT), W_WIDE+0.03, H_BOX,
        "Stage-II: residual learner\n(MLP/TCN/GRU/Transformer)",
        fc=COL["stage2"][0], ec=COL["stage2"][1], lw=1.5, fs=10.5
    )

    n_shr = add_box(
        ax, (X_SHR, Y_TOP), W_WIDE+0.05, H_BOX,
        "Shrinkage fusion\n$Z_{final}=Z_{prior}+\\lambda\\hat r$",
        fc=COL["shrink"][0], ec=COL["shrink"][1], lw=1.5
    )

    n_out = add_box(
        ax, (X_OUT, Y_TOP), 0.14, H_BOX*0.88,
        "Output: $Z_{final}$",
        fc=COL["output"][0], ec=COL["output"][1], lw=1.8
    )

    # Arrows (consistent)
    add_arrow(ax, (n_input["rx"], n_input["cy"] + 0.045), (n_s1["lx"], n_s1["cy"]),
              text="$x_{phy}$", lw=1.6)
    add_arrow(ax, (n_input["rx"], n_input["cy"] - 0.045), (n_oof["lx"], n_oof["cy"]),
              text="$x_{phy}$", lw=1.6)

    add_arrow(ax, (n_s1["rx"], n_s1["cy"]), (n_shr["lx"], n_shr["cy"]),
              text="$Z_{prior}$", lw=1.6)
    add_arrow(ax, (n_shr["rx"], n_shr["cy"]), (n_out["lx"], n_out["cy"]),
              text="$Z_{final}$", lw=1.8)

    add_arrow(ax, (n_oof["rx"], n_oof["cy"]), (n_res["lx"], n_res["cy"]),
              text="$Z_{prior}$", lw=1.6)
    add_arrow(ax, (n_res["rx"], n_res["cy"]), (n_s2["lx"], n_s2["cy"]),
              text="$r$", lw=1.6)

    add_arrow(ax, (n_ctx["rx"], n_ctx["cy"]), (n_s2["lx"], n_s2["cy"] + 0.03),
              text="cond.", color=COL["context"][1], lw=1.3, rad=-0.12, dashed=True, ms=11,
              text_offset=(0.02, 0.02))

    add_arrow(ax, (n_s2["cx"], n_s2["ty"]), (n_shr["cx"], n_shr["by"]),
              text="$\\hat r$", color=COL["residual"][1], lw=1.4, dashed=True, ms=12,
              text_offset=(0.05, 0.02))

    # Stage-I -> context (optional dashed)
    add_arrow(ax, (n_s1["cx"], n_s1["by"]), (n_ctx["cx"], n_ctx["ty"]),
              text="$Z_{prior}$", color=COL["context"][1], lw=1.2, dashed=True, ms=11,
              text_offset=(0.06, 0.015))

    out = "Fig3-1_Framework_EN.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"[OK] saved: {out}")
    plt.show()

if __name__ == "__main__":
    main()
