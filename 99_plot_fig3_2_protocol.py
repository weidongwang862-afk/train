# 99_plot_fig3_2_protocol_v3_pretty.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ==================== 1) Global style ====================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.size"] = 12
rcParams["axes.linewidth"] = 1.2

FIG_W, FIG_H = 13, 7.2
DPI = 300

# Palette (paper-friendly)
C_TEXT = "#1A202C"
C_EDGE = "#2D3748"
C_ARROW = "#2D3748"

C_FRAME_OUT = "#C53030"
C_FRAME_IN  = "#2B6CB0"

C_NODE_RAW   = "#EDF2F7"
C_NODE_SPLIT = "#FED7D7"
C_NODE_OOF   = "#DBEAFE"
C_NODE_HGBR  = "#E6F4FF"
C_NODE_PRIOR = "#BFDBFE"
C_NODE_RES   = "#FDE68A"
C_NODE_EVAL  = "#DCFCE7"

# ==================== 2) Canvas ====================
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = plt.gca()
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# ==================== 3) Helpers ====================
def draw_box(x, y, w, h, text, fc, ec=C_EDGE, lw=1.8, fs=12, bold=True):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=0.22",
        ec=ec, fc=fc, linewidth=lw, zorder=10
    )
    ax.add_patch(box)
    ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center",
        fontsize=fs, fontweight=("bold" if bold else "normal"),
        color=C_TEXT, zorder=11, linespacing=1.25
    )
    return {"x": x, "y": y, "w": w, "h": h,
            "cx": x + w/2, "cy": y + h/2,
            "lx": x, "ly": y + h/2,
            "rx": x + w, "ry": y + h/2,
            "tx": x + w/2, "ty": y + h,
            "bx": x + w/2, "by": y}

def arrow(p1, p2, text=None, rad=0.0):
    # 用稳定的箭头样式，避免大三角形
    a = mpatches.FancyArrowPatch(
        p1, p2,
        arrowstyle="->",              # 关键：不要用 -|> + head_width/head_length
        mutation_scale=14,            # 头部大小（适中）
        lw=1.8, color=C_ARROW,
        fill=False,                   # 关键：不填充，彻底杜绝黑色多边形
        connectionstyle=f"arc3,rad={rad}",
        zorder=1                      # 关键：箭头永远在底层
    )
    ax.add_patch(a)

    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(
            mx, my, text,
            ha="center", va="center",
            fontsize=10.5, color="#4A5568",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.95),
            zorder=20
        )


def frame(x, y, w, h, ec, ls, title, ty):
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=ec,
                              linewidth=2.0, linestyle=ls, zorder=2)
    ax.add_patch(rect)
    ax.text(
        x + w/2, ty, title,
        ha="center", va="center",
        fontsize=13, fontweight="bold",
        color=ec,
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
        zorder=30
    )

# ==================== 4) Frames ====================
frame(
    x=0.25, y=0.25, w=11.5, h=6.5,
    ec=C_FRAME_OUT, ls="--",
    title="Outer Loop: Leave-One-Group-Out (LOGO) by VIN",
    ty=6.62
)

frame(
    x=5.95, y=2.55, w=5.75, h=3.35,
    ec=C_FRAME_IN, ls=":",
    title="Inner Loop: OOF within Train VINs",
    ty=5.78
)

# ==================== 5) Nodes (aligned layout) ====================
# Row y positions
Y_TOP = 4.55
Y_MID = 3.25
Y_BOT = 1.35

# Node sizes
W1, H1 = 2.10, 1.05
W2, H2 = 2.20, 1.05
W3, H3 = 2.40, 0.95

raw   = draw_box(0.65, Y_MID, W1, H1, "Raw Dataset\n(grouped by VIN)",
                 fc=C_NODE_RAW, fs=11)
split = draw_box(3.35, Y_MID, W1, H1, "LOGO Split\n(Train vs Test)",
                 fc=C_NODE_SPLIT, ec=C_FRAME_OUT, fs=11)

kfold = draw_box(6.25, Y_TOP, W2, H2, "GroupKFold on\nTrain VINs",
                 fc=C_NODE_OOF, ec=C_FRAME_IN, fs=11)

hgbr  = draw_box(9.05, Y_TOP, W3, H3, "Train HGBR (K-1 folds)\nPredict on held-out fold",
                 fc=C_NODE_HGBR, ec="#4299E1", fs=10.5)

prior = draw_box(9.05, 3.05, W3, 0.80, "Aggregate OOF\n$Z_{prior}$",
                 fc=C_NODE_PRIOR, ec=C_FRAME_IN, fs=11)

res   = draw_box(6.25, Y_BOT, 2.55, 1.05, "Train Residual Net\nSelect $\\lambda$",
                 fc=C_NODE_RES, ec="#D69E2E", fs=11)

evaln = draw_box(3.35, Y_BOT, W1, 1.05, "Final Evaluation\n(unseen VINs)",
                 fc=C_NODE_EVAL, ec="#38A169", fs=11)

# ==================== 6) Arrows ====================
arrow((raw["rx"], raw["ry"]), (split["lx"], split["ly"]))

# Split -> KFold (train branch)
arrow((split["rx"], split["ry"] + 0.22), (kfold["lx"], kfold["ly"]), text="Train VINs", rad=0.0)

# KFold -> HGBR
arrow((kfold["rx"], kfold["ry"]), (hgbr["lx"], hgbr["ly"]))

# HGBR -> OOF prior (down)
arrow((hgbr["cx"], hgbr["by"]), (prior["cx"], prior["ty"]), rad=0.0)

# Prior -> Residual (smooth backflow curve)
arrow((prior["lx"], prior["ly"]), (res["rx"], res["ry"]), text="OOF $Z_{prior}$", rad=-0.28)

# Split -> Eval (test branch)
arrow((split["cx"], split["by"]), (evaln["cx"], evaln["ty"]), text="Test VINs", rad=0.0)

# Residual -> Eval (frozen)
arrow((res["lx"], res["ly"]), (evaln["rx"], evaln["ry"]), text="Frozen models", rad=0.15)

# ==================== 7) Save ====================
plt.tight_layout()
save_path = "Fig3-2_Protocol_Schema_Q1.png"
plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
print(f"Figure saved to: {save_path}")
plt.show()
