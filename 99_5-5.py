# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# ===== 1. 基础配置 =====
CN_FONT_CANDIDATES = ["SimSun", "STSong", "Songti SC", "NSimSun", "Microsoft YaHei"]
EN_FONT = "Times New Roman"

def _pick_available_font(cands):
    from matplotlib.font_manager import fontManager
    available = set([f.name for f in fontManager.ttflist])
    for c in cands:
        if c in available:
            return c
    return "SimSun"

CN_FONT = _pick_available_font(CN_FONT_CANDIDATES)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [CN_FONT]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 300

FP_CN = FontProperties(family=CN_FONT)
FP_EN = FontProperties(family=EN_FONT)

# ===== 颜色方案 =====
COLORS = {
    'input': ('#e3f2fd', '#1565c0'),
    'stage1': ('#f5f5f5', '#616161'),
    'oof': ('#fff8e1', '#ff8f00'),
    'residual': ('#ffebee', '#c62828'),
    'stage2': ('#e0f2f1', '#00695c'),
    'context': ('#f1f8e9', '#558b2f'),
    'shrinkage': ('#f3e5f5', '#7b1fa2'),
    'output': ('#e8f5e9', '#2e7d32'),
    'arrow': '#546e7a',
}

# ===== 2. 绘图组件 =====

def add_box(ax, center, w, h, text_cn, text_en=None, fc="#f7f7f7", ec="#333", 
            lw=1.5, fontsize=10, shadow=True):
    x = center[0] - w/2
    y = center[1] - h/2
    
    # 阴影：右下角
    if shadow:
        shadow_box = FancyBboxPatch(
            (x + 0.004, y - 0.004), w, h,
            boxstyle="Round,pad=0.01,rounding_size=0.02", 
            fc='#aaaaaa', ec='none', alpha=0.25, zorder=5
        )
        ax.add_patch(shadow_box)
    
    # 主盒子
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="Round,pad=0.01,rounding_size=0.02", 
        fc=fc, ec=ec, lw=lw, zorder=10
    )
    ax.add_patch(box)

    if text_en is None:
        ax.text(center[0], center[1], text_cn, ha="center", va="center",
                fontsize=fontsize, fontproperties=FP_CN, zorder=20)
    else:
        ax.text(center[0], center[1] + h*0.20, text_cn, ha="center", va="center",
                fontsize=fontsize, fontproperties=FP_CN, zorder=20)
        ax.text(center[0], center[1] - h*0.20, text_en, ha="center", va="center",
                fontsize=fontsize-1, fontproperties=FP_EN, color="#444", zorder=20)
    return box

def add_arrow(ax, start, end, text=None, color="#546e7a", lw=1.2, 
              style="-|>", curved=0, text_offset=(0, 0.015)):
    if curved == 0:
        conn_style = "arc3,rad=0"
    else:
        conn_style = f"arc3,rad={curved}"
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=conn_style,
        mutation_scale=12,
        lw=lw,
        color=color,
        zorder=15
    )
    ax.add_patch(arrow)
    
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center",
                fontsize=9, fontproperties=FP_EN, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
                zorder=25)

def add_polyline_arrow(ax, points, color="#546e7a", lw=1.2):
    """绘制折线箭头"""
    for i in range(len(points) - 1):
        is_last = (i == len(points) - 2)
        style = "-|>" if is_last else "-"
        arrow = FancyArrowPatch(
            points[i], points[i+1],
            arrowstyle=style,
            connectionstyle="arc3,rad=0",
            mutation_scale=12,
            lw=lw,
            color=color,
            zorder=15
        )
        ax.add_patch(arrow)

def main():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])  # 留出边距
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ===== 统一布局坐标 =====
    Y_TOP = 0.72
    Y_MID = 0.50
    Y_BOT = 0.28
    
    # 水平位置 - 所有元素往右移，给左边留空间
    X_INPUT = 0.10      # 从 0.06 改为 0.10
    X_STAGE1 = 0.30     # 从 0.28 改为 0.30
    X_OOF = 0.30
    X_RES = 0.50        # 从 0.48 改为 0.50
    X_CTX = 0.50
    X_STAGE2 = 0.70     # 从 0.68 改为 0.70
    X_SHRINK = 0.70
    X_OUT = 0.88

    # 统一盒子尺寸
    W_BOX = 0.14
    H_BOX = 0.12
    W_WIDE = 0.16
    
    # ===== 标题 =====
    ax.text(0.5, 0.94, "Fig 5-5  Two-stage Residual Learning Framework", 
            ha="center", fontsize=16, fontproperties=FP_EN, weight='bold')

    # ===== 背景分组区域 =====
    bg1 = FancyBboxPatch(
        (X_STAGE1 - 0.10, Y_BOT - 0.10), 0.20, Y_TOP - Y_BOT + 0.22,
        boxstyle="Round,pad=0.01,rounding_size=0.03",
        fc='#fafafa', ec='#bdbdbd', ls='--', lw=1, alpha=0.5, zorder=1
    )
    ax.add_patch(bg1)
    ax.text(X_STAGE1, Y_TOP + 0.10, "Stage 1: Base Prediction", 
            ha="center", fontsize=11, fontproperties=FP_EN, color='#757575')

    bg2 = FancyBboxPatch(
        (X_RES - 0.10, Y_BOT - 0.10), 0.42, Y_TOP - Y_BOT + 0.22,
        boxstyle="Round,pad=0.01,rounding_size=0.03",
        fc='#fafafa', ec='#bdbdbd', ls='--', lw=1, alpha=0.5, zorder=1
    )
    ax.add_patch(bg2)
    ax.text((X_RES + X_STAGE2)/2, Y_TOP + 0.10, "Stage 2: Residual Correction", 
            ha="center", fontsize=11, fontproperties=FP_EN, color='#757575')

    # ===== 绘制盒子 =====
    
    add_box(ax, (X_INPUT, Y_MID), W_BOX, 0.18,
        "输入：单车充电序列\n(预处理/对齐)", "Input: Charge Seqs",
        fc=COLORS['input'][0], ec=COLORS['input'][1], lw=2
    )

    add_box(ax, (X_STAGE1, Y_TOP), W_WIDE, H_BOX,
        "阶段1：强基线回归\n(HGBR 模型)", "Stage-1: HGBR",
        fc=COLORS['stage1'][0], ec=COLORS['stage1'][1], lw=1.5
    )

    add_box(ax, (X_SHRINK, Y_TOP), W_WIDE + 0.02, H_BOX,
        "Shrinkage 收缩融合\n" + r"$\hat{y} = \hat{y}_1 + \alpha \cdot \Delta r$", 
        "Fusion Module",
        fc=COLORS['shrinkage'][0], ec=COLORS['shrinkage'][1], lw=1.5
    )

    add_box(ax, (X_OUT, Y_TOP), W_BOX - 0.02, H_BOX - 0.02,
        "最终输出", "Final Output",
        fc=COLORS['output'][0], ec=COLORS['output'][1], lw=2
    )

    add_box(ax, (X_OOF, Y_BOT), W_WIDE, H_BOX,
        "OOF 预测生成\n(分组交叉验证)", "OOF Prediction",
        fc=COLORS['oof'][0], ec=COLORS['oof'][1], lw=1.5
    )

    add_box(ax, (X_RES, Y_BOT), W_BOX, H_BOX,
        "残差构造\n" + r"$r = y - \hat{y}_{oof}$", "Residual",
        fc=COLORS['residual'][0], ec=COLORS['residual'][1], lw=1.5
    )

    add_box(ax, (X_STAGE2, Y_BOT), W_WIDE + 0.02, H_BOX,
        "阶段2：残差学习\n(Transformer/TCN)", "Stage-2: Learner",
        fc=COLORS['stage2'][0], ec=COLORS['stage2'][1], lw=1.5
    )

    add_box(ax, (X_CTX, Y_MID), W_BOX, H_BOX - 0.02,
        "上下文注入\n(位置/工况)", "Context",
        fc=COLORS['context'][0], ec=COLORS['context'][1], lw=1.5
    )

    # ===== 绘制连线 =====
    
    arrow_color = COLORS['arrow']
    
    # 折线的转折点
    X_TURN = X_INPUT + W_BOX/2 + 0.025
    
    # Input -> Stage 1 (上路)
    p1 = (X_INPUT + W_BOX/2, Y_MID + 0.04)
    p2 = (X_TURN, Y_MID + 0.04)
    p3 = (X_TURN, Y_TOP)
    p4 = (X_STAGE1 - W_WIDE/2, Y_TOP)
    add_polyline_arrow(ax, [p1, p2, p3, p4], color=arrow_color)
    ax.text(X_TURN + 0.015, Y_TOP - 0.08, "X", ha="left", va="center",
            fontsize=9, fontproperties=FP_EN, color=arrow_color,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))

    # Input -> OOF (下路)
    p1 = (X_INPUT + W_BOX/2, Y_MID - 0.04)
    p2 = (X_TURN, Y_MID - 0.04)
    p3 = (X_TURN, Y_BOT)
    p4 = (X_OOF - W_WIDE/2, Y_BOT)
    add_polyline_arrow(ax, [p1, p2, p3, p4], color=arrow_color)
    ax.text(X_TURN + 0.015, Y_BOT + 0.08, "X", ha="left", va="center",
            fontsize=9, fontproperties=FP_EN, color=arrow_color,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))

    # Stage 1 -> Shrinkage
    add_arrow(ax, (X_STAGE1 + W_WIDE/2, Y_TOP), (X_SHRINK - W_WIDE/2 - 0.01, Y_TOP),
              text=r"$\hat{y}_1$", color=arrow_color)

    # Shrinkage -> Output
    add_arrow(ax, (X_SHRINK + W_WIDE/2 + 0.01, Y_TOP), (X_OUT - W_BOX/2 + 0.01, Y_TOP),
              color=arrow_color, lw=1.8)

    # OOF -> Residual
    add_arrow(ax, (X_OOF + W_WIDE/2, Y_BOT), (X_RES - W_BOX/2, Y_BOT),
              text=r"$\hat{y}_{oof}$", color=arrow_color)

    # Residual -> Stage 2
    add_arrow(ax, (X_RES + W_BOX/2, Y_BOT), (X_STAGE2 - W_WIDE/2 - 0.01, Y_BOT),
              text=r"$r$", color=arrow_color)

    # Stage 2 -> Shrinkage
    add_arrow(ax, (X_STAGE2, Y_BOT + H_BOX/2), (X_SHRINK, Y_TOP - H_BOX/2),
              text=r"$\Delta r$", color=COLORS['residual'][1], lw=1.5)

    # Stage 1 -> Context (虚线)
    ctx_points = [
        (X_STAGE1, Y_TOP - H_BOX/2),
        (X_STAGE1, Y_MID),
        (X_CTX - W_BOX/2, Y_MID)
    ]
    for i in range(len(ctx_points) - 1):
        is_last = (i == len(ctx_points) - 2)
        arr = FancyArrowPatch(
            ctx_points[i], ctx_points[i+1],
            arrowstyle="-|>" if is_last else "-",
            connectionstyle="arc3,rad=0",
            mutation_scale=10, lw=1, linestyle='--',
            color=COLORS['context'][1], zorder=15
        )
        ax.add_patch(arr)

    # Context -> Stage 2
    add_arrow(ax, (X_CTX + W_BOX/2, Y_MID), (X_STAGE2 - W_WIDE/2 - 0.01, Y_BOT + 0.02),
              curved=-0.2, color=COLORS['context'][1], text="Context",
              text_offset=(0.03, 0.02))

    # ===== 底部注释 =====
    note_text = (
        r"注：Input 经双路处理。上路(Top)提供稳定基线 $\hat{y}_1$，"
        r"下路(Bottom)通过 OOF 残差 $r$ 学习非线性偏差 $\Delta r$，"
        r"最终在 Shrinkage 模块通过 $\hat{y} = \hat{y}_1 + \alpha \Delta r$ 完成自适应融合。"
    )
    ax.text(0.5, 0.06, note_text, ha="center", fontsize=9, 
            fontproperties=FP_CN, color="#666")

    # ===== 保存 - 不使用 bbox_inches="tight" =====
    out = "Fig5-5_Final.png"
    fig.savefig(out, dpi=300, facecolor='white', pad_inches=0.1)
    print(f"[OK] saved: {out}")

if __name__ == "__main__":
    main()