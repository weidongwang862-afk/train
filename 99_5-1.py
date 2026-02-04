import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) 配置：改成你的预测文件父目录
# =========================
BASE_DIR = r"E:\RAW_DATA\outputs"
OUT_DIR  = r"E:\RAW_DATA"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Fig5-1_Model_Comparison_A.png")

# =========================
# 2) 对比集合 A：文件名模式匹配
# =========================
MODEL_FILES = [
    ("HGBR",        "predictions_test_FULL_pred_core_HGBR_OOF.csv"),
    ("GRU", "predictions_test_SEQ_GRU.csv"),
    ("BiGRU",    "predictions_test_SEQ_BIGRU.csv"),
    ("ConvGRU",  "predictions_test_SEQ_CONVGRU.csv"),
    ("Transformer",     "predictions_test_SEQ_TFMR.csv"),
    ("Fusion",   "predictions_test_SEQ_FUSION.csv"),
]

# 自动识别列名
VIN_CANDS  = ["vin", "VIN"]
TRUE_CANDS = ["y_true", "true", "label", "soh_true", "soh_trend"]
PRED_CANDS = ["y_pred", "pred", "pred_soh", "pred_core", "yhat"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def find_one_file(pattern):
    hits = glob.glob(os.path.join(BASE_DIR, "**", pattern), recursive=True)
    if len(hits) == 0:
        raise FileNotFoundError(f"未找到文件: {pattern}（搜索目录：{BASE_DIR}）")
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def eval_pred_file(name, path):
    df = pd.read_csv(path)

    vin_col  = pick_col(df, VIN_CANDS)
    true_col = pick_col(df, TRUE_CANDS)
    pred_col = pick_col(df, PRED_CANDS)

    if true_col is None or pred_col is None:
        raise ValueError(f"[{name}] 缺少必要列。现有列: {df.columns.tolist()}")

    use_cols = [c for c in [vin_col, true_col, pred_col] if c is not None]
    df = df[use_cols].copy()

    err = df[pred_col] - df[true_col]
    df["abs_error"] = err.abs()

    micro = float(df["abs_error"].mean())
    n_samples = int(df.shape[0])

    if vin_col is None:
        macro = micro
        vin_std = 0.0
        n_vins = 1
    else:
        vin_mae = df.groupby(vin_col)["abs_error"].mean()
        macro = float(vin_mae.mean())
        vin_std = float(vin_mae.std(ddof=1)) if vin_mae.shape[0] > 1 else 0.0
        n_vins = int(vin_mae.shape[0])

    return dict(name=name, path=path, micro=micro, macro=macro, vin_std=vin_std,
                n_samples=n_samples, n_vins=n_vins)

def main():
    # 字体：英文 Times + 中文宋体
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    stats = []
    for name, pat in MODEL_FILES:
        path = find_one_file(pat)
        stats.append(eval_pred_file(name, path))

    names   = [s["name"] for s in stats]
    macro   = np.array([s["macro"] for s in stats], dtype=float)
    micro   = np.array([s["micro"] for s in stats], dtype=float)
    vin_std = np.array([s["vin_std"] for s in stats], dtype=float)

    # =========================
    # 颜色叙事：基线灰 / 深度冷色 / 主角暖色
    # =========================
    colors = []
    for n in names:
        if n == "HGBR":
            colors.append("#9E9E9E")       # 灰：强基线
        elif n == "Fusion":
            colors.append("#E64A19")       # 暖：主角
        else:
            colors.append("#2E6F9E")       # 冷：普通深度模型（统一冷色调）

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12.4, 6.2), dpi=300)

    # 柱子 + 黑色误差棒（工字线）
    bars = ax.bar(
        x, macro, width=0.62, color=colors, alpha=0.92,
        edgecolor="black", linewidth=0.8,
        yerr=vin_std, capsize=5,
        error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2)
    )

    ax.set_title("强基线 vs 端到端序列模型：总体性能对比（测试集）", fontweight="bold", pad=12)
    ax.set_ylabel("macro MAE（按VIN均匀加权）", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.30)

    # y轴留白，给柱顶文字空间
    y_max = float((macro + vin_std).max())
    ax.set_ylim(0, y_max * 1.22)

    # =========================
    # 双指标标注：
    # - 柱顶：macro
    # - 柱内：micro
    # =========================
    for i, b in enumerate(bars):
        h = b.get_height()

        # 柱顶 macro
        ax.text(
            b.get_x() + b.get_width()/2, h + y_max*0.02,
            f"{macro[i]:.5f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

        # 柱内 micro（放在柱体中上部）
        ax.text(
            b.get_x() + b.get_width()/2, h * 0.55,
            f"micro\n{micro[i]:.5f}",
            ha="center", va="center", fontsize=10,
            color="white" if colors[i] != "#9E9E9E" else "black",
            fontweight="bold"
        )

    # 图例式说明（不搞复杂legend，直接图内注释）
    ax.text(0.01, 0.98,
            "柱高：macro MAE；柱内：micro MAE；误差棒：per-VIN MAE Std",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color="0.35")

    # 右下角补充样本与VIN数，评委常问
    info = []
    for s in stats:
        info.append(f"{s['name']}: vins={s['n_vins']}, n={s['n_samples']}")
    ax.text(0.99, 0.02, "\n".join(info),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="0.35")

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.show()

    print("Saved:", OUT_PNG)
    for s in stats:
        print(f"[{s['name']}] macro={s['macro']:.6f} micro={s['micro']:.6f} std={s['vin_std']:.6f} | {s['path']}")

if __name__ == "__main__":
    main()
