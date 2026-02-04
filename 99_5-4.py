# -*- coding: utf-8 -*-
"""
Fig 5-4 最差 VIN 案例剖析（True vs Pred + 残差 + 工况背景）
- 兼容你的真实文件：HGBR 预测CSV没有 t_end，Deep(FUSION等) 有 t_end
- 对齐策略：以 Deep 的时间轴为主；HGBR 按 VIN 内样本顺序“贴”到 Deep 的时间排序上
- 可选叠加工况背景：segment_summary_vinXX.parquet（若无/读不到则自动跳过）
- 字体：中文宋体(SimSun) + 英文 Times New Roman
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 0) 路径与参数：按需改这里
# =========================
VIN_TARGET = "vin125"  

PRED_HGBR_CSV = r"E:\RAW_DATA\outputs\09_seq_featcore\09_residual\predictions_test_FULL_pred_core_HGBR_OOF.csv"
PRED_DEEP_CSV = r"E:\RAW_DATA\outputs\09_seq_featcore\predictions_test_SEQ_TFMR.csv"  # 你也可以换成 ATTENTION/GRU 等
SEGMENT_SUMMARY_PARQUET = rf"E:\RAW_DATA\outputs\02_segments\segment_summary_{VIN_TARGET}.parquet"  # 没有就放空字符串 ""

OUT_FIG = rf"E:\RAW_DATA\Fig5-4_case_{VIN_TARGET}.png"

# Deep 模型名显示（用于图例）
DEEP_NAME = "Transformer"

# 允许对齐时的最小长度要求（Deep 与 HGBR 同 VIN 样本数差太多就直接用 index 轴）
ALIGN_MIN_RATIO = 0.85

# =========================
# 1) 字体与风格
# =========================
def setup_fonts():
    mpl.rcParams["axes.unicode_minus"] = False
    # 让英文优先 Times New Roman，中文 fallback 到 SimSun（宋体）
    mpl.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.titleweight"] = "regular"
    mpl.rcParams["figure.dpi"] = 160

setup_fonts()


# =========================
# 2) 通用工具：列名自动识别
# =========================
def pick_col(df, cands, required=False, name=""):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"[缺列] {name} 需要列之一 {cands}，但当前列为：{list(df.columns)}")
    return None


def normalize_vin(v):
    return str(v)


# =========================
# 3) 读取预测文件（CSV）
# =========================
def load_pred_csv(path, model_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{model_name}] 找不到文件：{path}")

    df = pd.read_csv(path)

    vin_col = pick_col(df, ["vin", "VIN"], required=True, name=f"{model_name}.vin")
    y_true_col = pick_col(df, ["y_true", "true", "label", "target", "soh_true", "SoH_true", "SoH_trend", "soh_trend"],
                          required=True, name=f"{model_name}.y_true")
    y_pred_col = pick_col(df, ["y_pred", "pred_core", "prediction", "y_hat", "soh_pred", "SoH_pred"],
                          required=True, name=f"{model_name}.y_pred")
    t_col = pick_col(df, ["t_end", "t", "t_idx", "tidx", "time", "terminaltime", "tend", "t_end_s", "t_end_min"],
                     required=False, name=f"{model_name}.t")

    out = pd.DataFrame({
        "vin": df[vin_col].map(normalize_vin),
        "y_true": pd.to_numeric(df[y_true_col], errors="coerce"),
        "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce"),
    })
    if t_col is not None:
        out["t"] = pd.to_numeric(df[t_col], errors="coerce")
    else:
        out["t"] = np.nan

    out = out.dropna(subset=["vin", "y_true", "y_pred"]).reset_index(drop=True)
    out["model"] = model_name
    return out


# =========================
# 4) 可选：读取工况分段（Parquet）
# =========================
def load_segments_parquet(path, vin_target):
    if (path is None) or (str(path).strip() == ""):
        return None
    if not os.path.exists(path):
        return None

    # 你的环境一般有 pyarrow；这里直接用 pandas.read_parquet
    seg = pd.read_parquet(path)

    vin_col = pick_col(seg, ["vin", "VIN"], required=False)
    if vin_col is not None:
        seg = seg[seg[vin_col].map(normalize_vin) == normalize_vin(vin_target)].copy()

    start_col = pick_col(seg, ["t_start", "start", "t0", "s_start", "start_t"], required=False)
    end_col = pick_col(seg, ["t_end", "end", "t1", "s_end", "end_t"], required=False)
    state_col = pick_col(seg, ["state", "segment", "seg", "seg_type", "mode", "label"], required=False)

    if start_col is None or end_col is None:
        # 没有边界列就没法画背景，直接跳过
        return None

    out = pd.DataFrame({
        "t_start": pd.to_numeric(seg[start_col], errors="coerce"),
        "t_end": pd.to_numeric(seg[end_col], errors="coerce"),
    })
    if state_col is not None:
        out["state"] = seg[state_col].astype(str).str.lower()
    else:
        out["state"] = "other"

    out = out.dropna(subset=["t_start", "t_end"]).sort_values(["t_start", "t_end"]).reset_index(drop=True)
    return out


# =========================
# 5) 对齐：用 Deep 的 t 作为主轴，把 HGBR 贴过去
# =========================
def align_on_deep_time(df_hgbr, df_deep, vin_target):
    a = df_hgbr[df_hgbr["vin"] == normalize_vin(vin_target)].copy()
    b = df_deep[df_deep["vin"] == normalize_vin(vin_target)].copy()

    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"[数据为空] {vin_target} 在 HGBR/Deep 中至少有一个为空：HGBR={len(a)}, Deep={len(b)}")

    # Deep 必须有 t 才能作为主轴，否则退化为 index 轴
    if b["t"].isna().all():
        return None, a.reset_index(drop=True), b.reset_index(drop=True)

    b = b.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    # HGBR 没 t：按 VIN 内原顺序，映射到 Deep 的排序上
    # 若 HGBR 自己也有 t，就按 t 排序再对齐
    if not a["t"].isna().all():
        a = a.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    else:
        a = a.reset_index(drop=True)

    # 长度不一致处理：尽量取共同长度；差距过大则用 index 轴
    n_a, n_b = len(a), len(b)
    ratio = min(n_a, n_b) / max(n_a, n_b)

    if ratio < ALIGN_MIN_RATIO:
        # 差太大：直接用 index 轴
        return None, a.reset_index(drop=True), b.reset_index(drop=True)

    n = min(n_a, n_b)
    a2 = a.iloc[:n].copy().reset_index(drop=True)
    b2 = b.iloc[:n].copy().reset_index(drop=True)

    x = b2["t"].to_numpy(dtype=float)
    return x, a2, b2


# =========================
# 6) 画图：True vs Pred + 残差 + 工况背景
# =========================
def infer_x_label(x):
    if x is None:
        return "样本序号 (Index)"
    xmax = float(np.nanmax(x)) if len(x) else 0.0
    # 经验规则：你这里的 3-2 用 Minutes，seq 也常见 minute/second tick
    if xmax >= 24 * 60 * 10:
        return "时间 (Minutes)"
    if xmax >= 1e5:
        return "时间 (Seconds)"
    return "时间 (t)"


def draw_segments(ax, seg_df, x_min, x_max):
    if seg_df is None or len(seg_df) == 0:
        return

    # 颜色：按你前面 3-2 的习惯（充电绿、行驶橙、静置灰、其他浅灰）
    cmap = {
        "charge": ("#96e9a8", 0.55),
        "charging": ("#96e9a8", 0.55),
        "drive": ("#eea149", 0.50),
        "driving": ("#eea149", 0.50),
        "rest": ("#b1b7bd", 0.60),
        "idle": ("#b1b7bd", 0.60),
        "other": ("#f2f2f2", 0.35),
    }

    for _, r in seg_df.iterrows():
        t0, t1 = float(r["t_start"]), float(r["t_end"])
        if t1 < x_min or t0 > x_max:
            continue
        st = str(r.get("state", "other")).lower()
        color, alpha = cmap.get(st, cmap["other"])
        ax.axvspan(max(t0, x_min), min(t1, x_max), color=color, alpha=alpha, lw=0)


def plot_case(vin_target, df_hgbr, df_deep, seg_df=None, out_path=None):
    x, a, b = align_on_deep_time(df_hgbr, df_deep, vin_target)

    # x 轴与序列
    if x is None:
        n = min(len(a), len(b))
        a = a.iloc[:n].reset_index(drop=True)
        b = b.iloc[:n].reset_index(drop=True)
        x = np.arange(n, dtype=float)

    y_true = b["y_true"].to_numpy(dtype=float)
    y_pred_h = a["y_pred"].to_numpy(dtype=float)
    y_pred_d = b["y_pred"].to_numpy(dtype=float)

    res_h = y_pred_h - y_true
    res_d = y_pred_d - y_true

    # 范围
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))

    fig = plt.figure(figsize=(11.8, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # 背景工况
    draw_segments(ax1, seg_df, x_min, x_max)
    draw_segments(ax2, seg_df, x_min, x_max)

    # 上图：True vs Pred
    ax1.plot(x, y_true, lw=2.0, color="#222222", label="True")
    ax1.plot(x, y_pred_h, lw=2.0, ls="--", color="#6c757d", label="HGBR (基线模型)")
    ax1.plot(x, y_pred_d, lw=2.2, color="#d62728", label=DEEP_NAME)

    ax1.set_ylabel("SoH / 电池容量 (归一化)", fontsize=13)
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # 下图：Residual
    ax2.axhline(0.0, color="#222222", lw=1.2, alpha=0.8)
    ax2.plot(x, res_h, lw=1.6, ls="--", color="#6c757d", label="HGBR 残差")
    ax2.plot(x, res_d, lw=1.8, color="#d62728", label=f"{DEEP_NAME} 残差")
    ax2.set_ylabel("残差\n(预测 - 真实)", fontsize=12)
    ax2.set_xlabel(infer_x_label(x), fontsize=13)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)

    # 标题
    fig.suptitle(f"最差VIN结果分析（以VIN125为例）: {vin_target}", fontsize=15, y=0.98)

    # 保存
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[保存完成] {out_path}")
    plt.close(fig)


# =========================
# 7) 主程序
# =========================
def main():
    df_h = load_pred_csv(PRED_HGBR_CSV, "HGBR (Baseline)")
    df_d = load_pred_csv(PRED_DEEP_CSV, DEEP_NAME)

    seg_df = None
    try:
        seg_df = load_segments_parquet(SEGMENT_SUMMARY_PARQUET, VIN_TARGET)
    except Exception as e:
        # 工况文件不影响主图，读不到就跳过
        print(f"[提示] 工况分段文件读取失败，已跳过：{e}")

    plot_case(VIN_TARGET, df_h, df_d, seg_df=seg_df, out_path=OUT_FIG)


if __name__ == "__main__":
    main()
