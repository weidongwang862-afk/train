# scripts/06_train_baseline_vinsplits_ycol.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

import config

OUT_ROOT = Path(config.OUT_DIR).resolve()

# ======= Settings =======
MODE = os.environ.get("MODE", "vin_splits").strip()  # holdout | fit_all | vin_splits
SEED = int(os.environ.get("SEED", "42"))
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.20"))

# target column in frozen dataset
Y_COL = os.environ.get("Y_COL", "SoH_trend").strip()

# 强制按车划分文件（两列：split, vin）
SPLITS_VINS = os.environ.get("SPLITS_VINS", "").strip()

# dataset path（默认给 project 输出路径；你也可以用 env 覆盖）
DATASET = os.environ.get(
    "DATASET",
    str(OUT_ROOT / "04_features" / "dataset_all_C_frozen.parquet")
).strip()

# 特征列表文件（单列：特征名）
FEAT_CSV = os.environ.get(
    "FEAT_CSV",
    str(OUT_ROOT / "04_features" / "features_FINAL_core.csv")
).strip()

OUT_TAG = os.environ.get("OUT_TAG", "HGBR_BASELINE").strip()

# 低 SoH 加权（可选）
LOW_THR = float(os.environ.get("LOW_THR", "0.90"))
LOW_W = float(os.environ.get("LOW_W", "1.0"))  # 1.0 表示不开低段加权

# gating（可选：若列存在就计算 gated_any）
# 你的 frozen 列表里有 any_hit / any_relax_n，这里默认用 any_hit
GATE_ANY_COL = os.environ.get("GATE_ANY_COL", "any_hit").strip()  # >0 表示存在 any relax
GATE_ANY_MIN = float(os.environ.get("GATE_ANY_MIN", "1.0"))

# keys（尽量写入预测文件）
KEYS = ["vin", "t_start", "t_end", "terminaltime"]


def _load_vin_splits() -> tuple[list[str], list[str], list[str], Path]:
    """
    读取 splits_vins.csv（两列：split, vin），返回 train/val/test 的 VIN 列表
    """
    default_path = OUT_ROOT / "09_seq_featcore" / "splits_vins.csv"
    fp = Path(SPLITS_VINS) if SPLITS_VINS else default_path
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing splits_vins.csv: {fp}\n"
            f"Set env SPLITS_VINS to your file path."
        )

    s = pd.read_csv(fp)
    if not {"split", "vin"}.issubset(set(s.columns)):
        raise RuntimeError(f"splits_vins.csv must contain columns: split, vin. got={list(s.columns)}")

    s["vin"] = s["vin"].astype(str)
    tr = s.loc[s["split"] == "train", "vin"].tolist()
    va = s.loc[s["split"] == "val", "vin"].tolist()
    te = s.loc[s["split"] == "test", "vin"].tolist()

    print(f"[splits] file={fp}")
    print(f"[splits] train n_vins={len(tr)} head={tr[:10]}")
    print(f"[splits] val   n_vins={len(va)} head={va[:10]}")
    print(f"[splits] test  n_vins={len(te)} head={te[:10]}")
    return tr, va, te, fp


def _read_feat_cols(feat_csv: str) -> list[str]:
    fp = Path(feat_csv)
    if not fp.exists():
        raise FileNotFoundError(f"Missing FEAT_CSV: {fp}")

    # 先按“有表头”读：常见是列名叫 feature
    df0 = pd.read_csv(fp)
    if df0.shape[1] >= 1:
        if "feature" in df0.columns:
            cols = df0["feature"].astype(str).tolist()
        else:
            # 没叫 feature 就取第一列
            cols = df0.iloc[:, 0].astype(str).tolist()
    else:
        cols = []

    # 如果读出来只有一行且像是表头失败，再按“无表头单列”读
    if len(cols) == 0:
        df1 = pd.read_csv(fp, header=None)
        cols = df1.iloc[:, 0].astype(str).tolist()

    # 清洗：去空、去 nan、去表头残留
    cols = [c.strip() for c in cols if str(c).strip() and str(c).strip().lower() != "nan"]
    cols = [c for c in cols if c.lower() not in {"feature", "features"}]

    # 去重保持顺序
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out



def _make_model(seed: int) -> HistGradientBoostingRegressor:
    # 保持你 baseline 的默认（先做公平对比）
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_depth=6,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.0,
        random_state=seed,
    )


def _build_xy(df: pd.DataFrame, feat_cols: list[str], y_col: str) -> tuple[np.ndarray, np.ndarray]:
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[y_col].to_numpy(dtype=np.float32, copy=False)
    return X, y


def _weights_low_soh(y: np.ndarray, thr: float, w: float) -> np.ndarray:
    if w <= 1.0:
        return np.ones_like(y, dtype=np.float32)
    ww = np.ones_like(y, dtype=np.float32)
    ww[y < thr] = w
    return ww


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _predict_frame(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, pred_name: str) -> pd.DataFrame:
    out = pd.DataFrame({"vin": df["vin"].astype(str).values})
    for k in ["t_end", "t_start", "terminaltime"]:
        if k in df.columns:
            out[k] = df[k].values
    out["y_true"] = y_true
    out[pred_name] = y_pred
    return out


def main():
    np.random.seed(SEED)

    feat_cols = _read_feat_cols(FEAT_CSV)
    if len(feat_cols) == 0:
        raise RuntimeError(f"Empty feature list in FEAT_CSV: {FEAT_CSV}")

    df = pd.read_parquet(DATASET)

    if "vin" not in df.columns:
        raise RuntimeError(f"dataset missing 'vin' column: {DATASET}")
    if Y_COL not in df.columns:
        raise RuntimeError(
            f"dataset missing target column Y_COL='{Y_COL}'. "
            f"available head={list(df.columns)[:60]}"
        )

    print(f"[dataset] path={DATASET}")
    print(f"[target] y_col={Y_COL}")
    print(f"[feats] n_feat_cols={len(feat_cols)} | feat_csv={FEAT_CSV}")

    df["vin"] = df["vin"].astype(str)

    # 保留必要列：keys + target + gate_col + feat_cols
    keep_cols = list(dict.fromkeys(KEYS + [Y_COL, GATE_ANY_COL] + feat_cols))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # 某些特征列如果不在 df 里会导致 KeyError：这里做一次硬检查，便于你定位
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise RuntimeError(
            f"Feature columns missing in dataset: n_missing={len(missing_feats)} "
            f"head={missing_feats[:20]}"
        )

    # ===== split =====
    if MODE == "holdout":
        vins = df["vin"].unique().tolist()
        rng = np.random.RandomState(SEED)
        rng.shuffle(vins)
        n_te = int(len(vins) * HOLDOUT_FRAC)
        vin_te = set(vins[:n_te])
        tr_df = df[~df["vin"].isin(vin_te)].copy()
        va_df = None
        te_df = df[df["vin"].isin(vin_te)].copy()
        print(f"[split] MODE=holdout | train_vins={tr_df['vin'].nunique()} test_vins={te_df['vin'].nunique()}")

    elif MODE == "fit_all":
        tr_df = df.copy()
        va_df = None
        te_df = None
        print(f"[split] MODE=fit_all | n_rows={len(tr_df)} n_vins={tr_df['vin'].nunique()}")

    elif MODE == "vin_splits":
        tr_v, va_v, te_v, fp = _load_vin_splits()
        tr_df = df[df["vin"].isin(tr_v)].copy()
        va_df = df[df["vin"].isin(va_v)].copy()
        te_df = df[df["vin"].isin(te_v)].copy()
        print(
            f"[split] MODE=vin_splits | "
            f"train={len(tr_df)} val={len(va_df)} test={len(te_df)} | "
            f"train_vins={tr_df['vin'].nunique()} val_vins={va_df['vin'].nunique()} test_vins={te_df['vin'].nunique()}"
        )
    else:
        raise ValueError(f"Unknown MODE={MODE}")

    # ===== train core =====
    Xtr, ytr = _build_xy(tr_df, feat_cols, y_col=Y_COL)
    wtr = _weights_low_soh(ytr, LOW_THR, LOW_W)

    core = _make_model(SEED)
    core.fit(Xtr, ytr, sample_weight=wtr)

    out_dir = OUT_ROOT / "04_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _eval_split(name: str, split_df: pd.DataFrame, out_suffix: str):
        X, y = _build_xy(split_df, feat_cols, y_col=Y_COL)
        p_core = core.predict(X).astype(np.float32)
        mae_core = _mae(y, p_core)

        out = _predict_frame(split_df, y, p_core, pred_name="pred_core")

        # ===== gated_any (optional) =====
        mae_any = np.nan
        if GATE_ANY_COL in split_df.columns:
            gate_vals = split_df[GATE_ANY_COL].fillna(0).to_numpy()
            mask_any = gate_vals >= GATE_ANY_MIN
        else:
            mask_any = np.zeros(len(split_df), dtype=bool)

        if mask_any.any():
            if GATE_ANY_COL in tr_df.columns:
                gate_tr = tr_df[GATE_ANY_COL].fillna(0).to_numpy()
                mask_any_tr = gate_tr >= GATE_ANY_MIN
            else:
                mask_any_tr = np.zeros(len(tr_df), dtype=bool)

            if mask_any_tr.any():
                Xtr_a, ytr_a = Xtr[mask_any_tr], ytr[mask_any_tr]
                wtr_a = _weights_low_soh(ytr_a, LOW_THR, LOW_W)

                gated = _make_model(SEED + 1)
                gated.fit(Xtr_a, ytr_a, sample_weight=wtr_a)

                p_any = np.full(len(split_df), np.nan, dtype=np.float32)
                p_any[mask_any] = gated.predict(X[mask_any]).astype(np.float32)

                out["pred_gated_any"] = p_any
                mae_any = _mae(y[mask_any], p_any[mask_any])
            else:
                out["pred_gated_any"] = np.nan
        else:
            out["pred_gated_any"] = np.nan

        print(f"[{name}] core_MAE={mae_core:.6f} | n={len(split_df)} | vins={split_df['vin'].nunique()}")
        if not np.isnan(mae_any):
            print(f"[{name}] gated_any_MAE={mae_any:.6f} | n_any={int(mask_any.sum())}")

        out_fp = out_dir / f"predictions_{out_suffix}_{OUT_TAG}_{Y_COL}.csv"
        out.to_csv(out_fp, index=False, encoding="utf-8-sig")
        print("Saved:", out_fp)

    if MODE == "vin_splits":
        _eval_split("VAL", va_df, out_suffix="val")
        _eval_split("TEST", te_df, out_suffix="test")
        _eval_split("TRAIN", tr_df, out_suffix="train")

    elif MODE == "holdout":
        _eval_split("TEST", te_df, out_suffix="test")

    elif MODE == "fit_all":
        Xall, yall = _build_xy(tr_df, feat_cols, y_col=Y_COL)
        pall = core.predict(Xall).astype(np.float32)
        out = _predict_frame(tr_df, yall, pall, pred_name="pred_core")
        out_fp = out_dir / f"predictions_all_{OUT_TAG}_{Y_COL}.csv"
        out.to_csv(out_fp, index=False, encoding="utf-8-sig")
        print("Saved:", out_fp)


if __name__ == "__main__":
    main()
