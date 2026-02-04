# scripts/06A_oof_hgbr_preds.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import config

OUT_ROOT = Path(config.OUT_DIR)

# ---------- paths ----------
DATASET = Path(os.environ.get("DATASET", r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"))
FEAT_CSV = Path(os.environ.get("FEAT_CSV", r"E:\RAW_DATA\outputs\04_features\features_FINAL_core.csv"))
SPLITS_VINS = Path(os.environ.get("SPLITS_VINS", str(OUT_ROOT / "09_seq_featcore" / "splits_vins.csv")))

OUT_DIR = Path(os.environ.get("OUT_DIR", str(OUT_ROOT / "09_seq_featcore" / "09_residual")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- settings ----------
SEED = int(os.environ.get("SEED", "42"))
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
Y_COL = os.environ.get("Y_COL", "SoH_trend").strip()

# gate indicator: your parquet has any_hit / any_relax_n; do NOT use n_any
GATE_COL = os.environ.get("GATE_COL", "").strip()  # optional override
GATE_MIN = float(os.environ.get("GATE_MIN", "1.0"))

OUT_TAG = os.environ.get("OUT_TAG", "HGBR_OOF").strip()


def _read_feat_cols(fp: Path) -> list[str]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing FEAT_CSV: {fp}")
    df = pd.read_csv(fp)
    if df.shape[1] == 1:
        c0 = str(df.columns[0]).lower()
        if c0 in ["feature", "features", "feat"]:
            cols = df.iloc[:, 0].astype(str).tolist()
        else:
            df2 = pd.read_csv(fp, header=None)
            cols = df2.iloc[:, 0].astype(str).tolist()
    else:
        cols = df.iloc[:, 0].astype(str).tolist()

    cols = [c for c in cols if c and c != "feature"]
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_splits(fp: Path) -> tuple[list[str], list[str], list[str]]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing splits_vins.csv: {fp}")
    s = pd.read_csv(fp)
    if not {"split", "vin"}.issubset(s.columns):
        raise RuntimeError(f"splits_vins.csv must contain columns split, vin. got={list(s.columns)}")
    s["vin"] = s["vin"].astype(str)
    tr = s.loc[s["split"] == "train", "vin"].tolist()
    va = s.loc[s["split"] == "val", "vin"].tolist()
    te = s.loc[s["split"] == "test", "vin"].tolist()
    print(f"[splits] file={fp}")
    print(f"[splits] train n_vins={len(tr)} head={tr[:10]}")
    print(f"[splits] val   n_vins={len(va)} head={va[:10]}")
    print(f"[splits] test  n_vins={len(te)} head={te[:10]}")
    return tr, va, te


def _pick_gate_col(cols: set[str]) -> str:
    if GATE_COL:
        if GATE_COL not in cols:
            raise RuntimeError(f"GATE_COL={GATE_COL} not in dataset columns")
        return GATE_COL
    for c in ["any_hit", "any_relax_n"]:
        if c in cols:
            return c
    raise RuntimeError("Cannot find gate column: need any_hit or any_relax_n in dataset.")


def _make_model(seed: int) -> HistGradientBoostingRegressor:
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


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def main():
    np.random.seed(SEED)

    feat_cols = _read_feat_cols(FEAT_CSV)
    print(f"[feats] n_feat_cols={len(feat_cols)} | feat_csv={FEAT_CSV}")

    tr_vins, va_vins, te_vins = _load_splits(SPLITS_VINS)

    schema_cols = set(pq.read_schema(DATASET).names)
    need_core = ["vin", "t_end", Y_COL]
    miss = [c for c in need_core if c not in schema_cols]
    if miss:
        raise RuntimeError(f"dataset missing required columns: {miss}")

    # read only existing columns
    want_cols = ["vin", "t_end", Y_COL] + feat_cols + ["any_hit", "any_relax_n"]
    use_cols = [c for c in want_cols if c in schema_cols]

    df_all = pd.read_parquet(DATASET, columns=use_cols)
    df_all["vin"] = df_all["vin"].astype(str)
    df_all["t_end"] = df_all["t_end"].astype(np.int64)

    gate_col = _pick_gate_col(set(df_all.columns))
    print(f"[gate] gate_col={gate_col} gate_min={GATE_MIN}")

    tr_df = df_all[df_all["vin"].isin(tr_vins)].copy()
    va_df = df_all[df_all["vin"].isin(va_vins)].copy()
    te_df = df_all[df_all["vin"].isin(te_vins)].copy()
    print(f"[data] train={len(tr_df)} val={len(va_df)} test={len(te_df)} | "
          f"train_vins={tr_df['vin'].nunique()} val_vins={va_df['vin'].nunique()} test_vins={te_df['vin'].nunique()}")

    # matrices
    X = tr_df[feat_cols].to_numpy(np.float32, copy=False)
    y = tr_df[Y_COL].to_numpy(np.float32, copy=False)
    groups = tr_df["vin"].to_numpy()

    # train gated mask
    gvals = tr_df[gate_col].fillna(0).to_numpy()
    if gate_col == "any_hit":
        m_any_tr = gvals >= 1.0
    else:
        m_any_tr = gvals >= GATE_MIN

    # ---------- OOF on TRAIN ----------
    oof_core = np.full(len(tr_df), np.nan, dtype=np.float32)
    oof_any = np.full(len(tr_df), np.nan, dtype=np.float32)

    gkf = GroupKFold(n_splits=min(N_SPLITS, tr_df["vin"].nunique()))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        core = _make_model(SEED + fold)
        core.fit(X[tr_idx], y[tr_idx])
        oof_core[va_idx] = core.predict(X[va_idx]).astype(np.float32)

        tr_any_idx = tr_idx[m_any_tr[tr_idx]]
        va_any_idx = va_idx[m_any_tr[va_idx]]
        if len(tr_any_idx) > 0 and len(va_any_idx) > 0:
            gated = _make_model(SEED + 100 + fold)
            gated.fit(X[tr_any_idx], y[tr_any_idx])
            oof_any[va_any_idx] = gated.predict(X[va_any_idx]).astype(np.float32)

        print(f"[fold {fold}] core_filled={np.isfinite(oof_core[va_idx]).sum()}/{len(va_idx)} "
              f"| any_filled={np.isfinite(oof_any[va_idx]).sum()}/{len(va_idx)}")

    if np.isnan(oof_core).any():
        raise RuntimeError(f"OOF pred_core has NaN rows: {int(np.isnan(oof_core).sum())}")

    out_tr = tr_df[["vin", "t_end", Y_COL]].copy()
    out_tr.rename(columns={Y_COL: "y_true"}, inplace=True)
    out_tr["pred_core"] = oof_core
    out_tr["pred_gated_any"] = oof_any

    fp_tr_core = OUT_DIR / f"predictions_train_OOF_pred_core_{OUT_TAG}.csv"
    fp_tr_any = OUT_DIR / f"predictions_train_OOF_pred_gated_any_{OUT_TAG}.csv"
    out_tr[["vin", "t_end", "y_true", "pred_core"]].to_csv(fp_tr_core, index=False, encoding="utf-8-sig")
    out_tr[["vin", "t_end", "y_true", "pred_gated_any"]].to_csv(fp_tr_any, index=False, encoding="utf-8-sig")
    print("Saved:", fp_tr_core)
    print("Saved:", fp_tr_any)

    # ---------- FULL fit on TRAIN, predict VAL/TEST ----------
    core_full = _make_model(SEED)
    core_full.fit(X, y)

    gated_full = None
    if m_any_tr.any():
        gated_full = _make_model(SEED + 999)
        gated_full.fit(X[m_any_tr], y[m_any_tr])

    def _pred_split(df_split: pd.DataFrame, name: str):
        Xs = df_split[feat_cols].to_numpy(np.float32, copy=False)
        ys = df_split[Y_COL].to_numpy(np.float32, copy=False)

        pred_core = core_full.predict(Xs).astype(np.float32)

        gv = df_split[gate_col].fillna(0).to_numpy()
        if gate_col == "any_hit":
            m_any = gv >= 1.0
        else:
            m_any = gv >= GATE_MIN

        pred_any = np.full(len(df_split), np.nan, dtype=np.float32)
        if gated_full is not None and m_any.any():
            pred_any[m_any] = gated_full.predict(Xs[m_any]).astype(np.float32)

        out = df_split[["vin", "t_end"]].copy()
        out["y_true"] = ys
        out["pred_core"] = pred_core
        out["pred_gated_any"] = pred_any

        fp_core = OUT_DIR / f"predictions_{name}_FULL_pred_core_{OUT_TAG}.csv"
        fp_any = OUT_DIR / f"predictions_{name}_FULL_pred_gated_any_{OUT_TAG}.csv"
        out[["vin", "t_end", "y_true", "pred_core"]].to_csv(fp_core, index=False, encoding="utf-8-sig")
        out[["vin", "t_end", "y_true", "pred_gated_any"]].to_csv(fp_any, index=False, encoding="utf-8-sig")

        print(f"[{name.upper()}] core_MAE={_mae(ys, pred_core):.6f} n={len(df_split)} vins={df_split['vin'].nunique()}")
        if np.isfinite(pred_any).any():
            m = np.isfinite(pred_any)
            print(f"[{name.upper()}] gated_any_MAE={_mae(ys[m], pred_any[m]):.6f} n_any={int(m.sum())}")
        print("Saved:", fp_core)
        print("Saved:", fp_any)

    _pred_split(va_df, "val")
    _pred_split(te_df, "test")


if __name__ == "__main__":
    main()
