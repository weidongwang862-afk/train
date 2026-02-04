# scripts/06E_groupkfold_gated_eval.py
from __future__ import annotations

import os
from pathlib import Path
import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

import config

OUT_ROOT = Path(config.OUT_DIR)
FEAT_DIR = OUT_ROOT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = os.environ.get("TARGET", "SoH_trend").strip()
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))

DEFAULT_DATASET = r"E:\RAW_DATA\outputs\04_features\dataset_all_C_frozen.parquet"
DATASET = Path(os.environ.get("DATASET", DEFAULT_DATASET)).expanduser()

# 需要你先跑 06D 得到两套 feature set（你可以直接把路径 set 进来）
CORE_FSET = Path(os.environ.get("CORE_FSET", str(FEAT_DIR / "features_FINAL_core.csv")))
RELAX_FSET = Path(os.environ.get("RELAX_FSET", str(FEAT_DIR / "features_FINAL_any.csv")))

GATE_COL = os.environ.get("GATE_COL", "has_any_relax").strip()
STRICT_COL = os.environ.get("STRICT_COL", "has_any_relax_strict").strip()
MIN_RELAX_TRAIN = int(os.environ.get("MIN_RELAX_TRAIN", "1000"))

OUT_TAG = os.environ.get("OUT_TAG", "GATED_C").strip()

POST_MIN_REST_S = int(os.environ.get("POST_MIN_REST_S", "1800"))
POST_MAX_GAP_S = int(os.environ.get("POST_MAX_GAP_S", "600"))
PRE_MIN_REST_S = int(os.environ.get("PRE_MIN_REST_S", "3600"))
PRE_MAX_GAP_S = int(os.environ.get("PRE_MAX_GAP_S", "3600"))

# 融合参数
DIFF_CLIP = float(os.environ.get("DIFF_CLIP", "0.008"))
W_BASE = float(os.environ.get("W_BASE", "0.30"))
W_STRICT_BONUS = float(os.environ.get("W_STRICT_BONUS", "0.35"))
W_MAX = float(os.environ.get("W_MAX", "0.70"))

# 额外质量因子：按 post rest 时长平滑（可关）
USE_DUR_WEIGHT = os.environ.get("USE_DUR_WEIGHT", "1").strip() == "1"
DUR_FULL_S = float(os.environ.get("DUR_FULL_S", "1800"))


def read_feature_list(p: Path) -> list[str]:
    if not p.exists():
        raise FileNotFoundError(f"Feature list not found: {p}")
    df = pd.read_csv(p)
    if "feature" in df.columns:
        feats = df["feature"].astype(str).tolist()
    else:
        feats = df.iloc[:, 0].astype(str).tolist()
    feats = [f for f in feats if f and f.lower() != "nan"]
    return list(dict.fromkeys(feats))


def build_gate_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_post = df["post_relax_n"].notna() if "post_relax_n" in df.columns else pd.Series(False, index=df.index)
    has_pre = df["pre_relax_n"].notna() if "pre_relax_n" in df.columns else pd.Series(False, index=df.index)

    if "any_hit" in df.columns:
        has_any = pd.to_numeric(df["any_hit"], errors="coerce").fillna(0).astype("int64") > 0
    else:
        has_any = (has_post | has_pre)

    df["has_any_relax"] = has_any.astype("int8")

    post_strict = has_post.copy()
    if "rst_duration_s" in df.columns:
        post_strict &= (pd.to_numeric(df["rst_duration_s"], errors="coerce") >= POST_MIN_REST_S)
    if "gap_s" in df.columns:
        post_strict &= (pd.to_numeric(df["gap_s"], errors="coerce") <= POST_MAX_GAP_S)

    pre_strict = has_pre.copy()
    if "pre_rest_dur_s" in df.columns:
        pre_strict &= (pd.to_numeric(df["pre_rest_dur_s"], errors="coerce") >= PRE_MIN_REST_S)
    if "pre_gap_s" in df.columns:
        pre_strict &= (pd.to_numeric(df["pre_gap_s"], errors="coerce") <= PRE_MAX_GAP_S)

    df["has_any_relax_strict"] = (post_strict | pre_strict).fillna(False).astype("int8")
    return df


def fit_predict_hgb(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, wtr: np.ndarray | None = None) -> np.ndarray:
    Xtr = Xtr.apply(pd.to_numeric, errors="coerce")
    Xva = Xva.apply(pd.to_numeric, errors="coerce")
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xva = Xva.fillna(med)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        l2_regularization=0.0,
        random_state=42,
    )
    if wtr is None:
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr, sample_weight=np.asarray(wtr, dtype="float64"))

    return model.predict(Xva)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_parquet(DATASET, engine="pyarrow")
    if "vin" not in df.columns:
        raise ValueError("Dataset must contain column 'vin'")

    df = build_gate_cols(df)

    core_feats = read_feature_list(CORE_FSET)
    relax_feats = read_feature_list(RELAX_FSET)

    missing_core = [c for c in core_feats if c not in df.columns]
    missing_relax = [c for c in relax_feats if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing core features: {missing_core[:10]}")
    if missing_relax:
        raise ValueError(f"Missing relax features: {missing_relax[:10]}")

    groups = df["vin"].astype(str).values
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))

    fold_rows, vin_rows, event_rows = [], [], []

    for fold, (itr, iva) in enumerate(gkf.split(df, groups=groups), start=1):
        dtr = df.iloc[itr].copy()
        dva = df.iloc[iva].copy()

        ytr = pd.to_numeric(dtr[TARGET], errors="coerce").to_numpy(dtype="float64")
        yva = pd.to_numeric(dva[TARGET], errors="coerce").to_numpy(dtype="float64")

        if "qc_weight" in dtr.columns:
            wtr_all = pd.to_numeric(dtr["qc_weight"], errors="coerce").fillna(1.0).to_numpy(dtype="float64")
        else:
            wtr_all = np.ones(len(dtr), dtype="float64")

        pred_core = fit_predict_hgb(dtr[core_feats], ytr, dva[core_feats], wtr=wtr_all)

        gate_tr = pd.to_numeric(dtr[GATE_COL], errors="coerce").fillna(0).astype("int64") > 0
        gate_va = pd.to_numeric(dva[GATE_COL], errors="coerce").fillna(0).astype("int64") > 0
        n_relax_tr = int(gate_tr.sum())
        n_relax_va = int(gate_va.sum())

        pred = pred_core.copy()
        used_relax_model = False

        if n_relax_tr >= MIN_RELAX_TRAIN and n_relax_va > 0:
            used_relax_model = True

            pred_relax = fit_predict_hgb(
                dtr.loc[gate_tr, relax_feats],
                ytr[gate_tr.values],
                dva.loc[gate_va, relax_feats],
                wtr=wtr_all[gate_tr.values],
            )

            gv = gate_va.values.astype(bool)

            strict_v = np.zeros(gv.sum(), dtype="int64")
            if STRICT_COL in dva.columns:
                strict_v = pd.to_numeric(dva.loc[gv, STRICT_COL], errors="coerce").fillna(0).astype("int64").values

            w = np.full(gv.sum(), W_BASE, dtype="float64")
            w += W_STRICT_BONUS * (strict_v > 0).astype("float64")
            w = np.minimum(w, W_MAX)

            if USE_DUR_WEIGHT and "rst_duration_s" in dva.columns:
                dur = pd.to_numeric(dva.loc[gv, "rst_duration_s"], errors="coerce").fillna(0).values.astype("float64")
                q = np.log1p(dur / 60.0) / np.log1p(DUR_FULL_S / 60.0)
                q = np.clip(q, 0.0, 1.0)
                w = w * q

            diff = np.abs(pred_relax - pred_core[gv])
            scale = np.minimum(1.0, DIFF_CLIP / np.maximum(diff, 1e-12))
            w = w * scale

            pred[gv] = pred_core[gv] + w * (pred_relax - pred_core[gv])

        mae_core = float(mean_absolute_error(yva, pred_core))
        mae_gated = float(mean_absolute_error(yva, pred))

        fold_rows.append({
            "fold": fold,
            "n_train": int(len(dtr)),
            "n_valid": int(len(dva)),
            "n_relax_train": n_relax_tr,
            "n_relax_valid": n_relax_va,
            "mae_core": mae_core,
            "mae_gated": mae_gated,
            "used_relax_model": int(used_relax_model),
        })

        tmp = dva[["vin", "t_start", "t_end"]].copy() if all(c in dva.columns for c in ["vin", "t_start", "t_end"]) else pd.DataFrame({"vin": dva["vin"].astype(str).values})
        tmp["fold"] = fold
        tmp["y"] = yva
        tmp["pred_core"] = pred_core
        tmp["pred_gated"] = pred
        tmp["ae_core"] = np.abs(tmp["y"] - tmp["pred_core"])
        tmp["ae_gated"] = np.abs(tmp["y"] - tmp["pred_gated"])
        tmp["improve"] = tmp["ae_core"] - tmp["ae_gated"]
        event_rows.append(tmp)

        for vin, g in tmp.groupby("vin"):
            vin_rows.append({
                "fold": fold,
                "vin": str(vin),
                "n": int(len(g)),
                "mae_core": float(mean_absolute_error(g["y"].values, g["pred_core"].values)),
                "mae_gated": float(mean_absolute_error(g["y"].values, g["pred_gated"].values)),
                "delta": float(mean_absolute_error(g["y"].values, g["pred_gated"].values) - mean_absolute_error(g["y"].values, g["pred_core"].values)),
            })

        print(f"[fold {fold}] core={mae_core:.6f} gated={mae_gated:.6f} (relax_train={n_relax_tr}, relax_valid={n_relax_va}, used={used_relax_model})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{OUT_TAG}_{ts}"

    df_fold = pd.DataFrame(fold_rows)
    df_vin = pd.DataFrame(vin_rows)
    df_event = pd.concat(event_rows, ignore_index=True)

    out_fold = FEAT_DIR / f"cv_fold_metrics_{prefix}.csv"
    out_vin = FEAT_DIR / f"cv_vin_metrics_{prefix}.csv"
    out_event = FEAT_DIR / f"cv_event_metrics_{prefix}.csv"

    df_fold.to_csv(out_fold, index=False, encoding="utf-8-sig")
    df_vin.to_csv(out_vin, index=False, encoding="utf-8-sig")
    df_event.to_csv(out_event, index=False, encoding="utf-8-sig")

    print("\n==== Summary ====")
    print("MAE(core)  mean/std:", float(df_fold["mae_core"].mean()), float(df_fold["mae_core"].std()))
    print("MAE(gated) mean/std:", float(df_fold["mae_gated"].mean()), float(df_fold["mae_gated"].std()))
    print("Saved:", out_fold)
    print("Saved:", out_vin)
    print("Saved:", out_event)


if __name__ == "__main__":
    main()
