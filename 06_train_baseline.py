# scripts/06_train_baseline.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import datetime

import config

OUT_ROOT = Path(config.OUT_DIR)
FEAT_DIR = OUT_ROOT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = os.environ.get("TARGET", "SoH_trend").strip()
KEYS = ["vin", "t_start", "t_end"]

DEFAULT_FROZEN = r"E:\RAW_DATA\outputs\04_features\dataset_all_step7plus_relax_qc.parquet"
DATASET = Path(os.environ.get("DATASET", DEFAULT_FROZEN)).expanduser()

MODE = os.environ.get("MODE", "holdout").strip().lower()   # holdout / fit_all
SEED = int(os.environ.get("SEED", "42"))
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.2"))

OUT_TAG = os.environ.get("OUT_TAG", "B150_BASELINE").strip()

# gated fusion params
DIFF_CLIP = float(os.environ.get("DIFF_CLIP", "0.008"))
W_BASE = float(os.environ.get("W_BASE", "0.30"))
W_STRICT_BONUS = float(os.environ.get("W_STRICT_BONUS", "0.35"))
W_MAX = float(os.environ.get("W_MAX", "0.70"))

POST_MIN_REST_S = int(os.environ.get("POST_MIN_REST_S", "1800"))
PRE_MIN_REST_S = int(os.environ.get("PRE_MIN_REST_S", "3600"))
PRE_MAX_GAP_S = int(os.environ.get("PRE_MAX_GAP_S", "3600"))


def load_feature_list(fp: Path) -> list[str]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing feature list: {fp}")
    df = pd.read_csv(fp)
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

    pre_strict = has_pre.copy()
    if "pre_rest_dur_s" in df.columns:
        pre_strict &= (pd.to_numeric(df["pre_rest_dur_s"], errors="coerce") >= PRE_MIN_REST_S)
    if "pre_gap_s" in df.columns:
        pre_strict &= (pd.to_numeric(df["pre_gap_s"], errors="coerce") <= PRE_MAX_GAP_S)

    df["has_any_relax_strict"] = (post_strict | pre_strict).fillna(False).astype("int8")
    return df


def build_Xy(df: pd.DataFrame, feature_cols: list[str]):
    y = pd.to_numeric(df[TARGET], errors="coerce").astype("float64").to_numpy()
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y


def fit_predict_hgb(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, wtr: np.ndarray | None):
    Xte = Xte.reindex(columns=Xtr.columns)
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=800,
        max_depth=3,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=SEED,
    )

    if wtr is None:
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr, sample_weight=wtr)

    return model, model.predict(Xte)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_parquet(DATASET, engine="pyarrow")
    if "vin" not in df.columns:
        raise ValueError("Dataset must contain column 'vin'")

    df = build_gate_cols(df)

    core_list_fp = FEAT_DIR / "features_FINAL_core.csv"
    any_list_fp = FEAT_DIR / "features_FINAL_any.csv"
    core_feats = load_feature_list(core_list_fp)
    any_feats = load_feature_list(any_list_fp)

    # 列交集，保证不会因为缺列直接炸
    core_feats = [c for c in core_feats if c in df.columns]
    any_feats = [c for c in any_feats if c in df.columns]
    if len(core_feats) < 5:
        raise ValueError(f"Too few core features: {len(core_feats)}")
    if len(any_feats) < 5:
        raise ValueError(f"Too few any features: {len(any_feats)}")

    w_all = None
    if "qc_weight" in df.columns:
        w_all = pd.to_numeric(df["qc_weight"], errors="coerce").fillna(1.0).to_numpy(dtype="float64")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if MODE == "fit_all":
        X_core, y = build_Xy(df, core_feats)
        model_core, pred_core = fit_predict_hgb(X_core, y, X_core, w_all)

        # gated 在 fit_all 下也可以导出，但没有测试 MAE 的意义
        gate = df["has_any_relax"].to_numpy(dtype=bool)
        strict = df["has_any_relax_strict"].to_numpy(dtype="int8")
        pred_gated = pred_core.copy()

        if gate.sum() >= 50:
            X_any, _ = build_Xy(df.loc[gate], any_feats)
            y_any = y[gate]
            w_any = w_all[gate] if w_all is not None else None

            model_any, pred_any_gate = fit_predict_hgb(X_any, y_any, X_any, w_any)

            w = (W_BASE + W_STRICT_BONUS * strict[gate]).astype("float64")
            w = np.clip(w, 0.0, W_MAX)
            diff = pred_any_gate - pred_core[gate]
            diff = np.clip(diff, -DIFF_CLIP, DIFF_CLIP)
            pred_gated[gate] = pred_core[gate] + w * diff

            joblib.dump(model_any, FEAT_DIR / f"model_any_{OUT_TAG}_{ts}.joblib")

        joblib.dump(model_core, FEAT_DIR / f"model_core_{OUT_TAG}_{ts}.joblib")

        out = df[[c for c in KEYS if c in df.columns]].copy()
        out["y_true"] = y
        out["pred_core"] = pred_core
        out["pred_gated"] = pred_gated
        out["has_any_relax"] = df["has_any_relax"].astype("int8").values
        out["has_any_relax_strict"] = df["has_any_relax_strict"].astype("int8").values

        out_fp = FEAT_DIR / f"predictions_fitall_{OUT_TAG}_{ts}.csv"
        out.to_csv(out_fp, index=False, encoding="utf-8-sig")
        print("MODE=fit_all Saved:", out_fp)
        return

    if MODE != "holdout":
        raise ValueError("MODE must be holdout or fit_all")

    # ===== HOLDOUT: 按 VIN 切分 =====
    rng = np.random.default_rng(SEED)
    vins = np.array(sorted(df["vin"].astype(str).unique().tolist()))
    rng.shuffle(vins)
    n_hold = max(1, int(round(HOLDOUT_FRAC * len(vins))))
    vin_test = set(vins[:n_hold].tolist())

    test = df[df["vin"].astype(str).isin(vin_test)].copy()
    train = df[~df["vin"].astype(str).isin(vin_test)].copy()

    wtr = None
    if w_all is not None:
        wtr = pd.to_numeric(train["qc_weight"], errors="coerce").fillna(1.0).to_numpy(dtype="float64")

    Xtr_core, ytr = build_Xy(train, core_feats)
    Xte_core, yte = build_Xy(test, core_feats)

    model_core, pred_core = fit_predict_hgb(Xtr_core, ytr, Xte_core, wtr)
    mae_core = float(mean_absolute_error(yte, pred_core))
    print(f"[core] TEST MAE = {mae_core:.6f} | n_features={Xtr_core.shape[1]} | n_test={len(test)} | vins_test={len(vin_test)}")

    # ===== gated: 子模型仅在 gate=1 的训练子集上训练 =====
    gate_tr = train["has_any_relax"].to_numpy(dtype=bool)
    gate_te = test["has_any_relax"].to_numpy(dtype=bool)
    strict_te = test["has_any_relax_strict"].to_numpy(dtype="int8")

    pred_gated = pred_core.copy()
    used = int(gate_te.sum())

    if gate_tr.sum() >= 50 and used > 0:
        Xtr_any, _ = build_Xy(train.loc[gate_tr], any_feats)
        ytr_any = ytr[gate_tr]
        Xte_any, _ = build_Xy(test.loc[gate_te], any_feats)

        wtr_any = wtr[gate_tr] if wtr is not None else None
        model_any, pred_any_gate = fit_predict_hgb(Xtr_any, ytr_any, Xte_any, wtr_any)

        w = (W_BASE + W_STRICT_BONUS * strict_te[gate_te]).astype("float64")
        w = np.clip(w, 0.0, W_MAX)

        diff = pred_any_gate - pred_core[gate_te]
        diff = np.clip(diff, -DIFF_CLIP, DIFF_CLIP)

        pred_gated[gate_te] = pred_core[gate_te] + w * diff
        mae_gated = float(mean_absolute_error(yte, pred_gated))
        print(f"[gated] TEST MAE = {mae_gated:.6f} | gate_test={used} | DIFF_CLIP={DIFF_CLIP} W_BASE={W_BASE} BONUS={W_STRICT_BONUS} W_MAX={W_MAX}")

        joblib.dump(model_any, FEAT_DIR / f"model_any_{OUT_TAG}_{ts}.joblib")
    else:
        mae_gated = mae_core
        print(f"[gated] skipped (gate_train={int(gate_tr.sum())}, gate_test={used}) -> pred_gated == pred_core")

    joblib.dump(model_core, FEAT_DIR / f"model_core_{OUT_TAG}_{ts}.joblib")

    out = test[[c for c in KEYS if c in test.columns]].copy()
    out["y_true"] = yte
    out["pred_core"] = pred_core
    out["pred_gated"] = pred_gated
    out["has_any_relax"] = test["has_any_relax"].astype("int8").values
    out["has_any_relax_strict"] = test["has_any_relax_strict"].astype("int8").values

    out_fp = FEAT_DIR / f"predictions_holdout_{OUT_TAG}_{ts}.csv"
    out.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print("Saved:", out_fp)


if __name__ == "__main__":
    main()