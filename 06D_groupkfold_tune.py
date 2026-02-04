# scripts/06D_groupkfold_tune.py
from __future__ import annotations

import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

import config

OUT_ROOT = Path(config.OUT_DIR)
FEAT_DIR = OUT_ROOT / "04_features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 冻结数据：统一指向全量总表 =====
DEFAULT_FROZEN = r"E:\RAW_DATA\outputs\04_features\dataset_all_step7plus_relax_qc.parquet"
DATASET = Path(os.environ.get("DATASET", DEFAULT_FROZEN)).expanduser()

TARGET = os.environ.get("TARGET", "SoH_trend").strip()
RELAX_MODE = os.environ.get("RELAX_MODE", "core_mask").strip().lower()  # core_mask / any_mask / post_mask / pre_mask
ADD_PROG = os.environ.get("ADD_PROG", "0").strip() == "1"              # 默认不加 event_frac，只可加 t_rel_days/event_idx

OUT_TAG = os.environ.get(
    "OUT_TAG",
    f"FROZEN_B150_{RELAX_MODE}"
).strip()

QUALITY_MAP = {"A_rest>=2h": 2, "B_rest<2h": 1, "C_no_rest": 0}

DROP_ALWAYS = {
    "vin", "t_start", "t_end",
    "C_trend", "SoH_trend",
    "C_est_ah", "SoH_raw",
}

# ===== 超参（环境变量可覆盖）=====
R_TH = float(os.environ.get("R_TH", "0.92"))
TOP_M = int(os.environ.get("TOP_M", "12"))
M_LIST = [int(x) for x in os.environ.get("M_LIST", "3,5,8,12,16").split(",") if x.strip()]
TOPK_PERM = int(os.environ.get("TOPK_PERM", "30"))
PERM_REPEATS = int(os.environ.get("PERM_REPEATS", "6"))
NSPLITS = int(os.environ.get("NSPLITS", "5"))
SEED = int(os.environ.get("SEED", "42"))

BASE_NAN_THR = float(os.environ.get("BASE_NAN_THR", "0.90"))
RELAX_NAN_THR = float(os.environ.get("RELAX_NAN_THR", "0.995"))

# HQ（高QC）阈值
POST_MIN_REST_S = int(os.environ.get("POST_MIN_REST_S", "1800"))
POST_MAX_GAP_S = int(os.environ.get("POST_MAX_GAP_S", "600"))
PRE_MIN_REST_S = int(os.environ.get("PRE_MIN_REST_S", "3600"))
PRE_MAX_GAP_S = int(os.environ.get("PRE_MAX_GAP_S", "3600"))


def assign_group(col: str) -> str:
    c = str(col).lower()

    if c.startswith("post_relax_"):
        return "relax_post"
    if c.startswith("pre_relax_"):
        return "relax_pre"
    if c.startswith("any_relax_"):
        return "relax_any"
    if c in {"any_hit", "any_relax_src", "any_relax_n"}:
        return "relax_any_meta"

    if c in {"gap_s", "rst_duration_s", "post_relax_n"}:
        return "relax_post_meta"
    if c in {"pre_gap_s", "pre_rest_dur_s", "pre_relax_n"}:
        return "relax_pre_meta"

    if c.startswith("ic_"):
        return "ic"
    if "dvdt" in c or ("dv_" in c and "dt" in c):
        return "dvdt"
    if "stage" in c or "cc_frac" in c or "hi_soc_frac" in c or "dv_cc_" in c or "dv_hi_" in c:
        return "stage"
    if "temp" in c or c.startswith("t_") or "mintemperature" in c or "maxtemperature" in c:
        return "temp"
    if "voltage" in c or c.startswith("v_") or "_v_" in c:
        return "voltage"
    if "current" in c or c.startswith("i_") or "_i_" in c:
        return "current"
    if "soc" in c:
        return "soc"
    if "event_" in c or "t_rel_" in c:
        return "progress"
    if "odo" in c:
        return "odo"
    return "other"


def add_progress_features(df: pd.DataFrame) -> pd.DataFrame:
    if "vin" not in df.columns or "t_start" not in df.columns or "t_end" not in df.columns:
        return df
    df = df.sort_values(["vin", "t_start"]).copy()
    t0 = df.groupby("vin")["t_start"].transform("min")
    df["t_rel_days"] = (df["t_end"] - t0) / 86400.0
    df["event_idx"] = df.groupby("vin").cumcount().astype("int32")
    return df


def safe_abs_corr(x: pd.Series, y: np.ndarray) -> float:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
    yv = np.asarray(y, dtype="float64")
    m = np.isfinite(xv) & np.isfinite(yv)
    if m.sum() < 3:
        return 0.0
    xv = xv[m]
    yv = yv[m]
    sx = float(np.std(xv))
    sy = float(np.std(yv))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cx = xv - float(np.mean(xv))
    cy = yv - float(np.mean(yv))
    r = float(np.mean(cx * cy) / (sx * sy))
    if not np.isfinite(r):
        return 0.0
    return abs(r)


def drop_constant_cols(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    nun = Xtr.nunique(dropna=True)
    drop_cols = nun[nun <= 1].index.tolist()
    return Xtr.drop(columns=drop_cols), Xva.drop(columns=drop_cols), drop_cols


def drop_hi_nan_cols_groupaware(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    nan_ratio = Xtr.isna().mean()
    drop_cols = []
    for c, r in nan_ratio.items():
        g = assign_group(c)
        thr = RELAX_NAN_THR if g.startswith("relax") else BASE_NAN_THR
        if float(r) > float(thr):
            drop_cols.append(c)
    return Xtr.drop(columns=drop_cols), Xva.drop(columns=drop_cols), drop_cols


def corr_prune_within_group(Xtr: pd.DataFrame, cols: list[str], ytr: np.ndarray, r_th: float) -> list[str]:
    if len(cols) <= 1:
        return cols
    scored = [(c, safe_abs_corr(Xtr[c], ytr)) for c in cols]
    cols_sorted = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    if len(cols_sorted) <= 1:
        return cols_sorted

    Xc = Xtr[cols_sorted].astype("float64")
    med = Xc.median(numeric_only=True)
    Xc = Xc.fillna(med)

    nun = Xc.nunique(dropna=True)
    cols_sorted = [c for c in cols_sorted if int(nun.get(c, 0)) > 1]
    if len(cols_sorted) <= 1:
        return cols_sorted

    cache = {c: Xc[c].to_numpy(dtype="float64") for c in cols_sorted}
    kept: list[str] = []
    for c in cols_sorted:
        ok = True
        for k in kept:
            r = safe_abs_corr(pd.Series(cache[c]), cache[k])
            if r >= r_th:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def topm_per_group(Xtr: pd.DataFrame, ytr: np.ndarray, cols: list[str], m: int) -> list[str]:
    if m <= 0 or len(cols) == 0:
        return []
    scored = [(c, safe_abs_corr(Xtr[c], ytr)) for c in cols]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:m]]


def fit_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=800,
        max_depth=3,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=SEED,
    )


def evaluate_mae(y_true, y_pred) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def qc_mask(df: pd.DataFrame) -> pd.Series:
    m_post = pd.Series(False, index=df.index)
    m_pre = pd.Series(False, index=df.index)

    if "post_relax_n" in df.columns and "rst_duration_s" in df.columns:
        m_post = df["post_relax_n"].notna() & (pd.to_numeric(df["rst_duration_s"], errors="coerce") >= POST_MIN_REST_S)
        if "gap_s" in df.columns:
            m_post = m_post & (pd.to_numeric(df["gap_s"], errors="coerce") <= POST_MAX_GAP_S)

    if "pre_relax_n" in df.columns and "pre_rest_dur_s" in df.columns and "pre_gap_s" in df.columns:
        m_pre = (df["pre_relax_n"].notna()
                 & (pd.to_numeric(df["pre_rest_dur_s"], errors="coerce") >= PRE_MIN_REST_S)
                 & (pd.to_numeric(df["pre_gap_s"], errors="coerce") <= PRE_MAX_GAP_S))

    return (m_post | m_pre).fillna(False)


def add_relax_mask_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    has_post = df["post_relax_n"].notna() if "post_relax_n" in df.columns else pd.Series(False, index=df.index)
    has_pre = df["pre_relax_n"].notna() if "pre_relax_n" in df.columns else pd.Series(False, index=df.index)

    if "any_hit" in df.columns:
        has_any = pd.to_numeric(df["any_hit"], errors="coerce").fillna(0).astype("int64") > 0
    else:
        has_any = (has_post | has_pre)

    df["has_post_relax"] = has_post.astype("int8")
    df["has_pre_relax"] = has_pre.astype("int8")
    df["has_any_relax"] = has_any.astype("int8")

    has_post_strict = has_post.copy()
    if "rst_duration_s" in df.columns:
        has_post_strict &= (pd.to_numeric(df["rst_duration_s"], errors="coerce") >= POST_MIN_REST_S)
    if "gap_s" in df.columns:
        has_post_strict &= (pd.to_numeric(df["gap_s"], errors="coerce") <= POST_MAX_GAP_S)

    has_pre_strict = has_pre.copy()
    if "pre_rest_dur_s" in df.columns:
        has_pre_strict &= (pd.to_numeric(df["pre_rest_dur_s"], errors="coerce") >= PRE_MIN_REST_S)
    if "pre_gap_s" in df.columns:
        has_pre_strict &= (pd.to_numeric(df["pre_gap_s"], errors="coerce") <= PRE_MAX_GAP_S)

    df["has_any_relax_strict"] = (has_post_strict | has_pre_strict).fillna(False).astype("int8")
    return df


def apply_relax_mode(X: pd.DataFrame) -> pd.DataFrame:
    mode = (RELAX_MODE or "core_mask").lower().strip()
    keep_mask = mode.endswith("_mask")
    base = mode[:-5] if keep_mask else mode

    pre_relax_cols = [c for c in X.columns if str(c).startswith("pre_relax_")]
    post_relax_cols = [c for c in X.columns if str(c).startswith("post_relax_")]
    any_relax_cols = [c for c in X.columns if str(c).startswith("any_relax_")]

    pre_meta = [c for c in ["pre_gap_s", "pre_rest_dur_s", "pre_relax_n"] if c in X.columns]
    post_meta = [c for c in ["gap_s", "rst_duration_s", "post_relax_n"] if c in X.columns]
    any_meta = [c for c in ["any_hit", "any_relax_src", "any_relax_n"] if c in X.columns]

    mask_cols = [c for c in [
        "has_post_relax", "has_pre_relax", "has_any_relax", "has_any_relax_strict"
    ] if c in X.columns]

    def drop_with_mask(drop_cols: list[str]) -> pd.DataFrame:
        if not keep_mask:
            drop_cols = drop_cols + mask_cols
        return X.drop(columns=drop_cols, errors="ignore")

    if base == "core":
        drop = pre_relax_cols + post_relax_cols + any_relax_cols + pre_meta + post_meta + any_meta
        return drop_with_mask(drop)
    if base == "post":
        drop = pre_relax_cols + any_relax_cols + pre_meta + any_meta
        return drop_with_mask(drop)
    if base == "pre":
        drop = post_relax_cols + any_relax_cols + post_meta + any_meta
        return drop_with_mask(drop)
    if base == "any":
        drop = pre_relax_cols + post_relax_cols + pre_meta + post_meta
        return drop_with_mask(drop)
    raise ValueError(f"Unknown RELAX_MODE={RELAX_MODE}")


def build_Xy(df: pd.DataFrame):
    df = df.copy()

    if "quality_flag" in df.columns and "quality_code" not in df.columns:
        df["quality_code"] = df["quality_flag"].map(QUALITY_MAP).fillna(0).astype("int8")

    df = add_relax_mask_features(df)

    y = pd.to_numeric(df[TARGET], errors="coerce").astype("float64").values
    X = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    trend_cols = [c for c in X.columns if str(c).lower().endswith("_trend")]
    X = X.drop(columns=trend_cols, errors="ignore")
    X = X.drop(columns=["C0_trend", "c0_trend"], errors="ignore")

    odo_cols = [c for c in X.columns if "odo" in str(c).lower()]
    X = X.drop(columns=odo_cols, errors="ignore")

    X = apply_relax_mode(X)
    return X, y


def run_cv(df: pd.DataFrame, tag: str):
    df = df.sort_values(["vin", "t_start"]).reset_index(drop=True)
    if ADD_PROG:
        df = add_progress_features(df)

    X_all, y_all = build_Xy(df)
    groups = df["vin"].astype(str).values
    vin_unique = np.unique(groups)
    n_splits = min(NSPLITS, len(vin_unique))
    if n_splits < 2:
        raise ValueError(f"[{tag}] Need at least 2 VINs for GroupKFold.")
    cv = GroupKFold(n_splits=n_splits)

    fold_rows, vin_rows, perm_rows, selected_rows, curve_rows, ablation_rows = [], [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_all, y_all, groups=groups), start=1):
        Xtr = X_all.iloc[tr_idx].copy()
        ytr = y_all[tr_idx]
        Xva = X_all.iloc[va_idx].copy()
        yva = y_all[va_idx]

        Xtr, Xva, dropped_const = drop_constant_cols(Xtr, Xva)
        Xtr, Xva, dropped_nan = drop_hi_nan_cols_groupaware(Xtr, Xva)

        cols_fold = list(Xtr.columns)
        grp2cols: dict[str, list[str]] = {}
        for c in cols_fold:
            grp2cols.setdefault(assign_group(c), []).append(c)

        pruned = {g: corr_prune_within_group(Xtr, cols, ytr, r_th=R_TH) for g, cols in grp2cols.items()}

        for m in M_LIST:
            sel = []
            for g, cols in pruned.items():
                sel.extend(topm_per_group(Xtr, ytr, cols, m=m))
            sel = list(dict.fromkeys(sel))
            if len(sel) < 3:
                continue
            model = fit_model()
            model.fit(Xtr[sel], ytr)
            pred = model.predict(Xva[sel])
            mae = evaluate_mae(yva, pred)
            curve_rows.append({"tag": tag, "fold": fold, "m": int(m), "n_features": int(len(sel)), "mae_val": float(mae)})

        sel_fold = []
        for g, cols in pruned.items():
            sel_fold.extend(topm_per_group(Xtr, ytr, cols, m=TOP_M))
        sel_fold = list(dict.fromkeys(sel_fold))
        if len(sel_fold) < 3:
            continue

        model = fit_model()
        model.fit(Xtr[sel_fold], ytr)
        pred = model.predict(Xva[sel_fold])
        mae_val = evaluate_mae(yva, pred)

        fold_rows.append({
            "tag": tag, "fold": fold,
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "n_features": int(len(sel_fold)),
            "mae_val": float(mae_val),
            "dropped_hi_nan_cols": int(len(dropped_nan)),
            "dropped_const_cols": int(len(dropped_const)),
        })

        df_val = pd.DataFrame({"vin": groups[va_idx], "y": yva, "pred": pred})
        for vin, g in df_val.groupby("vin"):
            vin_rows.append({
                "tag": tag, "fold": fold, "vin": str(vin),
                "n_val_vin": int(len(g)),
                "mae_val_vin": float(evaluate_mae(g["y"].values, g["pred"].values)),
            })

        for c in sel_fold:
            selected_rows.append({"tag": tag, "fold": fold, "feature": c, "group": assign_group(c)})

        r = permutation_importance(
            model, Xva[sel_fold], yva,
            n_repeats=PERM_REPEATS,
            random_state=SEED,
            scoring="neg_mean_absolute_error",
        )
        imp = pd.DataFrame({
            "feature": sel_fold,
            "perm_mean": r.importances_mean,
            "perm_std": r.importances_std,
        }).sort_values("perm_mean", ascending=False)

        for _, row in imp.head(TOPK_PERM).iterrows():
            perm_rows.append({
                "tag": tag, "fold": fold,
                "feature": row["feature"],
                "group": assign_group(row["feature"]),
                "perm_mean": float(row["perm_mean"]),
                "perm_std": float(row["perm_std"]),
            })

        base_mae = mae_val
        groups_in = sorted(set(assign_group(c) for c in sel_fold))
        for gname in groups_in:
            keep = [c for c in sel_fold if assign_group(c) != gname]
            if len(keep) < 3:
                continue
            m2 = fit_model()
            m2.fit(Xtr[keep], ytr)
            p2 = m2.predict(Xva[keep])
            mae2 = evaluate_mae(yva, p2)
            ablation_rows.append({
                "tag": tag, "fold": fold,
                "drop_group": gname,
                "mae_val_drop": float(mae2),
                "delta_mae": float(mae2 - base_mae),
                "n_features_keep": int(len(keep)),
            })

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{OUT_TAG}_{tag}_{ts}"

    df_fold = pd.DataFrame(fold_rows).sort_values(["tag", "fold"])
    df_vinm = pd.DataFrame(vin_rows).sort_values(["tag", "fold", "mae_val_vin"], ascending=[True, True, False])
    df_sel = pd.DataFrame(selected_rows).sort_values(["tag", "fold", "group", "feature"])
    df_perm = pd.DataFrame(perm_rows).sort_values(["tag", "fold", "perm_mean"], ascending=[True, True, False])
    df_curve = pd.DataFrame(curve_rows).sort_values(["tag", "fold", "m"])
    df_ab = pd.DataFrame(ablation_rows).sort_values(["tag", "fold", "delta_mae"], ascending=[True, True, False])

    df_fold.to_csv(FEAT_DIR / f"cv_fold_metrics_{prefix}.csv", index=False, encoding="utf-8-sig")
    df_vinm.to_csv(FEAT_DIR / f"cv_vin_metrics_{prefix}.csv", index=False, encoding="utf-8-sig")
    df_curve.to_csv(FEAT_DIR / f"cv_featurecount_curve_{prefix}.csv", index=False, encoding="utf-8-sig")
    df_ab.to_csv(FEAT_DIR / f"cv_group_ablation_{prefix}.csv", index=False, encoding="utf-8-sig")
    df_sel.to_csv(FEAT_DIR / f"cv_selected_features_by_fold_{prefix}.csv", index=False, encoding="utf-8-sig")
    df_perm.to_csv(FEAT_DIR / f"cv_perm_topk_by_fold_{prefix}.csv", index=False, encoding="utf-8-sig")

    if not df_perm.empty:
        stab = (df_perm.groupby(["feature", "group"], as_index=False)
                      .agg(freq=("fold", "nunique"),
                           perm_mean_avg=("perm_mean", "mean"),
                           perm_std_avg=("perm_std", "mean"))
                      .sort_values(["freq", "perm_mean_avg"], ascending=[False, False]))
        stab.to_csv(FEAT_DIR / f"cv_perm_stability_{prefix}.csv", index=False, encoding="utf-8-sig")

        need = int(np.ceil(0.6 * stab["freq"].max())) if stab["freq"].max() > 0 else 1
        final = stab[stab["freq"] >= need].copy()
        final.to_csv(FEAT_DIR / f"final_feature_set_{prefix}.csv", index=False, encoding="utf-8-sig")

    print("Dataset:", DATASET)
    print("Saved prefix:", prefix)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_parquet(DATASET, engine="pyarrow")
    if "vin" not in df.columns:
        raise ValueError("Dataset must contain column 'vin'")

    tag_all = "all"
    tag_hq = "hq"

    run_cv(df, tag=tag_all)

    m = qc_mask(df)
    n_hq = int(m.sum())
    if n_hq >= 200:
        run_cv(df.loc[m].copy(), tag=tag_hq)
    else:
        print("[HQ] too few rows, skip.")


if __name__ == "__main__":
    main()
