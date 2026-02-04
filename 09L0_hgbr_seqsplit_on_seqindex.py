# scripts/09L0_hgbr_seqsplit_on_seqindex_FIXED.py
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

import config

OUT_ROOT = Path(config.OUT_DIR)
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

DEVICE = os.environ.get("DEVICE", "cpu")

SEED = int(os.environ.get("SEED", "42"))

# HGBR params（先固定一套稳定的，后续再调）
MAX_ITER = int(os.environ.get("MAX_ITER", "500"))
LR = float(os.environ.get("LR", "0.05"))
MAX_LEAF = int(os.environ.get("MAX_LEAF", "64"))
MIN_SAMPLES_LEAF = int(os.environ.get("MIN_SAMPLES_LEAF", "30"))

# 低 SoH 权重（用于“放大学习信号”，可先开）
LOW_THR = float(os.environ.get("LOW_THR", "0.90"))
LOW_W = float(os.environ.get("LOW_W", "3.0"))  # y<LOW_THR 的样本权重
HIGH_W = float(os.environ.get("HIGH_W", "1.0"))

OUT_TAG = os.environ.get("OUT_TAG", "HGBR_SEQSPLIT_XTABMASKED").strip()


def _pick_index_csv() -> Path:
    p1 = SEQ_DIR / "seq_index_with_tend.csv"
    if p1.exists():
        return p1
    p2 = SEQ_DIR / "seq_index.csv"
    if p2.exists():
        return p2
    raise FileNotFoundError("No seq_index_with_tend.csv or seq_index.csv found in SEQ_DIR")


def _find_tabular_masked_npz() -> Path:
    # 你之前生成的“值+mask=60维”的文件名不确定，所以用 glob 自动找
    cands = sorted(SEQ_DIR.glob("*masked*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(
            "No '*masked*.npz' found in SEQ_DIR. Put your masked tabular aligned npz into 09_seq_featcore."
        )
    return cands[0]


def _load_X_tab_from_npz(npz_path: Path) -> np.ndarray:
    obj = np.load(npz_path, allow_pickle=True)
    # 兼容不同 key 命名
    for k in ["X_tab", "X", "x", "X_all"]:
        if k in obj.files:
            X = obj[k]
            X = np.asarray(X, dtype=np.float32)
            return X
    raise KeyError(f"Cannot find X array in {npz_path.name}. Keys={obj.files}")


def _build_y_from_vin_npz(index_df: pd.DataFrame) -> np.ndarray:
    # y 来自 vin_npz 的 y[t_idx]，保证和 seq_index 同源对齐
    vins = index_df["vin"].astype(str).values
    t_idx = index_df["t_idx"].astype(int).values

    y_out = np.empty(len(index_df), dtype=np.float32)

    cache = {}  # vin -> npz
    for i in range(len(index_df)):
        v = vins[i]
        if v not in cache:
            fp = VIN_DIR / f"{v}.npz"
            if not fp.exists():
                raise FileNotFoundError(f"Missing vin npz: {fp}")
            cache[v] = np.load(fp, allow_pickle=False)
        y_arr = cache[v]["y"]  # (T,)
        ti = t_idx[i]
        y_out[i] = np.float32(y_arr[ti])
    return y_out


def _split_masks_by_vin(index_df: pd.DataFrame, splits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vin_ser = index_df["vin"].astype(str)
    tr = vin_ser.isin([str(x) for x in splits["train"]]).values
    va = vin_ser.isin([str(x) for x in splits["val"]]).values
    te = vin_ser.isin([str(x) for x in splits["test"]]).values
    return tr, va, te


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main():
    np.random.seed(SEED)

    index_csv = _pick_index_csv()
    splits = json.loads((SEQ_DIR / "splits.json").read_text(encoding="utf-8"))

    index_df = pd.read_csv(index_csv)
    index_df["vin"] = index_df["vin"].astype(str)

    if "t_idx" not in index_df.columns:
        raise RuntimeError("seq_index missing 't_idx' column")

    # 1) X_tab：读你已经生成的 masked tabular aligned npz（60维）
    tab_npz = _find_tabular_masked_npz()
    X_all = _load_X_tab_from_npz(tab_npz)

    if len(X_all) != len(index_df):
        raise RuntimeError(f"X_tab rows != seq_index rows: {len(X_all)} vs {len(index_df)}")

    # 2) y：直接从 vin_npz 的 y[t_idx] 拿
    y_all = _build_y_from_vin_npz(index_df)

    # 3) split：按 splits.json 的 VIN 划分
    m_tr, m_va, m_te = _split_masks_by_vin(index_df, splits)

    Xtr, ytr = X_all[m_tr], y_all[m_tr]
    Xva, yva = X_all[m_va], y_all[m_va]
    Xte, yte = X_all[m_te], y_all[m_te]

    # 4) sample_weight：放大低 SoH
    wtr = np.where(ytr < LOW_THR, LOW_W, HIGH_W).astype(np.float32)

    print(f"[index] {index_csv.name} rows={len(index_df)} | X_tab={tab_npz.name} dim={X_all.shape[1]}")
    print(f"[split] train={len(ytr)} val={len(yva)} test={len(yte)} | low_thr={LOW_THR} low_w={LOW_W}")

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=LR,
        max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )

    model.fit(Xtr, ytr, sample_weight=wtr)

    p_va = model.predict(Xva).astype(np.float32)
    p_te = model.predict(Xte).astype(np.float32)

    mae_va = _mae(yva, p_va)
    mae_te = _mae(yte, p_te)

    print(f"[VAL]  MAE={mae_va:.6f} | n={len(yva)} | vins={len(set(index_df.loc[m_va,'vin']))}")
    print(f"[TEST] MAE={mae_te:.6f} | n={len(yte)} | vins={len(set(index_df.loc[m_te,'vin']))}")

    # 保存预测，便于你继续用 09F_error_by_bins.py 复用分析逻辑
    out_pred_va = SEQ_DIR / f"predictions_val_{OUT_TAG}.csv"
    out_pred_te = SEQ_DIR / f"predictions_test_{OUT_TAG}.csv"

    df_va = index_df.loc[m_va, ["vin"]].copy()
    df_va["y_true"] = yva
    df_va["y_pred"] = p_va
    df_va.to_csv(out_pred_va, index=False, encoding="utf-8-sig")

    df_te = index_df.loc[m_te, ["vin"]].copy()
    df_te["y_true"] = yte
    df_te["y_pred"] = p_te
    df_te.to_csv(out_pred_te, index=False, encoding="utf-8-sig")

    print("Saved:", out_pred_va)
    print("Saved:", out_pred_te)


if __name__ == "__main__":
    main()
