# scripts/06B_build_residual_index_from_oof.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

# === inputs ===
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
RES_DIR = SEQ_DIR / "09_residual"

DATASET = Path(os.environ.get("DATASET", str(OUT_ROOT / r"04_features/dataset_all_C_frozen.parquet")))
SEQ_INDEX = Path(os.environ.get("SEQ_INDEX", str(SEQ_DIR / "seq_index.csv")))  # 只有 vin,t_idx

# base 选择：pred_core 或 pred_gated_any
BASE_COL = os.environ.get("BASE_COL", "pred_core").strip()

# OOF / FULL 预测文件（A 脚本生成）
PRED_TRAIN = Path(os.environ.get(
    "PRED_TRAIN",
    str(RES_DIR / f"predictions_train_OOF_{BASE_COL}_HGBR_OOF.csv")
))
PRED_VAL = Path(os.environ.get(
    "PRED_VAL",
    str(RES_DIR / f"predictions_val_FULL_{BASE_COL}_HGBR_OOF.csv")
))
PRED_TEST = Path(os.environ.get(
    "PRED_TEST",
    str(RES_DIR / f"predictions_test_FULL_{BASE_COL}_HGBR_OOF.csv")
))

OUT_TAG = os.environ.get("OUT_TAG", f"OOF_{BASE_COL}").strip()

# === helpers ===
def _need_cols(df: pd.DataFrame, cols: list[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"[{name}] missing columns: {miss} | got={list(df.columns)}")

def _read_endpoints(seq_index_fp: Path) -> pd.DataFrame:
    ep = pd.read_csv(seq_index_fp)
    _need_cols(ep, ["vin", "t_idx"], f"endpoints:{seq_index_fp.name}")
    ep["vin"] = ep["vin"].astype(str)
    ep["t_idx"] = ep["t_idx"].astype(np.int32)
    print(f"[endpoints] {seq_index_fp} rows={len(ep)} unique_vins={ep['vin'].nunique()} "
          f"t_idx min/max={ep['t_idx'].min()}/{ep['t_idx'].max()}")
    return ep[["vin", "t_idx"]].drop_duplicates()

def _build_frozen_map(dataset_fp: Path) -> pd.DataFrame:
    # 只读最少列
    df = pd.read_parquet(dataset_fp, columns=["vin", "t_end"])
    df["vin"] = df["vin"].astype(str)
    # 注意：这里的 t_end 必须是冻结数据那套刻度（你 A 脚本里也用的就是它）
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["t_end"]).copy()
    df["t_end"] = df["t_end"].astype(np.int64)

    # 每个 vin 内按 t_end 排序，生成 t_idx（与 npz 构建时的顺序保持一致的前提是你当时也是按时间顺序写入）
    df = df.sort_values(["vin", "t_end"], kind="mergesort").reset_index(drop=True)
    df["t_idx"] = df.groupby("vin").cumcount().astype(np.int32)

    print(f"[frozen_map] {dataset_fp} rows={len(df)} unique_vins={df['vin'].nunique()} "
          f"t_end min/max={df['t_end'].min()}/{df['t_end'].max()}")
    return df[["vin", "t_end", "t_idx"]]

def _read_pred(pred_fp: Path, base_col: str) -> pd.DataFrame:
    df = pd.read_csv(pred_fp)
    _need_cols(df, ["vin", "t_end", "y_true", base_col], f"pred:{pred_fp.name}")
    df["vin"] = df["vin"].astype(str)

    # 统一 t_end dtype
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["t_end"]).copy()
    df["t_end"] = df["t_end"].astype(np.int64)

    # gated_any 里 base_pred 会有 NaN（没触发 gate），直接丢掉，否则 residual 没意义
    df[base_col] = pd.to_numeric(df[base_col], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df = df.dropna(subset=["y_true", base_col]).copy()

    print(f"[pred] {pred_fp} rows={len(df)} unique_vins={df['vin'].nunique()} "
          f"t_end min/max={df['t_end'].min()}/{df['t_end'].max()}")
    return df[["vin", "t_end", "y_true", base_col]]

def _make_residual(split_name: str, pred_df: pd.DataFrame, frozen_map: pd.DataFrame, endpoints: pd.DataFrame, base_col: str) -> pd.DataFrame:
    # pred + frozen_map -> 拿 t_idx
    m = pred_df.merge(frozen_map, on=["vin", "t_end"], how="left", validate="many_to_one")
    miss = int(m["t_idx"].isna().sum())
    if miss:
        # 只打印数量，避免刷屏
        print(f"[{split_name}] merge pred->frozen_map missing t_idx: {miss}/{len(m)}")
    m = m.dropna(subset=["t_idx"]).copy()
    m["t_idx"] = m["t_idx"].astype(np.int32)

    # 再过滤到 seq_index 的可取窗端点
    before = len(m)
    m = m.merge(endpoints, on=["vin", "t_idx"], how="inner")
    after = len(m)
    print(f"[{split_name}] filter by endpoints: {before} -> {after}")

    m = m.rename(columns={base_col: "base_pred"})
    m["residual"] = (m["y_true"] - m["base_pred"]).astype(np.float32)

    r = m["residual"].to_numpy()
    print(f"[{split_name}] residual stats: n={len(r)} mean={r.mean():.6f} std={r.std():.6f} "
          f"p05={np.quantile(r,0.05):.6f} p50={np.quantile(r,0.50):.6f} p95={np.quantile(r,0.95):.6f}")

    # 训练残差模型够用：vin,t_idx 是关键；t_end 留着便于回溯
    m = m[["vin", "t_idx", "t_end", "y_true", "base_pred", "residual"]].copy()
    return m

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Missing DATASET: {DATASET}")
    if not SEQ_INDEX.exists():
        raise FileNotFoundError(f"Missing SEQ_INDEX: {SEQ_INDEX}")
    for fp in [PRED_TRAIN, PRED_VAL, PRED_TEST]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing prediction file: {fp}")

    print(f"[BASE_COL] {BASE_COL}")
    print(f"[DATASET]  {DATASET}")
    print(f"[SEQ_INDEX] {SEQ_INDEX}")
    print(f"[PRED_TRAIN] {PRED_TRAIN.name}")
    print(f"[PRED_VAL]   {PRED_VAL.name}")
    print(f"[PRED_TEST]  {PRED_TEST.name}")

    endpoints = _read_endpoints(SEQ_INDEX)
    frozen_map = _build_frozen_map(DATASET)

    tr_pred = _read_pred(PRED_TRAIN, BASE_COL)
    va_pred = _read_pred(PRED_VAL, BASE_COL)
    te_pred = _read_pred(PRED_TEST, BASE_COL)

    tr = _make_residual("TRAIN", tr_pred, frozen_map, endpoints, BASE_COL)
    va = _make_residual("VAL", va_pred, frozen_map, endpoints, BASE_COL)
    te = _make_residual("TEST", te_pred, frozen_map, endpoints, BASE_COL)

    out_tr = RES_DIR / f"residual_index_train_{OUT_TAG}.csv"
    out_va = RES_DIR / f"residual_index_val_{OUT_TAG}.csv"
    out_te = RES_DIR / f"residual_index_test_{OUT_TAG}.csv"

    tr.to_csv(out_tr, index=False, encoding="utf-8-sig")
    va.to_csv(out_va, index=False, encoding="utf-8-sig")
    te.to_csv(out_te, index=False, encoding="utf-8-sig")

    print("Saved:", out_tr)
    print("Saved:", out_va)
    print("Saved:", out_te)

if __name__ == "__main__":
    main()
