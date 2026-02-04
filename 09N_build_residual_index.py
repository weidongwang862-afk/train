# scripts/09N_build_residual_index_v2.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import config

OUT_ROOT = Path(config.OUT_DIR)

# ===== 路径（默认按你当前项目真实位置）=====
SEQ_DIR = Path(os.environ.get("SEQ_DIR", r"E:\RAW_DATA\outputs\09_seq_featcore"))
SEQ_INDEX = Path(os.environ.get("SEQ_INDEX", str(SEQ_DIR / "seq_index.csv")))  # 深度学习真实用的端点集合

HGBR_DIR = Path(os.environ.get("HGBR_DIR", r"E:\RAW_DATA\outputs\04_features"))
PRED_TRAIN = Path(os.environ.get("PRED_TRAIN", str(HGBR_DIR / "predictions_train_HGBR_BASELINE_SoH_trend.csv")))
PRED_VAL   = Path(os.environ.get("PRED_VAL",   str(HGBR_DIR / "predictions_val_HGBR_BASELINE_SoH_trend.csv")))
PRED_TEST  = Path(os.environ.get("PRED_TEST",  str(HGBR_DIR / "predictions_test_HGBR_BASELINE_SoH_trend.csv")))

DATASET = Path(os.environ.get("DATASET", str(HGBR_DIR / "dataset_all_C_frozen.parquet")))
Y_COL = os.environ.get("Y_COL", "SoH_trend").strip()

# 用哪个 HGBR 输出当 base_pred：pred_core 或 pred_gated_any
PRED_COL = os.environ.get("PRED_COL", "pred_gated_any").strip()

OUT_DIR = Path(os.environ.get("OUT_DIR", str(SEQ_DIR / "09_residual")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stats(name: str, r: np.ndarray):
    if r.size == 0:
        print(f"[{name}] n=0 (EMPTY) -> check merge keys.")
        return
    print(
        f"[{name}] n={r.size} | mean={r.mean():.6f} std={r.std():.6f} "
        f"p05={np.quantile(r,0.05):.6f} p50={np.quantile(r,0.50):.6f} p95={np.quantile(r,0.95):.6f}"
    )


def _build_frozen_key_map() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing frozen dataset: {DATASET}")

    df = pd.read_parquet(DATASET)
    need = {"vin", "t_end", Y_COL}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"Frozen dataset must contain {need}, got missing={list(need - set(df.columns))}")

    df = df.copy()
    df["vin"] = df["vin"].astype(str)

    # 选择排序键，保证 t_idx 可复现
    sort_cols = ["vin", "t_end"]
    if "terminaltime" in df.columns:
        sort_cols.append("terminaltime")
    elif "t_start" in df.columns:
        sort_cols.append("t_start")

    df = df.sort_values(sort_cols, kind="mergesort")  # 稳定排序
    df["t_idx"] = df.groupby("vin").cumcount().astype(np.int32)

    # 组装对齐 key：优先 vin+t_end+terminaltime（若预测里也有 terminaltime）
    key_cols = ["vin", "t_end"]
    if "terminaltime" in df.columns:
        key_cols.append("terminaltime")

    keep_cols = key_cols + ["t_idx", Y_COL]
    m = df[keep_cols].copy()

    # 处理 key 重复：对齐时用 first
    m = m.drop_duplicates(subset=key_cols, keep="first")
    return m, key_cols


def _load_seq_endpoints() -> pd.DataFrame:
    if not SEQ_INDEX.exists():
        raise FileNotFoundError(f"Missing seq_index.csv: {SEQ_INDEX}")
    s = pd.read_csv(SEQ_INDEX)
    if not {"vin", "t_idx"}.issubset(set(s.columns)):
        raise RuntimeError(f"seq_index.csv must contain vin,t_idx. got={list(s.columns)}")
    s = s[["vin", "t_idx"]].copy()
    s["vin"] = s["vin"].astype(str)
    s["t_idx"] = s["t_idx"].astype(np.int32)
    s = s.drop_duplicates(subset=["vin", "t_idx"], keep="first")
    return s


def _build_one(split_name: str, pred_fp: Path, frozen_map: pd.DataFrame, key_cols: list[str], seq_ep: pd.DataFrame) -> pd.DataFrame:
    if not pred_fp.exists():
        raise FileNotFoundError(f"Missing HGBR prediction file: {pred_fp}")

    p = pd.read_csv(pred_fp)
    need = {"vin", "t_end", "y_true", PRED_COL}
    if not need.issubset(set(p.columns)):
        raise RuntimeError(f"{pred_fp} missing columns: {list(need - set(p.columns))} | got={list(p.columns)}")

    p = p.copy()
    p["vin"] = p["vin"].astype(str)

    # 预测文件里 t_end / terminaltime 的 dtype 统一
    p["t_end"] = pd.to_numeric(p["t_end"], errors="coerce").astype("Int64")

    if "terminaltime" in key_cols:
        if "terminaltime" not in p.columns:
            raise RuntimeError(f"Frozen map uses key {key_cols}, but prediction lacks terminaltime: {pred_fp}")
        p["terminaltime"] = pd.to_numeric(p["terminaltime"], errors="coerce").astype("Int64")

    # base_pred 有 NaN（gated_any）时先过滤
    base = pd.to_numeric(p[PRED_COL], errors="coerce")
    yt = pd.to_numeric(p["y_true"], errors="coerce")
    ok0 = base.notna() & yt.notna() & p["t_end"].notna()
    p = p.loc[ok0, :].copy()
    p["base_pred"] = base.loc[ok0].astype(np.float32).values
    p["y_true"] = yt.loc[ok0].astype(np.float32).values

    # merge 回 frozen 的 t_idx
    j = p.merge(frozen_map, on=key_cols, how="left", validate="many_to_one")

    n_all = len(j)
    n_miss = int(j["t_idx"].isna().sum())
    print(f"[{split_name}] merge->frozen_map rows={n_all} missing_t_idx={n_miss}")

    j = j[j["t_idx"].notna()].copy()
    j["t_idx"] = j["t_idx"].astype(np.int32)
    j["residual"] = j["y_true"].astype(np.float32) - j["base_pred"].astype(np.float32)
    j["split"] = split_name

    # 过滤到深度学习真实用过的端点集合（确保与 seq_index 一致）
    k = j.merge(seq_ep, on=["vin", "t_idx"], how="inner")
    print(f"[{split_name}] after filter by seq_index endpoints -> rows={len(k)}")

    out_cols = ["vin", "t_idx", "t_end"]
    if "terminaltime" in j.columns:
        out_cols.append("terminaltime")
    out_cols += ["y_true", "base_pred", "residual", "split"]
    return k[out_cols].copy()


def main():
    print(f"[PRED_COL] {PRED_COL}")
    print(f"[DATASET] {DATASET}")
    print(f"[SEQ_INDEX] {SEQ_INDEX}")

    frozen_map, key_cols = _build_frozen_key_map()
    print(f"[frozen_map] rows={len(frozen_map)} key_cols={key_cols} unique_vins={frozen_map['vin'].nunique()}")

    seq_ep = _load_seq_endpoints()
    print(f"[seq_endpoints] rows={len(seq_ep)} unique_vins={seq_ep['vin'].nunique()}")

    tr = _build_one("train", PRED_TRAIN, frozen_map, key_cols, seq_ep)
    va = _build_one("val",   PRED_VAL,   frozen_map, key_cols, seq_ep)
    te = _build_one("test",  PRED_TEST,  frozen_map, key_cols, seq_ep)

    # stats
    _safe_stats("TRAIN", tr["residual"].to_numpy(dtype=np.float32))
    _safe_stats("VAL",   va["residual"].to_numpy(dtype=np.float32))
    _safe_stats("TEST",  te["residual"].to_numpy(dtype=np.float32))

    tr_fp = OUT_DIR / f"residual_index_train_{PRED_COL}.csv"
    va_fp = OUT_DIR / f"residual_index_val_{PRED_COL}.csv"
    te_fp = OUT_DIR / f"residual_index_test_{PRED_COL}.csv"

    tr.to_csv(tr_fp, index=False, encoding="utf-8-sig")
    va.to_csv(va_fp, index=False, encoding="utf-8-sig")
    te.to_csv(te_fp, index=False, encoding="utf-8-sig")

    print("Saved:", tr_fp)
    print("Saved:", va_fp)
    print("Saved:", te_fp)


if __name__ == "__main__":
    main()
