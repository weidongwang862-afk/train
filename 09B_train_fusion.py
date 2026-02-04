# scripts/09B_train_fusion_true.py
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import config

# =========================
# Paths / Env
# =========================
OUT_ROOT = Path(config.OUT_DIR)

# 你现在的时序数据目录（你已生成 09_seq_featcore/vin_npz, stats.json, splits.json 等）
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

# 你已经生成的 seq_index_with_tend.csv（优先用它；没有就退回 seq_index.csv）
SEQ_INDEX_WITH_TEND = SEQ_DIR / "seq_index_with_tend.csv"
SEQ_INDEX_FALLBACK  = SEQ_DIR / "seq_index.csv"

# 冻结数据（用于读取 tabular 特征值），优先从环境变量指定
FROZEN_PARQUET = Path(os.environ.get("FROZEN_PARQUET", str(OUT_ROOT / "dataset_all_C_frozen.parquet")))

# 你的“特征名列表”文件（features_FINAL_core.csv 通常是一列特征名，不带 vin）
# 允许你用 FEAT_LIST 覆盖
FEAT_LIST = Path(os.environ.get("FEAT_LIST", str(OUT_ROOT / "04_features" / "features_FINAL_core.csv")))

# Tabular 对齐缓存：第一次会构建并保存，后续直接 load，避免每次都对齐
TAB_CACHE = Path(os.environ.get("TAB_CACHE", str(SEQ_DIR / "tabular_aligned_corefeat_by_tidx_masked.npz")))

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LR = float(os.environ.get("LR", "5e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.02"))
SEED = int(os.environ.get("SEED", "42"))

# Sequence encoder
SEQ_HIDDEN = int(os.environ.get("SEQ_HIDDEN", "64"))
SEQ_LAYERS = int(os.environ.get("SEQ_LAYERS", "2"))
SEQ_DROPOUT = float(os.environ.get("SEQ_DROPOUT", "0.50"))

# Tab encoder
TAB_HIDDEN = int(os.environ.get("TAB_HIDDEN", "128"))
TAB_DROPOUT = float(os.environ.get("TAB_DROPOUT", "0.20"))

# Fusion head
FUSION_HIDDEN = int(os.environ.get("FUSION_HIDDEN", "128"))
FUSION_DROPOUT = float(os.environ.get("FUSION_DROPOUT", "0.20"))

# 非对称惩罚（可关：ALPHA=0）
ALPHA = float(os.environ.get("ALPHA", "0.8"))
Y_THR = float(os.environ.get("Y_THR", "0.90"))  # 仅在 y<Y_THR 时对“高估”加罚

OUT_TAG = os.environ.get("OUT_TAG", "SEQ_FUSION").strip()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_feat_list(fp: Path) -> list[str]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing FEAT_LIST: {fp}")
    df = pd.read_csv(fp)
    # 兼容：一列特征名 / 多列（取第一列）
    col0 = df.columns[0]
    feats = df[col0].astype(str).tolist()
    feats = [c.strip() for c in feats if str(c).strip()]
    if len(feats) == 0:
        raise RuntimeError(f"Empty feature list: {fp}")
    return feats


def _load_seq_index() -> pd.DataFrame:
    if SEQ_INDEX_WITH_TEND.exists():
        df = pd.read_csv(SEQ_INDEX_WITH_TEND)
    else:
        df = pd.read_csv(SEQ_INDEX_FALLBACK)

    if "vin" not in df.columns or "t_idx" not in df.columns:
        raise RuntimeError("seq_index 缺少 vin 或 t_idx，无法训练。")

    df["vin"] = df["vin"].astype(str)
    df["t_idx"] = df["t_idx"].astype(int)

    # row_id：后面用于 tabular 的 train-only 标准化与快速切片
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    # t_end：如果你用了 seq_index_with_tend.csv 就已经有；否则允许无 t_end，但预测文件会缺 t_end
    if "t_end" in df.columns:
        df["t_end"] = df["t_end"].astype(np.int64)

    return df


def _build_or_load_tabular(index_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    输出：
      X_tab_all: (N, 2*d_val) float32，其中前 d_val 是特征值，后 d_val 是缺失mask(0/1)
      feat_cols: d_val 个特征名
    """
    if TAB_CACHE.exists():
        obj = np.load(TAB_CACHE, allow_pickle=True)
        X_tab = obj["X_tab"].astype(np.float32)
        feat_cols = obj["feat_cols"].tolist()
        return X_tab, feat_cols

    feat_cols = _read_feat_list(FEAT_LIST)

    if not FROZEN_PARQUET.exists():
        raise FileNotFoundError(f"Missing frozen parquet: {FROZEN_PARQUET}")

    frozen = pd.read_parquet(FROZEN_PARQUET)

    need_keys = ["vin", "t_idx"]
    for k in need_keys:
        if k not in frozen.columns:
            raise RuntimeError(
                f"冻结数据缺少列 {k}，无法用 (vin,t_idx) 对齐。"
                f"请确认 frozen parquet 是否包含 vin,t_idx；如果没有，必须先生成可对齐的键。"
            )

    frozen["vin"] = frozen["vin"].astype(str)
    frozen["t_idx"] = frozen["t_idx"].astype(int)

    missing_cols = [c for c in feat_cols if c not in frozen.columns]
    if len(missing_cols) > 0:
        raise RuntimeError(f"冻结数据缺少特征列（前 20 个）：{missing_cols[:20]}")

    # 用 MultiIndex 做 join
    frozen_small = frozen[["vin", "t_idx"] + feat_cols].copy()
    frozen_small = frozen_small.set_index(["vin", "t_idx"]).sort_index()

    keys = index_df[["vin", "t_idx"]].copy()
    keys = keys.set_index(["vin", "t_idx"])

    aligned = frozen_small.reindex(keys.index)  # 保持与 seq_index 行顺序一致

    Xv = aligned[feat_cols].to_numpy(dtype=np.float32)
    M = np.isnan(Xv).astype(np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0).astype(np.float32)

    X_tab = np.concatenate([Xv, M], axis=1).astype(np.float32)

    TAB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TAB_CACHE, X_tab=X_tab, feat_cols=np.array(feat_cols, dtype=object))
    return X_tab, feat_cols


def _train_only_normalize_tab(X_tab_all: np.ndarray, tr_row_ids: np.ndarray) -> np.ndarray:
    """
    只标准化前半段“特征值”，后半段 mask 保持 0/1。
    """
    d_tab = int(X_tab_all.shape[1])
    d_val = d_tab // 2

    Xv = X_tab_all[:, :d_val].astype(np.float32)
    Xm = X_tab_all[:, d_val:].astype(np.float32)

    mu = Xv[tr_row_ids].mean(axis=0)
    sd = Xv[tr_row_ids].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    mu = mu.astype(np.float32)

    Xv = (Xv - mu) / sd
    return np.concatenate([Xv.astype(np.float32), Xm], axis=1).astype(np.float32)


class FusionDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, seq_len: int, X_tab_all: np.ndarray):
        self.index = index_df.reset_index(drop=True)
        self.seq_len = int(seq_len)
        self.X_tab_all = X_tab_all
        self.cache: dict[str, np.lib.npyio.NpzFile] = {}

    def _load_vin(self, vin: str):
        if vin in self.cache:
            return self.cache[vin]
        fp = VIN_DIR / f"{vin}.npz"
        if not fp.exists():
            raise FileNotFoundError(f"Missing vin npz: {fp}")
        obj = np.load(fp)
        self.cache[vin] = obj
        return obj

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        vin = str(self.index.at[i, "vin"])
        t_idx = int(self.index.at[i, "t_idx"])
        row_id = int(self.index.at[i, "row_id"])

        obj = self._load_vin(vin)
        X = obj["X"]      # (T, d_seq) float32
        y = obj["y"]      # (T,) float32
        t_end = obj["t_end"]  # (T,) int64

        s = t_idx - self.seq_len + 1
        e = t_idx + 1
        x_seq = X[s:e]                      # (L, d_seq)
        y_t = y[t_idx].astype(np.float32)   # scalar
        te = int(t_end[t_idx])

        x_tab = self.X_tab_all[row_id]      # (d_tab,)

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(x_tab),
            torch.tensor(y_t, dtype=torch.float32),
            vin,
            torch.tensor(te, dtype=torch.long),
        )


class SeqEncoderGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):  # (B, L, d)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.ln(h)  # (B, hidden)


class TabEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, d_tab)
        return self.net(x)  # (B, hidden)


class FusionModel(nn.Module):
    def __init__(self, d_seq: int, d_tab: int):
        super().__init__()
        self.seq = SeqEncoderGRU(d_in=d_seq, hidden=SEQ_HIDDEN, layers=SEQ_LAYERS, dropout=SEQ_DROPOUT)
        self.tab = TabEncoder(d_in=d_tab, hidden=TAB_HIDDEN, dropout=TAB_DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(SEQ_HIDDEN + TAB_HIDDEN, FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_HIDDEN, 1),
        )

    def forward(self, x_seq, x_tab):
        hs = self.seq(x_seq)
        ht = self.tab(x_tab)
        h = torch.cat([hs, ht], dim=1)
        y = self.head(h).squeeze(-1)
        return y


@torch.no_grad()
def eval_mae(model, loader, device):
    model.eval()
    ys, ps, vins, tend = [], [], [], []
    for x_seq, x_tab, y, vin, te in loader:
        x_seq = x_seq.to(device)
        x_tab = x_tab.to(device)
        y = y.to(device)
        p = model(x_seq, x_tab)

        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
        vins.extend(list(vin))
        tend.extend(list(te.detach().cpu().numpy().tolist()))

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    mae = float(np.mean(np.abs(y_all - p_all)))

    pred_df = pd.DataFrame({"vin": vins, "t_end": tend, "y_true": y_all, "y_pred": p_all})
    vin_mae = (
        pred_df.assign(abs_err=(pred_df["y_true"] - pred_df["y_pred"]).abs())
        .groupby("vin", as_index=False)["abs_err"].mean()
        .rename(columns={"abs_err": "mae"})
    )
    vin_mae.columns = ["vin", "mae"]
    return mae, pred_df, vin_mae


def main():
    set_seed(SEED)

    stats = json.loads((SEQ_DIR / "stats.json").read_text(encoding="utf-8"))
    splits = json.loads((SEQ_DIR / "splits.json").read_text(encoding="utf-8"))

    seq_len = int(stats["seq_len"])
    d_seq = len(list(stats["features"]))

    index_df = _load_seq_index()

    # 先构建/加载 tabular aligned（值+缺失mask）
    X_tab_all, feat_cols = _build_or_load_tabular(index_df)

    # split
    tr = index_df[index_df["vin"].isin(splits["train"])].copy()
    va = index_df[index_df["vin"].isin(splits["val"])].copy()
    te = index_df[index_df["vin"].isin(splits["test"])].copy()

    # train-only 标准化 tabular 的“值半边”
    tr_row_ids = tr["row_id"].to_numpy(dtype=np.int64)
    X_tab_all = _train_only_normalize_tab(X_tab_all, tr_row_ids)

    d_tab = int(X_tab_all.shape[1])

    # dataset / loader
    ds_tr = FusionDataset(tr, seq_len, X_tab_all)
    ds_va = FusionDataset(va, seq_len, X_tab_all)
    ds_te = FusionDataset(te, seq_len, X_tab_all)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FusionModel(d_seq=d_seq, d_tab=d_tab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)

    loss_l1 = nn.L1Loss(reduction="none")

    best_va = 1e9
    best_path = SEQ_DIR / f"model_{OUT_TAG}.pt"

    print(
        f"Start FUSION TRUE | seq_len={seq_len} d_seq={d_seq} | "
        f"tab_feats={len(feat_cols)} tab_dim={d_tab}(val+mask) | "
        f"LR={LR} WD={WEIGHT_DECAY} EPOCHS={EPOCHS} | "
        f"ALPHA={ALPHA} Y_THR={Y_THR}"
    )

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for x_seq, x_tab, y, _, _ in dl_tr:
            x_seq = x_seq.to(DEVICE)
            x_tab = x_tab.to(DEVICE)
            y = y.to(DEVICE)

            p = model(x_seq, x_tab)

            per = loss_l1(p, y)                  # |p-y|
            over = torch.relu(p - y)             # 只保留高估部分
            mask_low = (y < Y_THR).float()       # 只在低 SoH 加罚

            loss = per.mean() + ALPHA * (mask_low * over).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]

        va_mae, _, _ = eval_mae(model, dl_va, DEVICE)
        print(f"Epoch {ep:02d} | lr={cur_lr:.2e} | train_loss={total_loss/max(steps,1):.6f} | val_MAE={va_mae:.6f}")

        if va_mae < best_va:
            best_va = va_mae
            torch.save({"model": model.state_dict(), "stats": stats, "feat_cols": feat_cols}, best_path)

    # load best and export
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    te_mae, pred_te, vin_te = eval_mae(model, dl_te, DEVICE)
    va_mae, pred_va, vin_va = eval_mae(model, dl_va, DEVICE)

    print(f"[TEST] MAE={te_mae:.6f} | n_test_samples={len(ds_te)} | n_test_vins={len(splits['test'])}")
    print(f"[VAL]  MAE={va_mae:.6f} | n_val_samples={len(ds_va)} | n_val_vins={len(splits['val'])}")

    out_pred_te = SEQ_DIR / f"predictions_test_{OUT_TAG}.csv"
    out_vin_te  = SEQ_DIR / f"vin_mae_test_{OUT_TAG}.csv"
    out_pred_va = SEQ_DIR / f"predictions_val_{OUT_TAG}.csv"
    out_vin_va  = SEQ_DIR / f"vin_mae_val_{OUT_TAG}.csv"

    pred_te.to_csv(out_pred_te, index=False, encoding="utf-8-sig")
    vin_te.to_csv(out_vin_te, index=False, encoding="utf-8-sig")
    pred_va.to_csv(out_pred_va, index=False, encoding="utf-8-sig")
    vin_va.to_csv(out_vin_va, index=False, encoding="utf-8-sig")

    print("Saved:", out_pred_va)
    print("Saved:", out_vin_va)
    print("Saved:", out_pred_te)
    print("Saved:", out_vin_te)
    print("Saved model:", best_path)


if __name__ == "__main__":
    main()
