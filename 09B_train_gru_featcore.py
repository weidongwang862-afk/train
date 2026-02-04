# scripts/09B_train_gru_seq.py
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # 调度器

import config

OUT_ROOT = Path(config.OUT_DIR)
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256")) 
EPOCHS = int(os.environ.get("EPOCHS", "20"))   
LR = float(os.environ.get("LR", "1e-3"))       
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01")) 
SEED = int(os.environ.get("SEED", "42"))

HIDDEN = int(os.environ.get("HIDDEN", "64"))   
LAYERS = int(os.environ.get("LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.40")) 

OUT_TAG = os.environ.get("OUT_TAG", "SEQ_GRU").strip()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class SeqIndexDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, seq_len: int):
        self.index = index_df.reset_index(drop=True)
        self.seq_len = int(seq_len)
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
        obj = self._load_vin(vin)

        X = obj["X"]      # (T, d)
        y = obj["y"]      # (T,)
        t_end = obj["t_end"]

        s = t_idx - self.seq_len + 1
        e = t_idx + 1
        x_win = X[s:e]
        y_t = y[t_idx].astype(np.float32)
        te = int(t_end[t_idx])

        return torch.from_numpy(x_win), torch.tensor(y_t), vin, te


class BiGRUWithAttention(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.noise = GaussianNoise(sigma=0.02)
        self.bn_in = nn.BatchNorm1d(d_in)
        
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.att_score = nn.Linear(hidden * 2, 1)
        
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.noise(x)
        x_in = x.transpose(1, 2) 
        x_in = self.bn_in(x_in)
        x_in = x_in.transpose(1, 2)
        
        out, _ = self.gru(x_in)
        
        scores = self.att_score(out)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(out * alpha, dim=1)
        
        y = self.head(context).squeeze(-1)
        return y


@torch.no_grad()
def eval_mae(model, loader, device):
    model.eval()
    ys, ps, vins, tend = [], [], [], []
    for x, y, vin, te in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
        vins.extend(list(vin))
        tend.extend(list(te.numpy().tolist()))
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    mae = float(np.mean(np.abs(y_all - p_all)))

    pred_df = pd.DataFrame({
        "vin": vins,
        "t_end": tend,
        "y_true": y_all,
        "y_pred": p_all,
    })

    vin_mae = (pred_df.assign(abs_err=(pred_df["y_true"] - pred_df["y_pred"]).abs())
               .groupby("vin", as_index=False)["abs_err"].mean()
               .rename(columns={"abs_err": "mae"}))
    vin_mae.columns = ["vin", "mae"]
    return mae, pred_df, vin_mae


def main():
    set_seed(SEED)

    stats = json.loads((SEQ_DIR / "stats.json").read_text(encoding="utf-8"))
    splits = json.loads((SEQ_DIR / "splits.json").read_text(encoding="utf-8"))

    seq_len = int(stats["seq_len"])
    d_in = len(list(stats["features"]))

    index_df = pd.read_csv(SEQ_DIR / "seq_index.csv")
    index_df["vin"] = index_df["vin"].astype(str)

    tr = index_df[index_df["vin"].isin(splits["train"])].copy()
    va = index_df[index_df["vin"].isin(splits["val"])].copy()
    te = index_df[index_df["vin"].isin(splits["test"])].copy()

    ds_tr = SeqIndexDataset(tr, seq_len)
    ds_va = SeqIndexDataset(va, seq_len)
    ds_te = SeqIndexDataset(te, seq_len)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiGRUWithAttention(d_in=d_in, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # [修复] 移除了 verbose=True，其余参数保持不变
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    loss_fn = nn.L1Loss()
    
    best_va = 1e9
    best_path = SEQ_DIR / f"model_{OUT_TAG}.pt"

    print(f"Start Training: V7 (Based on V5) | Plateau Scheduler | Hidden={HIDDEN}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        steps = 0
        
        for x, y, _, _ in dl_tr:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            p = model(x)
            loss = loss_fn(p, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            train_loss += loss.item()
            steps += 1
        
        va_mae, _, _ = eval_mae(model, dl_va, DEVICE)
        
        # 更新调度器
        scheduler.step(va_mae)
        
        # 手动打印 LR
        current_lr = opt.param_groups[0]['lr']

        print(f"Epoch {ep:02d} | LR={current_lr:.2e} | Train_Loss={train_loss/steps:.5f} | Val_MAE={va_mae:.6f}")

        if va_mae < best_va:
            best_va = va_mae
            print(f"  >>> New Best Val MAE: {best_va:.6f} (Saved)")
            torch.save({"model": model.state_dict(), "stats": stats}, best_path)

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    te_mae, pred_df, vin_mae = eval_mae(model, dl_te, DEVICE)
    print(f"[TEST] MAE={te_mae:.6f} | n_test_samples={len(ds_te)} | n_test_vins={len(splits['test'])}")

    out_pred = SEQ_DIR / f"predictions_test_{OUT_TAG}.csv"
    out_vin = SEQ_DIR / f"vin_mae_test_{OUT_TAG}.csv"
    pred_df.to_csv(out_pred, index=False, encoding="utf-8-sig")
    vin_mae.to_csv(out_vin, index=False, encoding="utf-8-sig")

    print(f"Saved model to: {best_path}")

if __name__ == "__main__":
    main()