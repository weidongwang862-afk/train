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
from torch.optim.lr_scheduler import CosineAnnealingLR # [改回] 标准余弦退火

import config

OUT_ROOT = Path(config.OUT_DIR)
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512")) # [增大] 增大Batch Size让梯度更稳
EPOCHS = int(os.environ.get("EPOCHS", "20"))   # [减少] 20轮足够了，要在过拟合前停止
LR = float(os.environ.get("LR", "1e-3"))       
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01")) 
SEED = int(os.environ.get("SEED", "42"))

HIDDEN = int(os.environ.get("HIDDEN", "32"))   # [缩小] 64->32，防止死记硬背
LAYERS = int(os.environ.get("LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.20")) 

OUT_TAG = os.environ.get("OUT_TAG", "SEQ_CONVGRU").strip()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# [保留] 高斯噪声，抗噪神器
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


class ConvBiGRUAttention(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.noise = GaussianNoise(sigma=0.015)
        self.bn_in = nn.BatchNorm1d(d_in)
        
        # [核心新增] 1D 卷积层：特征提取器
        # kernel_size=3: 每次看3个时间步，平滑数据
        self.conv = nn.Conv1d(in_channels=d_in, out_channels=hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        
        # GRU 输入维度现在是 hidden (因为经过了 conv)
        self.gru = nn.GRU(
            input_size=hidden, 
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
        # x: (B, L, d_in)
        x = self.noise(x)
        
        # BN & Conv1d
        # 需要 (B, d_in, L)
        x = x.transpose(1, 2)
        x = self.bn_in(x)
        x = self.conv(x)  # -> (B, hidden, L)
        x = self.act(x)
        x = x.transpose(1, 2) # -> (B, L, hidden)
        
        # GRU
        out, _ = self.gru(x) # (B, L, H*2)
        
        # Attention
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

    # 模型: Conv + BiGRU + Attn (更小，更强，更平滑)
    model = ConvBiGRUAttention(d_in=d_in, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # [改回] 标准退火，稳步下降
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    # [改回] L1 Loss。我们现在需要更强的梯度来推这最后一点误差。
    # 既然已经有 Dropout、Noise 和 Conv 层做正则化，就不怕 L1 Loss 过拟合了。
    loss_fn = nn.L1Loss()
    
    best_va = 1e9
    best_path = SEQ_DIR / f"model_{OUT_TAG}.pt"

    print(f"Start Training: Conv1d + BiGRU | Hidden={HIDDEN} | Batch={BATCH_SIZE}")

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
        
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        va_mae, _, _ = eval_mae(model, dl_va, DEVICE)
        print(f"Epoch {ep:02d} | LR={current_lr:.2e} | Train_Loss={train_loss/steps:.5f} | Val_MAE={va_mae:.6f}")

        # [关键] 只要 Val 变好就存，不要等到最后
        if va_mae < best_va:
            best_va = va_mae
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