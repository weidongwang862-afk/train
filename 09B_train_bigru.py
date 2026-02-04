# scripts/09B_train_gru_seq.py
from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config

OUT_ROOT = Path(config.OUT_DIR)
SEQ_DIR = OUT_ROOT / "09_seq_featcore"
VIN_DIR = SEQ_DIR / "vin_npz"

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "20")) # [微调] 稍微加两轮给 OneCycleLR 充分收敛
LR = float(os.environ.get("LR", "1e-3"))     # [微调] OneCycleLR 需要较高的最大学习率
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
SEED = int(os.environ.get("SEED", "42"))

HIDDEN = int(os.environ.get("HIDDEN", "128"))
LAYERS = int(os.environ.get("LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.20")) # [微调] 增加 Dropout 防止过拟合

OUT_TAG = os.environ.get("OUT_TAG", "SEQ_BIGRU").strip()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class BiGRURegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        # [核心修改1] 输入归一化。
        # BatchNorm1d 通常接受 (N, C, L)，所以 forward 里需要 permute
        self.bn_in = nn.BatchNorm1d(d_in)
        
        # [核心修改2] 双向 GRU (bidirectional=True)
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True
        )
        
        # 因为是双向，输出维度是 hidden * 2
        self.ln = nn.LayerNorm(hidden * 2)
        
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, L, d_in)
        # BN 需要 (B, d_in, L)
        x_in = x.transpose(1, 2) 
        x_in = self.bn_in(x_in)
        x_in = x_in.transpose(1, 2) # 变回 (B, L, d_in)
        
        out, _ = self.gru(x_in)  # (B, L, H*2)
        
        # 取最后一个时间步
        h = out[:, -1, :]        # (B, H*2)
        h = self.ln(h)
        y = self.head(h).squeeze(-1)
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

    # 模型初始化：BiGRU
    model = BiGRURegressor(d_in=d_in, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # [核心修改3] OneCycleLR
    # max_lr 设为 LR，total_steps 自动计算
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=LR, 
        steps_per_epoch=len(dl_tr), 
        epochs=EPOCHS,
        pct_start=0.3, # 前30% epoch 热身，后面下降
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # 回归 L1 Loss，因为这才是最终指标
    loss_fn = nn.L1Loss(reduction="none")
    ALPHA = float(os.environ.get("ALPHA", "1.0"))

    best_va = 1e9
    best_path = SEQ_DIR / f"model_{OUT_TAG}.pt"

    print(f"Start Training: Bi-GRU + InputBN + OneCycleLR")

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        steps = 0
        
        for x, y, _, _ in dl_tr:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            p = model(x)
            
            # 基础误差
            per_sample_loss = loss_fn(p, y)

            # [核心修改4] 强权重策略 (Simple & Strong)
            # 低于 0.9 的样本，权重直接乘 5.0。
            # 0.9 ~ 0.95 的样本，权重乘 2.0。
            # 高于 0.95 的样本，权重 1.0。
            w = torch.ones_like(y)
            w = torch.where(y < 0.90, 5.0, w)
            w = torch.where((y >= 0.90) & (y < 0.95), 2.0, w)
            
            # 只有在 高估 (p > y) 时才激活 ALPHA 惩罚
            diff = p - y
            mask_over = (diff > 0).float()
            # 对所有样本的高估都进行一定惩罚，低SoH样本惩罚更重（通过w放大）
            
            # Loss = 加权MAE + 额外的高估惩罚
            loss = (w * per_sample_loss).mean() + ALPHA * (w * mask_over * diff).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step() # OneCycleLR 是每个 step 更新
            
            train_loss += loss.item()
            steps += 1

        current_lr = scheduler.get_last_lr()[0]
        va_mae, _, _ = eval_mae(model, dl_va, DEVICE)
        print(f"Epoch {ep:02d} | LR={current_lr:.2e} | Train_Loss={train_loss/steps:.5f} | Val_MAE={va_mae:.6f}")

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