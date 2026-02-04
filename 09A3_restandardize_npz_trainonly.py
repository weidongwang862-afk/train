from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEQ_DIR = Path(r"E:\RAW_DATA\outputs\09_seq_featcore")
VIN_DIR = SEQ_DIR / "vin_npz"

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    stats = load_json(SEQ_DIR / "stats.json")
    splits = load_json(SEQ_DIR / "splits.json")

    mu_old = np.array(stats["norm_mu"], dtype=np.float64)
    sd_old = np.array(stats["norm_sd"], dtype=np.float64)

    train_vins = set(map(str, splits["train"]))

    # 1) 统计 train-only 的新 mu/sd（基于 X_raw）
    n = 0
    s1 = np.zeros_like(mu_old, dtype=np.float64)
    s2 = np.zeros_like(mu_old, dtype=np.float64)

    for fp in VIN_DIR.glob("*.npz"):
        vin = fp.stem
        if vin not in train_vins:
            continue
        obj = np.load(fp)
        X_old = obj["X"].astype(np.float64)            # (T,d)
        X_raw = X_old * sd_old + mu_old               # 反标准化
        n += X_raw.shape[0]
        s1 += X_raw.sum(axis=0)
        s2 += (X_raw ** 2).sum(axis=0)

    if n == 0:
        raise RuntimeError("No train vins found in vin_npz. Check splits.json and vin names.")

    mu_new = s1 / n
    var_new = s2 / n - mu_new ** 2
    var_new = np.maximum(var_new, 1e-12)
    sd_new = np.sqrt(var_new)

    # 2) 用新 mu/sd 重写所有 vin 的 X
    for fp in VIN_DIR.glob("*.npz"):
        obj = np.load(fp)
        X_old = obj["X"].astype(np.float64)
        y = obj["y"]
        t_end = obj["t_end"]

        X_raw = X_old * sd_old + mu_old
        X_new = ((X_raw - mu_new) / sd_new).astype(np.float32)

        np.savez_compressed(fp, X=X_new, y=y.astype(np.float32), t_end=t_end)

    # 3) 更新 stats.json
    stats["norm_mu"] = mu_new.tolist()
    stats["norm_sd"] = sd_new.tolist()
    save_json(SEQ_DIR / "stats.json", stats)

    print("Done. Updated npz standardized by train-only stats.")
    print("Train-only n_rows =", n)

if __name__ == "__main__":
    main()
