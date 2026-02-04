# scripts/03_segment_core1.py
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import OUT_DIR, ALL_FILES

# 阈值（先用初版，后面根据你数据再微调）
SPEED_STOP = 1.0
I_REST = 2.0
I_CHARGE = -5.0
I_DRIVE = 5.0
GAP_SEC = 600  # dt>600s 断段（10s采样下相当于>60帧缺失）

CORE_PATH = OUT_DIR / "01_clean_core"
SEG_OUT = OUT_DIR / "02_segments"
SEG_OUT.mkdir(parents=True, exist_ok=True)

CORE_USE = [
    "terminaltime", "soc", "speed", "totalodometer",
    "totalvoltage", "totalcurrent",
    "minvoltagebattery", "maxvoltagebattery",
    "mintemperaturevalue", "maxtemperaturevalue",
]

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False


# -------- 统一schema（关键修复点）--------
# 目的：保证每次写入 Parquet 的 dtype 一致，避免 pyarrow 报 schema mismatch
SUMMARY_COLS = [
    "seg_id", "state",
    "t_start", "t_end", "n",
    "soc_start", "soc_end",
    "odo_start", "odo_end",
    "I_sum", "I_min", "I_max",
    "V_sum", "V_min", "V_max",
    "T_min", "T_max",
    "duration_s", "delta_soc", "delta_odo",
    "I_mean", "V_mean",
]
SUMMARY_DTYPES = {
    "seg_id": "int64",
    "state": "string",

    "t_start": "int64",
    "t_end": "int64",
    "n": "int64",

    "soc_start": "float32",
    "soc_end": "float32",
    "odo_start": "float32",
    "odo_end": "float32",

    # sum / mean 固定 float64（防止 float32/float64 混写）
    "I_sum": "float64",
    "V_sum": "float64",
    "I_mean": "float64",
    "V_mean": "float64",

    # min/max 固定 float32（也可改 float64，但必须固定）
    "I_min": "float32",
    "I_max": "float32",
    "V_min": "float32",
    "V_max": "float32",
    "T_min": "float32",
    "T_max": "float32",

    "duration_s": "int64",
    "delta_soc": "float32",
    "delta_odo": "float32",
}


def iter_parquet_parts(vin: str):
    parts = sorted((CORE_PATH / vin).glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts for {vin} under {CORE_PATH / vin}")
    for p in parts:
        yield pd.read_parquet(p, columns=CORE_USE)


def classify_state(df: pd.DataFrame) -> pd.Series:
    speed = df["speed"].to_numpy()
    cur = df["totalcurrent"].to_numpy()

    state = np.full(len(df), "other", dtype=object)
    rest = (speed <= SPEED_STOP) & (np.abs(cur) <= I_REST)
    charge = (speed <= SPEED_STOP) & (cur <= I_CHARGE)
    drive = (speed > SPEED_STOP) & (cur >= I_DRIVE)

    state[rest] = "rest"
    state[charge] = "charge"
    state[drive] = "drive"
    return pd.Series(state, index=df.index)


def _cast_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    强制把 summary 的列转成固定 dtype，保证所有批次写入 schema 一致。
    """
    df = df.copy()

    # 确保列齐全
    for c in SUMMARY_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[SUMMARY_COLS]

    # 逐列 cast（对 NA 做安全处理）
    for c, dt in SUMMARY_DTYPES.items():
        if dt.startswith("int"):
            # int 列：NA 先填充再转（这里用 0；若你需要保留缺失可改成 Int64 可空类型）
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(dt)
        elif dt.startswith("float"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
        elif dt == "string":
            df[c] = df[c].astype("string")
        else:
            df[c] = df[c].astype(dt)

    return df


def _finalize_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # duration 用 int64（秒）
    df["duration_s"] = (df["t_end"] - df["t_start"]).astype("int64")

    # delta
    df["delta_soc"] = (df["soc_end"] - df["soc_start"]).astype("float32")
    df["delta_odo"] = (df["odo_end"] - df["odo_start"]).astype("float32")

    # mean：float64
    n = df["n"].astype("float64").clip(lower=1.0)
    df["I_mean"] = (df["I_sum"].astype("float64") / n).astype("float64")
    df["V_mean"] = (df["V_sum"].astype("float64") / n).astype("float64")

    return df


def _merge_carry_into_row(row: dict, carry: dict) -> dict:
    # 只做数值合并，不做 dtype 赋值；dtype 统一在 _cast_summary_df 完成
    row["t_start"] = carry["t_start"]
    row["soc_start"] = carry["soc_start"]
    row["odo_start"] = carry["odo_start"]

    row["n"] = int(row["n"]) + int(carry["n"])

    row["I_sum"] = float(row["I_sum"]) + float(carry["I_sum"])
    row["V_sum"] = float(row["V_sum"]) + float(carry["V_sum"])

    row["I_min"] = float(min(row["I_min"], carry["I_min"]))
    row["I_max"] = float(max(row["I_max"], carry["I_max"]))

    row["V_min"] = float(min(row["V_min"], carry["V_min"]))
    row["V_max"] = float(max(row["V_max"], carry["V_max"]))

    row["T_min"] = float(min(row["T_min"], carry["T_min"]))
    row["T_max"] = float(max(row["T_max"], carry["T_max"]))
    return row


def segment_one_vin(vin: str):
    seg_summary_path = SEG_OUT / f"segment_summary_{vin}.parquet"
    charge_dir = SEG_OUT / "charge_points" / vin
    charge_dir.mkdir(parents=True, exist_ok=True)

    seg_id_offset = 0
    prev_t = None
    prev_state = None

    carry = None  # dict 或 None

    if not _HAVE_PYARROW:
        raise RuntimeError("pyarrow is required for streaming write on large dataset (8.5e8 frames).")

    writer = None
    part_idx = 0

    def write_summary_df(df_out: pd.DataFrame):
        nonlocal writer
        if df_out is None or df_out.empty:
            return

        df_out = _finalize_derived(df_out)
        df_out = _cast_summary_df(df_out)  # 关键：强制 dtype 一致

        table = pa.Table.from_pandas(df_out, preserve_index=False)

        if writer is None:
            # 以第一批的 schema 作为“全局 schema”
            writer = pq.ParquetWriter(seg_summary_path.as_posix(), table.schema, compression="snappy")
        writer.write_table(table)

    for ch in tqdm(iter_parquet_parts(vin), desc=f"Segmenting {vin}"):
        if ch.empty:
            continue

        # 强制关键列 dtype（降低后续 dtype 漂移）
        ch = ch.copy()
        ch["terminaltime"] = pd.to_numeric(ch["terminaltime"], errors="coerce").astype("int64")
        for col in ["soc", "speed", "totalodometer", "totalvoltage", "totalcurrent",
                    "mintemperaturevalue", "maxtemperaturevalue"]:
            if col in ch.columns:
                ch[col] = pd.to_numeric(ch[col], errors="coerce")

        # 状态分类
        st = classify_state(ch)

        # dt（首行用 prev_t）
        t = ch["terminaltime"]
        dt = t.diff()
        if prev_t is None:
            dt.iloc[0] = 0
        else:
            dt.iloc[0] = int(t.iloc[0]) - int(prev_t)
        dt = dt.fillna(0).astype("int64")

        # 判断本 chunk 第一行是否断段
        first_break = False
        if prev_state is None:
            first_break = True
        else:
            if (st.iloc[0] != prev_state) or (dt.iloc[0] > GAP_SEC) or (dt.iloc[0] < 0):
                first_break = True

        # 如果断段，把 carry 写出
        if first_break and carry is not None:
            write_summary_df(pd.DataFrame([carry]))
            carry = None

        # 边界：状态变化 or 超大间隔 or 时间倒退
        boundary = (st != st.shift(1))
        boundary.iloc[0] = True if first_break else False
        boundary |= (dt > GAP_SEC) | (dt < 0)

        seg_id = boundary.cumsum().astype("int64") + seg_id_offset

        # 输出 charge_points（逐点）
        ch["state"] = st
        ch["seg_id"] = seg_id
        ch["dt"] = dt

        charge_points = ch[ch["state"] == "charge"]
        if not charge_points.empty:
            charge_points.to_parquet(charge_dir / f"part_{part_idx:05d}.parquet",
                                     index=False, compression="snappy")

        # 段级聚合（注意：sum 强制 float64，避免 float32/float64 漂移）
        g = ch.groupby(["seg_id", "state"], sort=False)
        tmp = g.agg(
            t_start=("terminaltime", "min"),
            t_end=("terminaltime", "max"),
            n=("terminaltime", "size"),
            soc_start=("soc", "first"),
            soc_end=("soc", "last"),
            odo_start=("totalodometer", "first"),
            odo_end=("totalodometer", "last"),
            I_sum=("totalcurrent", "sum"),
            I_min=("totalcurrent", "min"),
            I_max=("totalcurrent", "max"),
            V_sum=("totalvoltage", "sum"),
            V_min=("totalvoltage", "min"),
            V_max=("totalvoltage", "max"),
            T_min=("mintemperaturevalue", "min"),
            T_max=("maxtemperaturevalue", "max"),
        ).reset_index()

        if tmp.empty:
            prev_t = int(ch["terminaltime"].iloc[-1])
            prev_state = ch["state"].iloc[-1]
            seg_id_offset = int(ch["seg_id"].iloc[-1])
            part_idx += 1
            continue

        # 统一 tmp 的基础 dtype（减少后续 carry 合并引发的 dtype 漂移）
        tmp["t_start"] = pd.to_numeric(tmp["t_start"], errors="coerce").fillna(0).astype("int64")
        tmp["t_end"] = pd.to_numeric(tmp["t_end"], errors="coerce").fillna(0).astype("int64")
        tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce").fillna(0).astype("int64")

        # sum -> float64，min/max -> float32（固定）
        tmp["I_sum"] = pd.to_numeric(tmp["I_sum"], errors="coerce").astype("float64")
        tmp["V_sum"] = pd.to_numeric(tmp["V_sum"], errors="coerce").astype("float64")

        for c in ["soc_start", "soc_end", "odo_start", "odo_end",
                  "I_min", "I_max", "V_min", "V_max", "T_min", "T_max"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("float32")

        tmp["state"] = tmp["state"].astype("string")

        # carry 合并：如果第一段延续 carry，则把 carry 合到 tmp 对应行
        if (not first_break) and (carry is not None):
            cid = int(carry["seg_id"])
            hit = tmp["seg_id"] == cid
            if hit.any():
                idx = tmp.index[hit][0]
                row = tmp.loc[idx].to_dict()
                row = _merge_carry_into_row(row, carry)

                # 逐列赋值（避免 pandas 一次性塞 list 触发 dtype 混写）
                for k, v in row.items():
                    tmp.at[idx, k] = v
                carry = None
            else:
                # 异常情况：先把 carry 写出，避免丢段
                write_summary_df(pd.DataFrame([carry]))
                carry = None

        # 最后一段做 carry，不立即写出
        last_seg = int(ch["seg_id"].iloc[-1])
        is_last = tmp["seg_id"] == last_seg

        finished = tmp[~is_last]
        if not finished.empty:
            write_summary_df(finished)

        carry = tmp[is_last].iloc[0].to_dict()

        # 更新 prev 与 offset
        prev_t = int(ch["terminaltime"].iloc[-1])
        prev_state = ch["state"].iloc[-1]
        seg_id_offset = last_seg
        part_idx += 1

    # 文件结束：补写最后 carry（解决尾段丢失）
    if carry is not None:
        write_summary_df(pd.DataFrame([carry]))
        carry = None

    if writer is not None:
        writer.close()

    print("Wrote:", seg_summary_path)
    print("Charge points dir:", charge_dir)


def main():
    for fn in ALL_FILES:
        vin = fn.replace(".csv", "")
        segment_one_vin(vin)


if __name__ == "__main__":
    main()
