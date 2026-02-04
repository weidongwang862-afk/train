# scripts/02_clean_core_to_parquet.py
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config import RAW_DIR, OUT_DIR, ALL_FILES, CORE_COLS, DTYPES, RANGE_RULES, CHUNKSIZE
from dotenv import load_dotenv
load_dotenv()



# ---------- 可调参数 ----------
# 时间大回拨阈值：超过这个回拨量就认为发生了“新session/重启/拼接”，不在清洗阶段硬砍后续数据
# 你数据采样 10s/帧，建议先用 6 小时；如果你发现经常发生“回拨几分钟”但后续仍应过滤，则保留下面的轻微回拨过滤逻辑
ROLLBACK_RESET_SEC = 6 * 3600

# parquet压缩（8.5亿帧强烈建议）
PARQUET_COMPRESSION = "snappy"


def _read_csv_chunks(fp: Path):
    # pandas 原生支持 chunksize 迭代读取大文件
    common_kwargs = dict(
        usecols=CORE_COLS,
        dtype=DTYPES,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )
    # 某些环境 read_csv(engine="pyarrow") 更快
    try:
        return pd.read_csv(fp, engine="pyarrow", **common_kwargs)
    except Exception:
        return pd.read_csv(fp, **common_kwargs)


def clean_one_file(vin_csv: str):
    src = RAW_DIR / vin_csv
    vin = vin_csv.replace(".csv", "")
    out_dir = OUT_DIR / "01_clean_core" / vin
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    stat = {
        "vin": vin,
        "rows_in": 0,
        "rows_out": 0,
        "drop_na_key": 0,
        "drop_range": 0,
        "drop_dup_or_backwards_time": 0,
        "time_rollback_reset": 0,
        "drop_dup_terminaltime_in_chunk": 0,
    }

    part = 0
    last_t = None

    # 关键字段：缺了就没法切片/算标签
    KEY_COLS = ["terminaltime", "soc", "speed", "totalodometer", "totalvoltage", "totalcurrent"]

    for ch in tqdm(_read_csv_chunks(src), desc=f"Cleaning {vin_csv}"):
        stat["rows_in"] += len(ch)

        if ch.empty:
            continue

        # 0) terminaltime 容错：先转数值，再转为“整数秒”
        # 可能存在 '76913.0' / 空值 / 异常字符串等
        ch = ch.copy()
        ch.loc[:, "terminaltime"] = pd.to_numeric(ch["terminaltime"], errors="coerce")
        before = len(ch)
        ch = ch.dropna(subset=["terminaltime"]).copy()
        if ch.empty:
            stat["drop_na_key"] += before
            continue
        ch.loc[:, "terminaltime"] = ch["terminaltime"].astype("int64")

        # 1) 丢弃关键字段缺失
        before = len(ch)
        ch = ch.dropna(subset=KEY_COLS)
        stat["drop_na_key"] += (before - len(ch))
        if ch.empty:
            continue

        # 2) 物理范围过滤（宽松）
        before = len(ch)
        for col, (lo, hi) in RANGE_RULES.items():
            if col in ch.columns:
                ch = ch[(ch[col] >= lo) & (ch[col] <= hi)]
        stat["drop_range"] += (before - len(ch))
        if ch.empty:
            continue

        # 2.5) 剔除明显占位/无效编码
        # totalcurrent == -1000 常见为无效占位；totalvoltage <=0 也无物理意义
        if "totalvoltage" in ch.columns:
            ch = ch[ch["totalvoltage"] > 0]
        if "totalcurrent" in ch.columns:
            ch = ch[ch["totalcurrent"] != -1000]
        if ch.empty:
            continue

        # 3) chunk 内按时间排序（应对局部乱序）
        ch = ch.sort_values("terminaltime", kind="mergesort")

        # 3.1) chunk 内 terminaltime 去重（避免 dt=0 泛滥）
        before = len(ch)
        ch = ch.drop_duplicates(subset=["terminaltime"], keep="last")
        stat["drop_dup_terminaltime_in_chunk"] += (before - len(ch))
        if ch.empty:
            continue

        # 4) 跨 chunk 时间保护：
        # - 轻微回拨/重复：仍过滤 terminaltime > last_t
        # - 大回拨（>ROLLBACK_RESET_SEC）：认为数据进入新session，不在清洗阶段硬砍后续数据，直接 reset last_t
        if last_t is not None:
            tmin = int(ch["terminaltime"].iloc[0])
            if tmin <= last_t:
                rollback = int(last_t - tmin)
                if rollback > ROLLBACK_RESET_SEC:
                    stat["time_rollback_reset"] += 1
                    last_t = None
                else:
                    before = len(ch)
                    ch = ch[ch["terminaltime"] > last_t]
                    stat["drop_dup_or_backwards_time"] += (before - len(ch))
                    if ch.empty:
                        continue

        last_t = int(ch["terminaltime"].iloc[-1])

        # 5) 写 Parquet 分片（不用 index）
        part_path = out_dir / f"part_{part:05d}.parquet"
        ch.to_parquet(part_path, index=False, compression=PARQUET_COMPRESSION)
        stat["rows_out"] += len(ch)
        part += 1

    return stat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    err_rows = []

    for fn in ALL_FILES:
        try:
            logs.append(clean_one_file(fn))
        except Exception as e:
            err_rows.append({"file": fn, "error": repr(e)})
            print(f"[ERROR] {fn}: {e}")

    df = pd.DataFrame(logs)
    log_path = OUT_DIR / "01_clean_core" / "clean_log.csv"
    df.to_csv(log_path, index=False)

    if err_rows:
        err_df = pd.DataFrame(err_rows)
        err_path = OUT_DIR / "01_clean_core" / "clean_errors.csv"
        err_df.to_csv(err_path, index=False)
        print("Errors:", err_path)

    print("Done. Log:", log_path)



if __name__ == "__main__":
    main()
