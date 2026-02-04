# scripts/config.py
# scripts/config.py
from pathlib import Path
import os

# 项目根目录：E:\RAW_DATA（不依赖你在哪个目录运行脚本）
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "configs"

def _read_vins(list_path: str) -> list[str]:
    p = Path(list_path)
    if not p.is_absolute():
        p = ROOT / p  # 允许传入 configs/xxx.txt 这种相对路径
    text = p.read_text(encoding="utf-8")
    vins = []
    for line in text.splitlines():
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        s = s.replace(".csv", "").strip()
        vins.append(s)
    return vins

# 运行时用环境变量指定 VIN 清单（推荐）
VIN_LIST = os.environ.get("VIN_LIST", "").strip()
TRAIN_LIST = os.environ.get("TRAIN_LIST", "").strip()
TEST_LIST  = os.environ.get("TEST_LIST", "").strip()

if TRAIN_LIST or TEST_LIST:
    train_vins = _read_vins(TRAIN_LIST) if TRAIN_LIST else []
    test_vins  = _read_vins(TEST_LIST) if TEST_LIST else []
    TRAIN_FILES = [f"{v}.csv" for v in train_vins]
    TEST_FILES  = [f"{v}.csv" for v in test_vins]
    ALL_FILES   = TRAIN_FILES + TEST_FILES
elif VIN_LIST:
    vins = _read_vins(VIN_LIST)
    TRAIN_FILES = [f"{v}.csv" for v in vins]
    TEST_FILES  = []
    ALL_FILES   = TRAIN_FILES
else:
    # 兜底：不设置清单时，仍按你原来的 6 辆车跑（便于回归）
    TRAIN_FILES = ["vin35.csv", "vin9.csv", "vin39.csv", "vin27.csv"]
    TEST_FILES  = ["vin43.csv", "vin42.csv"]
    ALL_FILES   = TRAIN_FILES + TEST_FILES


# Step2：只保留“核心列”（先不碰两列巨大的列表字符串）
CORE_COLS = [
    "terminaltime", "soc", "speed", "totalodometer", "chargestatus",
    "totalvoltage", "totalcurrent",
    "minvoltagebattery", "maxvoltagebattery",
    "mintemperaturevalue", "maxtemperaturevalue"
]

DTYPES = {
    "terminaltime": "float64",
    "soc": "float32",
    "speed": "float32",
    "totalodometer": "float32",
    "chargestatus": "float32",
    "totalvoltage": "float32",
    "totalcurrent": "float32",
    "minvoltagebattery": "float32",
    "maxvoltagebattery": "float32",
    "mintemperaturevalue": "float32",
    "maxtemperaturevalue": "float32",
}

# 物理范围过滤（先宽松，后面再根据统计收紧）
RANGE_RULES = {
    "soc": (0.0, 100.0),
    "speed": (0.0, 200.0),
    "totalvoltage": (50.0, 1000.0),
    "totalcurrent": (-2000.0, 2000.0),
    "totalodometer": (0.0, 2_000_000.0),  # 单位如果是km，50万km也在范围内
    "minvoltagebattery": (0.0, 6.0),
    "maxvoltagebattery": (0.0, 6.0),
    "mintemperaturevalue": (-50.0, 120.0),
    "maxtemperaturevalue": (-50.0, 120.0),
}

CHUNKSIZE = 200_000
