from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = DATA / "outputs"
for p in (RAW, PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)

CENTER = (21.0285, 105.8542)          # Hồ Hoàn Kiếm
START = "2016-11-01"                  # trạm Đại sứ quán Mỹ bắt đầu
CAMS_START = "2022-08-01"             # Open-Meteo chỉ có lịch sử CAMS toàn cầu từ đây (kiểm tra lại khi tải)

OPENAQ = "https://api.openaq.org/v3"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
OM_PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
FC_START = "2024-01-01"                # previous-runs API có dữ liệu từ khoảng đầu 2024

HORIZONS = (24, 48, 72)
UNHEALTHY = 55.5                      # µg/m³, ngưỡng AQI "có hại cho sức khoẻ" (US EPA 2024: 55.5)
