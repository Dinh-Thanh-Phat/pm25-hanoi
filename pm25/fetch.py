"""Tải dữ liệu thô (chạy lại thì chỉ tải phần còn thiếu):

  PM2.5 theo giờ của các trạm Hà Nội  ← OpenAQ API v3 (cần key free trong .env)
  thời tiết theo giờ                   ← Open-Meteo archive (ERA5, free, không key)
  dự báo PM2.5 của CAMS                ← Open-Meteo air-quality (mốc so sánh)

  .venv/bin/python -m pm25.fetch [pm25|weather|cams|forecast|all]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd

from . import config


def _key() -> str:
    if os.environ.get("OPENAQ_API_KEY"):              # GitHub Actions: key nằm trong Secrets
        return os.environ["OPENAQ_API_KEY"]
    for line in (config.ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAQ_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("Thiếu OPENAQ_API_KEY trong .env")


def _get(url: str, headers: dict | None = None, tries: int = 9) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pm25-hanoi/1.0", **(headers or {})})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and "open-meteo" in url and i < tries - 1:
                print("   Open-Meteo hết lượt trong giờ, chờ 10 phút…", flush=True)
                time.sleep(600)                    # gói free: 5.000 lệnh/giờ (lệnh nặng tính nhiều lượt)
                continue
            if e.code in (408, 429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise


# ---------- trạm ----------
def stations() -> pd.DataFrame:
    """Mọi trạm OpenAQ có PM2.5 trong bán kính 25 km (tối đa của API) quanh Hoàn Kiếm."""
    p = config.RAW / "stations.json"
    if not p.exists():
        d = _get(f"{config.OPENAQ}/locations?coordinates={config.CENTER[0]},{config.CENTER[1]}&radius=25000&limit=1000",
                 {"X-API-Key": _key()})
        p.write_text(json.dumps(d, ensure_ascii=False))
    rows = []
    res = json.loads(p.read_text())["results"]
    reg = config.RAW / "stations_region.json"           # vùng rộng hơn (bbox), thêm trạm ngoại thành
    if reg.exists():
        ids = {r["id"] for r in res}
        res += [r for r in json.loads(reg.read_text())["results"] if r["id"] not in ids]
    for r in res:
        s = [x for x in r["sensors"] if x["parameter"]["name"] == "pm25"]
        if not s or not r.get("isMonitor") or not r.get("datetimeLast"):
            continue                       # chỉ lấy trạm quan trắc chuẩn, bỏ cảm biến giá rẻ
        rows.append({"location_id": r["id"], "sensor_id": s[0]["id"], "ten": r["name"],
                     "nguon": (r.get("provider") or {}).get("name"),
                     "lat": r["coordinates"]["latitude"], "lon": r["coordinates"]["longitude"],
                     "tu": r["datetimeFirst"]["utc"][:10], "den": r["datetimeLast"]["utc"][:10]})
    return pd.DataFrame(rows)


def fetch_pm25() -> None:
    st = stations()
    out = config.RAW / "pm25"
    out.mkdir(parents=True, exist_ok=True)
    k = _key()
    for i, s in enumerate(st.itertuples(), 1):
        f = out / f"{s.location_id}.parquet"
        if f.exists():
            print(f"[{i}/{len(st)}] {s.ten}: đã có", flush=True)
            continue
        recs, calls = [], 0
        # tải từng quý: trang sâu của một khoảng dài làm API quá giờ (408)
        for a in pd.date_range(pd.Timestamp(s.tu).to_period("Q").start_time, pd.Timestamp(s.den) + pd.Timedelta(days=1),
                               freq="QS"):
            b = a + pd.offsets.QuarterBegin(startingMonth=1)
            page = 1
            while True:
                q = urllib.parse.urlencode({"limit": 1000, "page": page,
                                            "datetime_from": a.isoformat(), "datetime_to": b.isoformat()})
                try:
                    d = _get(f"{config.OPENAQ}/sensors/{s.sensor_id}/hours?{q}", {"X-API-Key": k})
                except urllib.error.HTTPError as e:
                    print(f"   bỏ qua {a:%Y-%m} trang {page}: HTTP {e.code}", flush=True)
                    break
                calls += 1
                for x in d["results"]:
                    recs.append((x["period"]["datetimeFrom"]["utc"], x["value"], x["coverage"].get("observedCount")))
                time.sleep(1.1)            # giới hạn 60 lệnh/phút
                if len(d["results"]) < 1000:
                    break
                page += 1
        page = calls
        df = pd.DataFrame(recs, columns=["time", "pm25", "n_obs"])
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        df.drop_duplicates("time").sort_values("time").to_parquet(f, index=False)
        print(f"[{i}/{len(st)}] {s.ten}: {len(df)} giờ ({page} lệnh)", flush=True)
    st.to_csv(config.RAW / "stations.csv", index=False)


def update_pm25(lui_ngay: int = 2) -> dict:
    """Tải bù: với mỗi trạm đã có file, chỉ lấy các giờ từ (giờ cuối - lui_ngay) đến nay rồi gộp, giờ trùng lấy bản mới.
    Trạm đã ngừng chỉ tốn 1 lệnh. Trả về số giờ mới thêm theo trạm."""
    st = pd.read_csv(config.RAW / "stations.csv")
    out, k, now = config.RAW / "pm25", _key(), pd.Timestamp.now(tz="UTC").tz_convert(None).floor("h")
    added = {}
    for s in st.itertuples():
        f = out / f"{s.location_id}.parquet"
        if not f.exists():
            continue
        old = pd.read_parquet(f)
        a = (old.time.max() if len(old) else pd.Timestamp(s.tu)) - pd.Timedelta(days=lui_ngay)
        recs, page = [], 1
        while True:
            q = urllib.parse.urlencode({"limit": 1000, "page": page, "datetime_from": a.isoformat(),
                                        "datetime_to": (now + pd.Timedelta(hours=1)).isoformat()})
            try:
                d = _get(f"{config.OPENAQ}/sensors/{s.sensor_id}/hours?{q}", {"X-API-Key": k}, tries=3)
            except Exception as e:                       # một trạm lỗi thì bỏ qua, các trạm khác vẫn chạy
                print(f"   {s.ten}: lỗi {type(e).__name__}", flush=True)
                break
            recs += [(x["period"]["datetimeFrom"]["utc"], x["value"], x["coverage"].get("observedCount")) for x in d["results"]]
            time.sleep(1.1)                              # giới hạn 60 lệnh/phút
            if len(d["results"]) < 1000:
                break
            page += 1
        if not recs:
            continue
        new = pd.DataFrame(recs, columns=["time", "pm25", "n_obs"])
        new["time"] = pd.to_datetime(new["time"], utc=True).dt.tz_convert(None)
        df = pd.concat([old, new]).drop_duplicates("time", keep="last").sort_values("time")
        n = len(df) - len(old)
        if n or len(new):
            df.to_parquet(f, index=False)
        if n:
            added[s.ten] = n
    return added


# ---------- thời tiết + CAMS ----------
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
                "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "boundary_layer_height",
                "shortwave_radiation"]


def _years(start: str):
    y0, today = int(start[:4]), date.today()
    for y in range(y0, today.year + 1):
        a = max(f"{y}-01-01", start)
        b = min(f"{y}-12-31", today.isoformat())
        yield a, b


def _months(start: str):
    today = pd.Timestamp(date.today())
    for a in pd.date_range(start, today, freq="MS"):
        b = min(a + pd.offsets.MonthEnd(0), today)
        yield a.date().isoformat(), b.date().isoformat()


def _openmeteo(base: str, vars_: list[str], points: pd.DataFrame, start: str, name: str, monthly: bool = False) -> None:
    out = config.RAW / name
    out.mkdir(parents=True, exist_ok=True)
    cur = date.today().isoformat()[:7] if monthly else str(date.today().year)
    for a, b in (_months(start) if monthly else _years(start)):
        tag = a[:7] if monthly else a[:4]
        f = out / f"{tag}.parquet"
        old = pd.read_parquet(f) if f.exists() and tag != cur else None
        todo = points if old is None else points[~points.point_id.isin(set(old.point_id))]
        if todo.empty:
            continue                              # đủ mọi điểm rồi: chỉ tải bù điểm mới khi thêm trạm
        q = urllib.parse.urlencode({"latitude": ",".join(f"{x:.4f}" for x in todo.lat),
                                    "longitude": ",".join(f"{x:.4f}" for x in todo.lon),
                                    "start_date": a, "end_date": b, "hourly": ",".join(vars_), "timezone": "GMT"})
        d = _get(f"{base}?{q}")
        d = d if isinstance(d, list) else [d]
        frames = [] if old is None else [old]
        for pid, part in zip(todo.point_id, d):
            h = pd.DataFrame(part["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h["point_id"] = pid
            frames.append(h)
        pd.concat(frames).to_parquet(f, index=False)
        print(f"{name} {tag}: +{len(todo)} điểm", flush=True)
        time.sleep(8 if monthly else 1)


def weather_points() -> pd.DataFrame:
    # ERA5 ô ~25 km: gom trạm theo ô 0.1° để khỏi tải trùng
    st = pd.read_csv(config.RAW / "stations.csv")
    st["point_id"] = (st.lat.round(1).astype(str) + "_" + st.lon.round(1).astype(str))
    return st.groupby("point_id", as_index=False)[["lat", "lon"]].mean()


def fetch_weather() -> None:
    _openmeteo(config.OM_ARCHIVE, WEATHER_VARS, weather_points(), config.START, "weather")


def fetch_cams() -> None:
    _openmeteo(config.OM_AIR, ["pm2_5"], weather_points(), config.CAMS_START, "cams")


# dự báo thời tiết đã phát hành trước 1/2/3 ngày (để chấm trung thực: lúc dự báo chưa biết thời tiết thật)
FC_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "cloud_cover",
           "wind_speed_10m", "wind_direction_10m"]


def fetch_forecast() -> None:
    vars_ = [f"{v}_previous_day{d}" for v in FC_VARS for d in (1, 2, 3)]
    _openmeteo(config.OM_PREV_RUNS, vars_, weather_points(), config.FC_START, "forecast", monthly=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "update":
        print(update_pm25())
    if what in ("pm25", "all"):
        fetch_pm25()
    if what in ("weather", "all"):
        fetch_weather()
    if what in ("cams", "all"):
        fetch_cams()
    if what in ("forecast", "all"):
        fetch_forecast()
