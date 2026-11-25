"""Dự báo trực tiếp: số đo 5 ngày gần nhất (OpenAQ) + thời tiết quá khứ & dự báo (Open-Meteo) → mô hình."""
from __future__ import annotations

import json
import threading
import time
import urllib.parse

import joblib
import numpy as np
import pandas as pd

from . import config, dataset, fetch

CACHE = config.OUT / "live.json"
TTL = 30 * 60
_lock = threading.Lock()
WVARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure", "cloud_cover",
         "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "boundary_layer_height", "shortwave_radiation"]


def _recent_pm(st: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_csv(config.RAW / "stations.csv")
    k = fetch._key()
    series = {}
    for r in raw.itertuples():
        sid = dataset.MERGE.get(r.location_id, r.location_id)
        if sid not in set(st.sid) or pd.Timestamp(r.den) < since - pd.Timedelta(days=30):
            continue
        q = urllib.parse.urlencode({"limit": 1000, "datetime_from": since.isoformat()})
        try:
            d = fetch._get(f"{config.OPENAQ}/sensors/{r.sensor_id}/hours?{q}", {"X-API-Key": k}, tries=2)
        except Exception:
            continue
        s = pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["utc"]).tz_convert(None): x["value"]
                       for x in d["results"]}, dtype=float)
        if len(s):
            series[sid] = s
        time.sleep(1.1)
    return pd.DataFrame(series)


def _weather(points: pd.DataFrame) -> tuple[dict, dict]:
    q = urllib.parse.urlencode({"latitude": ",".join(f"{x:.4f}" for x in points.lat),
                                "longitude": ",".join(f"{x:.4f}" for x in points.lon),
                                "hourly": ",".join(WVARS), "past_days": 5, "forecast_days": 4, "timezone": "GMT"})
    d = fetch._get(f"https://api.open-meteo.com/v1/forecast?{q}")
    d = d if isinstance(d, list) else [d]
    q2 = urllib.parse.urlencode({"latitude": ",".join(f"{x:.4f}" for x in points.lat),
                                 "longitude": ",".join(f"{x:.4f}" for x in points.lon),
                                 "hourly": "pm2_5", "past_days": 1, "forecast_days": 4, "timezone": "GMT"})
    c = fetch._get(f"{config.OM_AIR}?{q2}")
    c = c if isinstance(c, list) else [c]
    W, C = {}, {}
    for pid, a, b in zip(points.point_id, d, c):
        W[pid] = pd.DataFrame(a["hourly"]).assign(time=lambda f: pd.to_datetime(f.time)).set_index("time")
        C[pid] = pd.DataFrame(b["hourly"]).assign(time=lambda f: pd.to_datetime(f.time)).set_index("time").pm2_5
    return W, C


def compute() -> dict:
    _, st = dataset.panel()
    now = pd.Timestamp.now("UTC").tz_convert(None).floor("h")
    pm = _recent_pm(st, now - pd.Timedelta(days=5))
    W, C = _weather(fetch.weather_points())
    idx = pd.date_range(now - pd.Timedelta(days=5), now + pd.Timedelta(hours=80), freq="h")
    P = pd.DataFrame({sid: dataset.clean(pm[sid].reindex(idx)) if sid in pm else np.nan for sid in st.sid}, index=idx)
    # thời điểm dự báo = giờ gần nhất mà đa số trạm đang chạy đã có số
    active = [s for s in st.sid if P[s].notna().sum() >= 24]
    t = P[active].dropna(how="all").index.max() if active else now
    out = {"luc": (t + pd.Timedelta(hours=7)).isoformat(), "tram": []}
    for h in config.HORIZONS:
        m, c = joblib.load(config.OUT / f"model_h{h}_full.joblib")
        X = dataset.features(P, st, W, h, "era5")
        X = X[X.time == t]
        pred = np.expm1(m.predict(X[c]))
        for sid, p, m24 in zip(X.sid, pred, X.pm_m24):
            out.setdefault("_p", {}).setdefault(int(sid), {})[h] = (None if np.isnan(m24) else float(p))
    names = st.set_index("sid")
    for sid in st.sid:
        s = P[sid]
        last = s.loc[:t].dropna()
        pid = names.loc[sid, "point_id"]
        cams = {h: float(C[pid].rolling(24, min_periods=18).mean().shift(-h).get(t, np.nan)) for h in config.HORIZONS}
        out["tram"].append({
            "sid": int(sid), "ten": names.loc[sid, "ten"], "lat": float(names.loc[sid, "lat"]),
            "lon": float(names.loc[sid, "lon"]),
            "hoat_dong": sid in active,
            "gan_nhat": None if last.empty else {"luc": (last.index[-1] + pd.Timedelta(hours=7)).isoformat(),
                                                 "pm25": float(last.iloc[-1])},
            "tb24": None if last.empty else float(s.loc[t - pd.Timedelta(hours=23):t].mean()),
            "du_bao": {str(h): out["_p"].get(int(sid), {}).get(h) for h in config.HORIZONS},
            "cams": {str(h): (None if np.isnan(v) else v) for h, v in cams.items()},
            "chuoi": [{"luc": (i + pd.Timedelta(hours=7)).isoformat(), "pm25": float(v)}
                      for i, v in s.loc[t - pd.Timedelta(hours=72):t].dropna().items()],
        })
    out.pop("_p", None)
    # thời tiết của điểm gần trung tâm: 24h vừa qua và 3 ngày tới (để giải thích vì sao bụi lên/xuống)
    pts = fetch.weather_points()
    c = pts.iloc[((pts.lat - config.CENTER[0]) ** 2 + (pts.lon - config.CENTER[1]) ** 2).argmin()].point_id
    w = W[c]
    def agg(a, b):
        x = w.loc[a:b]
        if x.empty:
            return None
        return {"gio": round(float(x.wind_speed_10m.mean()), 1), "mua": round(float(x.precipitation.sum()), 1),
                "blh": round(float(x.boundary_layer_height.mean())), "nhiet": round(float(x.temperature_2m.mean()), 1),
                "am": round(float(x.relative_humidity_2m.mean())),
                "huong": round(float(np.degrees(np.arctan2(np.sin(np.radians(x.wind_direction_10m)).mean(),
                                                           np.cos(np.radians(x.wind_direction_10m)).mean())) % 360))}
    out["thoi_tiet"] = {"qua": agg(t - pd.Timedelta(hours=23), t),
                        **{str(h): agg(t + pd.Timedelta(hours=h - 23), t + pd.Timedelta(hours=h)) for h in config.HORIZONS}}
    return out


def get(force: bool = False) -> dict:
    with _lock:
        if not force and CACHE.exists() and time.time() - CACHE.stat().st_mtime < TTL:
            return json.loads(CACHE.read_text())
        d = compute()
        d["tao_luc"] = time.time()
        CACHE.write_text(json.dumps(d, ensure_ascii=False))
        return d


if __name__ == "__main__":
    d = get(force=True)
    print("lúc", d["luc"])
    for s in d["tram"]:
        if s["hoat_dong"]:
            print(f"{s['ten'][:32]:32} TB24 {s['tb24']:6.1f} → " +
                  "  ".join(f"N{int(h) // 24}: {v:5.1f} (CAMS {s['cams'][h] or float('nan'):5.1f})"
                            for h, v in s["du_bao"].items() if v is not None))
