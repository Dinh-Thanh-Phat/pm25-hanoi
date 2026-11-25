"""Dựng bảng học: mỗi dòng = (trạm, thời điểm dự báo t), nhãn = PM2.5 trung bình 24h của ngày thứ 1/2/3 sau t.

Nguyên tắc chống rò rỉ: đặc trưng chỉ dùng số đo bụi ≤ t. Thời tiết tương lai (t+h) có 3 chế độ:
  none     – không dùng
  forecast – dự báo đã phát hành trước đó (Open-Meteo previous runs, chỉ có từ 2024) → con số trung thực
  era5     – thời tiết thật (tái phân tích) → cận trên, dùng khi huấn luyện
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from . import config

warnings.filterwarnings("ignore", "Mean of empty slice")

MERGE = {2539: 7441}          # StateAir 2016 và AirNow là cùng trạm Đại sứ quán Mỹ, đổi tên nguồn
MIN_HOURS = 3000              # trạm có ít hơn chừng này giờ sạch thì bỏ
LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 72)
FUT_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m"]


# ---------- làm sạch ----------
def clean(s: pd.Series) -> pd.Series:
    s = s.where((s > 0) & (s < 800))
    # cảm biến kẹt: cùng một giá trị ≥ 6 giờ liền
    run = s.groupby((s != s.shift()).cumsum()).transform("size")
    s = s.where(~((run >= 6) & s.notna()))
    # gai đơn lẻ: gấp 4 lần cả giờ trước và giờ sau, chênh > 100
    prev, nxt = s.shift(1), s.shift(-1)
    spike = (s > 4 * prev) & (s > 4 * nxt) & (s - prev.fillna(s) > 100)
    return s.where(~spike)


def stations() -> pd.DataFrame:
    st = pd.read_csv(config.RAW / "stations.csv")
    st["sid"] = st.location_id.replace(MERGE)
    st = st.groupby("sid", as_index=False).agg(ten=("ten", "last"), nguon=("nguon", "last"), lat=("lat", "mean"),
                                               lon=("lon", "mean"))
    st.loc[st.sid == 7441, "ten"] = "Đại sứ quán Mỹ"
    st["point_id"] = st.lat.round(1).astype(str) + "_" + st.lon.round(1).astype(str)
    return st


def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bảng PM2.5 theo giờ (UTC), mỗi cột một trạm; và danh sách trạm giữ lại."""
    st = stations()
    raw = pd.read_csv(config.RAW / "stations.csv")
    series = {}
    for r in raw.itertuples():
        f = config.RAW / "pm25" / f"{r.location_id}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f).set_index("time")["pm25"]
        sid = MERGE.get(r.location_id, r.location_id)
        series[sid] = d if sid not in series else series[sid].combine_first(d)
    idx = pd.date_range(config.START, pd.Timestamp.now("UTC").tz_convert(None).floor("h"), freq="h")
    P = pd.DataFrame({sid: clean(s[~s.index.duplicated()].reindex(idx)) for sid, s in series.items()}, index=idx)
    keep = [c for c in P if P[c].notna().sum() >= MIN_HOURS]
    st = st[st.sid.isin(keep)].reset_index(drop=True)
    st["so_gio"] = st.sid.map(P[keep].notna().sum())
    return P[keep], st


# ---------- thời tiết ----------
def _load(name: str) -> pd.DataFrame:
    fs = sorted((config.RAW / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates(["point_id", "time"]) if fs else pd.DataFrame()


def weather() -> dict[str, pd.DataFrame]:
    w = _load("weather")
    return {p: g.set_index("time").drop(columns="point_id").sort_index() for p, g in w.groupby("point_id")}


def forecast() -> dict[str, pd.DataFrame]:
    w = _load("forecast")
    return {p: g.set_index("time").drop(columns="point_id").sort_index() for p, g in w.groupby("point_id")}


def cams() -> dict[str, pd.Series]:
    w = _load("cams")
    return {p: g.set_index("time")["pm2_5"].sort_index() for p, g in w.groupby("point_id")}


# ---------- hình học ----------
def _bearing_km(a, b):
    """Hướng (độ, 0 = bắc, theo chiều kim đồng hồ) và khoảng cách km từ trạm a tới trạm b."""
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    dlo = lo2 - lo1
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    d = 6371 * math.acos(min(1, math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(dlo)))
    return brg, d


def target(P: pd.DataFrame, h: int) -> pd.DataFrame:
    """Trung bình 24h của khoảng (t+h-24, t+h], cần ≥ 18 giờ có số đo."""
    m = P.rolling(24, min_periods=18).mean()
    return m.shift(-h)


# ---------- đặc trưng ----------
def features(P: pd.DataFrame, st: pd.DataFrame, W: dict, h: int, fut: str = "era5",
             FC: dict | None = None, hide: set | None = None) -> pd.DataFrame:
    """Mỗi dòng = (trạm, t). hide: các trạm bị coi như không có số đo (thí nghiệm ước lượng chỗ không có trạm)."""
    hide = hide or set()
    Pv = P.drop(columns=[c for c in hide if c in P])
    rows = []
    roll24 = P.rolling(24, min_periods=12).mean()
    for s in st.itertuples():
        w = W[s.point_id].reindex(P.index)
        x = pd.DataFrame(index=P.index)
        own = P[s.sid] if s.sid not in hide else pd.Series(np.nan, index=P.index)
        for L in LAGS:
            x[f"pm_l{L}"] = own.shift(L)
        x["pm_m6"] = own.rolling(6, min_periods=3).mean()
        x["pm_m24"] = own.rolling(24, min_periods=12).mean()
        x["pm_m72"] = own.rolling(72, min_periods=36).mean()
        x["pm_max24"] = own.rolling(24, min_periods=12).max()
        x["pm_d24"] = x.pm_m24 - x.pm_m24.shift(24)

        # mạng trạm: trung bình các trạm khác, và trạm đầu gió theo hướng gió lúc t
        others = [o for o in st.itertuples() if o.sid != s.sid and o.sid in Pv]
        wd = np.radians(w["wind_direction_10m"].to_numpy())          # hướng gió THỔI TỪ
        num = np.zeros(len(P)); den = np.zeros(len(P)); allv = []
        num24 = np.zeros(len(P)); den24 = np.zeros(len(P))
        for o in others:
            brg, d = _bearing_km(s, o)
            v = Pv[o.sid].to_numpy()
            v24 = roll24[o.sid].to_numpy()
            allv.append(v)
            wt = np.clip(np.cos(np.radians(brg) - wd), 0, None) / (1 + d / 10)
            ok = ~np.isnan(v) & ~np.isnan(wt)
            num[ok] += wt[ok] * v[ok]; den[ok] += wt[ok]
            ok24 = ~np.isnan(v24) & ~np.isnan(wt)
            num24[ok24] += wt[ok24] * v24[ok24]; den24[ok24] += wt[ok24]
        with np.errstate(invalid="ignore", divide="ignore"):
            x["net_mean"] = np.nanmean(np.vstack(allv), 0) if allv else np.nan
            x["net_n"] = (~np.isnan(np.vstack(allv))).sum(0) if allv else 0
            x["upwind"] = np.where(den > 0.05, num / den, np.nan)
            x["upwind_m24"] = np.where(den24 > 0.05, num24 / den24, np.nan)
        x["own_minus_net"] = x.pm_l0 - x.net_mean

        # thời tiết lúc t (đã quan sát)
        for c in ["temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover", "wind_speed_10m",
                  "boundary_layer_height", "shortwave_radiation", "dew_point_2m"]:
            x[f"w_{c}"] = w[c]
        x["w_wd_sin"] = np.sin(np.radians(w.wind_direction_10m)); x["w_wd_cos"] = np.cos(np.radians(w.wind_direction_10m))
        x["w_rain24"] = w.precipitation.rolling(24, min_periods=1).sum()
        x["w_dp24"] = w.surface_pressure - w.surface_pressure.shift(24)
        x["w_blh_m24"] = w.boundary_layer_height.rolling(24, min_periods=12).mean()

        # thời tiết tương lai trong khoảng (t+h-24, t+h]
        if fut != "none":
            if fut == "era5":
                src = w[FUT_VARS]
            else:
                dd = min(3, math.ceil(h / 24))
                f = (FC or {}).get(s.point_id)
                src = (f[[f"{v}_previous_day{dd}" for v in FUT_VARS]].set_axis(FUT_VARS, axis=1).reindex(P.index)
                       if f is not None else pd.DataFrame(np.nan, index=P.index, columns=FUT_VARS))
            win = src.rolling(24, min_periods=12)
            fm = win.mean().shift(-h)
            x["f_temp"] = fm.temperature_2m; x["f_rh"] = fm.relative_humidity_2m; x["f_cloud"] = fm.cloud_cover
            x["f_ws"] = fm.wind_speed_10m; x["f_pres"] = fm.surface_pressure
            x["f_rain"] = src.precipitation.rolling(24, min_periods=12).sum().shift(-h)
            x["f_ws_min"] = src.wind_speed_10m.rolling(24, min_periods=12).min().shift(-h)
            ang = np.radians(src.wind_direction_10m)
            x["f_wd_sin"] = np.sin(ang).rolling(24, min_periods=12).mean().shift(-h)
            x["f_wd_cos"] = np.cos(ang).rolling(24, min_periods=12).mean().shift(-h)
            x["f_dtemp"] = x.f_temp - w.temperature_2m.rolling(24, min_periods=12).mean()
            x["f_dpres"] = x.f_pres - w.surface_pressure

        # lịch của ngày được dự báo (giờ Việt Nam = UTC+7)
        tt = (P.index + pd.Timedelta(hours=7 + h - 12))
        x["doy_sin"] = np.sin(2 * np.pi * tt.dayofyear / 365.25); x["doy_cos"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)
        x["dow"] = tt.dayofweek
        x["hour_issue"] = (P.index + pd.Timedelta(hours=7)).hour
        x["lat"] = s.lat; x["lon"] = s.lon
        x["sid"] = s.sid
        x["time"] = P.index
        rows.append(x)
    X = pd.concat(rows, ignore_index=True)
    return X


def build(h: int, fut: str = "era5", hide: set | None = None) -> pd.DataFrame:
    P, st = panel()
    W = weather()
    FC = forecast() if fut == "forecast" else None
    X = features(P, st, W, h, fut, FC, hide)
    Y = target(P, h)
    X["y"] = Y.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([X.time, X.sid])).to_numpy()
    # trung bình 24h gần nhất (mốc "giữ nguyên") và CAMS cho cùng khoảng
    X["persist"] = X.pm_m24
    C = cams()
    cm = {p: s.rolling(24, min_periods=18).mean().shift(-h) for p, s in C.items()}
    pid = X.sid.map(st.set_index("sid").point_id)
    X["cams"] = np.nan
    for p, s in cm.items():
        m = pid == p
        X.loc[m, "cams"] = s.reindex(X.loc[m, "time"]).to_numpy()
    return X
