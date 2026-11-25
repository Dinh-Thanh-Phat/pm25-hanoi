"""Ước lượng PM2.5 ở chỗ KHÔNG có trạm (bước 2).

Thí nghiệm "giấu trạm": chia trạm thành nhóm, mỗi lượt giấu một nhóm, chỉ dùng các trạm còn lại + thời tiết để
ước lượng trung bình 24h ở trạm bị giấu. So với: trạm gần nhất, trung bình mạng, nội suy nghịch đảo khoảng cách (IDW).
Giai đoạn 2024-02 → 2025-01 vì khi đó mạng trạm dày nhất.

  .venv/bin/python -m pm25.spatial
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config, dataset

PERIOD = ("2024-02-01", "2025-02-01")
STEP = 3                      # lấy mẫu mỗi 3 giờ (các giờ liền nhau gần như trùng thông tin)
K = 5                         # số nhóm trạm


def _xy(lat, lon):
    return np.array(lat) * 111.0, np.array(lon) * 104.0      # km xấp xỉ quanh Hà Nội


def neighbour_features(R: np.ndarray, sx, sy, tx, ty, wd=None) -> dict:
    """R: (T, S) trung bình 24h của các trạm được dùng (NaN = không có). (tx, ty): điểm cần ước lượng."""
    d = np.hypot(sx - tx, sy - ty) + 0.5                           # (S,)
    ok = ~np.isnan(R)
    w = np.where(ok, 1 / d ** 2, 0)
    Rz = np.where(ok, R, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        idw = (Rz * w).sum(1) / w.sum(1)
        mean = Rz.sum(1) / ok.sum(1)
        std = np.sqrt(np.where(ok, (R - mean[:, None]) ** 2, 0).sum(1) / ok.sum(1))
    dd = np.where(ok, d, np.inf)
    j = dd.argmin(1)
    near = R[np.arange(len(R)), j]
    near_d = dd[np.arange(len(R)), j]
    dd2 = dd.copy(); dd2[np.arange(len(R)), j] = np.inf
    j2 = dd2.argmin(1)
    out = {"idw": idw, "net_mean": mean, "net_std": std, "net_n": ok.sum(1), "near": near, "near_d": near_d,
           "near2": R[np.arange(len(R)), j2], "near2_d": dd2[np.arange(len(R)), j2]}
    # IDW 3 trạm gần nhất (cục bộ hơn)
    order = np.argsort(dd, 1)[:, :3]
    rr = np.take_along_axis(np.where(ok, R, np.nan), order, 1); ww = 1 / np.take_along_axis(dd, order, 1) ** 2
    with np.errstate(invalid="ignore"):
        out["idw3"] = np.nansum(rr * ww, 1) / np.nansum(np.where(np.isnan(rr), 0, ww), 1)
    if wd is not None:                                              # trạm đầu gió
        brg = (np.degrees(np.arctan2(sy - ty, sx - tx)) + 360) % 360  # hướng từ điểm tới trạm (0 = bắc; sx=bắc, sy=đông)
        wt = np.clip(np.cos(np.radians(brg[None, :] - wd[:, None])), 0, None) / (1 + d / 10)
        wt = np.where(ok, wt, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["upwind"] = np.where(wt.sum(1) > .05, (Rz * wt).sum(1) / wt.sum(1), np.nan)
    return out


def build():
    P, st = dataset.panel()
    W = dataset.weather()
    R = P.rolling(24, min_periods=18).mean()
    idx = R.loc[PERIOD[0]:PERIOD[1]].index[::STEP]
    R = R.loc[idx]
    st = st[st.sid.isin(R.columns[R.notna().sum() > 300])].reset_index(drop=True)
    R = R[st.sid]
    sx, sy = _xy(st.lat, st.lon)
    cx, cy = _xy(*config.CENTER)
    rows = []
    for i, s in st.iterrows():
        y = R[s.sid].to_numpy()
        others = R.drop(columns=s.sid).to_numpy()
        m = np.ones(len(st), bool); m[i] = False
        w = W[s.point_id].reindex(idx)
        f = neighbour_features(others, sx[m], sy[m], sx[i], sy[i], w.wind_direction_10m.to_numpy())
        df = pd.DataFrame(f)
        df["y"] = y; df["sid"] = s.sid; df["time"] = idx
        df["dist_center"] = np.hypot(sx[i] - cx, sy[i] - cy)
        df["lat"] = s.lat; df["lon"] = s.lon
        wblh = W[s.point_id].boundary_layer_height.rolling(24, min_periods=12).mean().reindex(idx)
        df["w_blh24"] = wblh.to_numpy()
        for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
            df[f"w_{c}"] = w[c].to_numpy()
        df["doy_sin"] = np.sin(2 * np.pi * idx.day_of_year / 365.25); df["doy_cos"] = np.cos(2 * np.pi * idx.day_of_year / 365.25)
        rows.append(df)
    X = pd.concat(rows, ignore_index=True)
    return X[X.y.notna() & X.idw.notna() & (X.net_n >= 3)].reset_index(drop=True), st


FEATS = ["idw", "idw3", "net_mean", "net_std", "net_n", "near", "near2", "upwind", "w_blh24", "w_temperature_2m", "w_relative_humidity_2m", "w_wind_speed_10m", "doy_sin", "doy_cos"]
# không dùng lat/lon, khoảng cách tới tâm, khoảng cách trạm gần nhất: các đặc trưng gần như cố định theo trạm khiến
# mô hình nhớ từng trạm (chỉ ~13 trạm để học) → hỏng khi gặp trạm mới (đã thử: MAE 13–14, kém IDW 9.75)


def run():
    X, st = build()
    rng = np.random.default_rng(0)
    sids = rng.permutation(st.sid.to_numpy())
    groups = {s: k % K for k, s in enumerate(sids)}
    X["fold"] = X.sid.map(groups)
    X["hgb"] = np.nan
    for k in range(K):
        tr, te = X.fold != k, X.fold == k
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X.loc[tr, FEATS], np.log1p(X.y[tr]) - np.log1p(X.idw[tr]))     # học phần hiệu chỉnh so với IDW
        X.loc[te, "hgb"] = np.expm1(m.predict(X.loc[te, FEATS]) + np.log1p(X.idw[te]))
    full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                         l2_regularization=1.0, random_state=0).fit(X[FEATS], np.log1p(X.y) - np.log1p(X.idw))
    joblib.dump((full, FEATS, "log1p(y)-log1p(idw)"), config.OUT / "spatial_model.joblib")
    meth = {"near": "Trạm gần nhất", "net_mean": "Trung bình mạng", "idw": "Nội suy IDW", "hgb": "Mô hình (đề tài)"}
    res = {"giai_doan": PERIOD, "so_tram": int(X.sid.nunique()), "so_dong": len(X), "tong": {}, "theo_tram": {}}
    for k, name in meth.items():
        e = (X[k] - X.y).abs()
        res["tong"][k] = {"ten": name, "mae": float(e.mean()),
                          "mae_tb_tram": float(e.groupby(X.sid).mean().mean()),
                          "r2": float(1 - ((X[k] - X.y) ** 2).sum() / ((X.y - X.y.mean()) ** 2).sum())}
    names = st.set_index("sid").ten
    for s, g in X.groupby("sid"):
        res["theo_tram"][str(s)] = {"ten": names[s], "n": len(g), "near_d": float(g.near_d.median()),
                                    **{k: float((g[k] - g.y).abs().mean()) for k in meth}}
    (config.OUT / "spatial_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"Giấu trạm: {res['so_tram']} trạm, {res['so_dong']} dòng, {PERIOD[0]} → {PERIOD[1]}")
    print(f"{'cách':22} {'MAE':>6} {'MAE TB/trạm':>12} {'R²':>6}")
    for k, v in res["tong"].items():
        print(f"{v['ten']:22} {v['mae']:6.2f} {v['mae_tb_tram']:12.2f} {v['r2']:6.2f}")
    worst = sorted(res["theo_tram"].values(), key=lambda v: -v["hgb"])[:5]
    print("trạm khó nhất:", "; ".join(f"{v['ten']} {v['hgb']:.1f} (IDW {v['idw']:.1f}, trạm gần nhất {v['near_d']:.0f} km)"
                                      for v in worst))


if __name__ == "__main__":
    run()
