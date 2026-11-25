"""Huấn luyện + chấm điểm. .venv/bin/python -m pm25.train

Chia theo thời gian: train < 2024-07-01 ≤ val < 2025-01-01 ≤ test.
Mỗi tầm dự báo h ∈ {24, 48, 72} một mô hình HistGradientBoosting, học log(1+y), gộp mọi trạm.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config, dataset

VAL_START, TEST_START = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")
NET = ["net_mean", "net_n", "upwind", "upwind_m24", "own_minus_net"]
DROP = {"sid", "time", "y", "persist", "cams"}


def cols(X, variant):
    c = [k for k in X.columns if k not in DROP]
    if variant == "own":
        c = [k for k in c if k not in NET]
    if variant == "nofut":
        c = [k for k in c if not k.startswith("f_")]
    return c


def fit(X, c, seed=0):
    tr = X[X.time < VAL_START]; va = X[(X.time >= VAL_START) & (X.time < TEST_START)]
    # dừng sớm theo tập val tách theo thời gian (không dùng mẫu ngẫu nhiên → tránh rò rỉ giữa các giờ liền nhau)
    best, best_it = 1e9, 0
    m = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.04, max_leaf_nodes=48, min_samples_leaf=60,
                                      l2_regularization=1.0, early_stopping=False, random_state=seed, warm_start=True)
    ytr, yva = np.log1p(tr.y), va.y.to_numpy()
    for it in range(100, 1501, 100):
        m.set_params(max_iter=it); m.fit(tr[c], ytr)
        e = np.mean(np.abs(np.expm1(m.predict(va[c])) - yva))
        if e < best - 0.01:
            best, best_it = e, it
        elif it - best_it >= 300:
            break
    m = HistGradientBoostingRegressor(max_iter=best_it, learning_rate=0.04, max_leaf_nodes=48, min_samples_leaf=60,
                                      l2_regularization=1.0, early_stopping=False, random_state=seed)
    m.fit(pd.concat([tr, va])[c], np.log1p(pd.concat([tr, va]).y))
    return m, best_it, best


def metrics(y, p, thr=config.UNHEALTHY):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    obs, pred = y >= thr, p >= thr
    hit = int((obs & pred).sum()); miss = int((obs & ~pred).sum()); fa = int((~obs & pred).sum())
    return {"n": int(ok.sum()), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "bias": float(np.mean(p - y)),
            "canh_bao": {"nguong": thr, "so_ngay_vuot": int(obs.sum()),
                         "bat_duoc": hit / (hit + miss) if hit + miss else None,
                         "bao_nham": fa / (hit + fa) if hit + fa else None,
                         "csi": hit / (hit + miss + fa) if hit + miss + fa else None}}


METHODS = ["giu_nguyen", "theo_mua", "cams", "cams_hc", "hgb_own_forecast", "hgb_full_forecast", "hgb_nofut_era5", "hgb_full_era5"]


def boot(P, a, b, n=1000, seed=0):
    """Δ MAE = MAE(a) − MAE(b), khoảng tin cậy 95% bằng bootstrap lấy lại theo ngày."""
    day = (P.time + pd.Timedelta(hours=7)).dt.floor("D")
    g = pd.DataFrame({"d": day, "ea": (P[a] - P.y).abs(), "eb": (P[b] - P.y).abs()}).groupby("d").agg(["sum", "count"])
    sa, sb, cnt = g[("ea", "sum")].to_numpy(), g[("eb", "sum")].to_numpy(), g[("ea", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (n, len(sa)))
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    return {"delta": float((sa.sum() - sb.sum()) / cnt.sum()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5))}


def climatology(X):
    """Trung bình nhiều năm theo tuần trong năm (giờ VN) từ tập train — mốc 'theo mùa'."""
    tr = X[X.time < VAL_START]
    wk = lambda t: ((t + pd.Timedelta(hours=7)).dt.isocalendar().week.clip(upper=52)).astype(int)
    tab = tr.groupby(wk(tr.time)).y.mean()
    return wk(X.time).map(tab).to_numpy()


def run():
    t0 = time.time()
    res, preds = {}, []
    for h in config.HORIZONS:
        X = dataset.build(h, "era5")
        X = X[X.y.notna() & X.pm_m24.notna()].reset_index(drop=True)
        X["clim"] = climatology(X)
        out = {}
        for variant in ("full", "own", "nofut"):
            c = cols(X.drop(columns="clim"), variant)
            m, it, vmae = fit(X, c)
            joblib.dump((m, c), config.OUT / f"model_h{h}_{variant}.joblib")
            out[variant] = (m, c, it, vmae)
            print(f"h={h} {variant}: {it} cây, val MAE {vmae:.2f}, {time.time() - t0:.0f}s", flush=True)

        # tập test chung: mọi cách chấm trên CÙNG các dòng (có dự báo thời tiết đã phát hành và có CAMS)
        XF = dataset.build(h, "forecast")
        XF = XF[(XF.time >= TEST_START) & XF.y.notna() & XF.pm_m24.notna() & XF.f_temp.notna()
                & XF.cams.notna()].reset_index(drop=True)
        T = X[X.time >= TEST_START].merge(XF[["time", "sid"]], on=["time", "sid"]).reset_index(drop=True)
        XF = XF.merge(T[["time", "sid"]], on=["time", "sid"]).reset_index(drop=True)
        T = T.sort_values(["sid", "time"]).reset_index(drop=True)
        XF = XF.sort_values(["sid", "time"]).reset_index(drop=True)
        assert (T.time.values == XF.time.values).all()
        P = pd.DataFrame({"time": T.time, "sid": T.sid, "h": h, "y": T.y, "giu_nguyen": T.persist,
                          "theo_mua": T.clim, "cams": T.cams})
        # CAMS hiệu chỉnh tuyến tính trên log, khớp bằng dữ liệu trước test (8/2022 → 12/2024)
        hc = X[(X.time < TEST_START) & X.cams.notna()]
        b1, b0 = np.polyfit(np.log1p(hc.cams), np.log1p(hc.y), 1)
        P["cams_hc"] = np.expm1(b0 + b1 * np.log1p(T.cams))
        r_hc = {"he_so": [float(b0), float(b1)], "so_dong": len(hc)}
        for variant in ("full", "own"):
            m, c = out[variant][:2]
            P[f"hgb_{variant}_forecast"] = np.expm1(m.predict(XF[c]))
        for variant in ("full", "nofut"):
            m, c = out[variant][:2]
            P[f"hgb_{variant}_era5"] = np.expm1(m.predict(T[c]))
        r = {"so_dong": {"train": int((X.time < VAL_START).sum()),
                         "val": int(((X.time >= VAL_START) & (X.time < TEST_START)).sum()), "test": len(P)},
             "test_tu": str(P.time.min()), "test_den": str(P.time.max())}
        for k in METHODS:
            r[k] = metrics(P.y, P[k])
        r["cay"] = {v: out[v][2] for v in out}
        r["cams_hc_fit"] = r_hc
        # khoảng tin cậy 95% cho mức giảm MAE (bootstrap theo khối ngày, vì các giờ liền nhau tương quan)
        r["ci"] = {f"{a}_vs_{b}": boot(P, a, b) for a, b in [("hgb_full_forecast", "hgb_own_forecast"),
                                                              ("hgb_full_forecast", "giu_nguyen"),
                                                              ("hgb_full_forecast", "cams_hc")]}
        r["theo_tram"] = {str(s): {k: float(np.mean(np.abs(g[k] - g.y))) for k in METHODS} | {"n": len(g)}
                          for s, g in P.groupby("sid")}
        mo = (P.time + pd.Timedelta(hours=7)).dt.month
        r["mua_dong"] = {k: metrics(P.y[mo.isin([11, 12, 1, 2, 3])], P[k][mo.isin([11, 12, 1, 2, 3])]) for k in METHODS}
        preds.append(P)
        res[str(h)] = r
    pd.concat(preds).to_parquet(config.OUT / "test_predictions.parquet", index=False)
    (config.OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    report(res)


def report(res=None):
    res = res or json.loads((config.OUT / "results.json").read_text())
    rows = [("giu_nguyen", "Giữ nguyên TB 24h vừa qua"), ("theo_mua", "Trung bình theo mùa"), ("cams", "CAMS (châu Âu), gốc"), ("cams_hc", "CAMS đã hiệu chỉnh lệch"),
            ("hgb_own_forecast", "Mô hình, chỉ trạm mình"), ("hgb_full_forecast", "Mô hình + mạng trạm"),
            ("hgb_nofut_era5", "Mô hình, không thời tiết tương lai"),
            ("hgb_full_era5", "Mô hình + thời tiết thật (cận trên)")]
    for h, r in res.items():
        print(f"\n== ngày thứ {int(h) // 24} (h={h}) — test {r['so_dong']['test']} dòng, {r.get('test_tu','')[:10]} → {r.get('test_den','')[:10]} ==")
        print(f"{'cách':38} {'MAE':>6} {'RMSE':>6} {'R²':>6}  bắt được  báo nhầm  CSI  (ngưỡng {config.UNHEALTHY})")
        for k, name in rows:
            if k not in r:
                continue
            m, cb = r[k], r[k]["canh_bao"]
            f = lambda v: f"{v:6.0%}" if v is not None else "     -"
            print(f"{name:38} {m['mae']:6.2f} {m['rmse']:6.2f} {m['r2']:6.2f}  {f(cb['bat_duoc'])}    {f(cb['bao_nham'])}  "
                  f"{(cb['csi'] or 0):.2f}  lệch {m['bias']:+.1f}")
        for k, v in r.get("ci", {}).items():
            print(f"  ΔMAE {k}: {v['delta']:+.2f} (95%: {v['lo']:+.2f} … {v['hi']:+.2f})")


if __name__ == "__main__":
    import sys
    report() if "--report" in sys.argv else run()
