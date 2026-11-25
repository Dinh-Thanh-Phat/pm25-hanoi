"""Mốc học sâu (bước 4): GRU đọc chuỗi 72 giờ (bụi + thời tiết) + thời tiết tương lai → dự báo cùng lúc 3 ngày.

Cùng cách chia thời gian và cùng tập test chung với train.py để so công bằng.
  .venv/bin/python -m pm25.deep
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config, dataset, train

SEQ = 72
HOR = config.HORIZONS
SEQ_W = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure",
         "boundary_layer_height", "cloud_cover"]


def _future(src: pd.DataFrame, h: int) -> pd.DataFrame:
    """Thời tiết trung bình của khoảng (t+h-24, t+h] — cùng cách với dataset.features."""
    win = src.rolling(24, min_periods=12)
    m = win.mean().shift(-h)
    ang = np.radians(src.wind_direction_10m)
    return pd.DataFrame({f"f{h}_temp": m.temperature_2m, f"f{h}_rh": m.relative_humidity_2m, f"f{h}_ws": m.wind_speed_10m,
                         f"f{h}_pres": m.surface_pressure, f"f{h}_cloud": m.cloud_cover,
                         f"f{h}_rain": src.precipitation.rolling(24, min_periods=12).sum().shift(-h),
                         f"f{h}_wds": np.sin(ang).rolling(24, min_periods=12).mean().shift(-h),
                         f"f{h}_wdc": np.cos(ang).rolling(24, min_periods=12).mean().shift(-h)})


def arrays(fut: str):
    """Trả (seq[N,72,F], stat[N,G], y[N,3], time, sid) cho mọi (trạm, t) có đủ dữ liệu."""
    P, st = dataset.panel()
    W = dataset.weather()
    FC = dataset.forecast() if fut == "forecast" else None
    Y = {h: dataset.target(P, h) for h in HOR}
    S, G, YY, T, SID = [], [], [], [], []
    for s in st.itertuples():
        if s.point_id not in W or (fut == "forecast" and s.point_id not in FC):
            continue                                   # điểm thời tiết chưa tải xong
        w = W[s.point_id].reindex(P.index)
        pm = P[s.sid]
        lp = np.log1p(pm).to_numpy()
        feats = np.column_stack([np.nan_to_num(lp), (~np.isnan(lp)).astype(float)] +
                                [w[c].to_numpy() for c in SEQ_W] +
                                [np.sin(np.radians(w.wind_direction_10m)), np.cos(np.radians(w.wind_direction_10m))])
        stat = []
        for h in HOR:
            if fut == "era5":
                src = w[dataset.FUT_VARS]
            else:
                d = min(3, math.ceil(h / 24))
                f = FC.get(s.point_id)
                src = (f[[f"{v}_previous_day{d}" for v in dataset.FUT_VARS]].set_axis(dataset.FUT_VARS, axis=1)
                       .reindex(P.index))
            stat.append(_future(src, h))
        stat = pd.concat(stat, axis=1)
        tt = P.index + pd.Timedelta(hours=7)
        stat["doy_s"] = np.sin(2 * np.pi * tt.dayofyear / 365.25); stat["doy_c"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)
        stat["hour"] = tt.hour / 23
        y = np.column_stack([Y[h][s.sid].to_numpy() for h in HOR])
        m24 = pm.rolling(24, min_periods=12).mean().to_numpy()
        ok = (~np.isnan(y).any(1)) & ~np.isnan(m24) & ~stat.isna().any(axis=1).to_numpy()
        ok[:SEQ] = False
        ii = np.where(ok)[0]
        if not len(ii):
            continue
        win = np.lib.stride_tricks.sliding_window_view(feats, SEQ, axis=0)   # (T-71, F, 72)
        S.append(np.transpose(win[ii - SEQ + 1], (0, 2, 1)).astype(np.float32))
        G.append(stat.to_numpy()[ii].astype(np.float32))
        YY.append(y[ii].astype(np.float32)); T.append(P.index[ii]); SID.append(np.full(len(ii), s.sid))
    return (np.concatenate(S), np.concatenate(G), np.concatenate(YY), pd.DatetimeIndex(np.concatenate(T)),
            np.concatenate(SID))


class Net(nn.Module):
    def __init__(self, f, g, hid=64):
        super().__init__()
        self.gru = nn.GRU(f, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hid + g, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, len(HOR)))

    def forward(self, s, g):
        _, h = self.gru(s)
        return self.head(torch.cat([h[-1], g], 1))


def run(epochs=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    S, G, Y, T, SID = arrays("era5")
    tr = T < train.VAL_START; va = (T >= train.VAL_START) & (T < train.TEST_START)
    tr_idx = np.where(tr)[0][::2]                    # lấy mẫu mỗi 2 giờ cho nhanh
    # chuẩn hoá theo tập train
    smu, ssd = np.nanmean(S[tr_idx], (0, 1)), np.nanstd(S[tr_idx], (0, 1)) + 1e-6
    smu[:2], ssd[:2] = 0, 1                          # log-bụi và mặt nạ để nguyên
    gmu, gsd = G[tr_idx].mean(0), G[tr_idx].std(0) + 1e-6
    norm = lambda s, g: (torch.tensor(np.nan_to_num((s - smu) / ssd)), torch.tensor(np.nan_to_num((g - gmu) / gsd)))
    net = Net(S.shape[2], G.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Yl = np.log1p(Y)
    def predict(s, g):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(s), 4096):
                a, b = norm(s[i:i + 4096], g[i:i + 4096])
                out.append(net(a.to(dev), b.to(dev)).cpu().numpy())
        return np.expm1(np.concatenate(out))
    best, state, bad = 1e9, None, 0
    va_idx = np.where(va)[0]
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            a, g = norm(S[b], G[b])
            loss = nn.functional.l1_loss(net(a.to(dev), g.to(dev)), torch.tensor(Yl[b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        mae = np.abs(predict(S[va_idx], G[va_idx]) - Y[va_idx]).mean(0)
        print(f"epoch {ep + 1}: val MAE {np.round(mae, 2).tolist()}", flush=True)
        if mae.mean() < best - 0.02:
            best, bad = mae.mean(), 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    torch.save({"state": state, "smu": smu, "ssd": ssd, "gmu": gmu, "gsd": gsd, "f": S.shape[2], "g": G.shape[1]},
               config.OUT / "gru.pt")


def evaluate():
    ck = torch.load(config.OUT / "gru.pt", weights_only=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net(ck["f"], ck["g"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    smu, ssd, gmu, gsd = ck["smu"], ck["ssd"], ck["gmu"], ck["gsd"]
    def predict(s, g):
        out = []
        with torch.no_grad():
            for i in range(0, len(s), 4096):
                a = torch.tensor(np.nan_to_num((s[i:i + 4096] - smu) / ssd)).to(dev)
                b = torch.tensor(np.nan_to_num((g[i:i + 4096] - gmu) / gsd)).to(dev)
                out.append(net(a, b).cpu().numpy())
        return np.expm1(np.concatenate(out))
    # test trung thực: thời tiết tương lai lấy từ dự báo đã phát hành, trên đúng các dòng test chung của train.py
    S2, G2, Y2, T2, SID2 = arrays("forecast")
    p = predict(S2, G2)
    key = pd.DataFrame({"time": T2, "sid": SID2})
    tp = pd.read_parquet(config.OUT / "test_predictions.parquet")
    res = json.loads((config.OUT / "results.json").read_text())
    for k, h in enumerate(HOR):
        g = key.assign(gru=p[:, k])
        q = tp[tp.h == h].merge(g, on=["time", "sid"], how="left")
        n_miss = int(q.gru.isna().sum())
        q = q[q.gru.notna()]
        r = res[str(h)]
        r["gru_forecast"] = train.metrics(q.y, q.gru) | {"thieu": n_miss}
        r["ci"]["hgb_full_forecast_vs_gru_forecast"] = train.boot(q, "hgb_full_forecast", "gru")
        print(f"h={h}: GRU MAE {r['gru_forecast']['mae']:.2f} vs HGB {train.metrics(q.y, q.hgb_full_forecast)['mae']:.2f}"
              f" trên {len(q)} dòng (thiếu {n_miss})", flush=True)
    (config.OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import sys
    if "--eval" not in sys.argv:
        run()
    evaluate()
