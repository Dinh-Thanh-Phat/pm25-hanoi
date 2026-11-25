import numpy as np
import pandas as pd

from pm25 import dataset, train
from types import SimpleNamespace as NS


def test_clean_removes_stuck_and_spikes():
    s = pd.Series([30, 31, 32, 400, 33, 34] + [50] * 7 + [40, -5, 900], dtype=float)
    c = dataset.clean(s)
    assert np.isnan(c[3])                       # gai đơn lẻ
    assert c[6:13].isna().all()                 # kẹt 7 giờ cùng giá trị
    assert np.isnan(c[14]) and np.isnan(c[15])  # âm, quá lớn
    assert c[0] == 30 and c[13] == 40


def test_bearing():
    a = NS(lat=21.0, lon=105.8)
    brg, d = dataset._bearing_km(a, NS(lat=21.1, lon=105.8))       # ngay phía bắc ~11 km
    assert abs(brg) < 1 and 10.5 < d < 11.6
    brg, _ = dataset._bearing_km(a, NS(lat=21.0, lon=105.9))       # phía đông
    assert abs(brg - 90) < 1


def test_target_is_future_24h_mean():
    idx = pd.date_range("2025-01-01", periods=96, freq="h")
    P = pd.DataFrame({1: np.arange(96, dtype=float)}, index=idx)
    y = dataset.target(P, 24)
    assert y.iloc[0, 0] == np.mean(np.arange(1, 25))              # trung bình giờ 1..24 sau t=0
    assert np.isnan(y.iloc[-1, 0])


def test_metrics_warning():
    m = train.metrics([10, 60, 70, 20], [12, 58, 40, 60])
    assert m["canh_bao"]["so_ngay_vuot"] == 2
    assert m["canh_bao"]["bat_duoc"] == 0.5 and m["canh_bao"]["bao_nham"] == 0.5
    assert abs(m["mae"] - (2 + 2 + 30 + 40) / 4) < 1e-9
