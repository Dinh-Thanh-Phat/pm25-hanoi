"""Phân tích lỗi (bước 3) trên tập test chung: sai ở trạm nào, tháng nào, ngày ô nhiễm có bị bỏ sót không.

  .venv/bin/python -m pm25.analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, dataset

M = "hgb_full_forecast"
BASE = ["giu_nguyen", "cams_hc", M]


def run():
    tp = pd.read_parquet(config.OUT / "test_predictions.parquet")
    names = dataset.stations()[["sid", "ten"]].drop_duplicates("sid").set_index("sid").ten
    W = dataset.weather()
    st = dataset.stations().drop_duplicates("sid").set_index("sid")
    tp["thang"] = (tp.time + pd.Timedelta(hours=7)).dt.to_period("M").astype(str)
    out = {}
    for h, g in tp.groupby("h"):
        r = {}
        # theo trạm
        r["theo_tram"] = {names.get(s, str(s)): {"n": len(x), **{m: round(float((x[m] - x.y).abs().mean()), 2) for m in BASE},
                                                 "tb_thuc": round(float(x.y.mean()), 1)}
                          for s, x in g.groupby("sid")}
        # theo tháng
        r["theo_thang"] = {k: {"n": len(x), "tb_thuc": round(float(x.y.mean()), 1),
                               **{m: round(float((x[m] - x.y).abs().mean()), 2) for m in BASE},
                               "bias": round(float((x[M] - x.y).mean()), 2)}
                           for k, x in g.groupby("thang")}
        # theo mức ô nhiễm thật
        bins = [0, 25, 50, 80, 150, 1e9]
        lab = ["<25", "25–50", "50–80", "80–150", ">150"]
        g = g.assign(muc=pd.cut(g.y, bins, labels=lab, right=False))
        r["theo_muc"] = {str(k): {"n": len(x), "mae": round(float((x[M] - x.y).abs().mean()), 2),
                                  "bias": round(float((x[M] - x.y).mean()), 2)}
                         for k, x in g.groupby("muc", observed=True)}
        # ngày vượt ngưỡng bị bỏ sót: điều kiện thời tiết lúc đó khác gì ngày bắt được
        ev = g[g.y >= config.UNHEALTHY].copy()
        ev["sot"] = ev[M] < config.UNHEALTHY
        wx = []
        for row in ev.itertuples():
            w = W.get(st.point_id.get(row.sid))
            if w is None:
                wx.append((np.nan, np.nan, np.nan)); continue
            a, b = row.time + pd.Timedelta(hours=h - 24), row.time + pd.Timedelta(hours=h)
            ww = w.loc[a:b]
            wx.append((ww.wind_speed_10m.mean(), ww.boundary_layer_height.mean(), ww.precipitation.sum()))
        ev[["gio", "blh", "mua"]] = wx
        r["vuot_nguong"] = {"tong": len(ev), "bo_sot": int(ev.sot.sum()),
                            "bo_sot_thieu_tb": round(float((ev.y - ev[M])[ev.sot].mean()), 1) if ev.sot.any() else None,
                            **{f"{k}_{c}": round(float(ev.loc[ev.sot == (k == 'sot'), c].mean()), 2)
                               for k in ("sot", "bat") for c in ("gio", "blh", "mua", "y")}}
        out[str(int(h))] = r
    (config.OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for h, r in out.items():
        print(f"\n===== dự báo +{int(h) // 24} ngày =====")
        print("Theo trạm (MAE giữ nguyên / CAMS hc / mô hình):")
        for k, v in sorted(r["theo_tram"].items(), key=lambda kv: -kv[1][M]):
            print(f"  {k:32} n={v['n']:5} thực TB {v['tb_thuc']:5}  {v['giu_nguyen']:6} {v['cams_hc']:6} {v[M]:6}")
        print("Theo tháng (thực TB, MAE mô hình, lệch):")
        for k, v in r["theo_thang"].items():
            print(f"  {k}  n={v['n']:4} thực {v['tb_thuc']:5}  MAE {v[M]:6}  lệch {v['bias']:+.1f}")
        print("Theo mức thật:", {k: (v["n"], v["mae"], v["bias"]) for k, v in r["theo_muc"].items()})
        print("Vượt 50:", r["vuot_nguong"])


if __name__ == "__main__":
    run()
