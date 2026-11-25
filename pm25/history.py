"""Chuỗi PM2.5 trung bình ngày của cả Hà Nội (2016 → nay) cho giao diện: lịch nhiệt, khí hậu theo tháng."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, dataset

CACHE = config.OUT / "history.json"


def compute() -> dict:
    P, st = dataset.panel()
    vn = P.copy()
    vn.index = vn.index + pd.Timedelta(hours=7)              # ngày theo giờ Việt Nam
    city_h = vn.mean(axis=1)                                 # trung bình các trạm có số trong giờ đó
    n_h = vn.notna().sum(axis=1)
    day = city_h.resample("D")
    d = pd.DataFrame({"v": day.mean(), "gio": city_h.notna().resample("D").sum(), "tram": n_h.resample("D").max()})
    d = d[d.gio >= 18]
    d = d[d.index >= "2017-01-01"]
    m = d.assign(thang=d.index.month, nam=d.index.year)
    thang = [{"thang": int(k), "tb": round(float(g.v.mean()), 1), "vuot": round(float((g.v > config.UNHEALTHY).mean()), 3),
              "so_ngay": len(g)} for k, g in m.groupby("thang")]
    nam = [{"nam": int(k), "tb": round(float(g.v.mean()), 1), "vuot": int((g.v > config.UNHEALTHY).sum()),
            "so_ngay": len(g)} for k, g in m.groupby("nam")]
    return {"tu": d.index.min().strftime("%Y-%m-%d"), "den": d.index.max().strftime("%Y-%m-%d"),
            "ngay": [i.strftime("%Y-%m-%d") for i in d.index], "v": [round(float(x), 1) for x in d.v],
            "tram": [int(x) for x in d.tram], "thang": thang, "nam": nam,
            "so_tram": int(P.notna().any().sum()), "so_gio": int(P.notna().sum().sum()),
            "nguong": config.UNHEALTHY}


def get() -> dict:
    src = max((config.RAW / "pm25").glob("*.parquet"), key=lambda p: p.stat().st_mtime).stat().st_mtime
    if CACHE.exists() and CACHE.stat().st_mtime > src:
        return json.loads(CACHE.read_text())
    d = compute()
    CACHE.write_text(json.dumps(d, ensure_ascii=False))
    return d


if __name__ == "__main__":
    d = get()
    print(d["tu"], d["den"], len(d["v"]), "ngày;", d["so_tram"], "trạm;", d["so_gio"], "giờ đo")
    print("theo năm:", [(x["nam"], x["tb"], x["vuot"]) for x in d["nam"]])
    print("theo tháng:", [(x["thang"], x["tb"], x["vuot"]) for x in d["thang"]])
