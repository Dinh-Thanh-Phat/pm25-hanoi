"""Giao diện + API. Chạy: .venv/bin/uvicorn pm25.api:app --port 8200"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, dataset, history, live

app = FastAPI(title="Bụi mịn Hà Nội — dự báo 3 ngày")
STATIC = Path(__file__).parent / "static"
app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})   # luôn lấy bản mới, tránh thấy CSS cũ


@app.get("/api/hien-tai")
def now(lam_moi: bool = False):
    try:
        return live.get(force=lam_moi)
    except Exception as e:                      # mạng hỏng: trả bản cũ nếu có, kèm cảnh báo
        if live.CACHE.exists():
            d = json.loads(live.CACHE.read_text())
            d["loi"] = f"Không lấy được số mới ({type(e).__name__}); đang hiện bản lúc {d['luc'][:16]}"
            return d
        raise HTTPException(503, f"Không lấy được số đo hoặc dự báo thời tiết: {e}")


@app.get("/api/ket-qua")
def results():
    f = config.OUT / "results.json"
    if not f.exists():
        raise HTTPException(404, "Chưa chạy huấn luyện (python -m pm25.train)")
    r = json.loads(f.read_text())
    st = dataset.stations().set_index("sid").ten.to_dict()
    extra = {k: json.loads((config.OUT / f).read_text()) for k, f in
             (("phan_tich", "analysis.json"), ("khong_gian", "spatial_results.json")) if (config.OUT / f).exists()}
    return {"ket_qua": r, "ten_tram": {str(k): v for k, v in st.items()}, "nguong": config.UNHEALTHY, **extra}


@app.get("/api/lich-su")
def hist():
    return history.get()


@app.get("/api/tram")
def stations_meta():
    """Mọi trạm (kể cả đã ngừng): khoảng thời gian có số, số giờ đo sạch."""
    global _ST
    if _ST is None:
        P, st = dataset.panel()
        _ST = [{"sid": int(r.sid), "ten": r.ten, "lat": float(r.lat), "lon": float(r.lon), "so_gio": int(r.so_gio),
                "tu": f"{P[r.sid].first_valid_index():%Y-%m-%d}", "den": f"{P[r.sid].last_valid_index():%Y-%m-%d}"}
               for r in st.itertuples()]
    return _ST


_ST = None


@app.get("/api/doi-chieu")
def backtest(sid: int, h: int = 24, ngay: int = 90):
    """Dự báo (bản trung thực) so với thực đo trên tập test, lấy 1 dòng/ngày lúc 7h sáng."""
    f = config.OUT / "test_predictions.parquet"
    if not f.exists():
        raise HTTPException(404, "Chưa có kết quả test")
    p = pd.read_parquet(f)
    p = p[(p.sid == sid) & (p.h == h)]
    if p.empty:
        raise HTTPException(404, "Trạm này không có dữ liệu test")
    vn = p.time + pd.Timedelta(hours=7)
    p = p[vn.dt.hour == 7].tail(ngay)
    tgt = p.time + pd.Timedelta(hours=7 + h - 12)
    return [{"ngay": d.strftime("%Y-%m-%d"), "thuc_do": round(float(y), 1),
             "du_bao": round(float(m), 1), "cams": None if np.isnan(c) else round(float(c), 1)}
            for d, y, m, c in zip(tgt, p.y, p.hgb_full_forecast, p.cams)]
