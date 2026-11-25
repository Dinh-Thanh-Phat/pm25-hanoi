"""Giao diện + API. Chạy: .venv/bin/uvicorn pm25.api:app --port 8200"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, dataset, fetch, history, live

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


# ---------- tải bù số đo mỗi đêm (1:30 giờ máy) để phần lịch sử tự dài thêm ----------
UPDATE_LOG = config.OUT / "update.log"


def nightly_update() -> None:
    global _ST
    t0 = time.time()
    try:
        added = fetch.update_pm25()
        history.get()                                   # file số đo mới hơn bộ đệm → tự tính lại
        _ST = None
        msg = f"ok {sum(added.values())} giờ mới {added}"
    except Exception as e:
        msg = f"lỗi {type(e).__name__}: {e}"
    with UPDATE_LOG.open("a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} {msg} ({time.time() - t0:.0f}s)\n")


def _scheduler() -> None:
    newest = max((config.RAW / "pm25").glob("*.parquet"), key=lambda p: p.stat().st_mtime).stat().st_mtime
    if time.time() - newest > 20 * 3600:               # máy tắt qua đêm: bù ngay khi bật
        nightly_update()
    while True:
        now = datetime.now()
        nxt = now.replace(hour=1, minute=30, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        time.sleep((nxt - now).total_seconds())
        nightly_update()
        push_vercel(full=True)                          # lịch sử vừa dài thêm → xuất lại toàn bộ


# ---------- đẩy bản tĩnh lên Vercel mỗi 30 phút (chỉ khi site/ đã link với Vercel) ----------
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DEPLOY_LOG = config.OUT / "deploy.log"
_last_hash = None
_deploy_lock = threading.Lock()


def push_vercel(full: bool = False) -> None:
    global _last_hash
    if not (SITE / ".vercel").exists():
        return
    with _deploy_lock:
        t0, msg = time.time(), ""
        try:
            r = subprocess.run([str(ROOT / ".venv/bin/python"), "tools/build_static.py"] + ([] if full else ["--live"]),
                               cwd=ROOT, capture_output=True, text=True, timeout=600)
            if r.returncode:
                raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"build exit {r.returncode}")
            h = hashlib.md5((SITE / "data/hien-tai.json").read_bytes()).hexdigest()
            if h == _last_hash and not full:
                msg = "bỏ qua: số không đổi"
            else:
                d = subprocess.run(["npx", "-y", "vercel@latest", "deploy", "--prod", "--yes"], cwd=SITE,
                                   capture_output=True, text=True, timeout=600)
                if d.returncode:
                    raise RuntimeError((d.stderr or d.stdout).strip().splitlines()[-1])
                _last_hash = h
                msg = "đã deploy" + (" (toàn bộ)" if full else "")
        except Exception as e:
            msg = f"lỗi {type(e).__name__}: {e}"
        with DEPLOY_LOG.open("a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} {msg} ({time.time() - t0:.0f}s)\n")


def _vercel_loop() -> None:
    push_vercel(full=True)                              # vừa bật server: đẩy bản mới nhất
    while True:
        time.sleep(30 * 60)
        push_vercel()


@app.on_event("startup")
def _start_scheduler() -> None:
    threading.Thread(target=_scheduler, daemon=True, name="tai-bu-dem").start()
    threading.Thread(target=_vercel_loop, daemon=True, name="day-vercel").start()
