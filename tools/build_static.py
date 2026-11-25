"""Xuất bản tĩnh cho Vercel: trang + ảnh + JSON lấy từ chính các API FastAPI (số liệu y hệt bản chạy máy).

  .venv/bin/python tools/build_static.py          # → site/
  .venv/bin/python tools/build_static.py --live   # chỉ làm mới data/hien-tai.json (nhanh, cho lịch 30 phút)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pm25 import api  # noqa: E402

SITE = ROOT / "site"
DATA = SITE / "data"


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def build(live_only: bool = False) -> dict:
    c = TestClient(api.app)                      # không chạy startup → không bật lịch tải bù
    r = c.get("/api/hien-tai")
    if r.status_code != 200:
        raise SystemExit(f"hien-tai lỗi {r.status_code}: {r.text[:200]}")
    _save(DATA / "hien-tai.json", r.json())
    if live_only:
        return {"hien-tai": r.json()["luc"]}
    html = (ROOT / "pm25/static/index.html").read_text()
    html = html.replace('<html lang="vi">', '<html lang="vi" data-static="1">', 1)
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html)
    shutil.copytree(ROOT / "pm25/static/img", SITE / "img", dirs_exist_ok=True)
    out = {}
    for name in ("ket-qua", "lich-su", "tram"):
        r = c.get(f"/api/{name}")
        if r.status_code == 200:
            _save(DATA / f"{name}.json", r.json())
            out[name] = "ok"
    n = 0
    for sid in {int(k) for h in ("24", "48", "72") for k in json.loads((DATA / "ket-qua.json").read_text())["ket_qua"][h]["theo_tram"]}:
        for h in (24, 48, 72):
            r = c.get("/api/doi-chieu", params={"sid": sid, "h": h})
            if r.status_code == 200:
                _save(DATA / "doi-chieu" / f"{sid}_{h}.json", r.json())
                n += 1
    out["doi-chieu"] = n
    (SITE / ".vercelignore").write_text(".env*\n.vercel\n.gitignore\n")   # token Vercel CLI không bao giờ lên mạng
    (SITE / "vercel.json").write_text(json.dumps({"headers": [{"source": "/data/(.*)", "headers": [
        {"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]}]}, indent=1))
    return out


if __name__ == "__main__":
    print(build(live_only="--live" in sys.argv))
    print("kích thước:", round(sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) / 1e6, 1), "MB")
