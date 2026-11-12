"""
Web Digital Twin – Flask.

    python web_app.py            ->  http://localhost:5000

API:
  GET  /api/meta                    danh sách kịch bản, cấu hình, tầng, cabin
  POST /api/run                     {scenario, level, capacity, config, seed, sensor_noise}
                                    -> chạy P0 và cấu hình được chọn trên CÙNG danh sách hành khách
  GET  /api/results                 kết quả thực nghiệm (results/*.json) nếu đã chạy
  POST /api/chat                    {message, history, context} -> trợ lý LLM (có công cụ chạy mô phỏng)
  GET  /api/llm                     nhà cung cấp LLM đang dùng
"""
from __future__ import annotations

import json
import os
import time

from flask import Flask, Response, jsonify, request

from config import IDX
from llm.assistant import chat as llm_chat
from llm.assistant import provider_name
from simulation.export import meta, run_payload
from simulation.runner import run_one
from simulation.scenarios import SCENARIOS, n_for

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env(path=os.path.join(HERE, ".env")):
    """Đọc file .env (KEY=value mỗi dòng) để khỏi phải đặt biến môi trường trong terminal."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
app = Flask(__name__)


@app.get("/")
def index():
    with open(os.path.join(HERE, "templates", "index.html"), encoding="utf-8") as fh:
        return Response(wrap_page(fh.read()), mimetype="text/html")


def wrap_page(body: str) -> str:
    return ('<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>body{margin:0}</style></head><body>' + body + "</body></html>")


@app.get("/api/meta")
def api_meta():
    return jsonify(meta())


@app.post("/api/run")
def api_run():
    q = request.get_json(force=True) or {}
    s = q.get("scenario", "U1")
    if s not in SCENARIOS:
        return jsonify(error=f"Không có kịch bản {s}"), 400
    level = q.get("level", "mid")
    if isinstance(level, str) and level.isdigit():
        level = int(level)
    cap = int(q.get("capacity", 15))
    seed = int(q.get("seed", 42))
    cfg = q.get("config", "P2")
    noise = bool(q.get("sensor_noise", q.get("camera_noise", True)))
    t0 = time.time()
    runs = {}
    for c in ("P0", cfg):
        sim, m = run_one(s, level, cap, c, seed, sensor_noise=noise, record=True)
        runs[c] = run_payload(sim, m)
    return jsonify(scenario=s, level=level, N=n_for(s, level), capacity=cap, seed=seed, compare=cfg,
                   runs=runs, elapsed=round(time.time() - t0, 2))


def _clean(o):
    """NaN/inf không hợp lệ trong JSON chuẩn (trình duyệt từ chối) -> null."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


@app.get("/api/results")
def api_results():
    out = {}
    for name in ("summary", "threshold", "classifier_report", "optuna_report", "pareto", "sensitivity", "stats"):
        p = os.path.join(HERE, "results", name + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh)
    if "summary" in out:
        out["summary"] = {"summary": out["summary"]["summary"], "ablation": out["summary"]["ablation"]}
    if "stats" in out:
        out["stats"] = {"summary": out["stats"]["summary"], "n_tests": out["stats"]["n_tests"]}
    return jsonify(_clean(out))


# ------------------------------------------------------------------- What-if
@app.post("/api/whatif")
def api_whatif():
    """Thử giả định trên mô phỏng: cabin hỏng, lượng khách tăng/giảm, sức chứa – so P0 và cấu hình tối ưu."""
    from simulation.whatif import run_whatif
    q = request.get_json(force=True) or {}
    if q.get("scenario", "U1") not in SCENARIOS:
        return jsonify(error="Không có kịch bản này"), 400
    return jsonify(_clean(run_whatif(q.get("scenario", "U1"), q.get("level", "mid"), int(q.get("capacity", 15)),
                                     float(q.get("demand", 1.0)), q.get("fault_car") or None,
                                     float(q.get("fault_start_min", 2)), float(q.get("fault_minutes", 10)),
                                     q.get("config", "P2"), int(q.get("seed", 42)))))


# ------------------------------------------------------------- Thời gian thực
from realtime.live import LiveManager  # noqa: E402

LIVE = LiveManager()


@app.post("/api/live/start")
def api_live_start():
    q = request.get_json(force=True) or {}
    info = LIVE.start(q.get("clock") or None, float(q.get("speed", 10)), q.get("control", "closed"),
                      int(q.get("capacity", 15)))
    return jsonify(info)


@app.post("/api/live/stop")
def api_live_stop():
    LIVE.stop()
    return jsonify(ok=True)


@app.get("/api/live/state")
def api_live_state():
    return jsonify(_clean(LIVE.snapshot()))


@app.post("/api/live/cmd")
def api_live_cmd():
    """Lệnh từ dashboard: control (closed/advisory), fault {car, minutes}, speed."""
    q = request.get_json(force=True) or {}
    kind = q.get("kind")
    if kind == "control":
        ok = LIVE.publish("cmd/control", {"mode": q.get("mode", "closed")})
    elif kind == "fault":
        ok = LIVE.publish("cmd/fault", {"car": q.get("car", "A1"), "minutes": float(q.get("minutes", 10))})
    elif kind == "speed":
        ok = LIVE.publish("cmd/speed", {"speed": float(q.get("speed", 10))})
    else:
        return jsonify(error="Lệnh không hợp lệ"), 400
    return jsonify(ok=ok)


@app.post("/api/live/report")
def api_live_report():
    """Báo cáo hàng chờ của người dùng (form QR): {floor: "8", count: 20}."""
    q = request.get_json(force=True, silent=True) or request.form.to_dict()
    floor = str(q.get("floor", "")).strip()
    if floor not in IDX:
        return jsonify(error="Tầng không hợp lệ"), 400
    count = max(0, min(300, int(float(q.get("count", 0)))))
    ok = LIVE.publish("report", {"floor": floor, "count": count})
    return jsonify(ok=ok, running=LIVE.running)


@app.get("/bao-cao")
def report_form():
    """Trang form trên điện thoại (dán mã QR ở sảnh thang) để báo số người đang chờ."""
    with open(os.path.join(HERE, "templates", "report.html"), encoding="utf-8") as fh:
        return Response(wrap_page(fh.read()), mimetype="text/html")


@app.get("/docs")
def docs_page():
    with open(os.path.join(HERE, "templates", "docs.html"), encoding="utf-8") as fh:
        return Response(wrap_page(fh.read()), mimetype="text/html")


# ------------------------------------------------------------------------ LLM
@app.get("/api/llm")
def api_llm():
    prov, model = provider_name()
    return jsonify(provider=prov, model=model)


@app.post("/api/chat")
def api_chat():
    q = request.get_json(force=True) or {}
    msg = (q.get("message") or "").strip()
    if not msg:
        return jsonify(error="Câu hỏi trống"), 400
    return jsonify(_clean(llm_chat(msg, q.get("history") or [], q.get("context") or {}, live=LIVE)))


if __name__ == "__main__":
    prov, model = provider_name()
    print(f"[Digital Twin] LLM: {prov} ({model})")
    if prov == "offline":
        print("[Digital Twin] Chưa thấy khóa LLM: tạo file .env cạnh web_app.py với dòng GROQ_API_KEY=gsk_...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
