"""
Trợ lý vận hành dùng LLM cho Digital Twin.

LLM được cấp "công cụ" để tự lấy số liệu thay vì bịa:
  - run_simulation(scenario, level, capacity, config, seed)  -> chạy mô phỏng thật, trả KPI
  - compare_configs(scenario, level, capacity)              -> chạy P0–P3 trên cùng hành khách
  - scenario_info(scenario)                                 -> mô tả kịch bản
  - experiment_results(scenario)                            -> kết quả nhiều seed (results/summary.json)
  - what_if(scenario, fault_car, fault_minutes, demand)      -> giả định: cabin hỏng, lượng khách khác
  - live_status()                                           -> trạng thái twin đang chạy trực tiếp (nếu có)

Nhà cung cấp (đặt biến môi trường trước khi chạy web_app.py):
  ANTHROPIC_API_KEY  [+ LLM_MODEL, mặc định claude-haiku-5-5]
  GROQ_API_KEY       [+ GROQ_MODEL, mặc định openai/gpt-oss-120b; model không còn -> tự chọn model khác] – Groq, miễn phí có giới hạn
  OPENAI_API_KEY     [+ OPENAI_BASE_URL, OPENAI_MODEL] – mọi API tương thích OpenAI (OpenAI, Gemini, Ollama…)
Không có khóa nào -> chế độ ngoại tuyến: trả lời bằng luật + vẫn chạy mô phỏng thật cho câu hỏi "nếu…".
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from simulation.runner import CONFIGS, run_one
from simulation.scenarios import LEVELS, SCENARIOS, n_for

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KPI_SHOW = ["AWT", "P95", "max_wait", "backlog", "throughput", "bypass", "stops", "energy",
            "clearance", "worst_floor_awt", "role_changes"]
KPI_VI = {"AWT": "chờ TB (s)", "P95": "chờ P95 (s)", "max_wait": "chờ lâu nhất (s)", "backlog": "backlog",
          "throughput": "thông lượng", "bypass": "bypass", "stops": "số dừng", "energy": "năng lượng (EU)",
          "clearance": "giải tỏa (s)", "worst_floor_awt": "AWT tầng tệ nhất (s)", "role_changes": "đổi vai"}

SYSTEM = """Bạn là trợ lý vận hành của Digital Twin hệ thống 6 thang máy (A1–A6) trong tòa nhà học 13 tầng (B2, B1, G, 1–10).
Hiện trạng P0: A1 dừng mọi tầng; A2, A3 dừng B2, B1, G, 5; A4–A6 dừng G, 8. Tầng 1–4, 6, 7, 9, 10 chỉ có A1 phục vụ.
Cấu hình: P0 hiện trạng; P1 phân vai tĩnh theo thời khóa biểu; P2 điều phối động (cảm biến cân tải + hồng ngoại ở cửa + AI nhận diện chế độ + phân vai Express/Zone/Local);
P3 thấp điểm/standby; P4 = P2 tiết kiệm năng lượng (chỉ giữ đủ số cabin cần, phạt năng lượng). 12 kịch bản: U1–U3 (Up-Peak), D1–D3 (Down-Peak), I1–I3 (Inter-Floor), O1–O3 (Off-Peak).
Mức tải: low, mid, high (hoặc số người cụ thể). Sức chứa hữu dụng 10 hoặc 15.
Có chế độ trực tiếp: tòa nhà ảo gửi dữ liệu cảm biến cho twin qua MQTT, twin gửi lệnh phân vai ngược lại.
Khi được yêu cầu viết báo cáo vận hành: gọi live_status (nếu đang chạy trực tiếp) và experiment_results, viết ngắn gọn theo mục:
tình hình hiện tại, tầng/khung giờ cần chú ý, đề xuất phân vai, rủi ro.
Quy tắc: luôn dùng công cụ để lấy số liệu, không tự bịa con số. Trả lời bằng tiếng Việt, ngắn gọn, nêu số cụ thể và đơn vị.
Nhắc rằng đây là kết quả mô phỏng với tham số giả định khi đưa ra kết luận. Năng lượng là đơn vị tương đối EU."""

TOOLS = [
    {"name": "run_simulation", "description": "Chạy một lượt mô phỏng và trả về KPI.",
     "input_schema": {"type": "object", "properties": {
         "scenario": {"type": "string", "enum": list(SCENARIOS)},
         "level": {"type": "string", "description": "low | mid | high hoặc số người, ví dụ '450'"},
         "capacity": {"type": "integer", "enum": [10, 15]},
         "config": {"type": "string", "enum": list(CONFIGS)},
         "seed": {"type": "integer"}}, "required": ["scenario", "config"]}},
    {"name": "compare_configs", "description": "Chạy P0, P1, P2, P3 trên CÙNG danh sách hành khách và so sánh KPI.",
     "input_schema": {"type": "object", "properties": {
         "scenario": {"type": "string", "enum": list(SCENARIOS)},
         "level": {"type": "string"}, "capacity": {"type": "integer", "enum": [10, 15]}},
         "required": ["scenario"]}},
    {"name": "scenario_info", "description": "Mô tả một kịch bản (nhóm, khung giờ, luồng khách, số người).",
     "input_schema": {"type": "object", "properties": {"scenario": {"type": "string", "enum": list(SCENARIOS)}},
                      "required": ["scenario"]}},
    {"name": "what_if", "description": "Thử giả định trên mô phỏng: một cabin hỏng trong vài phút và/hoặc lượng khách nhân hệ số; so P0 với cấu hình tối ưu, bình thường vs có giả định.",
     "input_schema": {"type": "object", "properties": {
         "scenario": {"type": "string", "enum": list(SCENARIOS)},
         "fault_car": {"type": "string", "enum": ["A1", "A2", "A3", "A4", "A5", "A6"]},
         "fault_minutes": {"type": "number"}, "demand": {"type": "number", "description": "hệ số lượng khách, ví dụ 1.5"},
         "config": {"type": "string", "enum": list(CONFIGS)}}, "required": ["scenario"]}},
    {"name": "live_status", "description": "Trạng thái twin đang chạy trực tiếp: giờ, chế độ AI, phân vai đề xuất, ước lượng hàng chờ, dự báo 10 phút, lớp sắp vào/tan.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "experiment_results", "description": "Kết quả thí nghiệm nhiều seed (trung bình ± KTC 95 %) của một kịch bản.",
     "input_schema": {"type": "object", "properties": {"scenario": {"type": "string", "enum": list(SCENARIOS)},
                                                       "capacity": {"type": "integer", "enum": [10, 15]}},
                      "required": ["scenario"]}},
]


# --------------------------------------------------------------------------- tools
def _level(v):
    if v is None or v == "":
        return "mid"
    v = str(v).strip()
    return int(v) if v.isdigit() else (v if v in LEVELS else "mid")


def _kpi(m):
    return {k: (None if m[k] is None else round(float(m[k]), 1)) for k in KPI_SHOW}


def tool_run_simulation(scenario, config="P2", level="mid", capacity=15, seed=42, **_):
    lv = _level(level)
    _, m = run_one(scenario, lv, int(capacity or 15), config, int(seed or 42))
    return {"scenario": scenario, "N": n_for(scenario, lv), "capacity": capacity, "config": config, "kpi": _kpi(m),
            "A1_pax_share": round(m["cars"]["A1"]["pax"] / max(1, sum(c["pax"] for c in m["cars"].values())), 2)}


def tool_compare_configs(scenario, level="mid", capacity=15, **_):
    lv = _level(level)
    out = {"scenario": scenario, "N": n_for(scenario, lv), "capacity": capacity, "configs": {}}
    for c in CONFIGS:
        _, m = run_one(scenario, lv, int(capacity or 15), c, 42)
        out["configs"][c] = _kpi(m)
    return out


def tool_scenario_info(scenario, **_):
    s = SCENARIOS[scenario]
    return {"scenario": scenario, "group": s["group"], "name": s["name"], "time": s["time"],
            "levels": dict(zip(LEVELS, s["levels"])), "flow": s["note"]}


def tool_experiment_results(scenario, capacity=15, **_):
    p = os.path.join(HERE, "results", "summary.json")
    if not os.path.exists(p):
        return {"error": "Chưa có kết quả thí nghiệm – chạy python -m experiments.run_experiments"}
    with open(p, encoding="utf-8") as fh:
        rows = json.load(fh)["summary"]
    out = {}
    for r in rows:
        if r["scenario"] == scenario and r["capacity"] == int(capacity or 15):
            out.setdefault(r["level"], {})[r["config"]] = {
                "AWT": round(r["AWT"], 1), "AWT_ci95": round(r["AWT_ci"], 1), "P95": round(r["P95"], 0),
                "energy": round(r["energy"], 0), "backlog": round(r["backlog"], 1), "n_seeds": r["n_seeds"]}
    return {"scenario": scenario, "capacity": capacity, "by_level": out}


def tool_what_if(scenario, fault_car=None, fault_minutes=10, demand=1.0, config="P2", **_):
    from simulation.whatif import run_whatif
    r = run_whatif(scenario, "mid", 15, float(demand or 1.0), fault_car, 2.0, float(fault_minutes or 10), config, 42,
                   series=False)
    return {k: v for k, v in r.items() if k != "runs"} | {"kpi": {k: v["kpi"] for k, v in r["runs"].items()}}


def make_executor(live=None):
    def live_status(**_):
        if live is None or not live.running:
            return {"running": False, "note": "Chưa bật chế độ trực tiếp trên dashboard."}
        snap = live.snapshot()
        tw = snap.get("twin") or {}
        est = tw.get("est_waiting", [])
        from config import fname
        hot = sorted(((round(v, 1), fname(i)) for i, v in enumerate(est) if v >= 1), reverse=True)[:4]
        return {"running": True, "clock": tw.get("clock"), "mode": tw.get("mode_vi"), "control": tw.get("control"),
                "recommended_roles": tw.get("recommended"), "waiting_estimate_top": hot,
                "lookahead_10min": tw.get("lookahead"), "upcoming_timetable": tw.get("upcoming", [])[:5],
                "recent_commands": tw.get("cmd_log", [])[-4:], "user_reports": tw.get("reports", [])[-3:],
                "twin_accuracy": snap.get("accuracy")}

    table = {"run_simulation": tool_run_simulation, "compare_configs": tool_compare_configs,
             "scenario_info": tool_scenario_info, "experiment_results": tool_experiment_results,
             "what_if": tool_what_if, "live_status": live_status}

    def run(name, args):
        try:
            if "scenario" in args and args["scenario"] not in SCENARIOS:
                return {"error": f"Không có kịch bản {args['scenario']}"}
            return table[name](**args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
    return run


# --------------------------------------------------------------------- providers
class LLMError(RuntimeError):
    pass


def _post(url, body, headers, timeout=60):
    # Groq/Cloudflare chặn User-Agent mặc định "Python-urllib" (HTTP 403, mã 1010) -> đặt User-Agent riêng
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "User-Agent": "elevator-digital-twin/1.0", **headers})
    try:
        with _opener().open(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        # Máy có cấu hình proxy (Windows / biến HTTP(S)_PROXY) nhưng proxy không chạy -> WinError 10061.
        # Thử lại kết nối thẳng, bỏ qua proxy.
        if urllib.request.getproxies() and os.environ.get("LLM_PROXY", "auto") == "auto":
            _NO_PROXY["on"] = True
            with _opener().open(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        raise


_NO_PROXY = {"on": os.environ.get("LLM_PROXY", "auto") == "off"}


def _opener():
    """LLM_PROXY=auto (mặc định): dùng proxy hệ thống, lỗi thì đi thẳng · off: luôn đi thẳng · http://host:port: proxy riêng."""
    px = os.environ.get("LLM_PROXY", "auto")
    if px not in ("auto", "off"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": px, "https": px}))
    if _NO_PROXY["on"]:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _context_text(ctx):
    if not ctx:
        return ""
    return "\nNgười dùng đang xem trên dashboard: " + json.dumps(ctx, ensure_ascii=False)


def chat_anthropic(message, history, ctx, execute, max_steps=4):
    key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-8:]] + [{"role": "user", "content": message}]
    calls = []
    for _ in range(max_steps):
        res = _post("https://api.anthropic.com/v1/messages",
                    {"model": model, "max_tokens": 1200, "system": SYSTEM + _context_text(ctx), "tools": TOOLS, "messages": msgs},
                    {"x-api-key": key, "anthropic-version": "2023-06-01"})
        blocks = res.get("content", [])
        if res.get("stop_reason") != "tool_use":
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), calls, model
        msgs.append({"role": "assistant", "content": blocks})
        results = []
        for b in blocks:
            if b.get("type") == "tool_use":
                out = execute(b["name"], b.get("input") or {})
                calls.append({"tool": b["name"], "input": b.get("input"), "output": out})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": json.dumps(out, ensure_ascii=False)})
        msgs.append({"role": "user", "content": results})
    return "Đã gọi công cụ nhiều lần nhưng chưa kết luận được; hãy hỏi cụ thể hơn.", calls, model


def _openai_cfg():
    if os.environ.get("GROQ_API_KEY"):
        return (os.environ["GROQ_API_KEY"], "https://api.groq.com/openai/v1",
                os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    return (os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))


PREFERRED = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
             "llama-3.1-8b-instant"]
_MODEL_CACHE = {}


def _pick_model(base, key):
    """Model mặc định không còn -> hỏi danh sách model của nhà cung cấp và chọn model hỗ trợ gọi công cụ."""
    if base in _MODEL_CACHE:
        return _MODEL_CACHE[base]
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}",
                                                              "User-Agent": "elevator-digital-twin/1.0"})
    with _opener().open(req, timeout=20) as r:
        ids = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
    pick = next((m for m in PREFERRED if m in ids), None) or next(
        (m for m in ids if not any(x in m for x in ("whisper", "tts", "guard", "embed", "vision"))), None)
    if not pick:
        raise LLMError("Không tìm thấy model phù hợp trong tài khoản")
    _MODEL_CACHE[base] = pick
    return pick


def chat_openai(message, history, ctx, execute, max_steps=4):
    key, base, model = _openai_cfg()
    model = _MODEL_CACHE.get(base, model)
    try:
        return _chat_openai(message, history, ctx, execute, key, base, model, max_steps)
    except LLMError as e:
        if "model" not in str(e) or not ("404" in str(e) or "400" in str(e)):
            raise
        alt = _pick_model(base, key)          # tự đổi sang model còn hoạt động
        return _chat_openai(message, history, ctx, execute, key, base, alt, max_steps)


def _chat_openai(message, history, ctx, execute, key, base, model, max_steps):
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    msgs = [{"role": "system", "content": SYSTEM + _context_text(ctx)}] + \
           [{"role": h["role"], "content": h["content"]} for h in history[-8:]] + [{"role": "user", "content": message}]
    calls = []
    for _ in range(max_steps):
        res = _post(base + "/chat/completions", {"model": model, "messages": msgs, "tools": tools},
                    {"Authorization": f"Bearer {key}"})
        msg = res["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return msg.get("content") or "", calls, model
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tcs})
        for tc in tcs:
            args = json.loads(tc["function"].get("arguments") or "{}")
            out = execute(tc["function"]["name"], args)
            calls.append({"tool": tc["function"]["name"], "input": args, "output": out})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, ensure_ascii=False)})
    return "Đã gọi công cụ nhiều lần nhưng chưa kết luận được; hãy hỏi cụ thể hơn.", calls, model


# ----------------------------------------------------------------------- offline
def _find_scenario(text, ctx):
    m = re.search(r"\b([UDIO][1-3])\b", text.upper())
    if m:
        return m.group(1)
    return (ctx or {}).get("scenario") or "U1"


def _fmt_cmp(a, b, keys=("AWT", "P95", "backlog", "energy")):
    parts = []
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        d = "" if not x else f" ({(y - x) / x * 100:+.0f} %)"
        parts.append(f"{KPI_VI[k]}: {x:g} → {y:g}{d}")
    return "; ".join(parts)


def chat_offline(message, ctx, execute):
    """Không có khóa API: hiểu ý định bằng luật đơn giản, số liệu vẫn lấy từ mô phỏng thật."""
    t = message.lower()
    calls = []
    sc = _find_scenario(message, ctx)

    def call(name, args):
        out = execute(name, args)
        calls.append({"tool": name, "input": args, "output": out})
        return out

    num = re.search(r"(\d{2,4})\s*(người|ng|sinh viên|sv|khách)", t)
    cap = 10 if re.search(r"sức chứa\s*10|10 người/chuyến|cap\s*10", t) else 15
    if "báo cáo" in t or "tình hình" in t or "trực tiếp" in t:
        st = call("live_status", {})
        if not st.get("running"):
            return ("Chưa bật chế độ trực tiếp. Bấm Bắt đầu ở mục Thời gian thực, rồi hỏi lại để nhận báo cáo vận hành."), calls
        hot = ", ".join(f"tầng {f} ~{v:g} người" for v, f in st["waiting_estimate_top"]) or "không tầng nào đông"
        la = st.get("lookahead_10min") or {}
        nxt = "; ".join(f"{e['time']} tầng {e['floor'] - 2 if isinstance(e['floor'], int) else e['floor']} {e['kind']} (~{e['n']} người)"
                        for e in st["upcoming_timetable"][:3]) or "không có lớp nào"
        fc = (f" Dự báo 10 phút: giữ nguyên phân vai AWT ≈ {la['keep']['AWT']} s, theo twin ≈ {la['twin']['AWT']} s."
              if la.get("keep") and la.get("twin") else "")
        return (f"Báo cáo lúc {st['clock']}: chế độ {st['mode']}, điều khiển {'khép kín' if st['control'] == 'closed' else 'tư vấn'}. "
                f"Hàng chờ ước lượng: {hot}. Sắp tới: {nxt}.{fc}"), calls
    fault = re.search(r"\b(A[1-6])\b", message.upper())
    if fault and ("hỏng" in t or "sự cố" in t or "dừng" in t):
        mins = re.search(r"(\d+)\s*phút", t)
        out = call("what_if", {"scenario": sc, "fault_car": fault.group(1), "fault_minutes": float(mins.group(1)) if mins else 10})
        k = out["kpi"]
        return (f"{sc}: nếu {fault.group(1)} hỏng {out['fault_minutes']:g} phút – P0: AWT {k['P0_normal']['AWT']:g} → {k['P0_whatif']['AWT']:g} s; "
                f"P2: {k['P2_normal']['AWT']:g} → {k['P2_whatif']['AWT']:g} s (P2 tự phân vai lại cho các cabin còn lại)."), calls
    if num or "nếu" in t or "what" in t or "tăng" in t:
        lv = num.group(1) if num else ("high" if "cao" in t else "low" if "thấp" in t else "mid")
        out = call("compare_configs", {"scenario": sc, "level": lv, "capacity": cap})
        c = out["configs"]
        best = min(c, key=lambda k: c[k]["AWT"])
        return (f"Mô phỏng {sc} với {out['N']} người, sức chứa {cap}: "
                f"P0 → P2: {_fmt_cmp(c['P0'], c['P2'])}. Cấu hình có AWT thấp nhất là {best} "
                f"({c[best]['AWT']:g} s). Đây là kết quả 1 seed, tham số giả định."), calls
    if ("kịch bản" in t or "là gì" in t) and not any(w in t for w in ("vì sao", "tại sao", "tốt hơn", "so sánh")):
        out = call("scenario_info", {"scenario": sc})
        return f"{sc} · {out['name']} ({out['group']}, {out['time']}): {out['flow']}.", calls
    out = call("experiment_results", {"scenario": sc, "capacity": cap})
    if "error" in out:
        return out["error"], calls
    hi = out["by_level"].get("high") or next(iter(out["by_level"].values()))
    p0, p2 = hi["P0"], hi["P2"]
    why = ("Ở P0 các tầng 1–4, 6, 7, 9, 10 chỉ có A1 phục vụ nên A1 quá tải trong khi A2–A6 rảnh; "
           "P2 giao vai cho từng cabin và gom năng lực chung nên khách chờ ít hơn.")
    return (f"{sc}, tải cao, sức chứa {cap} (trung bình {p0['n_seeds']} seed): AWT P0 {p0['AWT']} s ± {p0['AWT_ci95']} → "
            f"P2 {p2['AWT']} s ± {p2['AWT_ci95']}; năng lượng {p0['energy']:g} → {p2['energy']:g} EU. {why} "
            "Muốn trả lời bằng LLM, đặt GROQ_API_KEY (hoặc ANTHROPIC_API_KEY / OPENAI_API_KEY) trong file .env rồi chạy lại web."), calls


def provider_name():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ.get("LLM_MODEL", "claude-haiku-5-5")
    if os.environ.get("GROQ_API_KEY"):
        return "groq", _MODEL_CACHE.get("https://api.groq.com/openai/v1", os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return "offline", "luật + mô phỏng"


def chat(message, history=None, ctx=None, live=None):
    execute = make_executor(live)
    prov, model = provider_name()
    try:
        if prov == "anthropic":
            ans, calls, model = chat_anthropic(message, history or [], ctx, execute)
        elif prov in ("groq", "openai"):
            ans, calls, model = chat_openai(message, history or [], ctx, execute)
        else:
            ans, calls = chat_offline(message, ctx, execute)
    except Exception as exc:  # noqa: BLE001  – lỗi mạng/khóa -> vẫn trả lời ngoại tuyến
        ans, calls = chat_offline(message, ctx, execute)
        ans = f"(Không gọi được {prov}: {exc}. Trả lời ngoại tuyến.) " + ans
        prov, model = "offline", "luật + mô phỏng"
    return {"answer": ans, "provider": prov, "model": model,
            "tools": [{"tool": c["tool"], "input": c["input"]} for c in calls]}
