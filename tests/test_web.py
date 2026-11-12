"""Kiểm thử API web: chạy mô phỏng và trợ lý (chế độ ngoại tuyến)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(k, None)

import web_app  # noqa: E402

web_app.provider_name = lambda: ("offline", "luật + mô phỏng")
import llm.assistant as A  # noqa: E402

A.provider_name = lambda: ("offline", "luật + mô phỏng")
app = web_app.app


def test_chat_offline_runs_real_simulation():
    c = app.test_client()
    r = c.post("/api/chat", json={"message": "Nếu D3 có 200 người thì sao?"}).get_json()
    assert r["provider"] == "offline"
    assert r["tools"][0]["tool"] == "compare_configs"
    assert "D3" in r["answer"] and "200" in r["answer"]


def test_run_endpoint_returns_both_configs():
    c = app.test_client()
    r = c.post("/api/run", json={"scenario": "D3", "level": "low", "capacity": 15, "config": "P2"}).get_json()
    assert set(r["runs"]) == {"P0", "P2"} and r["runs"]["P2"]["kpi"]["AWT"] is not None
