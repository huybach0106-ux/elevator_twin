"""
Xuất dashboard thành MỘT file HTML tĩnh (không cần máy chủ) với dữ liệu chạy sẵn:
8 tình huống × P0–P3, mức tải trung bình, sức chứa 15, seed 42 + kết quả thực nghiệm.
    python export_static.py            -> dist/digital_twin.html
"""
from __future__ import annotations

import json
import os

from simulation.export import meta, run_payload
from simulation.runner import run_one
from simulation.scenarios import SCENARIOS, n_for

HERE = os.path.dirname(os.path.abspath(__file__))


def build(seed=42, level="mid", capacity=15):
    runs = {}
    for s in SCENARIOS:
        runs[s] = {"N": n_for(s, level)}
        for cfg in ("P0", "P1", "P2", "P3", "P4"):
            sim, m = run_one(s, level, capacity, cfg, seed, record=True)
            runs[s][cfg] = run_payload(sim, m)
    results = {}
    for name in ("summary", "threshold", "classifier_report", "optuna_report", "pareto", "sensitivity", "stats"):
        p = os.path.join(HERE, "results", name + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                results[name] = json.load(fh)
    if "summary" in results:
        results["summary"] = {"summary": results["summary"]["summary"],
                              "ablation": [{k: r[k] for k in ("variant", "scenario", "AWT")} for r in results["summary"]["ablation"]]}
    if "stats" in results:
        results["stats"] = {"summary": results["stats"]["summary"], "n_tests": results["stats"]["n_tests"]}
    pre = {"meta": meta(), "runs": runs, "results": results, "default": {"scenario": "U1"}}
    with open(os.path.join(HERE, "templates", "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    data = json.dumps(pre, ensure_ascii=False, separators=(",", ":"))
    return page.replace("/*__PRELOAD__*/", "window.PRELOAD = " + data + ";", 1)


def main():
    from web_app import wrap_page
    body = build()
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    with open(os.path.join(HERE, "dist", "digital_twin_body.html"), "w", encoding="utf-8") as fh:
        fh.write(body)          # bản không khung (dùng để đăng artifact)
    with open(os.path.join(HERE, "dist", "digital_twin.html"), "w", encoding="utf-8") as fh:
        fh.write(wrap_page(body))
    print("Đã ghi dist/digital_twin.html", round(len(body) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
