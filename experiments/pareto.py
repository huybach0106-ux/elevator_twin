"""
A1 · Đánh đổi thời gian chờ – năng lượng (đường Pareto).

Quét hệ số sử dụng mục tiêu u_target của chế độ eco (P2 chỉ giữ đủ số cabin cần):
  u_target nhỏ -> nhiều cabin hoạt động -> chờ ít, tốn năng lượng; lớn -> ngược lại.
So cùng hành khách với P0, P1, P2, P3. Điểm "không bị trội" (không cấu hình nào vừa chờ ít hơn vừa tốn ít hơn) = Pareto.
Chạy:  python -m experiments.pareto
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.runner import run_one  # noqa: E402
from simulation.scenarios import SCENARIOS  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(1, 6)
U_VALUES = [0.35, 0.5, 0.7, 0.9, 1.1]


def variants():
    v = [("P0", "P0", None), ("P1", "P1", None), ("P2", "P2", None), ("P3", "P3", None)]
    v += [(f"eco u={u}", "P2", {"u_target": u, "w_energy": 3.0}) for u in U_VALUES]
    return v


def _job(a):
    s, name, cfg, par, seed = a
    _, m = run_one(s, "high", 15, cfg, seed, params=par)
    return s, name, m["AWT"], m["P95"], m["energy"]


def pareto_front(points):
    front = []
    for p in points:
        if not any(q["AWT"] <= p["AWT"] and q["energy"] <= p["energy"] and
                   (q["AWT"] < p["AWT"] or q["energy"] < p["energy"]) for q in points):
            front.append(p["name"])
    return front


def main():
    jobs = [(s, n, c, p, sd) for s in SCENARIOS for n, c, p in variants() for sd in SEEDS]
    with Pool(os.cpu_count() or 1) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    agg = {}
    for s, n, awt, p95, en in res:
        agg.setdefault((s, n), []).append((awt, p95, en))
    out = {"scenarios": {}, "groups": {}, "u_values": U_VALUES}
    for s in SCENARIOS:
        pts = []
        for n, _, _ in variants():
            v = np.array(agg[(s, n)])
            pts.append({"name": n, "AWT": float(v[:, 0].mean()), "P95": float(v[:, 1].mean()), "energy": float(v[:, 2].mean())})
        front = pareto_front(pts)
        for p in pts:
            p["pareto"] = p["name"] in front
        out["scenarios"][s] = pts
    # tổng hợp theo nhóm: tỷ lệ so với P0
    for g in ("Up-Peak", "Down-Peak", "Inter-Floor", "Off-Peak"):
        rows = {}
        for s, sc in SCENARIOS.items():
            if sc["group"] != g:
                continue
            base = next(p for p in out["scenarios"][s] if p["name"] == "P0")
            for p in out["scenarios"][s]:
                r = rows.setdefault(p["name"], [[], []])
                r[0].append(p["AWT"] / base["AWT"])
                r[1].append(p["energy"] / base["energy"])
        out["groups"][g] = {n: {"AWT_ratio": float(np.mean(a)), "energy_ratio": float(np.mean(e))} for n, (a, e) in rows.items()}
    with open(os.path.join(HERE, "results", "pareto.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("Tỷ lệ so với P0 (AWT, năng lượng) theo nhóm:")
    for g, rows in out["groups"].items():
        print(" ", g, " | ".join(f"{n}: {r['AWT_ratio']:.2f}/{r['energy_ratio']:.2f}" for n, r in rows.items()))


if __name__ == "__main__":
    main()
