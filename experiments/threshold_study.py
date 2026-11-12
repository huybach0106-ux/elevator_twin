"""
Nghiên cứu NGƯỠNG – trả lời "khi nào nên tăng thang / chuyển chế độ" (mục 1 & 7 tài liệu).

(A) D3 thuần: N khách 8 → G, chỉ k cabin Express G–8 hoạt động (k = 1, 2, 3; A1 tắt để so với công thức).
    So sánh thời gian giải tỏa MÔ PHỎNG với CÔNG THỨC mục 4:
        lượt = ceil(N / c);  vòng/cabin = ceil(lượt / k);  T ≈ vòng × T_vòng
        T_vòng = 2 × 8 tầng × 2,2 s + 2 × (t_cửa + c × t_khách)
    -> kiểm chứng mô hình + chỉ ra điểm lợi ích giảm dần khi thêm cabin.
(B) U1 vào tiết 1: quét N, so sánh P2 luôn chia vùng (Express/Zone) và P2 luôn phục vụ chung (pooling)
    -> điểm cắt = ngưỡng chuyển chế độ.
Chạy:  python -m experiments.threshold_study
"""
from __future__ import annotations

import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.baseline import BaselineDispatcher  # noqa: E402
from config import DOOR_TIME, F, TRANSFER_TIME, TRAVEL_TIME_PER_FLOOR  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.metrics import compute  # noqa: E402
from simulation.passenger_generator import _times_burst, build  # noqa: E402
from simulation.runner import run_one  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(1, 6)


def pure_s1(n, seed):
    rng = np.random.default_rng([seed, 801, n])
    return build([(float(t), F(8), F("G"), False) for t in _times_burst(rng, n, spread=20, mean_delay=10)])


def analytic(n, k, cap):
    t_round = 2 * 8 * TRAVEL_TIME_PER_FLOOR + 2 * (DOOR_TIME + cap * TRANSFER_TIME)
    trips = math.ceil(n / cap)
    rounds = math.ceil(trips / k)
    # vòng cuối không cần quay lại tầng 8: trừ nửa vòng + một lần dừng tầng 8
    return rounds * t_round - (t_round / 2 - (DOOR_TIME + cap * TRANSFER_TIME)) + 8 * TRAVEL_TIME_PER_FLOOR


def _s1_job(a):
    n, k, cap, a1, seed = a
    disabled = [c for c in ("A4", "A5", "A6")[k:]] + ([] if a1 else ["A1"]) + ["A2", "A3"]
    sim = Simulation(pure_s1(n, seed), BaselineDispatcher(disabled), capacity=cap, horizon=3000,
                     seed=seed, record=False).run()
    m = compute(sim)
    last_arr = max(p.t_arr for p in sim.passengers)
    return (n, k, cap, a1), (sim.clearance or float("nan")) - 0.0, m["AWT"], last_arr


def _zone_job(a):
    scen, n, mode, seed = a
    params = {"pool_util": 0.0} if mode == "zoning" else {"pool_util": 99.0}
    _, m = run_one(scen, n, 15, "P2", seed, params=params)
    return (scen, n, mode), m["AWT"], m["P95"], m["energy"]


def main():
    out = {"s1": [], "s3": []}
    with Pool(os.cpu_count() or 1) as pool:
        jobs = [(n, k, cap, a1, sd) for n in range(25, 326, 25) for k in (1, 2, 3) for cap in (10, 15)
                for a1 in (False, True) for sd in SEEDS]
        res = {}
        for key, clr, awt, _ in pool.map(_s1_job, jobs, chunksize=4):
            res.setdefault(key, []).append((clr, awt))
        for (n, k, cap, a1), v in sorted(res.items()):
            out["s1"].append({"N": n, "k": k, "capacity": cap, "with_A1": a1,
                              "clear_sim": float(np.nanmean([x[0] for x in v])),
                              "awt_sim": float(np.mean([x[1] for x in v])),
                              "clear_formula": analytic(n, k, cap) if not a1 else None})
        # (B) chia vùng hay phục vụ chung – mở rộng tới tải rất cao, cả cao điểm lên (U1) và xuống (D2)
        ranges = {"U1": range(60, 961, 90), "D2": range(90, 991, 90)}
        jobs = [(sc, n, mode, sd) for sc, rg in ranges.items() for n in rg for mode in ("zoning", "pooling") for sd in SEEDS]
        res = {}
        for key, awt, p95, en in pool.map(_zone_job, jobs, chunksize=2):
            res.setdefault(key, []).append((awt, p95, en))
        out["d2"] = []
        for (sc, n, mode), v in sorted(res.items()):
            row = {"N": n, "mode": mode, "AWT": float(np.mean([x[0] for x in v])),
                   "P95": float(np.mean([x[1] for x in v])), "energy": float(np.mean([x[2] for x in v]))}
            out["s3" if sc == "U1" else "d2"].append(row)
    with open(os.path.join(HERE, "results", "threshold.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("D3 (cap 15, không A1): N | k=1 sim/công thức | k=2 | k=3")
    for n in range(25, 326, 50):
        cells = []
        for k in (1, 2, 3):
            r = next(x for x in out["s1"] if x["N"] == n and x["k"] == k and x["capacity"] == 15 and not x["with_A1"])
            cells.append(f"{r['clear_sim']:.0f}/{r['clear_formula']:.0f}")
        print(n, " | ".join(cells))
    for key, lab in (("s3", "U1"), ("d2", "D2")):
        print(f"{lab}: N | AWT chia vùng | AWT phục vụ chung")
        for n in sorted({r["N"] for r in out[key]}):
            z = next(x for x in out[key] if x["N"] == n and x["mode"] == "zoning")
            p = next(x for x in out[key] if x["N"] == n and x["mode"] == "pooling")
            print(" ", n, f"{z['AWT']:.1f}", f"{p['AWT']:.1f}", "<- chia vùng thắng" if z["AWT"] < p["AWT"] else "")


if __name__ == "__main__":
    main()
