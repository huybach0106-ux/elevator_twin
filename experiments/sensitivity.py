"""
A2 · Phân tích độ nhạy: kết luận "P2 tốt hơn P0" có còn đúng khi tham số giả định sai ±20 %?

Thay đổi từng tham số một (giữ nguyên các tham số khác): thời gian chạy mỗi tầng, thời gian đóng/mở cửa,
thời gian mỗi khách lên/xuống, lượng khách. Mỗi mức chạy P0 và P2 trên cùng hành khách, 5 seed,
4 kịch bản đại diện 4 nhóm (U1, D1, I1, O1) ở mức tải trung bình.
Chạy:  python -m experiments.sensitivity
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from simulation.runner import run_one  # noqa: E402
from simulation.scenarios import n_for  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = ["U1", "D1", "I1", "O1"]
SEEDS = range(1, 6)
PARAMS = {
    "TRAVEL_TIME_PER_FLOOR": "Thời gian chạy 1 tầng",
    "DOOR_TIME": "Thời gian đóng + mở cửa",
    "TRANSFER_TIME": "Thời gian mỗi khách lên/xuống",
    "DEMAND": "Lượng khách",
}
FACTORS = [0.8, 1.0, 1.2]
BASE = {k: getattr(config, k) for k in PARAMS if k != "DEMAND"}


def _set(name, value):
    """Đổi hằng số ở MỌI module đã import nó (engine, planner, optimized…)."""
    for mod in list(sys.modules.values()):
        if mod is not None and hasattr(mod, name) and getattr(mod, "__name__", "").split(".")[0] in (
                "config", "simulation", "algorithms", "experiments"):
            try:
                setattr(mod, name, value)
            except (AttributeError, TypeError):
                pass


def _job(a):
    param, f, s, cfg, seed = a
    for k, v in BASE.items():
        _set(k, v)
    level = "mid"
    if param == "DEMAND":
        level = int(round(n_for(s, "mid") * f))
    elif f != 1.0:
        _set(param, BASE[param] * f)
    _, m = run_one(s, level, 15, cfg, seed)
    return param, f, s, cfg, seed, m["AWT"], m["energy"]


def main():
    jobs = [(p, f, s, c, sd) for p in PARAMS for f in FACTORS for s in SCEN for c in ("P0", "P2") for sd in SEEDS]
    with Pool(os.cpu_count() or 1) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    d = {}
    for p, f, s, c, sd, awt, en in res:
        d[(p, f, s, c, sd)] = (awt, en)
    out = {"params": PARAMS, "factors": FACTORS, "scenarios": SCEN, "rows": []}
    for p in PARAMS:
        for f in FACTORS:
            for s in SCEN:
                imp = [(d[(p, f, s, "P0", sd)][0] - d[(p, f, s, "P2", sd)][0]) / d[(p, f, s, "P0", sd)][0] * 100 for sd in SEEDS]
                out["rows"].append({"param": p, "factor": f, "scenario": s,
                                    "AWT_P0": float(np.mean([d[(p, f, s, "P0", sd)][0] for sd in SEEDS])),
                                    "AWT_P2": float(np.mean([d[(p, f, s, "P2", sd)][0] for sd in SEEDS])),
                                    "improve_pct": float(np.mean(imp)), "improve_min": float(np.min(imp)),
                                    "p2_wins": int(sum(1 for x in imp if x > 0))})
    rows = out["rows"]
    out["all_wins"] = all(r["p2_wins"] == len(SEEDS) for r in rows)
    out["min_improve"] = float(min(r["improve_min"] for r in rows))
    with open(os.path.join(HERE, "results", "sensitivity.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("Cải thiện AWT của P2 so với P0 (%):  tham số  ×0,8 / ×1,0 / ×1,2")
    for p, lab in PARAMS.items():
        for s in SCEN:
            vals = [next(r for r in rows if r["param"] == p and r["factor"] == f and r["scenario"] == s)["improve_pct"] for f in FACTORS]
            print(f"  {lab:32s} {s}: " + " / ".join(f"{v:5.1f}" for v in vals))
    print("P2 thắng ở mọi lượt chạy:", out["all_wins"], "| cải thiện nhỏ nhất:", round(out["min_improve"], 1), "%")


if __name__ == "__main__":
    main()
