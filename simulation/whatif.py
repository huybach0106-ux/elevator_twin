"""What-if: chạy cùng kịch bản 'bình thường' và 'có giả định' (cabin hỏng, lượng khách khác) cho P0 và cấu hình tối ưu."""
from __future__ import annotations

from simulation.runner import run_one
from simulation.scenarios import n_for

KEYS = ("AWT", "P95", "max_wait", "backlog", "energy", "worst_floor_awt", "clearance")


def run_whatif(scenario="U1", level="mid", capacity=15, demand=1.0, fault_car=None, fault_start_min=2.0,
               fault_minutes=10.0, config="P2", seed=42, series=True):
    base_n = n_for(scenario, level)
    n = max(1, int(round(base_n * float(demand))))
    t0 = float(fault_start_min) * 60
    outages = [(fault_car, t0, t0 + float(fault_minutes) * 60)] if fault_car else []
    out = {"scenario": scenario, "N_normal": base_n, "N": n, "capacity": capacity, "fault": fault_car,
           "fault_minutes": fault_minutes if fault_car else 0, "runs": {}}
    for cfg in ("P0", config):
        for tag, outs, lv in (("normal", [], base_n), ("whatif", outages, n)):
            sim, m = run_one(scenario, lv, int(capacity), cfg, int(seed), record=series, outages=outs)
            r = {"config": cfg, "tag": tag,
                 "kpi": {k: (None if m[k] is None else round(float(m[k]), 1)) for k in KEYS}}
            if series:
                r["series"] = [[f["t"], sum(u + d for u, d, _ in f["f"])] for f in sim.frames[::5]]
            out["runs"][f"{cfg}_{tag}"] = r
    return out
