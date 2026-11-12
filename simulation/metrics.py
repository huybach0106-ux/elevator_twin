"""
KPI (mục 8 tài liệu): AWT, P90/P95, ATT, Total, Throughput, Backlog, Bypass, Stop, Energy proxy,
công bằng giữa các tầng, thời gian giải tỏa, RTT, tải từng cabin.

AWT_all tính cả khách chưa được đón (thời gian chờ bị cắt tại thời điểm kết thúc) để không
"giấu" backlog; AWT_served chỉ tính khách đã lên cabin. Cả hai đều được báo cáo.
"""
from __future__ import annotations

import numpy as np

from config import N_FLOORS, fname


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def compute(sim) -> dict:
    T = sim.t
    arrived = [p for p in sim.passengers if p.t_arr <= T]
    served = [p for p in arrived if p.t_board is not None]
    done = [p for p in arrived if p.state == "done"]
    w_served = np.array([p.t_board - p.t_arr for p in served])
    w_all = np.array([(p.t_board if p.t_board is not None else T) - p.t_arr for p in arrived])
    ride = np.array([p.t_alight - p.t_board for p in done])
    total = np.array([p.t_alight - p.t_arr for p in done])

    # công bằng theo tầng xuất phát (dùng chờ có kiểm duyệt)
    floor_awt = {}
    for f in range(N_FLOORS):
        ws = [(p.t_board if p.t_board is not None else T) - p.t_arr for p in arrived if p.origin == f]
        if len(ws) >= 3:
            floor_awt[fname(f)] = float(np.mean(ws))
    vals = np.array(list(floor_awt.values())) if floor_awt else np.array([0.0])
    jain = float(vals.sum() ** 2 / (len(vals) * (vals ** 2).sum())) if (vals ** 2).sum() > 0 else 1.0

    cars = {}
    for c in sim.cars:
        cars[c.id] = {
            "pax": c.pax_served, "stops": c.stops, "useless_stops": c.useless_stops,
            "floors": c.floors_moved, "energy": round(c.energy, 1),
            "util": round(c.busy_time / max(T, 1e-9), 3),
        }
    rtts = [r for c in sim.cars for r in c.rtts]
    energy = sum(c.energy for c in sim.cars)
    return {
        "n": len(sim.passengers),
        "arrived": len(arrived),
        "AWT": float(w_all.mean()) if len(w_all) else 0.0,
        "AWT_served": float(w_served.mean()) if len(w_served) else 0.0,
        "P90": _pct(w_all, 90),
        "P95": _pct(w_all, 95),
        "max_wait": float(w_all.max()) if len(w_all) else 0.0,
        "pct_wait_gt60": float((w_all > 60).mean() * 100) if len(w_all) else 0.0,
        "ATT": float(ride.mean()) if len(ride) else 0.0,
        "total_time": float(total.mean()) if len(total) else 0.0,
        "throughput": len(done),
        "backlog": len(arrived) - len(done),
        "unserved": len(arrived) - len(served),
        "bypass": int(sum(sim.bypass_pass) + sum(sim.bypass_left)),
        "bypass_pass": int(sum(sim.bypass_pass)),
        "bypass_left": int(sum(sim.bypass_left)),
        "bypass_by_floor": {fname(f): sim.bypass_pass[f] + sim.bypass_left[f] for f in range(N_FLOORS)
                            if sim.bypass_pass[f] + sim.bypass_left[f]},
        "stops": int(sum(c.stops for c in sim.cars)),
        "useless_stops": int(sum(c.useless_stops for c in sim.cars)),
        "energy": float(energy),
        "energy_per_pax": float(energy / max(1, len(done))),
        "RTT": float(np.mean(rtts)) if rtts else float("nan"),
        "clearance": sim.clearance,
        "floor_awt": floor_awt,
        "worst_floor_awt": float(vals.max()),
        "jain": jain,
        "cars": cars,
        "role_changes": sum(1 for e in sim.events if e["type"] == "role" and e["t"] > 0),
    }


def wait_list(sim):
    T = sim.t
    return [round((p.t_board if p.t_board is not None else T) - p.t_arr, 1) for p in sim.passengers if p.t_arr <= T]


def improvement(base, opt, higher_is_better=False):
    """Công thức tài liệu: (B − O)/B ×100 (KPI càng thấp càng tốt), (O − B)/B ×100 cho thông lượng."""
    if base in (0, None) or base != base:
        return None
    return ((opt - base) if higher_is_better else (base - opt)) / base * 100
