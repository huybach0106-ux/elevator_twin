"""Đóng gói kết quả một lượt chạy thành JSON gọn cho web (dashboard & bản xuất tĩnh)."""
from __future__ import annotations

from config import CARS, FLOOR_NAMES
from simulation.metrics import wait_list

KPI_KEYS = ["AWT", "AWT_served", "P90", "P95", "max_wait", "pct_wait_gt60", "ATT", "total_time", "throughput",
            "backlog", "unserved", "bypass", "stops", "useless_stops", "energy", "energy_per_pax", "RTT",
            "clearance", "worst_floor_awt", "jain", "role_changes", "n", "arrived"]


def _r(v, nd=1):
    if v is None or v != v:
        return None
    return round(float(v), nd)


def run_payload(sim, m) -> dict:
    fr = sim.frames
    payload = {
        "config": sim.meta.get("config"),
        "title": getattr(sim.dispatcher, "title", sim.meta.get("config")),
        "t": [f["t"] for f in fr],
        # cabin: [vị trí, tải, trạng thái, vai, chiều] gộp phẳng 6 × 5 mỗi khung
        "c": [[v for car in f["c"] for v in car] for f in fr],
        # tầng: [chờ lên, chờ xuống, chờ lâu nhất] gộp phẳng 13 × 3
        "f": [[v for fl in f["f"] for v in fl] for f in fr],
        "a": [f["a"] for f in fr],
        "d": [f["d"] for f in fr],
        "roles": sim.role_table,
        "events": [e for e in sim.events if e["type"] in ("role", "mode", "fallback")][:1500],
        "stops": sim.stop_log,
        "obs": [[o["t"], o["car"], o["floor"], o["dir"], o["n_in"], o["n_out"], o["load"], o["left"]] for o in sim.obs],
        "modes": getattr(sim.dispatcher, "mode_log", []),
        "waits": wait_list(sim),
        "kpi": {k: (_r(m[k], 3) if k == "jain" else _r(m[k])) for k in KPI_KEYS},
        "floor_awt": {k: _r(v) for k, v in m["floor_awt"].items()},
        "cars": m["cars"],
        "bypass_by_floor": m["bypass_by_floor"],
    }
    return payload


def meta() -> dict:
    from simulation.runner import CONFIGS
    from simulation.scenarios import SCENARIOS
    return {
        "floors": FLOOR_NAMES,
        "cars": [{"id": c["id"], "alias": c["alias"]} for c in CARS],
        "scenarios": {k: {"name": v["name"], "group": v["group"], "time": v["time"], "levels": v["levels"],
                          "mode": v["mode"], "note": v["note"],
                          "horizon": v["horizon"]} for k, v in SCENARIOS.items()},
        "configs": CONFIGS,
    }
