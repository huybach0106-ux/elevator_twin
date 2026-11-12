"""Chạy một lượt mô phỏng: (kịch bản, mức tải, sức chứa, cấu hình, seed) -> (sim, KPI)."""
from __future__ import annotations

from algorithms.baseline import BaselineDispatcher
from algorithms.optimized import OptimizedDispatcher
from simulation.engine import Simulation
from simulation.metrics import compute
from simulation.scenarios import SCENARIOS, make_passengers, planned_demand

CONFIGS = {
    "P0": "P0 – Hiện trạng",
    "P1": "P1 – Ưu tiên tầng cao (phân vai tĩnh)",
    "P2": "P2 – Điều phối động + AI",
    "P3": "P3 – Thấp điểm / Standby",
    "P4": "P4 – P2 tiết kiệm năng lượng (eco)",
}
# P4 = P2 + chỉ giữ số cabin cần cho hệ số sử dụng mục tiêu + phạt năng lượng trong hàm chi phí
ECO_PARAMS = {"u_target": 0.7, "w_energy": 3.0}


def make_dispatcher(config, scenario, level, seed, params=None, use_model=True, use_prior=True):
    if config == "P0":
        return BaselineDispatcher()
    plan = planned_demand(scenario, level, seed) if config in ("P1", "P2", "P4") and use_prior else None
    if config == "P4":
        d = OptimizedDispatcher("P2", params={**ECO_PARAMS, **(params or {})}, plan_passengers=plan,
                                scenario_mode=SCENARIOS[scenario]["mode"], use_model=use_model)
        d.name, d.title = "P4", CONFIGS["P4"]
        return d
    return OptimizedDispatcher(config, params=params, plan_passengers=plan,
                               scenario_mode=SCENARIOS[scenario]["mode"], use_model=use_model)


def run_one(scenario="U1", level="mid", capacity=15, config="P0", seed=42, sensor_noise=True,
            record=False, params=None, use_model=True, passengers=None, horizon=None, use_prior=True,
            outages=()):
    ps = passengers if passengers is not None else make_passengers(scenario, level, seed)
    disp = make_dispatcher(config, scenario, level, seed, params, use_model, use_prior)
    sim = Simulation(ps, disp, capacity=capacity, horizon=horizon or SCENARIOS[scenario]["horizon"],
                     seed=seed, sensor_noise=sensor_noise, record=record,
                     meta={"scenario": scenario, "level": level, "config": config, "seed": seed},
                     outages=outages)
    sim.run()
    return sim, compute(sim)
