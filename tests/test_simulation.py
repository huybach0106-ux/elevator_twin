"""
Kiểm thử bắt buộc (mục 9 tài liệu):
  - không bỏ khách (horizon đủ dài -> mọi khách tới đích),
  - không vượt sức chứa,
  - Express không mở cửa ở tầng trung gian,
  - cùng seed -> cùng danh sách hành khách cho mọi cấu hình,
  - khách chỉ đi cabin phục vụ được cả tầng đi và tầng đến,
  - cảm biến cân tải ước số người hợp lý.
Chạy:  python -m pytest -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.baseline import BaselineDispatcher  # noqa: E402
from algorithms.roles import group_role, local_role  # noqa: E402
from config import F, G  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.runner import make_dispatcher, run_one  # noqa: E402
from simulation.scenarios import SCENARIOS, make_passengers  # noqa: E402

CONFIGS = ["P0", "P1", "P2", "P3"]


class Watch:
    """Bọc dispatcher để kiểm tra bất biến ở mỗi bước."""

    def __init__(self, inner):
        self.inner = inner
        self.update_interval = inner.update_interval
        self.max_load = 0
        self.violations = []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def attach(self, sim):
        self.sim = sim
        self.inner.attach(sim)

    def update(self, t):
        self.inner.update(t)
        for c in self.sim.cars:
            self.max_load = max(self.max_load, c.load)
            for p in c.pax:
                if p.origin not in c.baseline | c.role.stops and self.inner.name == "P0":
                    self.violations.append((t, c.id, p.id))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("config", CONFIGS)
def test_no_passenger_lost_and_capacity(scenario, config):
    ps = make_passengers(scenario, "high", 3)
    w = Watch(make_dispatcher(config, scenario, "high", 3))
    sim = Simulation(ps, w, capacity=10, horizon=4000, seed=3, record=False).run()
    assert all(p.state == "done" for p in sim.passengers), "có khách không tới đích"
    assert w.max_load <= 10, "vượt sức chứa hữu dụng"
    assert not w.violations


def test_same_seed_same_passengers():
    a = make_passengers("U1", "mid", 7)
    b = make_passengers("U1", "mid", 7)
    assert [(p.t_arr, p.origin, p.dest) for p in a] == [(p.t_arr, p.origin, p.dest) for p in b]
    s1, _ = run_one("U1", "mid", 15, "P0", 7)
    s2, _ = run_one("U1", "mid", 15, "P2", 7)
    assert [(p.t_arr, p.origin, p.dest) for p in s1.passengers] == [(p.t_arr, p.origin, p.dest) for p in s2.passengers]


def test_express_never_stops_at_intermediate_floor():
    sim, _ = run_one("D3", "high", 15, "P0", 5, record=True)
    for t, ci, f, n_in, n_out in sim.stop_log:
        car = sim.cars[ci]
        if car.id in ("A4", "A5", "A6"):
            assert f in (G, F(8)), f"Express {car.id} mở cửa ở tầng {f}"


def test_passengers_only_ride_valid_cars():
    sim, _ = run_one("D1", "high", 15, "P0", 2)
    for p in sim.passengers:
        if p.car is None:      # còn trong backlog lúc kết thúc
            continue
        car = sim.car_by_id[p.car]
        assert p.origin in car.baseline and p.dest in car.baseline


def test_role_exclusion_keeps_basement_trips():
    a1 = local_role(frozenset({F(8)}))
    assert not a1.can_serve(G, F(8))        # G -> 8 nhường cabin Express
    assert a1.can_serve(F("B1"), F(8))      # B1 -> 8 vẫn được A1 nhận
    assert group_role([F(8)]).can_serve(G, F(8))


def test_load_cell_estimate_is_close():
    from simulation.sensors import CabinSensors
    sn = CabinSensors(seed=1)
    errs = [abs(sn.load(n) - n) for n in range(0, 16) for _ in range(50)]
    assert sum(errs) / len(errs) < 1.0          # sai số trung bình dưới 1 người
    assert sn.landing(0) == 0 and sn.landing(7) == 1


def test_optimized_beats_baseline_on_awt():
    for s in ("D1", "U1", "D2"):
        _, m0 = run_one(s, "high", 15, "P0", 9)
        _, m2 = run_one(s, "high", 15, "P2", 9)
        assert m2["AWT"] < m0["AWT"]
