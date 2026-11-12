"""
P0 – Hiện trạng mô phỏng (Baseline): tuyến dừng cố định + FCFS.

- A1/X dừng mọi tầng; A2–A3 dừng B2,B1,G,5; A4–A6 dừng G,8.
- Cuộc gọi được xử lý theo thứ tự đến: cabin rảnh gần nhất trong nhóm phục vụ được nhận cuộc gọi cũ nhất.
- Cabin đang chạy đón thêm khách cùng chiều trên đường đi (điều khiển tập hợp).
- Chỉ có cảm biến cân tải: báo đầy khi vượt 80 % tải định mức (kg) -> với sức chứa hữu dụng
  10–15 người, cabin vẫn dừng dù không còn chỗ đứng.
"""
from __future__ import annotations

from algorithms.roles import base_role
from config import AVG_PAX_KG, RATED_LOAD_KG, WEIGHT_BYPASS_RATIO


class BaselineDispatcher:
    name = "P0"
    title = "P0 – Hiện trạng"
    update_interval = 1.0

    def __init__(self, disabled=()):
        self.disabled = set(disabled)   # cabin tạm ngừng (dùng cho thí nghiệm số cabin Express)

    def attach(self, sim):
        self.sim = sim
        self.assigned: dict = {}   # (f, d, group) -> car id

    def initial_role(self, car):
        if car.id in self.disabled:
            from algorithms.roles import standby_role
            return standby_role((), car.home)
        return base_role(car.baseline)

    def on_arrival(self, p, t):
        pass

    def on_observation(self, ob):
        pass

    def may_board(self, car, p) -> bool:
        return True

    def _weight_full(self, car) -> bool:
        return car.load * AVG_PAX_KG >= WEIGHT_BYPASS_RATIO * RATED_LOAD_KG

    def update(self, t):
        sim = self.sim
        calls = sim.group_calls()
        # dọn phân công cũ
        for key, cid in list(self.assigned.items()):
            car = sim.car_by_id[cid]
            if key not in calls or (key[0], key[1]) not in car.pickups:
                self.assigned.pop(key, None)
        busy = {cid for cid in self.assigned.values()}
        for key, (age, n) in sorted(calls.items(), key=lambda kv: -kv[1][0]):   # FCFS: cũ nhất trước
            if key in self.assigned:
                continue
            f, d, group = key
            best = None
            for car in sim.cars:
                if car.id not in group or car.id in busy:
                    continue
                if car.state not in ("idle",) or car.pax or sim.requests(car):
                    continue
                dist = abs(car.pos - f)
                if best is None or dist < best[0]:
                    best = (dist, car)
            if best:
                car = best[1]
                car.pickups.add((f, d))
                self.assigned[key] = car.id
                busy.add(car.id)

    def wants_stop(self, car, f, d, t) -> str:
        if not self.sim.eligible(car, f, d):
            return "pass"
        if self._weight_full(car):
            return "bypass"
        return "stop"
