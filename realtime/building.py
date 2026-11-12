"""
Tòa nhà ẢO – đóng vai tòa nhà thật trong kiểm chứng software-in-the-loop.

Chạy mô phỏng theo đồng hồ thật (hoặc tua nhanh `speed` lần), hành khách sinh theo thời khóa biểu.
Bộ điều khiển nhóm thang của tòa nhà = FCFS như hiện trạng, nhưng nhận LỆNH PHÂN VAI từ twin qua bus
(chế độ 'closed'); ở chế độ 'advisory' tòa nhà bỏ qua lệnh, twin chỉ đề xuất.
Tòa nhà chỉ gửi ra ngoài những gì cảm biến thang thật có: vị trí, cửa, cân tải, nút gọi, đếm lượt qua cửa.

Chạy riêng (qua MQTT):  python -m realtime.building --clock 06:40 --speed 10
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from algorithms.baseline import BaselineDispatcher
from algorithms.roles import local_role, role_from_dict, role_to_dict
from config import AVG_PAX_KG, LOCAL_CAR, N_FLOORS
from realtime.bus import get_bus, prefix
from realtime.timetable import Timetable, clock_str, hhmm
from simulation.engine import Simulation


class RoleFollowerDispatcher(BaselineDispatcher):
    """Bộ điều khiển của tòa nhà: gán cuộc gọi FCFS, áp dụng vai do twin gửi, có luật an toàn riêng."""
    name = "building"

    def __init__(self):
        super().__init__()
        self.pending = {}

    def apply_roles(self, roles: dict):
        for cid, r in roles.items():
            car = self.sim.car_by_id.get(cid)
            if car is None:
                continue
            if r.standby and (car.pax or car.car_calls):
                self.pending[cid] = r
            else:
                self.pending.pop(cid, None)
                self.sim.set_role(car, r)

    def update(self, t):
        for cid, r in list(self.pending.items()):
            car = self.sim.car_by_id[cid]
            if not car.pax and not car.car_calls:
                self.sim.set_role(car, r)
                self.pending.pop(cid)
        # an toàn: khách không có cabin nào chở được > 60 s -> A1 phục vụ mọi tầng
        for (f, d, group), (age, _) in self.sim.group_calls().items():
            if not group and age > 60:
                self.sim.set_role(self.sim.car_by_id[LOCAL_CAR], local_role())
                break
        super().update(t)


class VirtualBuilding:
    def __init__(self, bus, clock0: float, speed: float = 1.0, capacity: int = 15, seed: int = 7,
                 control: str = "closed", timetable: Timetable | None = None):
        self.bus = bus
        self.pfx = prefix()
        self.clock0 = clock0
        self.speed = speed
        self.control = control
        self.tt = timetable or Timetable()
        self.rng = np.random.default_rng(seed)
        self.disp = RoleFollowerDispatcher()
        self.sim = Simulation([], self.disp, capacity=capacity, horizon=10**9, seed=seed, record=False)
        self.gen_until = 0.0
        self.n_obs_sent = 0
        self.running = False
        self.log = []
        self.lock = threading.Lock()
        bus.subscribe(self.pfx + "/cmd/#", self._on_cmd)

    # ------------------------------------------------------------- commands
    def _on_cmd(self, topic, msg):
        with self.lock:
            if topic.endswith("/cmd/roles"):
                if self.control == "closed":
                    self.disp.apply_roles({cid: role_from_dict(r) for cid, r in msg.get("roles", {}).items()})
                    self.log.append({"t": self.clock(), "msg": f"Áp dụng lệnh phân vai của twin ({len(msg.get('roles', {}))} cabin)"})
                else:
                    self.log.append({"t": self.clock(), "msg": "Nhận đề xuất của twin (chế độ tư vấn: không áp dụng)"})
            elif topic.endswith("/cmd/fault"):
                car, mins = msg.get("car"), float(msg.get("minutes", 10))
                if car in self.sim.car_by_id:
                    self.sim.outages.append((car, self.sim.t, self.sim.t + mins * 60))
                    self.log.append({"t": self.clock(), "msg": f"Sự cố cabin {car} trong {mins:g} phút"})
            elif topic.endswith("/cmd/control"):
                self.control = msg.get("mode", self.control)
                self.log.append({"t": self.clock(), "msg": "Chế độ: " + ("khép kín" if self.control == "closed" else "tư vấn")})
            elif topic.endswith("/cmd/speed"):
                self.speed = float(msg.get("speed", self.speed))
            self.log = self.log[-30:]

    def clock(self):
        return clock_str(self.clock0 + self.sim.t)

    # ----------------------------------------------------------------- loop
    def _generate(self):
        if self.sim.t + 120 < self.gen_until:
            return
        c0 = self.clock0 + self.gen_until
        ps = self.tt.passengers_between(c0, c0 + 300, self.clock0, self.rng, id_start=len(self.sim.passengers))
        self.sim.add_passengers(ps)
        self.gen_until += 300

    def _publish(self):
        s = self.sim
        cars = []
        for c in s.cars:
            kg = int(sum(self.rng.normal(AVG_PAX_KG, 12, c.load))) if c.load else 0
            cars.append({"id": c.id, "pos": round(c.pos_float(), 2), "floor": c.pos, "dir": c.dir, "state": c.state,
                         "door": c.state == "dwell", "load_kg": max(0, kg), "car_calls": sorted(c.car_calls),
                         "role": role_to_dict(c.role), "out": bool(getattr(c, "out", False))})
        calls = [[f, d, round(a, 1)] for (f, d), a in s.hall_calls().items()]
        self.bus.publish(self.pfx + "/state", {"t": round(s.t, 1), "clock": self.clock(), "cars": cars,
                                               "hall_calls": calls, "control": self.control, "speed": self.speed})
        while self.n_obs_sent < len(s.obs):
            ob = dict(s.obs[self.n_obs_sent])
            self.n_obs_sent += 1
            self.bus.publish(self.pfx + "/stop", ob)
        # số liệu thật (twin không đọc) – chỉ để web đối chiếu độ chính xác của twin
        waiting = [[sum(1 for p in s.waiting[f] if p.direction > 0), sum(1 for p in s.waiting[f] if p.direction < 0),
                    int(max((s.t - p.t_arr for p in s.waiting[f]), default=0))] for f in range(N_FLOORS)]
        boarded = [p for p in s.passengers if p.t_board is not None]
        recent = [p for p in boarded if p.t_board > s.t - 600]
        self.bus.publish(self.pfx + "/truth", {
            "t": round(s.t, 1), "clock": self.clock(), "waiting": waiting, "arrived": s.n_arrived, "done": s.n_done,
            "riding": sum(c.load for c in s.cars),
            "awt_10min": round(float(np.mean([p.t_board - p.t_arr for p in recent])), 1) if recent else None,
            "awt_all": round(float(np.mean([p.t_board - p.t_arr for p in boarded])), 1) if boarded else None,
            "energy": round(sum(c.energy for c in s.cars), 1), "log": self.log[-8:]})

    def run_forever(self):
        self.running = True
        wall0 = time.monotonic()
        sim0 = self.sim.t
        last_pub = -1.0
        while self.running:
            target = sim0 + (time.monotonic() - wall0) * self.speed
            n = 0
            with self.lock:
                while self.sim.t < target and n < 3000:
                    self._generate()
                    self.sim.step()
                    n += 1
                    if self.sim.t - last_pub >= 1.0:
                        last_pub = self.sim.t
                        self._publish()
            if n >= 3000:                     # không theo kịp -> bỏ qua phần trễ thay vì dồn
                wall0, sim0 = time.monotonic(), self.sim.t
            time.sleep(0.05)

    def start(self):
        th = threading.Thread(target=self.run_forever, daemon=True)
        th.start()
        return th

    def stop(self):
        self.running = False

    def set_speed(self, speed):
        self.speed = speed


def main():
    ap = argparse.ArgumentParser(description="Tòa nhà ảo – gửi dữ liệu qua MQTT")
    ap.add_argument("--clock", default=None, help="giờ bắt đầu HH:MM (mặc định giờ hiện tại)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--control", default="closed", choices=["closed", "advisory"])
    a = ap.parse_args()
    clock0 = hhmm(a.clock) if a.clock else (time.localtime().tm_hour * 3600 + time.localtime().tm_min * 60)
    b = VirtualBuilding(get_bus(), clock0, a.speed, control=a.control)
    print(f"Tòa nhà ảo chạy từ {clock_str(clock0)} × {a.speed} – chủ đề {prefix()}/*")
    b.run_forever()


if __name__ == "__main__":
    main()
