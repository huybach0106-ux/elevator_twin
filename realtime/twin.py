"""
Digital Twin chạy thời gian thực.

Twin KHÔNG đọc bên trong tòa nhà. Nó chỉ nghe bus (MQTT hoặc nội bộ):
  - state: vị trí / cửa / cân tải / nút gọi của từng cabin  -> dựng bản sao trạng thái (MirrorSim)
  - stop : lượt lên/xuống đếm bằng hồng ngoại ở cửa        -> ước lượng nhu cầu, AI nhận diện chế độ
  - report: người dùng báo hàng chờ qua form QR             -> hiệu chỉnh ước lượng
Mỗi giây chạy bộ điều phối P2 trên bản sao; khi phân vai thay đổi thì gửi lệnh cmd/roles cho tòa nhà.
Định kỳ chạy mô phỏng NHÌN TRƯỚC 10 phút từ trạng thái hiện tại (giữ nguyên vai vs theo twin)
để dự báo hàng chờ.

Chạy riêng (qua MQTT):  python -m realtime.twin --clock 06:40
"""
from __future__ import annotations

import argparse
import math
import queue
import threading
import time

import numpy as np

from ai.features import MODE_VI
from algorithms.baseline import BaselineDispatcher
from algorithms.optimized import OptimizedDispatcher
from algorithms.roles import role_from_dict, role_to_dict
from config import AVG_PAX_KG, CARS, F, G, LOBBY, N_FLOORS, UPPER, fname
from realtime.bus import get_bus, prefix
from realtime.timetable import Timetable, clock_str, hhmm
from simulation.engine import Simulation
from simulation.metrics import compute
from simulation.passenger_generator import Passenger


class MirrorCar:
    """Bản sao một cabin, dựng từ dữ liệu cảm biến (không có danh sách hành khách thật)."""

    def __init__(self, spec):
        self.id = spec["id"]
        self.alias = spec["alias"]
        self.baseline = frozenset(F(x) for x in spec["baseline"])
        self.reach = frozenset(spec["reach"])
        self.home = F(spec["home"])
        self.pos = self.home
        self.posf = float(self.home)
        self.dir = 0
        self.state = "idle"
        self.timer = 0.0
        self.pax: list = []
        self.car_calls: set = set()
        self.pickups: set = set()
        self.role = None
        self.role_since = 0.0
        self.sensed_load = 0
        self.out = False

    def pos_float(self):
        return self.posf


class MirrorSim:
    """Giao diện tối thiểu mà bộ điều phối P2 cần, dựng hoàn toàn từ tin nhắn cảm biến."""

    def __init__(self, capacity):
        self.t = 0.0
        self.capacity = capacity
        self.cars = [MirrorCar(s) for s in CARS]
        self.car_by_id = {c.id: c for c in self.cars}
        self.obs: list = []
        self.events: list = []
        self.calls: dict = {}
        self.commands: dict = {}

    def set_role(self, car, role, log=True, force=False):
        if car.role == role:
            return
        car.role = role
        car.role_since = self.t
        self.commands[car.id] = role

    def hall_calls(self):
        return dict(self.calls)

    def group_calls(self):
        return {}                      # gán cuộc gọi là việc của bộ điều khiển tòa nhà

    def requests(self, car):
        return set(car.car_calls)


class FixedRoles(BaselineDispatcher):
    """Dùng cho mô phỏng nhìn trước: giữ nguyên phân vai hiện tại suốt 10 phút."""

    def __init__(self, roles):
        super().__init__()
        self.roles = roles

    def initial_role(self, car):
        return self.roles.get(car.id) or super().initial_role(car)


class TwinService:
    def __init__(self, bus, clock0: float, capacity: int = 15, timetable: Timetable | None = None):
        self.bus = bus
        self.pfx = prefix()
        self.clock0 = clock0
        self.tt = timetable or Timetable()
        self.mirror = MirrorSim(capacity)
        rng = np.random.default_rng(3)
        prior = self.tt.passengers_between(clock0, clock0 + 1200, clock0, rng)
        self.planner = OptimizedDispatcher("P2", plan_passengers=prior or None,
                                           scenario_mode=self.tt.mode_hint(clock0))
        self.planner.attach(self.mirror)
        for c in self.mirror.cars:
            c.role = self.planner.initial_role(c)
        # gửi phân vai ban đầu (theo thời khóa biểu) ngay ở bản tin trạng thái đầu tiên
        self.mirror.commands = {c.id: c.role for c in self.mirror.cars}
        self.control = "closed"
        self.q = queue.Queue()
        self.running = False
        self.cmd_log: list = []
        self.lookahead: dict = {}
        self.reports: list = []
        self._last_look_wall = 0.0
        self._last_look_sim = -1e9
        self._last_sent: dict = {}
        self.view: dict = {}
        for t in ("/state", "/stop", "/report"):
            bus.subscribe(self.pfx + t, lambda topic, msg: self.q.put((topic, msg)))

    # ------------------------------------------------------------- ingest
    def _on_state(self, msg):
        m = self.mirror
        if not getattr(self, "_synced", False) and msg.get("clock"):
            h, mi, se = (int(x) for x in msg["clock"].split(":"))
            self.clock0 = h * 3600 + mi * 60 + se - msg["t"]       # đồng bộ đồng hồ theo tòa nhà
            self._synced = True
        m.t = msg["t"]
        self.control = msg.get("control", self.control)
        m.calls = {(f, d): a for f, d, a in msg.get("hall_calls", [])}
        for cd in msg["cars"]:
            c = m.car_by_id[cd["id"]]
            c.posf = cd["pos"]
            c.pos = cd["floor"]
            c.dir = cd["dir"]
            c.state = cd["state"]
            c.car_calls = set(cd["car_calls"])
            c.sensed_load = int(round(cd["load_kg"] / AVG_PAX_KG))
            c.pax = [None] * c.sensed_load
            c.out = bool(cd.get("out"))
            if self.control == "closed" or c.out:
                c.role = role_from_dict(cd["role"])      # vai thật đang chạy trong tòa nhà
        self.planner.update(m.t)
        self._send_commands()

    def _on_stop(self, msg):
        self.mirror.obs.append(msg)
        self.planner.on_observation(msg)
        if len(self.mirror.obs) > 5000:
            self.mirror.obs = self.mirror.obs[-3000:]

    def _on_report(self, msg):
        """Báo cáo hàng chờ của người dùng: coi như một quan sát số người còn chờ ở tầng đó."""
        f, n = F(str(msg["floor"]).strip()), int(msg["count"])     # tầng theo tên: "G", "8", "B1"
        d = int(msg.get("dir", -1 if f not in LOBBY else 1))
        self.planner.land[(f, d)] = (self.mirror.t, n)
        self.reports.append({"clock": clock_str(self.clock0 + self.mirror.t), "floor": fname(f), "count": n})
        self.reports = self.reports[-10:]

    # ------------------------------------------------------------- output
    def _send_commands(self):
        m = self.mirror
        if not m.commands:
            return
        roles = {cid: role_to_dict(r) for cid, r in m.commands.items()}
        changed = {cid: r for cid, r in roles.items() if self._last_sent.get(cid) != r}
        m.commands = {}
        if not changed:
            return
        self._last_sent.update(changed)
        self.bus.publish(self.pfx + "/cmd/roles", {"t": m.t, "roles": changed, "by": "twin"})
        self.cmd_log.append({"clock": clock_str(self.clock0 + m.t),
                             "text": ", ".join(f"{cid} → {r['label']}" for cid, r in sorted(changed.items()))})
        self.cmd_log = self.cmd_log[-12:]

    def _est_waiting(self):
        """Twin ước lượng số người chờ mỗi tầng từ nút gọi + tốc độ lên cabin + báo cáo người dùng."""
        est = [0.0] * N_FLOORS
        for (f, d), age in self.mirror.calls.items():
            est[f] += self.planner._est_crowd(f, d, age)
        return [round(x, 1) for x in est]

    def _publish_view(self):
        m = self.mirror
        clock = self.clock0 + m.t
        roles = {c.id: c.role.label for c in m.cars}
        rec = {c.id: (self._last_sent.get(c.id) or role_to_dict(c.role))["label"] for c in m.cars}
        self.view = {
            "t": m.t, "clock": clock_str(clock), "mode": self.planner.mode, "mode_vi": MODE_VI.get(self.planner.mode, ""),
            "control": self.control, "roles_now": roles, "recommended": rec,
            "est_waiting": self._est_waiting(), "cmd_log": self.cmd_log[-8:], "reports": self.reports[-5:],
            "est_calls": [[f, d, round(self.planner._est_crowd(f, d, a), 1), round(a)] for (f, d), a in m.calls.items()],
            "upcoming": self.tt.upcoming(clock, 900)[:8], "lookahead": self.lookahead,
            "cars": [{"id": c.id, "pos": c.pos_float(), "load": c.sensed_load, "state": c.state, "dir": c.dir,
                      "role": role_to_dict(c.role), "out": c.out} for c in m.cars],
            "n_obs": len(m.obs),
        }
        self.bus.publish(self.pfx + "/twin", self.view)

    # ------------------------------------------------------- look-ahead
    def _lookahead(self):
        """Mô phỏng nhìn trước 10 phút từ trạng thái hiện tại: giữ nguyên vai vs để twin điều phối."""
        m = self.mirror
        clock = self.clock0 + m.t
        rng = np.random.default_rng(int(m.t) % 10000)
        ps = []
        # khách đang chờ (ước lượng) – đích suy từ chiều đi
        for (f, d), age in m.calls.items():
            n = max(1, int(round(self.planner._est_crowd(f, d, age))))
            for _ in range(n):
                if d < 0:
                    dest = G if f not in LOBBY else int(rng.choice([x for x in range(f)]))
                else:
                    dest = int(rng.choice([x for x in UPPER if x > f])) if f < max(UPPER) else G
                ps.append(Passenger(len(ps), 0.0, f, dest))
        for p in self.tt.passengers_between(clock, clock + 600, clock, rng, id_start=len(ps)):
            ps.append(p)
        if not ps:
            self.lookahead = {"clock": clock_str(clock), "n": 0}
            return
        roles_now = {c.id: c.role for c in m.cars}
        out = {"clock": clock_str(clock), "n": len(ps), "horizon_min": 10}
        for key, disp in (("keep", FixedRoles(roles_now)),
                          ("twin", OptimizedDispatcher("P2", plan_passengers=ps, scenario_mode=self.planner.mode))):
            outages = [(c.id, 0, 10 ** 9) for c in m.cars if c.out]
            sim = Simulation(ps, disp, capacity=m.capacity, horizon=1500, seed=1, record=False, outages=outages)
            for c, mc in zip(sim.cars, m.cars):
                c.pos = mc.pos
            sim.run()
            k = compute(sim)
            worst = max(k["floor_awt"].items(), key=lambda kv: kv[1]) if k["floor_awt"] else ("–", 0)
            out[key] = {"AWT": round(k["AWT"], 1), "P95": round(k["P95"], 0), "backlog": k["backlog"],
                        "worst_floor": worst[0], "worst_awt": round(worst[1], 1)}
        self.lookahead = out

    # --------------------------------------------------------------- loop
    def run_forever(self):
        self.running = True
        last_view = 0.0
        while self.running:
            try:
                topic, msg = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            # chỉ xử lý trạng thái MỚI NHẤT nếu bị dồn (tua nhanh)
            pending_state = None
            batch = [(topic, msg)]
            while not self.q.empty() and len(batch) < 500:
                batch.append(self.q.get_nowait())
            for tp, ms in batch:
                if tp.endswith("/stop"):
                    self._on_stop(ms)
                elif tp.endswith("/report"):
                    self._on_report(ms)
                elif tp.endswith("/state"):
                    pending_state = ms
            if pending_state:
                self._on_state(pending_state)
            now = time.monotonic()
            if self.mirror.t - self._last_look_sim >= 60 and now - self._last_look_wall >= 15:
                self._last_look_sim, self._last_look_wall = self.mirror.t, now
                try:
                    self._lookahead()
                except Exception as exc:  # noqa: BLE001
                    self.lookahead = {"error": str(exc)}
            if now - last_view >= 0.5:
                last_view = now
                self._publish_view()

    def start(self):
        th = threading.Thread(target=self.run_forever, daemon=True)
        th.start()
        return th

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser(description="Digital Twin – nghe tòa nhà qua MQTT, gửi lệnh phân vai")
    ap.add_argument("--clock", default=None, help="giờ bắt đầu HH:MM, trùng với tòa nhà")
    a = ap.parse_args()
    clock0 = hhmm(a.clock) if a.clock else (time.localtime().tm_hour * 3600 + time.localtime().tm_min * 60)
    tw = TwinService(get_bus(), clock0)
    print(f"Twin đang nghe {prefix()}/* …")
    tw.run_forever()


if __name__ == "__main__":
    main()
