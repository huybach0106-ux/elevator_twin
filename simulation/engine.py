"""
Lõi mô phỏng sự kiện theo bước thời gian (DT = 0,1 s).

Mô hình:
- Cabin chạy 2,2 s/tầng; mỗi lần dừng tốn DOOR_TIME + TRANSFER_TIME × (lên + xuống).
- Hành khách chỉ lên cabin có vai trò phục vụ được cả tầng đi lẫn tầng đến (bảng hiển thị trên cửa).
- Điều khiển tập hợp (collective): cabin đi qua tầng có cuộc gọi cùng chiều thì bộ điều phối
  quyết định dừng / đi tiếp / bỏ qua (bypass do đầy tải).
- Mỗi lần mở cửa, cảm biến cabin (cân tải, hồng ngoại ở cửa, nút gọi) tạo một quan sát có sai số (simulation/sensors.py).
"""
from __future__ import annotations

from collections import deque

from config import (
    AVG_PAX_KG, CARS, COUNTERWEIGHT_RATIO, DOOR_TIME, DT, E_DOOR, E_FLOOR, E_IMBALANCE, E_START, F, G,
    LOG_EVERY, N_FLOORS, P_IDLE, P_STANDBY, RATED_LOAD_KG, STANDBY_WAKE_TIME, TRANSFER_TIME,
    TRAVEL_TIME_PER_FLOOR, USEFUL_CAPACITY,
)
from simulation.sensors import CabinSensors

STATE_CODE = {"idle": 0, "moving": 1, "dwell": 2, "standby": 3, "waking": 4}
FAKE_UP_PATIENCE = 45.0   # hành vi đi ngược chiều giữ chỗ (tùy chọn): sau 45 s không được chở lên, khách bỏ ý định đi ngược chiều


class Car:
    def __init__(self, spec: dict, capacity: int):
        self.id = spec["id"]
        self.alias = spec["alias"]
        self.baseline = frozenset(F(x) for x in spec["baseline"])
        self.reach = frozenset(spec["reach"])
        self.home = F(spec["home"])
        self.capacity = capacity
        self.pos = self.home
        self.next = self.home
        self.timer = 0.0
        self.state = "idle"
        self.dir = 0
        self.serve_dir = 0
        self.pax: list = []
        self.car_calls: set[int] = set()
        self.pickups: set[tuple[int, int]] = set()
        self.role = None
        self.role_since = 0.0
        self.parking_move = False
        # thống kê
        self.floors_moved = 0
        self.stops = 0
        self.useless_stops = 0
        self.departures = 0
        self.energy = 0.0
        self.pax_served = 0
        self.busy_time = 0.0
        self.g_depart = None
        self.rtts: list[float] = []
        self.sensed_load = 0

    @property
    def load(self) -> int:
        return len(self.pax)

    @property
    def space(self) -> int:
        return self.capacity - len(self.pax)

    def pos_float(self) -> float:
        if self.state == "moving":
            frac = 1.0 - max(0.0, self.timer) / TRAVEL_TIME_PER_FLOOR
            return self.pos + (self.next - self.pos) * frac
        return float(self.pos)


class Simulation:
    def __init__(self, passengers, dispatcher, capacity: int = USEFUL_CAPACITY, horizon: float = 900,
                 seed: int = 42, sensor_noise: bool = True, record: bool = True, meta: dict | None = None,
                 outages=()):
        self.passengers = [p.clone() for p in passengers]
        self.pending = deque(self.passengers)
        self.waiting: list[list] = [[] for _ in range(N_FLOORS)]
        self.capacity = capacity
        self.horizon = horizon
        self.t = 0.0
        self.cars = [Car(s, capacity) for s in CARS]
        self.car_by_id = {c.id: c for c in self.cars}
        self.sensors = CabinSensors(seed, sensor_noise)
        self.record = record
        self.meta = meta or {}
        self.obs: list[dict] = []          # nhật ký quan sát cảm biến
        self.events: list[dict] = []       # đổi vai, bypass, ...
        self.stop_log: list[tuple] = []    # (t, car_idx, floor, n_in, n_out)
        self.frames: list = []
        self.role_table: list[dict] = []   # đăng ký nhãn vai trò -> chỉ số (cho web)
        self._role_idx: dict[str, int] = {}
        self.bypass_pass = [0] * N_FLOORS  # cabin đầy đi qua không dừng
        self.bypass_left = [0] * N_FLOORS  # cabin dừng nhưng khách bị bỏ lại do đầy
        self.clearance = None
        self.n_arrived = 0
        self.n_done = 0
        self._log_every = int(round(LOG_EVERY / DT))
        self._disp_every = int(round(dispatcher.update_interval / DT))
        self.k = 0                          # số bước đã chạy (chế độ chạy từng bước / thời gian thực)
        # sự cố: [(cabin, t_bắt_đầu, t_kết_thúc)] – cabin trả hết khách đang chở rồi đứng yên
        self.outages = [tuple(o) for o in outages]
        self.dispatcher = dispatcher
        dispatcher.attach(self)
        for c in self.cars:
            self.set_role(c, dispatcher.initial_role(c), log=True)

    # ------------------------------------------------------------------ roles
    def role_index(self, role) -> int:
        key = role.label
        if key not in self._role_idx:
            from algorithms.roles import KIND_COLORS
            self._role_idx[key] = len(self.role_table)
            self.role_table.append({"label": role.label, "kind": role.kind, "color": KIND_COLORS[role.kind],
                                    "stops": sorted(role.stops)})
        return self._role_idx[key]

    def set_role(self, car: Car, role, log=True, force=False):
        if car.role == role:
            return
        if getattr(car, "out", False) and not force:
            return                           # cabin đang sự cố: bỏ qua lệnh đổi vai
        car.role = role
        car.role_since = self.t
        # bỏ các pickup không còn hợp lệ với vai mới
        car.pickups = {pk for pk in car.pickups if pk[0] in role.stops}
        if log:
            self.events.append({"t": round(self.t, 1), "type": "role", "car": car.id,
                                "role": self.role_index(role), "label": role.label})

    # -------------------------------------------------------------- queries
    def eligible(self, car: Car, f: int, d: int) -> list:
        """Hành khách đang chờ ở tầng f, chiều d, mà cabin (với vai hiện tại) chở được."""
        role = car.role
        if not role.accepts_direction(f, d):
            return []
        out = []
        for p in self.waiting[f]:
            if p.direction != d or not role.can_serve(p.origin, p.dest):
                continue
            if p.fake_up and not p.via_done and not any(s > f for s in role.stops):
                continue
            if not self.dispatcher.may_board(car, p):
                continue
            out.append(p)
        return out

    def hall_calls(self) -> dict:
        """(tầng, chiều) -> tuổi cuộc gọi (s) – thông tin nút gọi tầng mà hệ thống biết."""
        calls = {}
        for f in range(N_FLOORS):
            for p in self.waiting[f]:
                key = (f, p.direction)
                age = self.t - p.t_arr
                if age > calls.get(key, -1):
                    calls[key] = age
        return calls

    def group_calls(self) -> dict:
        """Cuộc gọi theo nhóm bảng gọi: (tầng, chiều, tập cabin phục vụ được) -> (tuổi, số người).
        Tương ứng thực tế: mỗi nhóm thang có bảng gọi riêng, khách bấm ở nhóm đi được tới đích."""
        out = {}
        for f in range(N_FLOORS):
            for p in self.waiting[f]:
                d = p.direction
                cars = frozenset(c.id for c in self.cars
                                 if c.role.accepts_direction(f, d) and c.role.can_serve(p.origin, p.dest)
                                 and not (p.fake_up and not p.via_done and not any(s > f for s in c.role.stops)))
                # cars rỗng = khách "mồ côi" (không cabin nào chở được với phân vai hiện tại)
                key = (f, d, cars)
                age, n = out.get(key, (0.0, 0))
                out[key] = (max(age, self.t - p.t_arr), n + 1)
        return out

    def requests(self, car: Car) -> set:
        return set(car.car_calls) | {f for f, _ in car.pickups}

    # ------------------------------------------------------------- main loop
    def add_passengers(self, passengers):
        """Thêm khách trong lúc đang chạy (chế độ thời gian thực). t_arr tính theo đồng hồ mô phỏng."""
        new = sorted(passengers, key=lambda p: p.t_arr)
        self.passengers.extend(new)
        merged = sorted(list(self.pending) + new, key=lambda p: p.t_arr)
        self.pending = deque(merged)

    def step(self):
        """Tiến một bước DT. Dùng cho chế độ thời gian thực; run() gọi lặp hàm này."""
        k = self.k
        self.t = round(k * DT, 3)
        while self.pending and self.pending[0].t_arr <= self.t + 1e-9:
            p = self.pending.popleft()
            self.waiting[p.origin].append(p)
            self.n_arrived += 1
            self.dispatcher.on_arrival(p, self.t)
        if k % 10 == 0:
            self._behaviour()
            self._check_outages()
        if k % self._disp_every == 0:
            self.dispatcher.update(self.t)
        for car in self.cars:
            self._step(car)
        if self.record and k % self._log_every == 0:
            self._log_frame()
        self.k += 1

    def _check_outages(self):
        from algorithms.roles import out_role
        for cid, t0, t1 in self.outages:
            car = self.car_by_id[cid]
            active = t0 <= self.t < t1
            if active and not getattr(car, "out", False):
                car.out = True
                car.pickups = set()
                self.set_role(car, out_role(), force=True)
                self.events.append({"t": round(self.t, 1), "type": "fault", "car": cid, "label": "Sự cố"})
            elif not active and getattr(car, "out", False) and self.t >= t1:
                car.out = False
                self.set_role(car, self.dispatcher.initial_role(car), force=True)
                self.events.append({"t": round(self.t, 1), "type": "fault_end", "car": cid, "label": "Hoạt động lại"})

    def run(self):
        n_steps = int(round(self.horizon / DT))
        self._log_every = int(round(LOG_EVERY / DT))
        self._disp_every = int(round(self.dispatcher.update_interval / DT))
        for _ in range(n_steps + 1):
            self.step()
            if not self.pending and self._all_done():
                if self.clearance is None:
                    self.clearance = self.t
                # chạy thêm 5 s cho hình ảnh rồi dừng; năng lượng chờ phần còn lại tính bù
                if self.t >= self.clearance + 5:
                    self._idle_tail()
                    break
        return self

    def _all_done(self):
        return all(not w for w in self.waiting) and all(not c.pax for c in self.cars)

    def _idle_tail(self):
        rest = self.horizon - self.t
        for c in self.cars:
            c.energy += (P_STANDBY if c.state == "standby" else P_IDLE) * rest

    def _behaviour(self):
        for f in range(N_FLOORS):
            for p in self.waiting[f]:
                if p.fake_up and not p.via_done and self.t - p.t_arr > FAKE_UP_PATIENCE:
                    p.fake_up = False

    # ------------------------------------------------------------- car logic
    def _step(self, car: Car):
        st = car.state
        if st == "moving":
            car.busy_time += DT
            car.timer -= DT
            if car.timer <= 1e-9:
                self._arrive(car)
        elif st == "dwell":
            car.busy_time += DT
            car.energy += P_IDLE * DT
            car.timer -= DT
            if car.timer <= 1e-9:
                n_in, _ = self._board(car, car.pos, car.serve_dir)
                if n_in:
                    car.timer = n_in * TRANSFER_TIME
                    return
                car.sensed_load = self.sensors.load(car.load)   # cân tải đo lại khi đóng cửa
                car.state = "idle"
                self._decide(car)
        elif st == "waking":
            car.energy += P_IDLE * DT
            car.timer -= DT
            if car.timer <= 1e-9:
                car.state = "idle"
                self._decide(car)
        elif st == "standby":
            car.energy += P_STANDBY * DT
            if not car.role.standby:
                car.state = "waking"
                car.timer = STANDBY_WAKE_TIME
        else:  # idle
            car.energy += P_IDLE * DT
            self._decide(car)

    def _decide(self, car: Car):
        # bỏ pickup đã hết người chờ hợp lệ
        car.pickups = {(f, d) for (f, d) in car.pickups if self.eligible(car, f, d)}
        reqs = self.requests(car)

        if not reqs:
            if car.space > 0:
                for d in self._here_dirs(car):
                    self._open(car, d)
                    return
            park = car.role.parking
            if park is not None and car.pos != park and not car.pax:
                car.parking_move = True
                self._start_move(car, 1 if park > car.pos else -1)
                return
            car.dir = 0
            if car.role.standby and not car.pax:
                car.state = "standby"
            return

        car.parking_move = False
        if car.pos in car.car_calls:
            self._open(car, car.dir or 1)
            return
        if car.dir != 0 and any((f - car.pos) * car.dir > 0 for f in reqs):
            d = car.dir
        elif any(f == car.pos for f, _ in car.pickups):
            dd = next(d for f, d in car.pickups if f == car.pos)
            if car.space > 0 and self.eligible(car, car.pos, dd):
                self._open(car, dd)
                return
            car.pickups.discard((car.pos, dd))
            return
        else:
            nearest = min(reqs, key=lambda f: (abs(f - car.pos), f))
            d = 1 if nearest > car.pos else -1
        # đón khách cùng chiều tại tầng hiện tại trước khi rời đi
        if car.space > 0 and self.eligible(car, car.pos, d):
            self._open(car, d)
            return
        self._start_move(car, d)

    def _here_dirs(self, car):
        dirs = []
        for d in (1, -1):
            el = self.eligible(car, car.pos, d)
            if el:
                dirs.append((min(p.t_arr for p in el), d))
        return [d for _, d in sorted(dirs)]

    def _start_move(self, car: Car, d: int):
        nxt = car.pos + d
        if nxt < 0 or nxt >= N_FLOORS:
            car.dir = 0
            car.state = "idle"
            return
        if car.state != "moving":
            car.energy += E_START
            car.departures += 1
            if car.pos == G:
                car.g_depart = self.t
        car.dir = d
        car.next = nxt
        car.timer = TRAVEL_TIME_PER_FLOOR
        car.state = "moving"

    def _arrive(self, car: Car):
        car.pos = car.next
        car.floors_moved += 1
        load_kg = car.load * AVG_PAX_KG
        car.energy += E_FLOOR * (1 + E_IMBALANCE * abs(load_kg - COUNTERWEIGHT_RATIO * RATED_LOAD_KG) / RATED_LOAD_KG)
        f, d = car.pos, car.dir
        if f == G and car.g_depart is not None:
            car.rtts.append(self.t - car.g_depart)
            car.g_depart = None

        if f in car.car_calls:
            self._open(car, self._serve_dir_after(car, f, d))
            return

        dec = self.dispatcher.wants_stop(car, f, d, self.t) if f in car.role.stops else "pass"
        if dec == "stop":
            self._open(car, d)
            return
        if dec == "bypass":
            self.bypass_pass[f] += 1
            for p in self.eligible(car, f, d):
                p.left_behind += 1
            self.events.append({"t": round(self.t, 1), "type": "bypass", "car": car.id, "floor": f})

        reqs = self.requests(car)
        if car.parking_move and car.role.parking is not None and car.role.parking != f and not reqs:
            self._continue(car, d)
            return
        if any((r - f) * d > 0 for r in reqs):
            self._continue(car, d)
            return
        # điểm quay đầu: đón chiều ngược lại nếu có
        if f in car.role.stops and car.space > 0:
            dec2 = self.dispatcher.wants_stop(car, f, -d, self.t)
            if dec2 == "stop":
                self._open(car, -d)
                return
        car.parking_move = False
        car.state = "idle"
        self._decide(car)

    def _continue(self, car, d):
        nxt = car.pos + d
        if 0 <= nxt < N_FLOORS:
            car.next = nxt
            car.timer = TRAVEL_TIME_PER_FLOOR
        else:
            car.state = "idle"
            self._decide(car)

    def _serve_dir_after(self, car, f, d):
        """Chiều phục vụ khi dừng: giữ chiều nếu còn việc phía trước, ngược lại đảo chiều."""
        remaining = {r for r in car.car_calls if r != f} | {pf for pf, _ in car.pickups if pf != f}
        if any((r - f) * d > 0 for r in remaining):
            return d
        if self.eligible(car, f, d):
            return d
        if self.eligible(car, f, -d) or any((r - f) * -d > 0 for r in remaining):
            return -d
        return d

    def _open(self, car: Car, sdir: int):
        f = car.pos
        car.state = "dwell"
        car.parking_move = False
        car.stops += 1
        car.energy += E_DOOR
        # khách xuống
        n_out = 0
        stay = []
        for p in car.pax:
            if p.cur_dest == f:
                if p.fake_up and not p.via_done:
                    p.via_done = True
                    p.cur_dest = p.dest
                    car.car_calls.add(p.dest)
                    stay.append(p)
                else:
                    p.state = "done"
                    p.t_alight = self.t
                    n_out += 1
                    self.n_done += 1
                    car.pax_served += 1
            else:
                stay.append(p)
        car.pax = stay
        car.car_calls.discard(f)
        if car.pax:
            ahead = [r for r in car.car_calls]
            if ahead:
                sdir = 1 if ahead[0] > f else -1
        car.serve_dir = sdir
        car.dir = sdir
        n_in, n_left = self._board(car, f, sdir)
        if n_left:
            self.bypass_left[f] += 1
            self.events.append({"t": round(self.t, 1), "type": "left", "car": car.id, "floor": f, "n": n_left})
        if n_in == 0 and n_out == 0:
            car.useless_stops += 1
        car.timer = DOOR_TIME + TRANSFER_TIME * (n_in + n_out)
        car.pickups.discard((f, sdir))
        # quan sát cảm biến
        left_true = sum(1 for p in self.waiting[f] if p.direction == sdir)
        car.sensed_load = self.sensors.load(car.load)
        ob = {"t": round(self.t, 1), "car": car.id, "floor": f, "dir": sdir,
              "n_in": self.sensors.door_count(n_in), "n_out": self.sensors.door_count(n_out),
              "load": car.sensed_load, "left": self.sensors.landing(left_true),
              "dwell": round(car.timer + float(self.sensors.rng.normal(0, 0.3)), 1)}
        self.obs.append(ob)
        self.dispatcher.on_observation(ob)
        if self.record:
            self.stop_log.append((round(self.t, 1), self.cars.index(car), f, n_in, n_out))

    def _board(self, car: Car, f: int, sdir: int):
        el = self.eligible(car, f, sdir)
        if not el:
            return 0, 0
        take = el[: max(0, car.space)]
        for p in take:
            self.waiting[f].remove(p)
            p.state = "riding"
            p.t_board = self.t
            p.car = car.id
            if p.fake_up and not p.via_done:
                p.cur_dest = max(s for s in car.role.stops if s > f)
            car.car_calls.add(p.cur_dest)
            car.pax.append(p)
        left = len(el) - len(take)
        for p in el[len(take):]:
            p.left_behind += 1
        return len(take), left

    # --------------------------------------------------------------- logging
    def _log_frame(self):
        cars = []
        for c in self.cars:
            cars.append([round(c.pos_float(), 2), c.load, STATE_CODE[c.state], self.role_index(c.role), c.dir])
        floors = []
        for f in range(N_FLOORS):
            up = dn = 0
            oldest = 0.0
            for p in self.waiting[f]:
                if p.direction > 0:
                    up += 1
                else:
                    dn += 1
                oldest = max(oldest, self.t - p.t_arr)
            floors.append([up, dn, int(oldest)])
        self.frames.append({"t": round(self.t, 1), "c": cars, "f": floors, "a": self.n_arrived, "d": self.n_done})
