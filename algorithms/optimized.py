"""
Phương án tối ưu P1–P3 (giả định mô phỏng: hệ thống điều khiển được điểm dừng & vai trò cabin).

P1 – Ưu tiên tầng cao (phân vai TĨNH theo thời khóa biểu):
     nhu cầu dự kiến -> planner chia nhóm tầng -> Express/Zone cho A2–A6, A1 Local.
P2 – Điều phối động (Digital Twin): mỗi 20 s đọc cảm biến cabin (cân tải, hồng ngoại ở cửa) + nút gọi tầng,
     AI nhận diện chế độ giao thông, ước lượng nhu cầu, tính lại phân vai (có hysteresis,
     thời gian giữ vai tối thiểu), standby khi thấp điểm.
P3 – Thấp điểm: A1 + A2 dừng mọi tầng theo yêu cầu, các cabin khác standby, đánh thức khi chờ lâu.

Chung cho cả ba:
- Gán cuộc gọi theo hàm chi phí: ETA + tải (cân tải) + số điểm dừng + phạt dùng A1 cho
  hành trình đã có cabin chuyên trách; cuộc gọi chờ lâu (ước lượng đông người) được gán thêm cabin.
- Cân tải ước số người >= sức chứa hữu dụng -> không dừng vô ích (bypass có kiểm soát).
- A1 luôn là Local; ở sảnh A1 nhường các tầng đã có Express/Zone, nhưng nếu cuộc gọi chờ quá
  fallback_wait thì A1 nhận mọi hành trình (không bỏ mặc khách).
"""
from __future__ import annotations

import math
from collections import defaultdict

from ai.features import MODE_VI, window_features
from ai.scenario_classifier import ModeDetector
from algorithms.planner import assign_cars, plan_groups, score_of, utilization
from algorithms.roles import Role, base_role, group_role, local_role, shuttle_role, standby_role
from config import (
    E_DOOR, E_FLOOR, E_START,
    DOOR_TIME, F, G, LOBBY, LOCAL_CAR, OPT_PARAMS, STANDBY_WAKE_TIME, TRANSFER_TIME,
    TRAVEL_TIME_PER_FLOOR, UPPER,
)

STOP_COST = DOOR_TIME + 4 * TRANSFER_TIME
REPORT_TTL = 180.0   # s – báo cáo hàng chờ còn hiệu lực trong 3 phút (mô phỏng)


class OptimizedDispatcher:
    update_interval = 1.0

    def __init__(self, variant: str = "P2", params: dict | None = None, plan_passengers=None,
                 scenario_mode: str | None = None, use_model: bool = True):
        self.variant = variant
        self.name = variant
        self.title = {"P1": "P1 – Ưu tiên tầng cao", "P2": "P2 – Điều phối động",
                      "P3": "P3 – Thấp điểm/Standby"}[variant]
        base = dict(OPT_PARAMS)
        if variant == "P3":   # thấp điểm: ưu tiên năng lượng – chỉ đánh thức khi chờ ≥ 45 s
            base.update(wake_wait=45.0)
        self.p = dict(base, **(params or {}))
        self.plan_passengers = plan_passengers
        self.scenario_mode = scenario_mode
        self.use_model = use_model

    # ------------------------------------------------------------ lifecycle
    def attach(self, sim):
        self.sim = sim
        self.cap = sim.capacity
        self.assign: dict = {}
        self.land: dict = {}            # (f,d) -> (t, còn người chờ hay không – từ nút gọi)
        self.reported: dict = {}        # (f,d) -> (t, số người) – báo cáo hàng chờ của người dùng (form QR)
        self.mode = "TWO_WAY"
        self.mode_log: list = []
        self.plan = []                  # [(group, n)]
        self.labels: dict = {}          # car id -> nhãn slot
        self.last_replan = -1e9
        self.last_apply = -1e9
        self.pending: dict = {}         # car id -> role chờ áp dụng khi cabin rỗng
        self.idle_since: dict = {}
        self.offpeak_votes = 0
        self.pooled = False
        self.last_rho = 0.0
        self.fallback_until = -1.0
        self.span = None
        self.active_ids = None
        self.detector = ModeDetector(self.use_model, self.p["offpeak_rate"]) if self.variant == "P2" else None
        self.locals = [c for c in sim.cars if c.id == LOCAL_CAR]
        self.others = [c for c in sim.cars if c.id != LOCAL_CAR]
        self.init_roles = {}
        if self.variant == "P1":
            mode = self.scenario_mode or "TWO_WAY"
            if mode == "OFF_PEAK":
                mode = "TWO_WAY"
            D, span = self._planned_demand(mode)
            self.mode = mode
            self.init_roles = self._make_roles(mode, D, span)
        elif self.variant == "P3":
            self.init_roles = self._offpeak_roles()
            self.mode = "OFF_PEAK"
        elif self.plan_passengers:   # P2 có thời khóa biểu: khởi động bằng phân vai dự kiến, sau đó tự thích nghi
            mode = self.scenario_mode or "TWO_WAY"
            self.mode = mode
            if mode == "OFF_PEAK":
                self.init_roles = self._offpeak_roles()
            else:
                D, span = self._planned_demand(mode)
                ts = [p.t_arr for p in self.plan_passengers]
                self._prior_window = max(60.0, (max(ts) - min(ts)) * 0.7) if ts else 600.0
                self.init_roles = self._make_roles(mode, D, span, window=self._prior_window)
        else:  # P2 không có thời khóa biểu: khởi động bằng cấu hình gốc, A1 Local
            self.init_roles = {c.id: (local_role() if c.id == LOCAL_CAR else base_role(c.baseline)) for c in sim.cars}
        if self.plan_passengers and self.variant == "P2":
            self.last_replan = 0.0   # tin thời khóa biểu trong chu kỳ đầu, chờ có quan sát
        self.mode_log.append({"t": 0.0, "mode": self.mode})

    def initial_role(self, car):
        return self.init_roles[car.id]

    def on_arrival(self, p, t):
        pass

    def on_observation(self, ob):
        self.land[(ob["floor"], ob["dir"])] = (ob["t"], ob["left"])

    def may_board(self, car, p) -> bool:
        return True

    # ------------------------------------------------------------- demand
    def _planned_demand(self, mode):
        D = defaultdict(float)
        inter = []
        for p in self.plan_passengers or []:
            o, d = p.origin, p.dest
            if o in LOBBY and d not in LOBBY:
                if mode in ("UP_PEAK", "TWO_WAY"):
                    D[d] += 1
            elif d in LOBBY and o not in LOBBY:
                if mode in ("DOWN_PEAK", "TWO_WAY"):
                    D[o] += 1
            elif o not in LOBBY and d not in LOBBY:
                inter += [o, d]
        span = (min(inter), max(inter)) if inter else None
        return D, span

    def _est_crowd(self, f, d, age):
        rate = self._rate.get((f, d), 0.0)
        base = 1.0
        L = self.land.get((f, d))
        if L and self.sim.t - L[0] <= age + 1:
            base = max(1.0, L[1])
        R = self.reported.get((f, d))
        if R and self.sim.t - R[0] <= REPORT_TTL:
            base = max(base, float(R[1]))
        return base + rate * min(age, self.p["window"]) / 60.0

    def _observed_demand(self, t, mode, calls):
        W = self.p["window"]
        D = defaultdict(float)
        self._rate = defaultdict(float)
        inter_floors = []
        for ob in reversed(self.sim.obs):
            if ob["t"] < t - W:
                break
            w = math.exp(-(t - ob["t"]) / W)
            f = ob["floor"]
            self._rate[(f, ob["dir"])] += ob["n_in"] / (W / 60.0)
            if f in LOBBY:
                continue
            if ob["n_out"] and mode in ("UP_PEAK", "TWO_WAY"):
                D[f] += w * ob["n_out"]
            if ob["dir"] < 0 and mode in ("DOWN_PEAK", "TWO_WAY"):
                D[f] += w * ob["n_in"]
            if ob["n_in"] or ob["n_out"]:
                inter_floors.append(f)
        for (f, d), age in calls.items():
            if f in LOBBY:
                continue
            inter_floors.append(f)
            if (d < 0 and mode in ("DOWN_PEAK", "TWO_WAY")) or mode == "INTERFLOOR":
                D[f] += self._est_crowd(f, d, age)
        # tầng đích đã bấm trong cabin (nút trong cabin) – thông tin sớm cho cao điểm lên
        if mode in ("UP_PEAK", "TWO_WAY"):
            for c in self.sim.cars:
                for f in c.car_calls:
                    if f not in LOBBY and c.dir > 0:
                        D[f] += 2.0
        span = (min(inter_floors), max(inter_floors)) if inter_floors else None
        return D, span

    # -------------------------------------------------------------- roles
    def _offpeak_roles(self):
        roles = {}
        keep = self.p["standby_keep"]
        for c in self.sim.cars:
            if c.id == LOCAL_CAR:
                roles[c.id] = local_role()
            elif keep > 1 and c.id == "A2":
                r = local_role()
                roles[c.id] = r.__class__("LOCAL", r.stops, "Local theo yêu cầu", r.exclude, 0, self.p.get("offpeak_park"))
            else:
                roles[c.id] = standby_role((), G)
        return roles

    # ------------------------------------------------ tiết kiệm năng lượng (eco)
    def _choose_active(self, D):
        """Chế độ eco: chỉ giữ đủ số cabin để hệ số sử dụng ≈ u_target, còn lại standby.
        Số cabin cần = λ × T_vòng / (tải mỗi vòng × u_target)."""
        u = self.p.get("u_target")
        if not u:
            self.active_ids = None
            return
        need = utilization(D, self.p["window"], 1, self.cap)          # số "cabin tương đương" cần
        n_act = int(min(len(self.sim.cars), max(self.p.get("eco_min_active", 2), math.ceil(need / u))))
        def rank(c):
            busy = 1 if (c.pax or c.car_calls) else 0
            awake = 0 if c.role.standby else 1
            return (-busy, -awake, abs(c.pos - G), c.id)
        keep = sorted(self.others, key=rank)[: n_act - 1]
        self.active_ids = {LOCAL_CAR} | {c.id for c in keep}

    def _act_others(self):
        ok = [c for c in self.others if not getattr(c, "out", False)]      # bỏ cabin đang sự cố
        if not getattr(self, "active_ids", None):
            return ok
        return [c for c in ok if c.id in self.active_ids]

    def _standby_rest(self, roles):
        if getattr(self, "active_ids", None):
            for c in self.sim.cars:
                if c.id not in self.active_ids:
                    roles[c.id] = standby_role((), G)
        return roles

    def _pooled_roles(self, mode):
        """Tải vừa: mọi cabin phục vụ chung các tầng trong tầm với (gộp năng lực, không chia vùng)."""
        park = G if mode == "UP_PEAK" else None
        roles = {}
        act = {c.id for c in self._act_others()} | {LOCAL_CAR}
        for c in self.sim.cars:
            if c.id not in act:
                roles[c.id] = standby_role((), G)
            elif c.id == LOCAL_CAR:
                roles[c.id] = local_role(parking=park)
            else:
                stops = frozenset(c.reach) | frozenset(c.baseline)
                roles[c.id] = Role("LOCAL", stops, "Local chung", frozenset(), 0, park)
        return roles

    def _make_roles(self, mode, D, span, window=None):
        cap = self.cap
        others = self._act_others()
        K = len(others)
        self.pooled = False
        if self.variant == "P2" and mode in self.p.get("pool_modes", ("UP_PEAK", "DOWN_PEAK", "TWO_WAY")):
            rho = utilization(D, window or self.p["window"], K + 1, cap)
            self.last_rho = rho
            if rho < self.p["pool_util"]:
                self.pooled = True
                self.plan = []
                return self._pooled_roles(mode)
        slots, slot_role, target, slot_floors = [], {}, {}, {}
        exclude = set()
        dirp = {"UP_PEAK": 1, "DOWN_PEAK": -1}.get(mode, 0)
        plan = []
        k_lobby = K
        self.span = span if mode == "INTERFLOOR" else None
        if mode == "INTERFLOOR" and span:
            lo, hi = span
            floors = list(range(lo, hi + 1))
            n_sh = max(min(2, K), min(K, round(K * 0.8)))
            for i in range(n_sh):
                lab = f"SH{i}"
                slots.append(lab)
                slot_role[lab] = ("shuttle", floors)
                target[lab] = (lo + hi) // 2
                slot_floors[lab] = floors
            k_lobby = K - n_sh
        if k_lobby > 0:
            score, plan, rest = plan_groups(D, k_lobby, cap, self.p["min_floor_share"])
            for gi, (g, n) in enumerate(plan):
                if mode != "INTERFLOOR":
                    exclude |= set(g)
                for j in range(n):
                    lab = f"G{gi}_{j}"
                    slots.append(lab)
                    slot_role[lab] = ("group", g)
                    target[lab] = max(g) if mode == "DOWN_PEAK" else G
                    slot_floors[lab] = g
        self.plan = plan
        # còn cabin chưa có vai (không có nhu cầu) -> giữ tuyến gốc
        while len(slots) < K:
            lab = f"B{len(slots)}"
            slots.append(lab)
            slot_role[lab] = ("base", None)
            target[lab] = G
            slot_floors[lab] = []
        mapping = assign_cars(slots, others, self.labels, target, slot_floors)
        self.labels = mapping
        roles = {}
        for c in others:
            lab = mapping[c.id]
            kind, fl = slot_role[lab]
            native = [f for f in c.baseline if f in (F("B1"), F("B2"))]
            if kind == "shuttle":
                roles[c.id] = shuttle_role(fl, native, parking=target[lab])
            elif kind == "group":
                park = target[lab] if mode in ("DOWN_PEAK", "UP_PEAK") else None
                roles[c.id] = group_role(fl, native, dir_priority=dirp, parking=park)
            else:
                roles[c.id] = base_role(c.baseline)
        if self.sim.t < self.fallback_until:
            exclude = set()
        roles[LOCAL_CAR] = local_role(frozenset(exclude))
        return self._standby_rest(roles)

    def _apply(self, roles, force=False):
        for c in self.sim.cars:
            r = roles.get(c.id)
            if r is None or getattr(c, "out", False):
                continue                   # cabin sự cố: giữ nguyên trạng thái sự cố
            if r.standby and (c.pax or c.car_calls):
                self.pending[c.id] = r
                continue
            self.pending.pop(c.id, None)
            self.sim.set_role(c, r)

    # ------------------------------------------------------------- update
    def update(self, t):
        sim = self.sim
        calls = sim.hall_calls()
        if not hasattr(self, "_rate"):
            self._rate = defaultdict(float)

        if self.variant == "P2" and t - self.last_replan >= self.p["replan_interval"]:
            self.last_replan = t
            x = window_features(sim.obs, t, self.p["window"], calls)
            mode = self.detector.predict(x)
            # hysteresis cho thấp điểm: cần 2 lần liên tiếp
            if mode == "OFF_PEAK":
                self.offpeak_votes += 1
                if self.offpeak_votes < 2:
                    mode = self.mode if self.mode != "OFF_PEAK" else mode
            else:
                self.offpeak_votes = 0
            D, span = self._observed_demand(t, mode, calls)
            if mode == "OFF_PEAK":
                new_roles = self._offpeak_roles()
                changed = self.mode != "OFF_PEAK"
                if changed:
                    self._apply(new_roles)
                    self.last_apply = t
            else:
                was_pooled = getattr(self, "pooled", False)
                old_span = getattr(self, "span", None)
                old_active = getattr(self, "active_ids", None)
                self._choose_active(D)
                cur_score = score_of(D, self.plan, self.cap) if self.plan else float("inf")
                saved = (self.plan, self.labels, was_pooled, old_span)
                new_roles = self._make_roles(mode, D, span)
                new_score = score_of(D, self.plan, self.cap) if self.plan else float("inf")
                span_grew = (mode == "INTERFLOOR" and self.span and
                             (old_span is None or self.span[0] < old_span[0] or self.span[1] > old_span[1]))
                structural = (mode != self.mode or self.pooled != was_pooled or bool(span_grew)
                              or self.active_ids != old_active)
                better = (not self.pooled) and new_score < (1 - self.p["improve_ratio"]) * cur_score
                hold_ok = t - self.last_apply >= self.p["min_hold"]
                if (structural and (hold_ok or mode != self.mode)) or (better and hold_ok):
                    self._apply(new_roles)
                    self.last_apply = t
                else:
                    self.plan, self.labels, self.pooled, self.span = saved
                    self.active_ids = old_active
            if mode != self.mode:
                self.mode_log.append({"t": round(t, 1), "mode": mode})
                sim.events.append({"t": round(t, 1), "type": "mode", "mode": mode, "label": MODE_VI[mode]})
            self.mode = mode

        # áp dụng vai chờ (standby khi cabin đã rỗng)
        for cid, r in list(self.pending.items()):
            c = sim.car_by_id[cid]
            if not c.pax and not c.car_calls:
                sim.set_role(c, r)
                self.pending.pop(cid)

        self._fallback(t, calls)
        self._standby_logic(t, calls)
        self._assign_calls(t)

    def _fallback(self, t, calls):
        """A1 nhận mọi hành trình nếu khách của nhóm chuyên trách chờ quá lâu."""
        a1 = self.sim.car_by_id[LOCAL_CAR]
        if not a1.role.exclude:
            return
        for (f, d, group), (age, _) in self.sim.group_calls().items():
            if LOCAL_CAR not in group and age > self.p["fallback_wait"]:
                self.sim.set_role(a1, local_role())
                self.fallback_until = t + 30.0
                self.sim.events.append({"t": round(t, 1), "type": "fallback", "car": LOCAL_CAR})
                return

    def _standby_logic(self, t, calls):
        sim = self.sim
        sb = [c for c in sim.cars if c.role.standby and not getattr(c, "out", False)]
        if not sb:
            return
        oldest = max(calls.values(), default=0.0)
        wake = self.p.get("eco_wake_wait", 60.0) if self.p.get("u_target") else self.p["wake_wait"]
        if oldest > wake:
            # đánh thức cabin standby gần cuộc gọi cũ nhất
            (f, d) = max(calls, key=calls.get)
            car = min(sb, key=lambda c: abs(c.pos - f))
            sim.set_role(car, base_role(car.baseline) if f in car.baseline else local_role())
            self.idle_since[car.id] = t
            return
        # cabin đã được đánh thức, rảnh lâu -> về standby (chỉ ở chế độ thấp điểm)
        if self.mode == "OFF_PEAK":
            target = self._offpeak_roles()
            for c in sim.cars:
                if target[c.id].standby and not c.role.standby:
                    busy = c.pax or sim.requests(c) or c.state != "idle"
                    if busy:
                        self.idle_since[c.id] = t
                    elif t - self.idle_since.get(c.id, t) > 60:
                        sim.set_role(c, target[c.id])

    # ------------------------------------------------------- call assignment
    def eta(self, car, f, d):
        sim = self.sim
        pos = car.pos_float()
        reqs = sim.requests(car)
        extra = STANDBY_WAKE_TIME if car.state in ("standby", "waking") else 0.0
        if car.state == "dwell":
            extra += max(0.0, car.timer)
        cd = car.dir if (car.state == "moving" or car.pax) else 0
        if cd == 0 or not reqs:
            return extra + abs(pos - f) * TRAVEL_TIME_PER_FLOOR
        if (f - pos) * cd >= 0 and d == cd:
            n_st = sum(1 for r in reqs if (r - pos) * cd > 0 and (f - r) * cd > 0)
            return extra + abs(f - pos) * TRAVEL_TIME_PER_FLOOR + n_st * STOP_COST
        far = max(reqs) if cd > 0 else min(reqs)
        far = far if (far - pos) * cd > 0 else pos
        return extra + (abs(far - pos) + abs(far - f)) * TRAVEL_TIME_PER_FLOOR + len(reqs) * STOP_COST

    def _cost(self, car, f, d, group):
        p = self.p
        c = p["w_eta"] * self.eta(car, f, d)
        c += p["w_load"] * (car.sensed_load / self.cap)
        c += p["w_stops"] * len(self.sim.requests(car))
        if car.id == LOCAL_CAR and len(group) > 1:
            c += p["w_role"]
        if car.sensed_load >= self.cap:
            c += 1e4
        if p.get("w_energy"):
            c += p["w_energy"] * self._energy_cost(car, f, d)
        return c

    def _energy_cost(self, car, f, d):
        """Năng lượng tăng thêm (EU) nếu giao cuộc gọi cho cabin này: khởi động + quãng đường đi thêm."""
        reqs = self.sim.requests(car)
        idle = car.state in ("idle", "standby", "waking") and not reqs and not car.pax
        if idle:
            return E_START + abs(car.pos - f) * E_FLOOR
        cd = car.dir
        if cd and (f - car.pos_float()) * cd >= 0 and d == cd:
            return E_START + E_DOOR                       # dừng thêm trên đường đang đi
        far = (max(reqs) if cd > 0 else min(reqs)) if reqs and cd else car.pos
        return E_START + E_DOOR + max(0.0, abs(far - f) - abs(far - car.pos)) * E_FLOOR

    def _assign_calls(self, t):
        sim = self.sim
        calls = sim.group_calls()
        new_assign = {}
        for key, (age, _) in sorted(calls.items(), key=lambda kv: -kv[1][0]):
            f, d, group = key
            cands = [sim.car_by_id[cid] for cid in group if not sim.car_by_id[cid].role.standby]
            if not cands:
                continue
            scored = sorted(cands, key=lambda c: self._cost(c, f, d, group))
            need = min(3, max(1, math.ceil(self._est_crowd(f, d, age) / self.cap)))
            prev = self.assign.get(key, [])
            chosen = []
            # giữ phân công cũ nếu không tệ hơn nhiều (ổn định)
            for cid in prev:
                c = sim.car_by_id[cid]
                if c in cands and self._cost(c, f, d, group) <= self._cost(scored[0], f, d, group) + 8:
                    chosen.append(c)
            for c in scored:
                if len(chosen) >= need:
                    break
                if c not in chosen:
                    chosen.append(c)
            new_assign[key] = [c.id for c in chosen[:need]]
        self.assign = new_assign
        pick = defaultdict(set)
        for (f, d, _), cids in new_assign.items():
            for cid in cids:
                pick[cid].add((f, d))
        for c in sim.cars:
            c.pickups = pick.get(c.id, set())

    def wants_stop(self, car, f, d, t) -> str:
        if not self.sim.eligible(car, f, d):
            return "pass"
        if car.sensed_load >= self.cap:   # cân tải: ước số người đã đủ sức chứa -> không dừng vô ích
            return "bypass"
        return "stop"