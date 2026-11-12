"""Quản lý một phiên chạy trực tiếp trong web_app: tòa nhà ảo + twin + thu thập dữ liệu cho dashboard."""
from __future__ import annotations

import threading
import time

from realtime.building import VirtualBuilding
from realtime.bus import get_bus, prefix
from realtime.timetable import Timetable, clock_str, hhmm
from realtime.twin import TwinService


class LiveManager:
    def __init__(self):
        self.bus = None
        self.building = None
        self.twin = None
        self.latest = {}
        self.history = []          # (clock, twin ước lượng tổng chờ, thật tổng chờ)
        self.started = None
        self.lock = threading.Lock()

    @property
    def running(self):
        return self.building is not None

    def start(self, clock: str | None = None, speed: float = 10.0, control: str = "closed", capacity: int = 15):
        self.stop()
        tt = Timetable()
        lt = time.localtime()
        clock0 = hhmm(clock) if clock else lt.tm_hour * 3600 + lt.tm_min * 60
        self.bus = get_bus()
        pfx = prefix()
        for t in ("state", "truth", "twin", "cmd/roles"):
            self.bus.subscribe(f"{pfx}/{t}", self._collect)
        self.twin = TwinService(self.bus, clock0, capacity, tt)
        self.building = VirtualBuilding(self.bus, clock0, speed, capacity, control=control, timetable=tt)
        self.twin.start()
        self.building.start()
        self.history = []
        self.started = {"clock": clock_str(clock0), "speed": speed, "control": control, "bus": self.bus.kind,
                        "prefix": pfx, "n_classes": len(tt.rows)}
        return self.started

    def stop(self):
        if self.building:
            self.building.stop()
        if self.twin:
            self.twin.stop()
        if self.bus:
            self.bus.close()
        self.building = self.twin = self.bus = None
        self.latest = {}

    def _collect(self, topic, msg):
        key = topic.rsplit("/", 1)[-1] if "cmd/" not in topic else "cmd"
        with self.lock:
            self.latest[key] = msg
            if key == "truth" and "twin" in self.latest:
                real = sum(u + d for u, d, _ in msg["waiting"])
                est = sum(self.latest["twin"].get("est_waiting", []))
                if not self.history or msg["t"] - self.history[-1]["t"] >= 5:
                    self.history.append({"t": msg["t"], "clock": msg["clock"], "real": real, "est": round(est, 1)})
                    self.history = self.history[-240:]

    def snapshot(self):
        with self.lock:
            h = self.history
            err = None
            if len(h) >= 5:
                pairs = [(x["real"], x["est"]) for x in h[-60:]]
                mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
                mean_real = sum(a for a, _ in pairs) / len(pairs)
                err = {"mae": round(mae, 1), "mean_real": round(mean_real, 1)}
            return {"running": self.running, "session": self.started, "state": self.latest.get("state"),
                    "truth": self.latest.get("truth"), "twin": self.latest.get("twin"), "history": h[-120:],
                    "accuracy": err}

    def publish(self, sub, payload):
        if self.bus:
            self.bus.publish(f"{prefix()}/{sub}", payload)
            return True
        return False
