"""
Thời khóa biểu -> nhu cầu thang máy theo giờ thật.

Nguồn: data/timetable.csv, hoặc Google Sheet công khai (File → Chia sẻ → Công bố lên web → CSV)
đặt trong biến môi trường / .env:  TIMETABLE_URL=https://docs.google.com/.../pub?output=csv
Cột: tang, phong, mon, bat_dau (HH:MM), ket_thuc (HH:MM), si_so

Mỗi lớp sinh ra:
  - trước giờ học 15 phút: si_so × tỷ lệ chọn thang khách từ sảnh (G/B1/B2) lên tầng của lớp, dồn về cuối;
  - sau giờ tan: cùng số khách từ tầng đó về sảnh (90 %) hoặc tầng khác (10 %), phần lớn trong 3–4 phút đầu.
Cộng thêm nhu cầu nền rải rác mọi tầng (0,5 người/phút).
"""
from __future__ import annotations

import csv
import io
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from config import F, G, N_FLOORS, UPPER
from simulation.passenger_generator import Passenger

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELEVATOR_SHARE = 0.7          # tỷ lệ chọn thang (giả định – thay bằng khảo sát)
BACKGROUND_PER_MIN = 0.5


def hhmm(s: str) -> int:
    h, m = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60


def clock_str(sec: float) -> str:
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


@dataclass
class ClassRow:
    floor: int
    room: str
    subject: str
    start: int
    end: int
    size: int


def load(path_or_url: str | None = None) -> list[ClassRow]:
    src = path_or_url or os.environ.get("TIMETABLE_URL") or os.path.join(HERE, "data", "timetable.csv")
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"User-Agent": "elevator-digital-twin/1.0"})
        text = urllib.request.urlopen(req, timeout=15).read().decode("utf-8-sig")
    else:
        with open(src, encoding="utf-8-sig") as fh:
            text = fh.read()
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            rows.append(ClassRow(F(r["tang"].strip()), r.get("phong", ""), r.get("mon", ""),
                                 hhmm(r["bat_dau"]), hhmm(r["ket_thuc"]), int(float(r["si_so"]))))
        except (KeyError, ValueError):
            continue
    return rows


def _lobby(rng):
    r = rng.random()
    return G if r < 0.82 else (F("B1") if r < 0.94 else F("B2"))


class Timetable:
    def __init__(self, rows=None, seed=0):
        self.rows = rows if rows is not None else load()
        self.seed = seed

    def trips_between(self, c0: float, c1: float, rng) -> list[tuple]:
        """Hành trình (giờ_đồng_hồ_s, tầng_đi, tầng_đến) có thời điểm đến sảnh trong [c0, c1)."""
        out = []
        for r in self.rows:
            n = int(round(r.size * ELEVATOR_SHARE))
            # lên lớp: [start-15', start+2']
            a, b = r.start - 900, r.start + 120
            if b > c0 and a < c1:
                times = a + rng.beta(2.2, 1.3, n) * (b - a)
                for t in times[(times >= c0) & (times < c1)]:
                    out.append((float(t), _lobby(rng), r.floor))
            # tan lớp: phần lớn trong 3–4 phút
            a, b = r.end, r.end + 600
            if b > c0 and a < c1:
                times = np.minimum(a + rng.uniform(0, 150, n) + rng.exponential(60, n), b)
                for t in times[(times >= c0) & (times < c1)]:
                    d = _lobby(rng) if rng.random() < 0.9 else int(rng.choice([f for f in UPPER if f != r.floor]))
                    out.append((float(t), r.floor, d))
        n_bg = rng.poisson(BACKGROUND_PER_MIN * (c1 - c0) / 60)
        for t in rng.uniform(c0, c1, n_bg):
            o, d = rng.choice(N_FLOORS, 2, replace=False)
            out.append((float(t), int(o), int(d)))
        return sorted(out)

    def passengers_between(self, c0, c1, clock0, rng, id_start=0) -> list[Passenger]:
        """Như trips_between nhưng trả Passenger với t_arr tính từ clock0 (giờ bắt đầu mô phỏng)."""
        return [Passenger(id_start + i, round(t - clock0, 2), o, d)
                for i, (t, o, d) in enumerate(self.trips_between(c0, c1, rng))]

    def upcoming(self, clock: float, horizon: float = 900) -> list[dict]:
        """Sự kiện trong `horizon` giây tới: lớp sắp vào / sắp tan, số khách thang dự kiến."""
        ev = []
        for r in self.rows:
            n = int(round(r.size * ELEVATOR_SHARE))
            if clock - 300 <= r.start <= clock + horizon:
                ev.append({"time": clock_str(r.start), "sec": r.start, "floor": r.floor, "kind": "vào lớp",
                           "room": r.room, "subject": r.subject, "n": n})
            if clock - 300 <= r.end <= clock + horizon:
                ev.append({"time": clock_str(r.end), "sec": r.end, "floor": r.floor, "kind": "tan lớp",
                           "room": r.room, "subject": r.subject, "n": n})
        return sorted(ev, key=lambda e: e["sec"])

    def mode_hint(self, clock: float, window: float = 900) -> str:
        """Chế độ giao thông dự kiến theo thời khóa biểu (làm thông tin trước cho P2)."""
        rng = np.random.default_rng(1)
        trips = self.trips_between(clock - 120, clock + window, rng)
        if len(trips) < 12:
            return "OFF_PEAK"
        from config import LOBBY
        up = sum(1 for _, o, d in trips if o in LOBBY and d not in LOBBY)
        dn = sum(1 for _, o, d in trips if d in LOBBY and o not in LOBBY)
        n = len(trips)
        if up / n > 0.65:
            return "UP_PEAK"
        if dn / n > 0.65:
            return "DOWN_PEAK"
        return "TWO_WAY"
