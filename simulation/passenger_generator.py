"""
Sinh hành khách (OD theo thời gian) cho các tình huống.

Công thức nhu cầu (mục 3 tài liệu):
    N(tầng, giờ) = số lớp × sĩ số trung bình × tỷ lệ tham gia × tỷ lệ chọn thang

Có thể nạp danh sách hành khách từ CSV khảo sát (load_csv) thay cho bộ sinh ngẫu nhiên.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

from config import F, G, LOBBY, N_FLOORS, fname


@dataclass
class Passenger:
    id: int
    t_arr: float
    origin: int
    dest: int
    fake_up: bool = False          # tùy chọn: bấm/đi ngược chiều lên trước để giữ chỗ
    # trạng thái runtime
    state: str = "waiting"         # waiting | riding | done
    cur_dest: int = -1
    car: str | None = None
    t_board: float | None = None
    t_alight: float | None = None
    left_behind: int = 0           # số lần bị bỏ lại do cabin đầy
    via_done: bool = False

    def __post_init__(self):
        self.cur_dest = self.dest

    @property
    def direction(self) -> int:
        if self.fake_up and not self.via_done:
            return 1
        return 1 if self.dest > self.origin else -1

    def clone(self) -> "Passenger":
        return Passenger(self.id, self.t_arr, self.origin, self.dest, self.fake_up)


def demand(n_classes: int, class_size: float, participation: float = 1.0,
           elevator_share: float = 1.0) -> int:
    """N = số lớp × sĩ số × tỷ lệ tham gia × tỷ lệ chọn thang."""
    return int(round(n_classes * class_size * participation * elevator_share))


# ---------------------------------------------------------------------------
# Các khối xây dựng
# ---------------------------------------------------------------------------
def _times_burst(rng, n, start=0.0, mean_delay=25.0, spread=40.0):
    """Tan học đồng loạt: sinh viên ra khỏi lớp trong ~1 phút, đi bộ tới thang."""
    return start + rng.uniform(0, spread, n) + rng.exponential(mean_delay, n)


def _times_profile(rng, n, duration, a=2.2, b=1.3):
    """Đầu ca: cường độ tăng dần, dồn về cuối cửa sổ (phân phối Beta)."""
    return rng.beta(a, b, n) * duration


def _times_uniform(rng, n, duration):
    return rng.uniform(0, duration, n)


def _pick(rng, weights: dict, n):
    keys = list(weights.keys())
    p = np.array([weights[k] for k in keys], dtype=float)
    p /= p.sum()
    return [keys[i] for i in rng.choice(len(keys), size=n, p=p)]


def background_trips(rng, n, duration):
    """Nhu cầu nền rải mọi tầng (bảo đảm không bỏ qua B2/B1 và tầng 1–4)."""
    out = []
    for t in _times_uniform(rng, n, duration):
        o, d = rng.choice(N_FLOORS, 2, replace=False)
        out.append((float(t), int(o), int(d), False))
    return out


def lobby_choice(rng):
    """Điểm ra/vào sảnh: phần lớn G, một phần B1/B2 (bãi xe)."""
    r = rng.random()
    return G if r < 0.82 else (F("B1") if r < 0.94 else F("B2"))


def build(trips) -> list[Passenger]:
    trips = sorted(trips, key=lambda x: x[0])
    return [Passenger(i, round(t, 2), o, d, fk) for i, (t, o, d, fk) in enumerate(trips)]


def load_csv(path) -> list[Passenger]:
    """CSV cột: t_arr, origin, dest[, fake_up]  (origin/dest dùng tên tầng: B2,B1,G,1..10)."""
    trips = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            trips.append((float(row["t_arr"]), F(row["origin"]), F(row["dest"]),
                          str(row.get("fake_up", "0")).strip() in ("1", "true", "True")))
    return build(trips)


def save_csv(passengers, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "t_arr", "origin", "dest", "fake_up"])
        for p in passengers:
            w.writerow([p.id, p.t_arr, fname(p.origin), fname(p.dest), int(p.fake_up)])
