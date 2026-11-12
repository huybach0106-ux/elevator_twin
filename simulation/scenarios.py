"""
12 kịch bản đã chốt: 4 nhóm giao thông × 3 kịch bản quen thuộc ở trường.

  Up-Peak     (80–85 % khách đi lên từ G) : U1 vào tiết 1 · U2 vào ca chiều · U3 hội thảo tầng 10
  Down-Peak   (80–85 % khách trở về G)    : D1 tan tiết trưa · D2 tan ca cuối ngày · D3 hết giờ thi tầng 8
  Inter-Floor (40–50 % hành trình liên tầng): I1 đổi phòng · I2 thư viện/lab/văn phòng khoa · I3 ngày hội CLB tầng 5–7
  Off-Peak    (< 20 % tải tham chiếu)     : O1 giữa buổi chiều · O2 buổi tối · O3 cuối tuần

Tải tham chiếu = 180 khách / 5 phút (tải mẫu của tài liệu) -> Off-Peak < 36 khách / 5 phút.
Mức "mid" = số khách trong bảng kịch bản; "low" = 0,6×, "high" = 1,4× (Off-Peak: 0,5× / 1,5×).
Số khách là GIÁ TRỊ THIẾT KẾ THÍ NGHIỆM, không phải kết quả khảo sát.
"""
from __future__ import annotations

import zlib

import numpy as np

from config import F, G, N_FLOORS, UPPER
from simulation.passenger_generator import _pick, build

LEVELS = ("low", "mid", "high")
B1, B2 = F("B1"), F("B2")
HIGH = [F(i) for i in range(5, 11)]
CLASS_W = {F(5): 1.3, F(6): 1.0, F(7): 1.0, F(8): 1.4, F(9): 1.1, F(10): 1.0}   # số lớp tương đối mỗi tầng


def lobby(rng, p_g=0.82, p_b1=0.12):
    """Điểm ra/vào sảnh: G hoặc tầng hầm (bãi xe)."""
    r = rng.random()
    return G if r < p_g else (B1 if r < p_g + p_b1 else B2)


def mix(rng, times, p_up, p_down, up_w, down_w, inter_floors, p_g=0.82, p_b1=0.12, other="random"):
    """Sinh hành trình theo tỷ lệ chiều đi.
    p_up: sảnh -> tầng (đích theo up_w) · p_down: tầng (theo down_w) -> sảnh · còn lại: liên tầng trong inter_floors
    (other='random' -> phần còn lại là OD ngẫu nhiên mọi tầng, dùng cho '5 % còn lại')."""
    trips = []
    ups = _pick(rng, up_w, len(times))
    downs = _pick(rng, down_w, len(times))
    for i, t in enumerate(times):
        r = rng.random()
        if r < p_up:
            o, d = lobby(rng, p_g, p_b1), ups[i]
        elif r < p_up + p_down:
            o, d = downs[i], lobby(rng, p_g, p_b1)
        elif inter_floors:
            o, d = (int(x) for x in rng.choice(inter_floors, 2, replace=False))
        else:
            o, d = (int(x) for x in rng.choice(N_FLOORS, 2, replace=False))
        trips.append((float(t), int(o), int(d), False))
    return trips


def t_profile(rng, n, dur, a=2.2, b=1.3):
    """Dồn về cuối cửa sổ (đến trước giờ học)."""
    return rng.beta(a, b, n) * dur


def t_release(rng, n, dur, burst=180, mean=90):
    """Tan học: phần lớn ra trong vài phút đầu, rải dần tới hết cửa sổ."""
    return np.minimum(rng.uniform(0, burst, n) + rng.exponential(mean, n), dur)


def t_uniform(rng, n, dur):
    return rng.uniform(0, dur, n)


ALL_UP = {**CLASS_W, F(1): 0.08, F(2): 0.08, F(3): 0.08, F(4): 0.08}
LOW_HIGH = {**{f: 1.0 for f in UPPER}}

# ------------------------------------------------------------------ Up-Peak
def gen_u1(rng, n):   # Vào tiết 1 buổi sáng, 6:45–7:00
    return mix(rng, t_profile(rng, n, 900), 0.85, 0.05, ALL_UP, LOW_HIGH, HIGH)


def gen_u2(rng, n):   # Vào ca chiều sau nghỉ trưa, 12:40–13:00 (ăn trưa ở sảnh -> chủ yếu từ G)
    return mix(rng, t_profile(rng, n, 1200, 2.0, 1.5), 0.80, 0.08, ALL_UP, LOW_HIGH, HIGH, p_g=0.95, p_b1=0.04)


def gen_u3(rng, n):   # Vào hội thảo tầng 10: 85 % G -> 10, 10 % G -> tầng khác, 5 % còn lại
    times = t_profile(rng, n, 600)
    trips = []
    for t in times:
        r = rng.random()
        if r < 0.85:
            o, d = lobby(rng), F(10)
        elif r < 0.95:
            o, d = lobby(rng), int(rng.choice([f for f in UPPER if f != F(10)]))
        else:
            o, d = (int(x) for x in rng.choice(N_FLOORS, 2, replace=False))
        trips.append((float(t), o, d, False))
    return trips


# ---------------------------------------------------------------- Down-Peak
def gen_d1(rng, n):   # Tan tiết cuối buổi sáng, 11:25–11:40
    return mix(rng, t_release(rng, n, 900), 0.05, 0.85, ALL_UP, CLASS_W, HIGH)


def gen_d2(rng, n):   # Tan ca cuối ngày: 1/3 xuống B1/B2 lấy xe
    return mix(rng, t_release(rng, n, 900), 0.06, 0.82, ALL_UP, CLASS_W, HIGH, p_g=0.67, p_b1=0.20)


def gen_d3(rng, n):   # Hết giờ thi, cả tầng 8 ra cùng lúc (5 phút)
    times = t_release(rng, n, 300, burst=60, mean=40)
    trips = []
    for t in times:
        r = rng.random()
        if r < 0.80:
            o, d = F(8), lobby(rng)
        elif r < 0.95:
            o, d = F(8), int(rng.choice([f for f in UPPER if f != F(8)]))
        else:
            o, d = (int(x) for x in rng.choice(N_FLOORS, 2, replace=False))
        trips.append((float(t), o, d, False))
    return trips


# -------------------------------------------------------------- Inter-Floor
def gen_i1(rng, n):   # Đổi phòng giữa hai tiết (lớp -> lab), 9:25–9:35
    return mix(rng, t_release(rng, n, 600, burst=240, mean=90), 0.25, 0.30, CLASS_W, CLASS_W, HIGH)


def gen_i2(rng, n):   # Thư viện, lab, văn phòng khoa giữa buổi – có cả tầng 1–4
    return mix(rng, t_uniform(rng, n, 900), 0.30, 0.30, LOW_HIGH, LOW_HIGH, UPPER)


def gen_i3(rng, n):   # Ngày hội CLB, gian hàng tầng 5–7
    booths = {F(5): 1.0, F(6): 1.0, F(7): 1.0, F(8): 0.2, F(9): 0.2, F(10): 0.2}
    return mix(rng, t_uniform(rng, n, 1800), 0.30, 0.20, booths, booths, [F(5), F(6), F(7)])


# ---------------------------------------------------------------- Off-Peak
def _offpeak(rng, n, dur=900):
    return mix(rng, t_uniform(rng, n, dur), 0.40, 0.40, LOW_HIGH, LOW_HIGH, UPPER)


def gen_o1(rng, n):   # Giữa buổi chiều khi đang học
    return _offpeak(rng, n)


def gen_o2(rng, n):   # Buổi tối sau 19:00: lớp tối, lao công, bảo vệ
    return _offpeak(rng, n)


def gen_o3(rng, n):   # Cuối tuần / ngày nghỉ: thư viện, người trực
    return _offpeak(rng, n)


def _lv(mid, lo=0.6, hi=1.4):
    return (int(round(mid * lo)), mid, int(round(mid * hi)))


SCENARIOS = {
    "U1": dict(group="Up-Peak", name="Vào tiết 1 buổi sáng", time="6:45–7:00", levels=_lv(300), horizon=1500,
               mode="UP_PEAK", gen=gen_u1, note="85 % từ G/B1/B2 lên tầng 5–10; 10 % liên tầng; 5 % đi xuống · 300 người/15 phút"),
    "U2": dict(group="Up-Peak", name="Vào ca chiều sau giờ nghỉ trưa", time="12:40–13:00", levels=_lv(240), horizon=1800,
               mode="UP_PEAK", gen=gen_u2, note="80 % từ G lên; 12 % liên tầng; 8 % đi xuống · 240 người/20 phút"),
    "U3": dict(group="Up-Peak", name="Vào hội thảo tầng 10", time="10 phút trước giờ bắt đầu", levels=_lv(200), horizon=1200,
               mode="UP_PEAK", gen=gen_u3, note="85 % G → tầng 10; 10 % G → tầng khác; 5 % còn lại · 200 người/10 phút"),
    "D1": dict(group="Down-Peak", name="Tan tiết cuối buổi sáng", time="11:25–11:40", levels=_lv(300), horizon=1500,
               mode="DOWN_PEAK", gen=gen_d1, note="85 % từ tầng 5–10 về G; 10 % liên tầng; 5 % đi lên · 300 người/15 phút"),
    "D2": dict(group="Down-Peak", name="Tan ca cuối ngày", time="17:00–17:15", levels=_lv(360), horizon=1500,
               mode="DOWN_PEAK", gen=gen_d2, note="82 % về sảnh (1/3 xuống B1/B2); 12 % liên tầng; 6 % đi lên · 360 người/15 phút"),
    "D3": dict(group="Down-Peak", name="Hết giờ thi, cả tầng 8 ra cùng lúc", time="Ngay sau giờ thi", levels=_lv(120), horizon=900,
               mode="DOWN_PEAK", gen=gen_d3, note="80 % tầng 8 → G; 15 % tầng 8 → tầng khác; 5 % còn lại · 120 người/5 phút"),
    "I1": dict(group="Inter-Floor", name="Đổi phòng giữa hai tiết (lớp → lab)", time="9:25–9:35", levels=_lv(150), horizon=1200,
               mode="INTERFLOOR", gen=gen_i1, note="45 % liên tầng 5–10; 30 % xuống G; 25 % từ G lên · 150 người/10 phút"),
    "I2": dict(group="Inter-Floor", name="Thư viện, lab, văn phòng khoa giữa buổi", time="8:00–11:00", levels=_lv(100), horizon=1500,
               mode="INTERFLOOR", gen=gen_i2, note="40 % liên tầng (cả tầng 1–4); 30 % lên; 30 % xuống · 100 người/15 phút"),
    "I3": dict(group="Inter-Floor", name="Ngày hội câu lạc bộ tầng 5–7", time="8:30–11:00", levels=_lv(200), horizon=2400,
               mode="INTERFLOOR", gen=gen_i3, note="50 % đi lại giữa tầng 5, 6, 7; 30 % từ G lên; 20 % xuống · 200 người/30 phút"),
    "O1": dict(group="Off-Peak", name="Giữa buổi chiều khi đang học", time="14:30–15:30", levels=_lv(30, 0.5, 1.5), horizon=1200,
               mode="OFF_PEAK", gen=gen_o1, note="Lác đác, đủ mọi chiều · 30 người/15 phút (≈ 6 % tải tham chiếu)"),
    "O2": dict(group="Off-Peak", name="Buổi tối sau 19:00", time="19:00–21:00", levels=_lv(15, 0.5, 1.5), horizon=1200,
               mode="OFF_PEAK", gen=gen_o2, note="Lớp tối, lao công, bảo vệ · 15 người/15 phút (≈ 3 % tải tham chiếu)"),
    "O3": dict(group="Off-Peak", name="Cuối tuần hoặc ngày nghỉ", time="Cả ngày", levels=_lv(8, 0.5, 1.5), horizon=1200,
               mode="OFF_PEAK", gen=gen_o3, note="Thư viện, người trực · 8 người/15 phút (≈ 1,5 % tải tham chiếu)"),
}


def n_for(scenario: str, level: str | int) -> int:
    if isinstance(level, int):
        return level
    return SCENARIOS[scenario]["levels"][LEVELS.index(level)]


def make_passengers(scenario: str, level="mid", seed: int = 42):
    """Danh sách hành khách tái lập được: cùng (kịch bản, N, seed) -> cùng hành khách cho mọi cấu hình."""
    sc = SCENARIOS[scenario]
    n = n_for(scenario, level)
    rng = np.random.default_rng([seed, zlib.crc32(scenario.encode()), n])
    return build(sc["gen"](rng, n))


def planned_demand(scenario: str, level="mid", seed: int = 42):
    """Nhu cầu dự kiến từ thời khóa biểu (P1, P2 dùng làm thông tin trước): một mẫu độc lập cùng phân bố."""
    return make_passengers(scenario, level, seed + 10_000)
