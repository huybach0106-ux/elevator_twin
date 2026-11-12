"""
Đặc trưng cửa sổ trượt từ QUAN SÁT CẢM BIẾN + nút gọi tầng (không dùng dữ liệu thật của mô phỏng).
Dùng chung cho: bộ nhận diện kịch bản (luật / Random Forest) và ước lượng nhu cầu của P2.
"""
from __future__ import annotations

import numpy as np

from config import LOBBY, N_FLOORS, UPPER

MODES = ["UP_PEAK", "DOWN_PEAK", "TWO_WAY", "INTERFLOOR", "OFF_PEAK"]
MODE_VI = {"UP_PEAK": "Cao điểm lên", "DOWN_PEAK": "Cao điểm xuống", "TWO_WAY": "Hai chiều",
           "INTERFLOOR": "Liên tầng", "OFF_PEAK": "Thấp điểm"}

FEATURE_NAMES = [
    "rate_board", "share_lobby_up", "share_lobby_down", "share_inter",
    "calls_lobby_up", "calls_upper_down", "calls_upper_up", "age_up_max", "age_down_max",
    "left_obs", "upper_up_board_share",
]


def window_features(obs, t, window, calls: dict) -> np.ndarray:
    """obs: list quan sát cảm biến cabin; calls: {(f,d): tuổi} từ nút gọi tầng.

    Ước lượng cơ cấu hành trình từ số đếm lên/xuống (U: sảnh→trên, Dn: trên→sảnh, I: liên tầng):
        lên ở sảnh = U, xuống ở sảnh = Dn, lên ở tầng trên = Dn + I, xuống ở tầng trên = U + I
        => I ≈ min(lên_trên − xuống_sảnh, xuống_trên − lên_sảnh)   (min để chống trễ đầu đợt)
    """
    b_lobby = b_upper = a_lobby = a_upper = b_upper_up = 0
    left = 0
    for ob in reversed(obs):
        if ob["t"] < t - window:
            break
        if ob["floor"] in LOBBY:
            if ob["dir"] > 0:
                b_lobby += ob["n_in"]
            a_lobby += ob["n_out"]
        else:
            b_upper += ob["n_in"]
            a_upper += ob["n_out"]
            if ob["dir"] > 0:
                b_upper_up += ob["n_in"]
        left += ob["left"]
    inter = max(0, min(b_upper - a_lobby, a_upper - b_lobby))
    total = max(1, b_lobby + b_upper)
    share_up = b_lobby / total
    share_inter = inter / total
    share_down = max(0, b_upper - inter) / total
    c_lu = sum(1 for (f, d) in calls if f in LOBBY and d > 0)
    c_ud = sum(1 for (f, d) in calls if f not in LOBBY and d < 0)
    c_uu = sum(1 for (f, d) in calls if f not in LOBBY and d > 0)
    age_up = max([a for (f, d), a in calls.items() if d > 0], default=0.0)
    age_dn = max([a for (f, d), a in calls.items() if d < 0], default=0.0)
    return np.array([
        (b_lobby + b_upper) / (window / 60.0), share_up, share_down, share_inter,
        c_lu, c_ud, c_uu, min(age_up, 300), min(age_dn, 300), left, b_upper_up / total,
    ], dtype=float)


def rule_mode(x: np.ndarray, offpeak_rate=7.2) -> str:
    """Luật ngưỡng (dự phòng khi chưa huấn luyện mô hình)."""
    rate, s_up, s_dn, s_in, c_lu, c_ud, c_uu, a_up, a_dn, left, _ = x
    busy_calls = c_ud + c_uu + c_lu
    if rate < offpeak_rate and busy_calls <= 2 and max(a_up, a_dn) < 30 and left < 3:
        return "OFF_PEAK"
    if rate < 3 and busy_calls >= 3:           # đầu đợt: chưa có quan sát, dựa vào nút gọi
        if c_ud >= 2 * max(1, c_uu) and c_lu <= 1:
            return "DOWN_PEAK"
        if c_lu and c_ud <= 1:
            return "UP_PEAK"
        return "TWO_WAY"
    if s_in > 0.36:
        return "INTERFLOOR"
    if s_up > 0.62:
        return "UP_PEAK"
    if s_dn > 0.62:
        return "DOWN_PEAK"
    return "TWO_WAY"


def true_mode(passengers, t, window, offpeak_rate=7.2) -> str:
    """Nhãn THẬT của cửa sổ (từ OD thật) – chỉ dùng để huấn luyện/đánh giá bộ nhận diện."""
    ps = [p for p in passengers if t - window <= p.t_arr <= t]
    rate = len(ps) / (window / 60.0)
    if rate < offpeak_rate:
        return "OFF_PEAK"
    up = sum(1 for p in ps if p.origin in LOBBY and p.dest not in LOBBY)
    dn = sum(1 for p in ps if p.dest in LOBBY and p.origin not in LOBBY)
    inter = len(ps) - up - dn
    n = max(1, len(ps))
    if inter / n >= 0.38:          # nhóm Inter-Floor: 40–50 % liên tầng
        return "INTERFLOOR"
    if up / n > 0.65:
        return "UP_PEAK"
    if dn / n > 0.65:
        return "DOWN_PEAK"
    return "TWO_WAY"
