"""
Bộ phân vai (role planner) – trái tim của phương án tối ưu.

Bài toán: chia các tầng có nhu cầu thành các NHÓM LIÊN TIẾP, mỗi nhóm giao n_g cabin
(Express nếu nhóm 1 tầng, Zone nếu nhiều tầng), A1 luôn Local lo phần còn lại.

Hàm mục tiêu (mục 4 tài liệu, dạng thời gian giải tỏa):
    W_g = D_g × T_vòng(g) / (n_g × sức chứa)          (thời gian để nhóm g hết khách)
    score = max_g W_g  (+ phần A1)                       -> cân bằng tải giữa các cabin
    T_vòng(g) = 2 × (tầng cao nhất − G) × t_tầng + (số tầng dừng + 1) × t_cửa + 2 × c × t_khách

Duyệt toàn bộ phân hoạch liên tiếp (≤ 512) × cách chia cabin (≤ 6) -> nghiệm tối ưu chính xác,
đủ nhanh để chạy lại mỗi 20 s trong chế độ điều phối động.
"""
from __future__ import annotations

from itertools import combinations, permutations

from config import DOOR_TIME, G, N_FLOORS, TRANSFER_TIME, TRAVEL_TIME_PER_FLOOR, UPPER


def rtt(group, cap_eff) -> float:
    top = max(max(group), G)
    bot = min(min(group), G)
    travel = 2 * (top - bot) * TRAVEL_TIME_PER_FLOOR
    return travel + (len(group) + 1) * DOOR_TIME + 2 * cap_eff * TRANSFER_TIME


def clear_time(demand, group, n, cap) -> float:
    if n <= 0:
        return float("inf") if demand > 0 else 0.0
    cap_eff = min(cap, max(1.0, demand / n))
    trips = demand / (n * cap)
    return max(trips, 1.0 if demand > 0 else 0.0) * rtt(group, cap_eff)


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for cut in combinations(range(1, total), parts - 1):
        prev = 0
        out = []
        for c in cut:
            out.append(c - prev)
            prev = c
        out.append(total - prev)
        yield tuple(out)


def _partitions(floors):
    n = len(floors)
    for mask in range(1 << max(0, n - 1)):
        groups, cur = [], [floors[0]]
        for i in range(1, n):
            if mask >> (i - 1) & 1:
                groups.append(cur)
                cur = [floors[i]]
            else:
                cur.append(floors[i])
        groups.append(cur)
        yield groups


def plan_groups(D: dict, K: int, cap: int, min_share=0.06, a1_cap_factor=1.0):
    """D: tầng -> nhu cầu (người) trong cửa sổ.  Trả về (score, [(group, n_cars)], rest_floors)."""
    total = sum(D.get(f, 0) for f in UPPER)
    if total <= 0 or K <= 0:
        return 0.0, [], [f for f in UPPER]
    cand = [f for f in UPPER if D.get(f, 0) > 0 and D.get(f, 0) >= min_share * total]
    if not cand:
        return 0.0, [], [f for f in UPPER]
    best = None
    for groups in _partitions(cand):
        m = len(groups)
        if m > K:
            continue
        dem = [sum(D.get(f, 0) for f in g) for g in groups]
        for comp in _compositions(K, m):
            sc = max(clear_time(d, g, n, cap) for d, g, n in zip(dem, groups, comp))
            # tie-break: ít điểm dừng hơn, ít nhóm hơn
            key = (round(sc, 3), sum(len(g) for g in groups), m)
            if best is None or key < best[0]:
                best = (key, [(g, n) for g, n in zip(groups, comp)])
    rest = [f for f in UPPER if f not in cand]
    rest_dem = sum(D.get(f, 0) for f in rest)
    score = best[0][0]
    if rest_dem:
        score = max(score, clear_time(rest_dem, rest, a1_cap_factor, cap))
    return score, best[1], rest


def expected_stops(n_floors, load):
    """Số điểm dừng kỳ vọng khi `load` khách chọn ngẫu nhiên đều trong n tầng."""
    if n_floors <= 0:
        return 0.0
    return n_floors * (1 - (1 - 1 / n_floors) ** max(load, 0))


def utilization(D, window, n_cars, cap):
    """Hệ số sử dụng ước tính nếu MỌI cabin phục vụ chung (pooling):
    ρ = λ × T_vòng_chung / (n_cabin × tải_mỗi_vòng)."""
    floors = [f for f in UPPER if D.get(f, 0) > 0]
    total = sum(D.get(f, 0) for f in floors)
    if not floors or total <= 0:
        return 0.0
    lam = total / window
    load = cap * 0.8
    top = max(floors)
    s = expected_stops(len(floors), load)
    t_round = 2 * (top - G) * TRAVEL_TIME_PER_FLOOR + (s + 1) * DOOR_TIME + 2 * load * TRANSFER_TIME
    return lam * t_round / (n_cars * load)


def score_of(D, plan, cap):
    """Điểm của một phân vai bất kỳ (để so sánh với phân vai hiện tại – hysteresis)."""
    covered = set()
    sc = 0.0
    for g, n in plan:
        covered |= set(g)
        sc = max(sc, clear_time(sum(D.get(f, 0) for f in g), g, n, cap))
    rest = [f for f in UPPER if f not in covered and D.get(f, 0) > 0]
    if rest:
        sc = max(sc, clear_time(sum(D[f] for f in rest), rest, 1, cap))
    return sc


def assign_cars(slots, cars, current_labels, target_floor, slot_floors=None):
    """Gán cabin -> vị trí vai (slots) tối thiểu khoảng cách + giữ nguyên vai cũ nếu được.
    slots: list nhãn vai;  cars: list Car;  target_floor: nhãn -> tầng đại diện;
    slot_floors: nhãn -> tập tầng cần phục vụ (kiểm tra giới hạn giếng thang 'reach')."""
    slots = slots[: len(cars)]          # không bao giờ có nhiều vai hơn số cabin
    best = None
    for perm in permutations(range(len(cars)), len(slots)):
        cost = 0.0
        for slot_i, car_i in enumerate(perm):
            car = cars[car_i]
            lab = slots[slot_i]
            if slot_floors and not set(slot_floors[lab]) <= set(car.reach):
                cost += 1e6
            cost += abs(car.pos - target_floor[lab]) * 0.5
            if current_labels.get(car.id) != lab:
                cost += 4.0
        if best is None or cost < best[0]:
            best = (cost, perm)
    return {cars[ci].id: slots[si] for si, ci in enumerate(best[1])}
