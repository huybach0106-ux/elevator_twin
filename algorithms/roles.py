"""Vai trò (role) của từng cabin: Local / Express / Zone / Shuttle / Standby."""
from __future__ import annotations

from dataclasses import dataclass, field

from config import G, LOBBY, N_FLOORS, fname


@dataclass(frozen=True)
class Role:
    kind: str                     # BASE | LOCAL | EXPRESS | ZONE | SHUTTLE | STANDBY
    stops: frozenset
    label: str
    exclude: frozenset = frozenset()   # tầng mà hành trình sảnh<->tầng đó đã có cabin chuyên trách
    dir_priority: int = 0         # +1 ưu tiên đi lên, -1 ưu tiên đi xuống, 0 không
    parking: int | None = None    # tầng đỗ khi rảnh (None = đứng yên tại chỗ)
    standby: bool = False

    def can_serve(self, o: int, d: int) -> bool:
        """Cabin ở vai này có nhận hành trình o -> d không (biển báo/bảng hiển thị trên cửa)."""
        if o not in self.stops or d not in self.stops:
            return False
        if self.exclude:
            # chỉ nhường hành trình G <-> tầng chuyên trách (mọi cabin chuyên trách đều dừng G);
            # hành trình từ B1/B2 vẫn do A1 nhận để không ai bị bỏ lại
            if o == G and d in self.exclude:
                return False
            if d == G and o in self.exclude:
                return False
        return True

    def accepts_direction(self, floor: int, direction: int) -> bool:
        """Ưu tiên chiều: không đón khách ngược chiều ở tầng trên (trừ sảnh)."""
        if self.dir_priority and direction != self.dir_priority and floor not in LOBBY:
            return False
        return True

    def short(self) -> str:
        return self.label


KIND_COLORS = {
    "BASE": "#8a8f98", "LOCAL": "#2f6fdb", "EXPRESS": "#e8743b", "ZONE": "#8d5bd6",
    "SHUTTLE": "#169a8a", "STANDBY": "#c3c7cf", "OUT": "#d64545",
}


def floors_label(stops) -> str:
    s = sorted(stops)
    return ",".join(fname(f) for f in s)


def base_role(stops) -> Role:
    return Role("BASE", frozenset(stops), "Gốc " + floors_label(stops))


def local_role(exclude=frozenset(), dir_priority=0, parking=None) -> Role:
    lab = "Local" + (f" (sảnh: trừ {floors_label(exclude)})" if exclude else "")
    return Role("LOCAL", frozenset(range(N_FLOORS)), lab, frozenset(exclude), dir_priority, parking)


def group_role(group, native_extra=(), dir_priority=0, parking=None, lobby=None) -> Role:
    """Express (1 tầng) hoặc Zone (nhiều tầng) phục vụ sảnh + nhóm tầng."""
    from config import G
    lobby = {G} if lobby is None else set(lobby)
    stops = frozenset(set(group) | lobby | set(native_extra))
    kind = "EXPRESS" if len(group) == 1 else "ZONE"
    lab = ("Express " if kind == "EXPRESS" else "Zone ") + floors_label(group)
    return Role(kind, stops, lab, frozenset(), dir_priority, parking)


def shuttle_role(floors, native_extra=(), parking=None) -> Role:
    from config import G
    stops = frozenset(set(floors) | {G} | set(native_extra))
    return Role("SHUTTLE", stops, "Shuttle " + f"{fname(min(floors))}–{fname(max(floors))}", frozenset(), 0, parking)


def standby_role(stops, parking) -> Role:
    return Role("STANDBY", frozenset(stops), "Standby", frozenset(), 0, parking, True)


def out_role() -> Role:
    """Cabin sự cố: không nhận khách mới, chỉ trả khách đang chở, rồi đứng yên tại chỗ."""
    return Role("OUT", frozenset(), "Sự cố", frozenset(), 0, None, True)


def role_to_dict(r: Role) -> dict:
    return {"kind": r.kind, "stops": sorted(r.stops), "label": r.label, "exclude": sorted(r.exclude),
            "dir": r.dir_priority, "parking": r.parking, "standby": r.standby}


def role_from_dict(d: dict) -> Role:
    return Role(d["kind"], frozenset(d["stops"]), d["label"], frozenset(d.get("exclude", ())), d.get("dir", 0),
                d.get("parking"), bool(d.get("standby", False)))
