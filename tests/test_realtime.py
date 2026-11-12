"""Kiểm thử phần mở rộng: sự cố cabin, chế độ eco, thời khóa biểu, vòng thời gian thực tòa nhà ảo ↔ twin."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.runner import run_one  # noqa: E402


def test_outage_car_carries_nobody_new():
    sim, m = run_one("U1", "mid", 15, "P0", 1, outages=[("A4", 0, 10 ** 9)], horizon=2400)
    assert m["cars"]["A4"]["pax"] == 0
    assert all(p.state == "done" for p in sim.passengers)       # các cabin khác vẫn chở hết khách


def test_p2_reroutes_when_local_car_fails():
    _, m0 = run_one("U1", "mid", 15, "P0", 2, outages=[("A1", 60, 660)])
    _, m2 = run_one("U1", "mid", 15, "P2", 2, outages=[("A1", 60, 660)])
    assert m2["AWT"] < m0["AWT"] / 3


def test_eco_saves_energy():
    _, m2 = run_one("U2", "high", 15, "P2", 1)
    _, m4 = run_one("U2", "high", 15, "P4", 1)
    assert m4["energy"] < m2["energy"]


def test_timetable_generates_class_flows():
    from realtime.timetable import Timetable, hhmm
    tt = Timetable()
    trips = tt.trips_between(hhmm("06:45"), hhmm("07:00"), np.random.default_rng(0))
    from config import LOBBY
    up = sum(1 for _, o, d in trips if o in LOBBY and d not in LOBBY)
    assert len(trips) > 100 and up / len(trips) > 0.8
    assert any(e["kind"] == "tan lớp" for e in tt.upcoming(hhmm("09:20"), 900))


def test_live_loop_building_and_twin():
    from realtime.live import LiveManager
    os.environ.pop("MQTT_HOST", None)
    lm = LiveManager()
    lm.start(clock="06:45", speed=120)
    try:
        for _ in range(40):
            time.sleep(0.5)
            s = lm.snapshot()
            if s["truth"] and s["twin"] and s["truth"]["arrived"] > 50 and s["twin"]["cmd_log"]:
                break
        assert s["truth"]["arrived"] > 50
        assert s["twin"]["cmd_log"], "twin chưa gửi lệnh phân vai nào"
        assert s["twin"]["mode"] in ("UP_PEAK", "TWO_WAY", "OFF_PEAK", "DOWN_PEAK", "INTERFLOOR")
    finally:
        lm.stop()


def test_whatif_endpoint():
    from web_app import app
    r = app.test_client().post("/api/whatif", json={"scenario": "D3", "level": "low", "fault_car": "A4",
                                                     "fault_minutes": 5}).get_json()
    assert set(r["runs"]) == {"P0_normal", "P0_whatif", "P2_normal", "P2_whatif"}
