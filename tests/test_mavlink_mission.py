import random
import socket

import pytest

from src.twin_core import DigitalTwin, FieldGrid
from src.twin_core import mavlink_mission as mm
from src.twin_core.sim_vehicle import EmulatedCopter, EmulatedFleet
from src.twin_core.swarm import Assignment, Fleet, Task, allocate, tasks_from_twin
from src.twin_core.virtual_field import VirtualField

C = mm.mavlink


def _free_base_port(n: int = 4) -> int:
    for _ in range(50):
        base = random.randint(20000, 40000)
        socks = []
        try:
            for i in range(n):
                s = socket.socket()
                s.bind(("127.0.0.1", base + 10 * i))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free ports")


@pytest.fixture
def grid():
    return FieldGrid.centered_on(10.0, 78.0, rows=20, cols=20, cell_size_m=5.0)


def _spray_task():
    legs = [(0.0, 10.0), (20.0, 10.0), (20.0, 5.0), (0.0, 5.0), (0.0, 0.0), (20.0, 0.0)]
    return Task("spray-z3", "spray", 3, legs, priority=1.0, payload_kg=2.0, alt_m=3.0)


def test_compile_mission_structure(grid):
    fleet = Fleet.for_field(grid, 1)
    scout = Task("scout-z0", "scout", 0, [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0)], alt_m=10.0)
    m = mm.compile_mission(Assignment("drone_01", [scout, _spray_task()]), fleet.frame)
    cmds = [it.command for it in m.items]
    assert cmds[1] == C.MAV_CMD_NAV_TAKEOFF and cmds[-1] == C.MAV_CMD_NAV_RETURN_TO_LAUNCH
    pumps = [it.params[1] for it in m.items if it.command == C.MAV_CMD_DO_SET_SERVO]
    assert pumps == [mm.SPRAY_PWM_ON, mm.SPRAY_PWM_OFF] * 3          # one on/off pair per lane
    assert all(it.params[0] == mm.SPRAY_SERVO for it in m.items if it.command == C.MAV_CMD_DO_SET_SERVO)
    lo, hi = m.spray_span["spray-z3"]
    assert m.items[lo].params[1] == mm.SPRAY_PWM_ON and m.items[hi].params[1] == mm.SPRAY_PWM_OFF
    assert set(m.task_ranges()) == {"scout-z0", "spray-z3"}
    wpl = mm.to_qgc_wpl(m).splitlines()
    assert wpl[0] == "QGC WPL 110" and len(wpl) == len(m.items) + 1
    assert wpl[1].split("\t")[1] == "1"                              # home flagged as current


def test_spray_speed_slows_for_bigger_doses():
    t = _spray_task()
    speeds = []
    for dose in (0.5, 1.0, 2.0, 4.0, 40.0):
        t.payload_kg = dose
        speeds.append(mm.spray_speed_for(t))
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] == mm.MAX_SPRAY_SPEED_M_S and speeds[-1] == mm.MIN_SPRAY_SPEED_M_S


def test_emulator_rejects_takeoff_outside_guided():
    port = _free_base_port(1)
    v = EmulatedCopter(port, 10.0, 78.0, time_scale=10).start()
    link = None
    try:
        link = mm.VehicleLink(f"tcp:127.0.0.1:{port}", timeout_s=5)
        link.wait_heartbeat()
        link.arm()
        with pytest.raises(mm.MavlinkError):
            link.command(C.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 10.0, timeout=3)
        link.set_mode("GUIDED")
        link.command(C.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 10.0, timeout=3)
    finally:
        if link:
            link.close()
        v.stop()


def test_fleet_flies_scout_and_spray_over_mavlink(grid):
    vf = VirtualField(grid, seed=2)
    twin = DigitalTwin("t", grid=grid)
    fleet = Fleet.for_field(grid, 2)
    tasks = (tasks_from_twin(twin, "scout", fleet.frame, zones=[5, 6, 9, 10])
             + tasks_from_twin(vf.truth, "spray", fleet.frame, fungus_threshold=0.15))
    plan = allocate(tasks, fleet.drone_info())
    homes = {d: fleet.frame.to_latlon(*fleet.pads[d]) for d in fleet.drones}
    emu = EmulatedFleet(homes, base_port=_free_base_port(2), time_scale=40)
    captures = []
    try:
        rep = mm.execute_plan_mavlink(plan, fleet.frame, emu.connections, homes=homes,
                                      on_capture=lambda *a: captures.append(a), timeout_s=10)
    finally:
        emu.stop()
    assert rep["errors"] == {}
    planned = {t.task_id for a in plan.assignments.values() for t in a.tasks}
    done = {tid for d in rep["drones"].values() for tid in d["tasks_done"]}
    assert done == planned
    assert all(d["completed"] for d in rep["drones"].values())
    assert len(captures) > 20
    assert {s["zone_id"] for s in rep["completed_sprays"]} == {t.zone_id for t in tasks if t.kind == "spray"}
    for s in rep["completed_sprays"]:
        assert s["applied_kg"] == pytest.approx(s["planned_kg"], rel=0.25)
    for d_id, v in emu.vehicles.items():
        assert not v.armed and v.sim.pos[2] < 0.5                      # landed and disarmed
    for track in rep["track"].values():
        assert track[0][0] == 0.0 and track[-1][0] > 30.0             # simulated seconds


def test_api_mavlink_backend_and_sitl_routes(twin, monkeypatch):
    from fastapi.testclient import TestClient

    import digital_twin_api
    from src.config.config import settings

    monkeypatch.setattr(settings, "twin_mavlink_mode", "emulated")
    monkeypatch.setattr(settings, "twin_mavlink_base_port", _free_base_port(3))
    monkeypatch.setattr(settings, "twin_mavlink_time_scale", 40.0)
    with TestClient(digital_twin_api.create_app(twin)) as client:
        client.post("/sim/reset", json={"seed": 4})
        homes = client.get("/swarm/sitl-homes").json()
        assert len(homes["instances"]) == 3
        assert homes["instances"][1]["connection"].endswith(str(settings.twin_mavlink_base_port + 10))
        assert "--custom-location=" in homes["instances"][0]["command"]
        files = client.post("/swarm/mission-files", json={"kind": "scout", "zones": [0, 1]}).json()
        assert files["missions"] and all(m["waypoints"].startswith("QGC WPL 110")
                                         for m in files["missions"].values())
        r = client.post("/sim/mission", json={"kind": "auto", "backend": "mavlink",
                                              "fungus_threshold": 0.15, "zones": [5, 6, 9, 10]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["backend"] == "mavlink" and body["frames_assimilated"] > 0
        assert body["sprays_queued"]
        assert body["error_after"]["fungus"]["rmse"] < body["error_before"]["fungus"]["rmse"]


@pytest.fixture
def twin(grid, tmp_path):
    return DigitalTwin.load_or_create("mav_field", tmp_path, grid=grid)
