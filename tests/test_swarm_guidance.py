import math

import numpy as np
import pytest

from src.digital_twin import swarm_guidance as sg
from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator


# The original two-drone formulas from src/dashboard/dashboard.py, verbatim.
def _orig_coordinated(t):
    sweep_time, cycles = 60.0, 8.0
    t_norm = (t % sweep_time) / sweep_time
    y_a = 25.0 - t_norm * 50.0
    x_a = 25.0 * math.sin(t_norm * math.pi * 2.0 * cycles)
    dx = 25.0 * math.pi * 2.0 * cycles * math.cos(t_norm * math.pi * 2.0 * cycles)
    y_b = -25.0 + t_norm * 50.0
    return ((x_a, y_a, 8.0), math.atan2(-50.0, dx)), ((x_a, y_b, 8.0), math.atan2(50.0, dx))


def _orig_scouting(t):
    a = (-20.0 + 15.0 * math.sin(t * 0.2), 20.0 * math.cos(t * 0.05), 12.0)
    b = (20.0 + 15.0 * math.cos(t * 0.2), 20.0 * math.sin(t * 0.05), 12.0)
    return (a, math.atan2(a[1], a[0])), (b, math.atan2(b[1], b[0]))


def _orig_orbit(t):
    a = (-8.0 + 12.0 * math.cos(t * 0.3), 12.0 * math.sin(t * 0.3), 10.0)
    b = (8.0 + 12.0 * math.cos(t * 0.3 + math.pi), 12.0 * math.sin(t * 0.3 + math.pi), 10.0)
    return (a, t * 0.3 + math.pi / 2), (b, t * 0.3 + math.pi / 2 + math.pi)


def _orig_holding(t):
    a = (18.0 * math.sin(t * 0.4), 18.0 * math.cos(t * 0.4), 10.0)
    b = (18.0 * math.sin(t * 0.4 + math.pi), 18.0 * math.cos(t * 0.4 + math.pi), 10.0)
    return (a, math.atan2(a[1], a[0])), (b, math.atan2(b[1], b[0]))


@pytest.mark.parametrize("mode,orig", [
    ("coordinated_spraying", _orig_coordinated),
    ("synchronized_scouting", _orig_scouting),
    ("orbit_avoidance_test", _orig_orbit),
    ("waypoint_fallback", _orig_holding),
])
def test_two_drone_patterns_match_original_dashboard(mode, orig):
    for t in (0.0, 3.7, 17.25, 59.9, 133.3):
        for i, (pos, yaw) in enumerate(orig(t)):
            got_pos, got_yaw, _ = sg.target_for(mode, i, 2, t)
            np.testing.assert_allclose(got_pos, pos, atol=1e-9)
            assert math.isclose(math.remainder(got_yaw - yaw, 2 * math.pi), 0.0, abs_tol=1e-9)


def test_ids_labels_colors_preserve_alpha_beta():
    assert sg.drone_ids(2) == ["drone_alpha", "drone_beta"]
    assert sg.drone_color(0) == "#38bdf8" and sg.drone_color(1) == "#ec4899"
    assert sg.drone_label(0) == "Drone Alpha (Sector A)"
    assert sg.drone_label(1) == "Drone Beta (Sector B)"
    ids = sg.drone_ids(sg.MAX_DRONES)
    assert len(set(ids)) == sg.MAX_DRONES
    assert len({sg.drone_color(i) for i in range(sg.MAX_DRONES)}) == sg.MAX_DRONES
    assert sg.drone_ids(0) == ["drone_alpha"] and len(sg.drone_ids(99)) == sg.MAX_DRONES
    np.testing.assert_allclose(sg.spawn_position(0, 2), [-15.0, -15.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(sg.spawn_position(1, 2), [15.0, 15.0, 10.0], atol=1e-9)


def _factory(state):
    return lambda d_id: UAVFlightDynamicsSimulator(state, 11.0, 79.0, drone_id=d_id)


def test_ensure_fleet_grows_shrinks_and_keeps_primary():
    state = {"MULTIPLAYER_DRONES": {}}
    ids = sg.ensure_fleet(state, 4, _factory(state))
    assert ids == sg.drone_ids(4)
    assert list(state["PHYSICS_SIMULATORS"]) == ids
    assert set(state["MULTIPLAYER_DRONES"]) == set(ids)
    primary = state["PHYSICS_SIMULATORS"]["drone_alpha"]
    assert state["PHYSICS_SIMULATOR"] is primary

    sg.ensure_fleet(state, 2, _factory(state))
    assert set(state["PHYSICS_SIMULATORS"]) == {"drone_alpha", "drone_beta"}
    assert set(state["MULTIPLAYER_DRONES"]) == {"drone_alpha", "drone_beta"}
    assert state["PHYSICS_SIMULATORS"]["drone_alpha"] is primary   # not recreated

    sg.ensure_fleet(state, 1, _factory(state))
    assert list(state["PHYSICS_SIMULATORS"]) == ["drone_alpha"]


def test_patterns_spread_larger_fleets():
    for mode in ("coordinated_spraying", "synchronized_scouting", "orbit_avoidance_test", "fallback"):
        for t in (0.0, 11.0, 42.0):
            pts = [sg.target_for(mode, i, 6, t)[0][:2] for i in range(6)]
            d = min(np.linalg.norm(np.subtract(a, b)) for k, a in enumerate(pts) for b in pts[k + 1:])
            assert d > 1.0, (mode, t, d)                          # no two drones share a target
            assert all(abs(x) <= 40 and abs(y) <= 40 for x, y in pts)


def test_spray_dose():
    ndvi = sg.synthetic_ndvi_map()
    assert sg.spray_dose(ndvi, 0.0, -27.0) == 0.0                  # healthy background (0.6)
    assert sg.spray_dose(ndvi, 20.0, 0.0) == pytest.approx((0.35 - 0.22) / 0.35, abs=1e-6)  # centre-right patch
    stressed = sg.spray_dose(ndvi, -20.0, 22.0)                     # top-left patch (0.15)
    assert stressed == pytest.approx((0.35 - 0.15) / 0.35, abs=1e-6)
    assert sg.spray_dose(None, 0.0, 0.0) == 0.0


def test_four_drone_physics_loop_is_stable_and_separated():
    """The dashboard loop in miniature: N drones following a pattern in shared physics."""
    state = {"MULTIPLAYER_DRONES": {}, "SWARM_WARNINGS": [], "CURRENT_ENV": {"wind_speed": 3.0}}
    ids = sg.ensure_fleet(state, 4, _factory(state))
    sims = state["PHYSICS_SIMULATORS"]
    t = 0.0
    for _ in range(400):                                            # 20 simulated seconds
        t += 0.05
        for i, d in enumerate(ids):
            sims[d].target_pos, sims[d].target_yaw, sims[d].autopilot_mode = \
                sg.target_for("synchronized_scouting", i, len(ids), t)
        for d in ids:
            sims[d].step(dt=0.05)
    pos = np.array([sims[d].pos for d in ids])
    assert np.all(np.isfinite(pos))
    gaps = [np.linalg.norm(pos[a, :2] - pos[b, :2]) for a in range(4) for b in range(a + 1, 4)]
    assert min(gaps) > 3.0
