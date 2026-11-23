"""Section 8: spray decisions under uncertainty, adaptive scouting, and fleet limits (wind, tank, refills)."""
import numpy as np
import pytest

from src.twin_core import DigitalTwin, FieldGrid
from src.twin_core.decisions import scouting_targets, spray_options, zone_risk
from src.twin_core.swarm import Task, allocate, split_for_tank, wind_factors

LAT, LON = 10.79, 79.14
WEATHER = [{"temperature": 28.0, "humidity": 92.0, "precipitation": 0.0, "wind_speed": 6.0}] * 21


@pytest.fixture
def twin(tmp_path):
    grid = FieldGrid.centered_on(LAT, LON, rows=20, cols=20, cell_size_m=5.0)
    return DigitalTwin.load_or_create("dec", tmp_path, grid=grid)


def _harvest(t):
    from digital_twin_api import expected_harvest_t_ha
    return expected_harvest_t_ha(t)


def _infect(twin, zone, load, var):
    m = twin.zone_labels == zone
    twin.state.mean["fungus"][m] = load
    twin.state.var["fungus"][m] = var


def test_scouting_targets_prefer_uncertain_zones_near_threshold(twin):
    twin.state.mean["fungus"][:] = 0.02
    twin.state.var["fungus"][:] = 1e-4                      # confident: healthy
    _infect(twin, 3, 0.18, 0.01)                            # near the threshold, unsure
    _infect(twin, 7, 0.60, 1e-4)                            # clearly infected, known
    targets = scouting_targets(twin, 0.2)
    assert targets[0] == 3 and 7 not in targets
    risk = {r["zone_id"]: r for r in zone_risk(twin, 0.2)}
    assert risk[7]["infected_share"] > 0.99 and risk[0]["infected_share"] < 0.01


def test_spray_pays_on_an_outbreak_and_not_on_a_healthy_field(twin):
    twin.state.mean["fungus"][:] = 0.01
    twin.state.var["fungus"][:] = 1e-4
    healthy = spray_options(twin, _harvest, WEATHER, samples=3, horizon_days=10)
    assert healthy["recommendation"]["option"] == "no spray" and not healthy["options"]

    _infect(twin, 5, 0.35, 0.004)
    res = spray_options(twin, _harvest, WEATHER, samples=4, horizon_days=14)
    assert res["zones"] == [5] and res["spray_cost_rs"] > 0
    opts = {o["option"]: o for o in res["options"]}
    now, later = opts["spray tomorrow"], opts["spray in 3 days"]
    assert now["gain_rs"] > 0 and now["gain_rs"] >= later["gain_rs"]     # waiting lets it spread
    assert res["recommendation"]["option"] == "spray tomorrow"
    assert now["gain_rs_p10"] <= now["gain_rs"] <= now["gain_rs_p90"]


def test_rain_and_wind_make_a_spray_day_infeasible(twin):
    _infect(twin, 5, 0.35, 0.004)
    wet = [dict(WEATHER[0], precipitation=20.0)] + [dict(WEATHER[0], wind_speed=25.0)] * 2 + WEATHER
    res = spray_options(twin, _harvest, wet, samples=2, horizon_days=10)
    opts = {o["option"]: o for o in res["options"]}
    assert not opts["spray tomorrow"]["feasible"] and "rain" in opts["spray tomorrow"]["notes"]
    assert not opts["spray in 3 days"]["feasible"] and "wind" in opts["spray in 3 days"]["notes"]
    assert res["recommendation"]["option"] == "wait"


def test_wind_slows_and_blocks_flights():
    lanes = [(0.0, y) for y in range(0, 40, 5)]
    legs = [p for i, (x, y) in enumerate(lanes) for p in ([(0.0, y), (40.0, y)] if i % 2 == 0 else [(40.0, y), (0.0, y)])]
    scout = Task("scout-z0", "scout", 0, legs, priority=1.0)
    spray = Task("spray-z0", "spray", 0, legs, priority=1.0, payload_kg=4.0)
    drones = {"d": {"home": (0.0, 0.0), "battery": 100.0, "tank_kg": 10.0}}
    calm, windy = allocate([spray], drones), allocate([spray], drones, wind_m_s=4.0)
    tf, df = wind_factors(4.0)
    assert windy.makespan_s == pytest.approx(calm.makespan_s * tf, rel=0.05) and df > 1
    assert windy.assignments["d"].est_battery_pct > calm.assignments["d"].est_battery_pct
    blocked = allocate([spray], drones, wind_m_s=6.0)
    assert blocked.blocked and "spray" in blocked.blocked and blocked.unassigned == [spray]
    assert not allocate([scout], drones, wind_m_s=6.0).blocked                 # scouting still allowed


def test_refills_split_big_jobs_into_sorties():
    legs = [p for i, y in enumerate(range(0, 60, 5)) for p in ([(0.0, y), (50.0, y)] if i % 2 == 0 else [(50.0, y), (0.0, y)])]
    big = Task("spray-z0", "spray", 0, legs, priority=1.0, payload_kg=25.0)
    parts = split_for_tank(big, 10.0)
    assert len(parts) >= 3 and all(p.payload_kg <= 10.0 + 1e-9 for p in parts)
    assert sum(p.payload_kg for p in parts) == pytest.approx(25.0, abs=0.01)
    drones = {"d": {"home": (0.0, 0.0), "battery": 100.0, "tank_kg": 10.0}}
    assert allocate([big], drones).unassigned == [big]                          # one tank is not enough
    plan = allocate([big], drones, refills=True)
    a = plan.assignments["d"]
    assert not plan.unassigned and a.sorties >= 3 and a.payload_kg == pytest.approx(25.0, abs=0.01)


def test_api_spray_decision_and_adaptive_scouting(twin):
    from fastapi.testclient import TestClient
    from digital_twin_api import create_app
    twin.state.mean["fungus"][:] = 0.02
    twin.state.var["fungus"][:] = 1e-4
    _infect(twin, 3, 0.18, 0.01)
    with TestClient(create_app(twin)) as c:
        tgt = c.get("/twin/scouting-targets").json()
        assert tgt["zones"][0] == 3
        plan = c.post("/swarm/plan", json={"kind": "scout", "adaptive": True}).json()
        assert {t["zone_id"] for t in plan["tasks"]} == set(tgt["zones"])
        assert c.post("/swarm/plan", json={"kind": "spray", "wind_m_s": 8, "fungus_threshold": 0.1}).json()["blocked"]
        r = c.post("/twin/spray-decision", json={"samples": 2, "horizon_days": 5, "weather_source": "synthetic"})
        assert r.status_code == 200 and "recommendation" in r.json()
