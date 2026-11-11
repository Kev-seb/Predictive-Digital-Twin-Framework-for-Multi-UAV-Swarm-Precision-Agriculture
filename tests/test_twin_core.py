import numpy as np
import pytest

from src.twin_core import DigitalTwin, FieldGrid, FieldState
from src.twin_core.field_state import PADDY_DEFAULTS
from src.twin_core.assimilation import assimilate_grid, assimilate_point, inflate_process_noise

LAT, LON = 10.0, 78.0


@pytest.fixture
def grid():
    return FieldGrid.centered_on(LAT, LON, rows=20, cols=20, cell_size_m=5.0)


@pytest.fixture
def twin(grid, tmp_path):
    return DigitalTwin.load_or_create("test_field", tmp_path, grid=grid)


# ------------------------------------------------------------------ grid
def test_grid_georeference_round_trip(grid):
    for r, c in [(0, 0), (7, 13), (19, 19)]:
        lat, lon = grid.cell_center_latlon(r, c)
        assert grid.latlon_to_cell(lat, lon) == (r, c)
    r, c = grid.latlon_to_cell(*grid.center_latlon())   # centre sits on a cell corner
    assert r in (9, 10) and c in (9, 10)
    assert grid.latlon_to_cell(LAT + 0.01, LON) is None


def test_resample_preserves_mean(grid):
    s = FieldState.initial(grid)
    coarse = s.resample("ndvi", 10, 10)
    assert coarse.shape == (10, 10)
    assert coarse.mean() == pytest.approx(s.mean["ndvi"].mean(), abs=1e-5)
    fine = s.resample("ndvi", 40, 40)
    assert fine.shape == (40, 40)


def test_state_save_load(grid, tmp_path):
    s = FieldState.initial(grid)
    s.sim_day = 5
    s.growth_stage = "Tillering"
    s.save(tmp_path / "s.npz")
    t = FieldState.load(tmp_path / "s.npz")
    assert t.sim_day == 5 and t.growth_stage == "Tillering" and t.grid == grid
    np.testing.assert_allclose(t.mean["fungus"], s.mean["fungus"])


# ---------------------------------------------------------- assimilation
def test_grid_assimilation_moves_toward_obs_and_shrinks_variance(grid):
    s = FieldState.initial(grid, seed=None)
    before_var = s.var["ndvi"].mean()
    res = assimilate_grid(s, "ndvi", np.full(grid.shape, 0.3), source="multispectral")
    assert res.cells_updated == grid.rows * grid.cols
    assert 0.3 < s.mean["ndvi"].mean() < 0.72
    assert s.mean["ndvi"].mean() < 0.4          # precise sensor -> strong pull
    assert s.var["ndvi"].mean() < before_var


def test_grid_assimilation_ignores_nan(grid):
    s = FieldState.initial(grid, seed=None)
    obs = np.full(grid.shape, np.nan, dtype=np.float32)
    obs[0, 0] = 0.1
    res = assimilate_grid(s, "ndvi", obs)
    assert res.cells_updated == 1
    assert s.mean["ndvi"][5, 5] == pytest.approx(0.72)


def test_point_assimilation_is_local(grid):
    s = FieldState.initial(grid, seed=None)
    lat, lon = grid.cell_center_latlon(10, 10)
    assimilate_point(s, "stress", 0.9, lat, lon, source="drone_rgb")
    assert s.mean["stress"][10, 10] > 0.6
    prior = PADDY_DEFAULTS["stress"]
    assert s.mean["stress"][10, 11] > s.mean["stress"][10, 13] > prior - 1e-6
    assert s.mean["stress"][0, 0] == pytest.approx(prior)


def test_process_noise_grows_variance(grid):
    s = FieldState.initial(grid)
    v0 = s.var["moisture"].mean()
    inflate_process_noise(s, days=3)
    assert s.var["moisture"].mean() > v0


# ------------------------------------------------------------------ twin
def test_twin_state_carries_over_between_steps(twin):
    m0 = twin.state.mean["moisture"].mean()
    twin.step(days=2, weather={"precipitation": 0.0, "temperature": 34.0})
    m2 = twin.state.mean["moisture"].mean()
    twin.step(days=2, weather={"precipitation": 0.0, "temperature": 34.0})
    m4 = twin.state.mean["moisture"].mean()
    assert m0 > m2 > m4                    # keeps drying, not re-initialised
    assert twin.state.sim_day == 4


def test_twin_persists_and_reloads(twin, grid, tmp_path):
    twin.step(days=3)
    snapshot = twin.state.mean["ndvi"].copy()
    again = DigitalTwin.load_or_create("test_field", tmp_path, grid=grid)
    assert again.state.sim_day == 3
    np.testing.assert_allclose(again.state.mean["ndvi"], snapshot)
    assert any(h["reason"] == "predict" for h in again.history)


def test_stress_observation_survives_next_step(twin):
    prior = float(twin.state.mean["stress"][5, 5])
    lat, lon = twin.state.grid.cell_center_latlon(5, 5)
    res = twin.observe_point(lat, lon, {"stress": 0.85}, source="drone_rgb")
    assert res[0]["layer"] == "stress->ndvi"
    observed = float(twin.state.mean["stress"][5, 5])
    assert observed > prior + 0.2                 # pulled toward the reading
    twin.step(days=1)
    assert twin.state.mean["stress"][5, 5] > prior + 0.2   # not wiped by the model
    assert twin.state.mean["stress"][15, 15] < 0.35


def test_observe_point_outside_field_raises(twin):
    with pytest.raises(ValueError):
        twin.observe_point(LAT + 1.0, LON, {"stress": 0.5})


def test_fork_does_not_touch_live_twin(twin):
    before = twin.state.mean["ndvi"].copy()
    day = twin.state.sim_day
    twin.fork().step(days=3)
    np.testing.assert_array_equal(twin.state.mean["ndvi"], before)
    assert twin.state.sim_day == day


def test_what_if_irrigation_raises_moisture(twin):
    hot_dry = {"temperature": 36.0, "precipitation": 0.0, "wind_speed": 20.0}
    plan = [{"day": 1, "zone_id": z, "type": "Precision Irrigation"} for z in twin.zone_names]
    out = twin.what_if(days=3, interventions=plan, weather=hot_dry)
    assert out["delta"]["moisture"] > 0.1


def test_subscribers_receive_events(twin):
    seen = []
    unsub = twin.subscribe(seen.append)
    twin.step(days=1)
    unsub()
    twin.step(days=1)
    assert [e["type"] for e in seen] == ["twin.stepped"]


# ------------------------------------------------------------------- API
@pytest.fixture
def client(twin):
    from fastapi.testclient import TestClient
    from digital_twin_api import create_app

    with TestClient(create_app(twin)) as c:
        yield c


def test_api_ue5_contract(client):
    assert client.get("/").json()["status"] == "ok"
    status = client.get("/field/status").json()
    assert len(status["instances"]) == 400
    inst = status["instances"][21]
    assert (inst["row"], inst["col"]) == (1, 1)
    assert 0.0 <= inst["ndvi"] <= 1.0 and 0.0 <= inst["soil_moisture"] <= 100.0
    heat = client.get("/field/heatmap").json()
    assert len(heat["ndvi_grid"]) == 20 and len(heat["ndvi_grid"][0]) == 20
    assert client.get("/field/instance/400").status_code == 404


def test_api_field_update_is_assimilated(client):
    before = client.get("/field/summary").json()["ndvi"]["mean"]
    instances = [{"index": i, "row": i // 20, "col": i % 20, "ndvi": 0.2, "stress": 0.8,
                  "soil_moisture": 40.0, "temperature_c": 30.0} for i in range(400)]
    r = client.post("/field/update", json={"day": 3, "instances": instances}).json()
    assert r["accepted"] == 400
    assert r["avg_ndvi"] < before
    assert r["new_anomalies"] > 0


def test_api_step_observe_and_what_if(client, twin):
    r = client.post("/twin/step", json={"days": 2, "weather": {"temperature": 30}})
    assert r.status_code == 200 and r.json()["sim_day"] == 2
    lat, lon = twin.state.grid.center_latlon()
    r = client.post("/twin/observe/point", json={"lat": lat, "lon": lon, "values": {"fungus": 0.6},
                                                 "source": "drone_rgb"})
    assert r.status_code == 200 and r.json()["results"][0]["cells_updated"] > 0
    assert client.post("/twin/observe/point", json={"lat": lat, "lon": lon,
                                                     "values": {"bogus": 1}}).status_code == 400
    r = client.post("/twin/what-if", json={"days": 2, "interventions": [
        {"day": 1, "zone_id": 0, "type": "Fungicide Spray"}]})
    assert r.status_code == 200 and "delta" in r.json()
    assert client.post("/twin/step", json={"interventions": [
        {"zone_id": 99, "type": "Fungicide Spray"}]}).status_code == 400
    layer = client.get("/twin/layers/stress?rows=10&cols=10&which=std").json()
    assert len(layer["values"]) == 10


def test_api_websocket_streams_step_events(client):
    with client.websocket_connect("/ws/twin") as ws:
        assert ws.receive_json()["type"] == "twin.hello"
        client.post("/twin/step", json={"days": 1})
        assert ws.receive_json()["type"] == "twin.stepped"


# ------------------------------------------------- ingestion & virtual field
from src.twin_core.ingest import (  # noqa: E402
    bands_to_layers, disease_coverage, footprint_radius_m, ingest_rgb_result,
    rgb_result_to_observation,
)
from src.twin_core.virtual_field import VirtualField, serpentine_waypoints  # noqa: E402
from src.twin_core.weather_feed import get_weather, openmeteo_daily_to_twin, synthetic_weather  # noqa: E402


def test_footprint_and_coverage():
    assert footprint_radius_m(0.0) == pytest.approx(1.5)
    assert footprint_radius_m(10.0) == pytest.approx(7.0, abs=0.01)
    boxes = [{"x1": 0, "y1": 0, "x2": 320, "y2": 240, "confidence": 1.0}]
    assert disease_coverage(boxes, (640, 480)) == pytest.approx(0.25)


def test_rgb_observation_rejects_bad_frames():
    good = {"crop_stress_score": 0.7, "model_confidence": 0.9, "quality_score": 0.9,
            "disease_detections": []}
    assert rgb_result_to_observation(good, {"lat": 0.0, "lon": 0.0}) is None
    assert rgb_result_to_observation({**good, "quality_score": 0.1}, {"lat": 10, "lon": 78}) is None
    obs = rgb_result_to_observation(good, {"lat": 10, "lon": 78, "alt": 10})
    assert obs.values == {"stress": 0.7, "fungus": 0.0}
    low = rgb_result_to_observation({**good, "model_confidence": 0.3}, {"lat": 10, "lon": 78})
    assert low.obs_var["stress"] > obs.obs_var["stress"]


def test_ingest_rgb_updates_twin(twin):
    lat, lon = twin.state.grid.cell_center_latlon(3, 3)
    out = ingest_rgb_result(twin, {"crop_stress_score": 0.8, "quality_score": 1.0,
                                   "model_confidence": 1.0},
                            {"lat": lat, "lon": lon, "alt": 3.0})
    assert out["assimilated"]
    assert twin.state.mean["stress"][3, 3] > 0.3


def test_virtual_bands_invert_to_truth(grid):
    vf = VirtualField(grid, seed=1)
    layers = bands_to_layers(vf.multispectral_survey(noise=0.0))
    for name in ("ndvi", "moisture", "nitrogen"):
        np.testing.assert_allclose(layers[name], vf.truth.state.mean[name], atol=2e-3)


def test_virtual_field_hidden_outbreak(grid):
    vf = VirtualField(grid, seed=1)
    twin = DigitalTwin("t", grid=grid)
    assert vf.truth.state.mean["fungus"].max() > 0.4
    assert vf.twin_error(twin)["fungus"]["rmse"] > 0.03


def test_assimilation_beats_open_loop(grid):
    vf = VirtualField(grid, seed=3)
    open_loop, assim = DigitalTwin("o", grid=grid), DigitalTwin("a", grid=grid)
    wps = serpentine_waypoints(grid, lane_spacing_m=10.0)
    for day in range(1, 5):
        fc = synthetic_weather(1, seed=3, start_day=day)
        vf.advance(1, fc)
        open_loop.step(1, fc)
        assim.step(1, fc)
    for lat, lon in wps:
        ingest_rgb_result(assim, vf.drone_camera(lat, lon, alt=10.0), {"lat": lat, "lon": lon, "alt": 10.0})
    eo, ea = vf.twin_error(open_loop), vf.twin_error(assim)
    assert ea["stress"]["rmse"] < 0.6 * eo["stress"]["rmse"]
    assert ea["fungus"]["rmse"] < 0.7 * eo["fungus"]["rmse"]
    assert ea["stress"]["coverage_95"] > 0.8


def test_serpentine_covers_field(grid):
    wps = serpentine_waypoints(grid, lane_spacing_m=10.0, step_m=5.0)
    cells = {grid.latlon_to_cell(lat, lon) for lat, lon in wps}
    assert None not in cells
    assert len(wps) == 10 * 20


def test_weather_sources():
    w = synthetic_weather(10, seed=5)
    assert len(w) == 10 and w == synthetic_weather(10, seed=5)
    assert all(20 < d["temperature"] < 40 and d["precipitation"] >= 0 for d in w)
    conv = openmeteo_daily_to_twin({
        "dates": ["2026-11-11"], "temperature_2m_max": [32.0], "temperature_2m_min": [24.0],
        "relative_humidity_2m_max": [95], "relative_humidity_2m_min": [65],
        "precipitation_sum": [12.5], "windspeed_10m_max": [14.0]})
    assert conv[0]["temperature"] == 28.0 and conv[0]["humidity"] == 80.0
    assert conv[0]["precipitation"] == 12.5
    weather, used = get_weather(10.0, 78.0, 3, source="synthetic")
    assert used == "synthetic" and len(weather) == 3


def test_api_sim_loop(client):
    r = client.post("/sim/reset", json={"seed": 7}).json()
    start = r["error"]["fungus"]["rmse"]
    r = client.post("/sim/advance", json={"days": 2}).json()
    assert r["sim_day"] == r["truth_day"] == 2 and r["weather_source"] == "synthetic"
    r = client.post("/sim/drone-pass", json={"altitude_m": 10, "lane_spacing_m": 20}).json()
    assert r["frames_assimilated"] > 0
    assert r["error_after"]["fungus"]["rmse"] < r["error_before"]["fungus"]["rmse"]
    r = client.post("/sim/survey", json={}).json()
    assert r["error_after"]["ndvi"]["rmse"] < r["error_before"]["ndvi"]["rmse"]
    assert client.get("/sim/error").json()["error"]["fungus"]["rmse"] < start
    assert len(client.get("/sim/truth/layers/fungus?rows=20&cols=20").json()["values"]) == 20


def test_api_ingest_routes(client, twin):
    lat, lon = twin.state.grid.center_latlon()
    r = client.post("/twin/ingest/rgb", json={
        "result": {"crop_stress_score": 0.7, "quality_score": 0.9, "model_confidence": 0.9},
        "telemetry": {"lat": lat, "lon": lon, "alt": 5}}).json()
    assert r["assimilated"]
    band = [[0.45] * 4 for _ in range(4)]
    r = client.post("/twin/ingest/multispectral", json={
        "green": band, "red": [[0.05] * 4] * 4, "red_edge": [[0.1] * 4] * 4, "nir": band})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    assert client.post("/twin/ingest/multispectral", json={
        "green": band, "red": [[0.05]], "red_edge": band, "nir": band}).status_code == 400
    r = client.post("/twin/step", json={"days": 2, "weather_source": "synthetic"}).json()
    assert r["weather_source"] == "synthetic"


def test_twin_forwarder_posts_to_api(twin, monkeypatch):
    import time as _t
    from fastapi.testclient import TestClient
    from digital_twin_api import create_app
    from src.twin_core.twin_client import TwinForwarder

    tc = TestClient(create_app(twin))
    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: tc.post(
        url.replace("http://twin", ""), json=json))
    fwd = TwinForwarder("http://twin", min_interval_s=0.0)
    lat, lon = twin.state.grid.center_latlon()
    assert fwd({"crop_stress_score": 0.9, "quality_score": 1.0, "model_confidence": 1.0,
                "heatmap_rgb": "dropped"}, {"lat": lat, "lon": lon, "alt": 300.0})
    for _ in range(50):
        if fwd.sent:
            break
        _t.sleep(0.05)
    assert fwd.stats()["sent"] == 1 and fwd.failed == 0
    assert any(e["type"] == "twin.observed" for e in twin.events)


# ------------------------------------------------------------------- swarm
from src.twin_core.swarm import (  # noqa: E402
    Fleet, LocalFrame, Task, allocate, lawnmower, tasks_from_twin,
)


def test_local_frame_round_trip():
    f = LocalFrame(10.0, 78.0)
    lat, lon = f.to_latlon(30.0, -20.0)
    x, y = f.to_local(lat, lon)
    assert x == pytest.approx(30.0, abs=1e-6) and y == pytest.approx(-20.0, abs=1e-6)


def test_lawnmower_alternates_lanes():
    pts = lawnmower((0.0, 50.0, 0.0, 20.0), spacing=10.0)
    assert len(pts) == 4
    assert pts[0][0] < pts[1][0] and pts[2][0] > pts[3][0]     # serpentine
    assert all(0.0 <= x <= 50.0 and 0.0 <= y <= 20.0 for x, y in pts)


def test_fleet_registry(grid):
    fleet = Fleet.for_field(grid, 3)
    assert sorted(fleet.drones) == ["drone_01", "drone_02", "drone_03"]
    assert len(set(fleet.pads.values())) == 3
    fleet.remove_drone("drone_02")
    fleet.add_drone("drone_99")
    assert len(set(fleet.pads.values())) == 3
    with pytest.raises(ValueError):
        fleet.add_drone("drone_99")
    snap = fleet.snapshot()
    assert set(snap["drones"]) == {"drone_01", "drone_03", "drone_99"}
    assert {"lat", "lon", "alt", "battery", "is_spraying"} <= set(snap["drones"]["drone_01"])


def test_allocation_balances_and_respects_limits(twin):
    fleet = Fleet.for_field(twin.state.grid, 3)
    tasks = tasks_from_twin(twin, "scout", fleet.frame)
    assert len(tasks) == 16
    three = allocate(tasks, fleet.drone_info())
    one = allocate(tasks, {"d": {"home": (0.0, 0.0), "battery": 100.0, "tank_kg": 10.0}})
    assert not three.unassigned and not one.unassigned
    assert three.makespan_s < 0.5 * one.makespan_s
    assert sum(len(a.tasks) for a in three.assignments.values()) == 16

    big = Task("spray-big", "spray", 0, [(0.0, 20.0), (40.0, 20.0)], priority=1.0, payload_kg=12.0)
    assert allocate([big], fleet.drone_info()).unassigned == [big]          # exceeds tank
    weak = {"w": {"home": (0.0, 0.0), "battery": 36.0, "tank_kg": 10.0}}
    assert allocate(tasks, weak).unassigned                                  # battery reserve


def test_fleet_scout_mission_feeds_twin(grid):
    vf = VirtualField(grid, seed=2)
    twin = DigitalTwin("t", grid=grid)
    fleet = Fleet.for_field(grid, 3)
    plan = allocate(tasks_from_twin(twin, "scout", fleet.frame), fleet.drone_info())
    before = vf.twin_error(twin)["fungus"]["rmse"]

    def capture(_d, lat, lon, alt):
        frame = vf.drone_camera(lat, lon, alt)
        if frame:
            ingest_rgb_result(twin, frame, {"lat": lat, "lon": lon, "alt": alt})

    rep = fleet.execute(plan, on_capture=capture)
    assert not rep["timed_out"]
    for st in rep["drones"].values():
        assert st["completed"] and st["captures"] > 0
        assert st["flight_s"] == pytest.approx(st["est_time_s"], rel=0.3)   # planner is honest
    assert vf.twin_error(twin)["fungus"]["rmse"] < 0.7 * before
    for s in fleet.snapshot()["drones"].values():
        assert s["alt"] < 1.5                                                 # back on the pad


def test_fleet_spray_mission_doses_and_reports(grid):
    vf = VirtualField(grid, seed=2)
    fleet = Fleet.for_field(grid, 2)
    tasks = tasks_from_twin(vf.truth, "spray", fleet.frame, fungus_threshold=0.15)
    assert tasks and all(t.payload_kg > 0 for t in tasks)
    rep = fleet.execute(allocate(tasks, fleet.drone_info()))
    assert {s["zone_id"] for s in rep["completed_sprays"]} == {t.zone_id for t in tasks}
    for s in rep["completed_sprays"]:
        assert s["applied_kg"] == pytest.approx(s["planned_kg"], rel=0.3)
        assert s["applied_kg"] <= s["planned_kg"] + 1e-6


def test_api_swarm_and_closed_loop(client):
    client.post("/sim/reset", json={"seed": 4})
    assert len(client.get("/swarm/state").json()["drones"]) == 3
    assert client.post("/swarm/drones", json={"drone_id": "drone_04"}).status_code == 200
    assert client.post("/swarm/drones", json={"drone_id": "drone_04"}).status_code == 409
    assert client.delete("/swarm/drones/drone_04").status_code == 200
    assert client.delete("/swarm/drones/nope").status_code == 404
    plan = client.post("/swarm/plan", json={"kind": "scout"}).json()
    assert len(plan["tasks"]) == 16 and not plan["unassigned"]
    assert client.post("/swarm/plan", json={"kind": "scout", "zones": [99]}).status_code == 400

    r = client.post("/sim/mission", json={"kind": "auto", "fungus_threshold": 0.15}).json()
    assert r["frames_assimilated"] > 0
    assert r["error_after"]["fungus"]["rmse"] < r["error_before"]["fungus"]["rmse"]
    assert [m["kind"] for m in r["missions"]] == ["scout", "spray"]
    queued = r["sprays_queued"]
    assert queued
    adv = client.post("/sim/advance", json={"days": 1}).json()
    assert sorted(i["zone_id"] for i in adv["applied_fleet_interventions"]) == sorted(queued)
    assert client.get("/swarm/state").json()["pending_interventions"] == []


# ------------------------------------------------------- paddy crop model
from src.twin_core.calibration import (  # noqa: E402
    Flight, constant_temperature, fit_ndvi_curve, fit_phenology, read_flights,
)
from src.twin_core.paddy_model import (  # noqa: E402
    PaddyParams, moisture_from_water, ndvi_potential, stage_for_gdd, water_from_moisture,
)
import datetime as _dt  # noqa: E402

DRY = {"temperature": 31.0, "humidity": 65.0, "precipitation": 0.0, "wind_speed": 8.0}


def _paddy(grid, irrigation_interval=5, **values):
    p = PaddyParams(irrigation_interval_days=irrigation_interval)
    return DigitalTwin("p", grid=grid, params=p,
                       state=FieldState.initial(grid, values=values or None, seed=None))


def test_water_moisture_mapping_round_trips():
    p = PaddyParams()
    for m in (0.2, 0.55, 0.6, 0.8, 1.0):
        h, s = water_from_moisture(p, np.array([m], dtype=np.float32))
        assert moisture_from_water(p, h, s)[0] == pytest.approx(m, abs=1e-5)


def test_flooded_paddy_stays_wet_with_canal_rotation(grid):
    tw = _paddy(grid)
    tw.step(14, DRY)
    assert tw.state.mean["moisture"].min() > 0.6           # never drains below saturation
    assert tw.state.aux["water_mm"].max() <= tw.params.bund_mm
    out = tw.step(1, DRY)
    assert not any("Drought" in line for line in out["logs"])


def test_paddy_without_irrigation_drains_then_droughts(grid):
    tw = _paddy(grid, irrigation_interval=0)
    tw.step(7, DRY)
    assert tw.state.aux["water_mm"].mean() < 15.0           # ET + percolation ~ 7.5 mm/day
    ndvi_7 = float(tw.state.mean["ndvi"].mean())
    tw.step(14, DRY)
    assert tw.state.aux["soil_sat"].mean() < 0.85
    assert tw.state.mean["moisture"].mean() < 0.6
    assert tw.state.mean["ndvi"].mean() < ndvi_7            # drought hurts the canopy


def test_heavy_rain_overflows_bund_not_unbounded(grid):
    tw = _paddy(grid)
    tw.step(3, {**DRY, "precipitation": 80.0})
    assert tw.state.aux["water_mm"].max() == pytest.approx(tw.params.bund_mm)


def test_disease_grows_in_humid_weather_only(grid):
    humid = {"temperature": 26.0, "humidity": 93.0, "precipitation": 5.0, "wind_speed": 6.0}
    a = _paddy(grid, fungus=0.02)
    a.step(14, humid)
    b = _paddy(grid, fungus=0.02)
    b.step(14, DRY)
    assert a.state.mean["fungus"].mean() > 0.1             # ~r 0.15/day
    assert a.state.mean["fungus"].max() <= a.params.max_severity + 1e-6
    assert b.state.mean["fungus"].mean() < 0.03


def test_fungicide_kills_and_protects(grid):
    humid = {"temperature": 26.0, "humidity": 93.0, "precipitation": 5.0}
    tw = _paddy(grid, fungus=0.2)
    spray = [{"day": 1, "zone_id": z, "type": "Fungicide Spray"} for z in tw.zone_names]
    tw.step(1, humid, spray)
    assert tw.state.mean["fungus"].mean() < 0.1
    after_spray = float(tw.state.mean["fungus"].mean())
    tw.step(5, humid)
    unsprayed = _paddy(grid, fungus=after_spray)
    unsprayed.step(5, humid)
    assert tw.state.mean["fungus"].mean() < unsprayed.state.mean["fungus"].mean()   # residual protection


def test_season_follows_paddy_ndvi_curve(grid):
    st = FieldState.initial(grid, values={"ndvi": 0.25, "nitrogen": 0.6}, seed=None, growth_stage="Nursery")
    tw = DigitalTwin("season", state=st)
    tw.state.scalars["gdd"] = 0.0
    weather = synthetic_weather(120, seed=1)
    ndvi, stages = [], []
    for d in range(120):
        iv = []
        if d + 1 in (21, 45, 65):
            iv += [{"day": 1, "zone_id": z, "type": "Nutrient Top-Dress"} for z in tw.zone_names]
        if d + 1 in (40, 60):
            iv += [{"day": 1, "zone_id": z, "type": "Fungicide Spray"} for z in tw.zone_names]
        tw.step(1, weather[d], iv)
        ndvi.append(float(tw.state.mean["ndvi"].mean()))
        stages.append(tw.state.growth_stage)
    peak = int(np.argmax(ndvi))
    assert 0.78 < max(ndvi) < 0.9
    assert stages[peak] in ("Vegetative", "Booting", "Flowering") and peak > 50
    assert 0.3 < ndvi[-1] < 0.6                               # senescence by harvest
    assert stages[-1] == "Mature" and "Booting" in stages and "Flowering" in stages


def test_stage_and_stress_relations():
    p = PaddyParams()
    assert stage_for_gdd(p, 0) == "Nursery"
    assert stage_for_gdd(p, p.gdd_onset["Booting"] + 1) == "Booting"
    assert ndvi_potential(p, 50) < 0.3 < ndvi_potential(p, p.gdd_onset["Booting"])
    assert ndvi_potential(p, p.gdd_harvest) == pytest.approx(p.ndvi_mature, abs=0.02)


def test_set_growth_stage_moves_thermal_time(twin):
    twin.set_growth_stage("Flowering")
    assert twin.state.growth_stage == "Flowering"
    assert twin.state.scalars["gdd"] == pytest.approx(twin.params.gdd_onset["Flowering"])


def test_paddy_state_persists(twin, grid, tmp_path):
    twin.step(3, DRY)
    again = DigitalTwin.load_or_create("test_field", tmp_path, grid=grid)
    np.testing.assert_allclose(again.state.aux["water_mm"], twin.state.aux["water_mm"])
    assert again.state.scalars["gdd"] == pytest.approx(twin.state.scalars["gdd"])


def test_legacy_crop_model_still_available(grid):
    tw = DigitalTwin("legacy", grid=grid, crop_model="legacy")
    tw.step(2)
    assert tw.state.sim_day == 2


# --------------------------------------------------------- calibration
def test_phenology_fit_on_repo_manifest():
    from pathlib import Path
    manifest = Path(__file__).resolve().parents[1] / "data" / "manifests" / "flight_manifest.csv"
    flights = read_flights(manifest)
    assert len(flights) == 20
    p, rep = fit_phenology(flights, PaddyParams(), constant_temperature(28.0))
    assert rep["flights_classified_correctly"] == "20/20"
    order = [p.gdd_onset[s] for s in ("Vegetative", "Booting", "Flowering", "Mature")]
    assert order == sorted(order)


def test_ndvi_curve_fit_recovers_parameters():
    true = PaddyParams(ndvi_max=0.8, ndvi_rise_mid_gdd=600.0, ndvi_mature=0.4)
    pts = [(g, ndvi_potential(true, g)) for g in range(0, 2300, 150)]
    fitted, rep = fit_ndvi_curve(pts, PaddyParams())
    assert rep["fitted"] and rep["rmse"] < 0.01
    assert fitted.ndvi_max == pytest.approx(0.8, abs=0.03)
    assert fitted.ndvi_rise_mid_gdd == pytest.approx(600.0, rel=0.1)
    assert fit_ndvi_curve([(0, 0.3)], PaddyParams())[1]["fitted"] is False


# ------------------------------------------------------------ UE5 stream
def _receive_until(ws, msg_type, limit=50):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == msg_type:
            return msg
    raise AssertionError(f"no {msg_type} message in {limit} messages")


def test_ws_field_snapshot_schema(client):
    with client.websocket_connect("/ws/field") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "field.snapshot" and snap["seq"] == 0
        assert snap["grid"]["rows"] == snap["grid"]["cols"] == 20
        for key in ("ndvi", "stress", "stress_std", "moisture", "fungus", "water_mm", "zone"):
            assert len(snap[key]) == 400, key
        assert all(0.0 <= v <= 1.0 for v in snap["ndvi"])
        assert set(snap["zone"]) == set(range(16))
        assert snap["growth_stage"] == "Vegetative" and snap["gdd"] > 0


def test_ws_field_pushes_on_change_and_coalesces(client):
    with client.websocket_connect("/ws/field") as ws:
        first = ws.receive_json()
        client.post("/twin/step", json={"days": 1})
        nxt = _receive_until(ws, "field.snapshot")
        assert nxt["seq"] > first["seq"] and nxt["sim_day"] == first["sim_day"] + 1
        ws.send_json({"type": "snapshot"})
        again = _receive_until(ws, "field.snapshot")
        assert again["seq"] > nxt["seq"]


def test_ws_field_streams_swarm_track(client):
    client.post("/sim/reset", json={"seed": 4})
    with client.websocket_connect("/ws/field") as ws:
        ws.receive_json()
        client.post("/sim/mission", json={"kind": "scout"})
        track = _receive_until(ws, "swarm.track")
        assert track["fields"] == ["t_s", "row", "col", "alt_m", "spraying"]
        assert len(track["drones"]) == 3
        pts = [p for d in track["drones"].values() for p in d]
        inside = [p for p in pts if 0 <= p[1] < 20 and 0 <= p[2] < 20]
        assert len(inside) > 0.6 * len(pts)          # mostly over the field
        assert max(p[3] for p in pts) > 8.0           # flew at scout altitude
    with client.websocket_connect("/ws/field") as ws:  # late joiners get the last track
        ws.receive_json()
        assert ws.receive_json()["type"] == "swarm.track"
