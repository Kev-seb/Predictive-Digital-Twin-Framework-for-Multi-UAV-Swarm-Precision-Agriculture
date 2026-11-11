import numpy as np
import pytest

from src.twin_core import DigitalTwin, FieldGrid, FieldState
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
    assert s.mean["stress"][10, 11] > s.mean["stress"][10, 13] > 0.15 - 1e-6
    assert s.mean["stress"][0, 0] == pytest.approx(0.15)


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
