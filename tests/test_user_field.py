"""Uploaded field -> live twin (POST /field/from-image, DELETE /field/user, GET /field/source)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import digital_twin_api as api


def _field(h=60, w=60, scale=1.0):
    """4-band reflectance with a low-NDVI (stressed) disc in the north-east."""
    rr, cc = np.mgrid[0:h, 0:w]
    disc = np.hypot(rr - h * 0.25, cc - w * 0.75) < h * 0.15
    nir = np.where(disc, 0.25, 0.60)
    red = np.where(disc, 0.15, 0.06)
    return {"green": (np.full((h, w), 0.08) * scale).tolist(), "red": (red * scale).tolist(),
            "red_edge": (np.full((h, w), 0.30) * scale).tolist(), "nir": (nir * scale).tolist()}


@pytest.fixture()
def client():
    from src.twin_core import DigitalTwin, FieldGrid
    twin = DigitalTwin(grid=FieldGrid.centered_on(10.0, 78.0), storage_dir=None)   # never writes to outputs/
    return TestClient(api.create_app(twin))


def test_upload_makes_the_image_the_field(client):
    assert client.get("/field/source").json()["source"] == "simulated"
    r = client.post("/field/from-image", json={"name": "my_field.tif", "bands": _field(),
                                               "center_lat": 11.25, "center_lon": 79.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "my_field.tif"
    assert abs(body["grid"]["center"][0] - 11.25) < 1e-4 and abs(body["grid"]["center"][1] - 79.5) < 1e-4
    frame = client.get("/ue5/frame?since=-1").json()
    assert frame["field_source"]["source"] == "upload"
    truth = np.array(frame["field"]["truth_ndvi"]).reshape(20, 20)
    twin = np.array(frame["field"]["ndvi"]).reshape(20, 20)
    # REALITY is the image: the stressed disc (north-east) has clearly lower NDVI than the rest
    assert truth[3:7, 13:17].mean() < truth[14:, :6].mean() - 0.2
    # the twin assimilated the image: it already sees the same pattern
    assert np.corrcoef(truth.ravel(), twin.ravel())[0, 1] > 0.8
    # no hidden disease on upload: reality is what the image shows
    assert max(frame["field"]["truth_fungus"]) < 0.2


def test_scaled_reflectance_and_missing_bands(client):
    ok = client.post("/field/from-image", json={"name": "dn.tif", "bands": _field(scale=10000.0)})
    assert ok.status_code == 200 and ok.json()["means"]["ndvi"] > 0.3     # 0-10000 rescaled to 0-1
    bad = _field()
    del bad["nir"]
    assert client.post("/field/from-image", json={"bands": bad}).status_code == 400


def test_field_size_from_the_georeference(client):
    r = client.post("/field/from-image", json={"bands": _field(), "center_lat": 11.0, "center_lon": 79.0,
                                               "width_m": 240.0, "height_m": 180.0}).json()
    assert abs(r["grid"]["cell_size_m"] - 6.0) < 1e-6        # 240 m over 40 cells
    assert abs(r["grid"]["area_ha"] - 5.76) < 1e-3          # the square grid that covers the field
    tiny = client.post("/field/from-image", json={"bands": _field(), "width_m": 12.0, "height_m": 12.0}).json()
    assert tiny["grid"]["cell_size_m"] == 2.5 and "default" in tiny["size_note"]


def test_new_flight_updates_the_twin(client):
    client.post("/field/from-image", json={"name": "plot7", "bands": _field()})
    # a repeat flight that agrees with the twin: the twin gets more certain
    same = client.post("/field/from-image", json={"bands": _field(), "mode": "update"}).json()
    assert same["ndvi_uncertainty"]["after"] < same["ndvi_uncertainty"]["before"]
    # the next flight: the stressed disc has recovered (a surprise the twin must follow)
    healed = _field()
    healed["nir"] = (np.full((60, 60), 0.6)).tolist()
    healed["red"] = (np.full((60, 60), 0.06)).tolist()
    r = client.post("/field/from-image", json={"name": "plot7 week 2", "bands": healed, "mode": "update"}).json()
    assert r["mode"] == "update" and r["flights"] == 3 and r["name"] == "plot7"
    twin = np.array(client.get("/ue5/frame?since=-1").json()["field"]["ndvi"]).reshape(20, 20)
    assert twin[3:7, 13:17].mean() > 0.5                     # the twin saw the recovery
    assert len(client.get("/field/flights").json()["flights"]) == 3


def test_update_needs_a_loaded_field(client):
    assert client.post("/field/from-image", json={"bands": _field(), "mode": "update"}).status_code == 409


def test_reset_keeps_the_uploaded_field_and_forget_goes_back(client):
    client.post("/field/from-image", json={"name": "keep.tif", "bands": _field()})
    r = client.post("/sim/reset", json={"seed": 7}).json()
    assert r["source"] == "upload"
    frame = client.get("/ue5/frame?since=-1").json()
    assert max(frame["field"]["truth_fungus"]) > 0.3          # demo story: one outbreak hidden in your field
    assert client.delete("/field/user").status_code == 200
    assert client.get("/field/source").json()["source"] == "simulated"
