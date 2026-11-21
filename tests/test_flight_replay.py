"""Real-flight replay (src/twin_core/flight_replay.py, scripts/replay_flights.py) on a synthetic DJI-style dataset."""

import datetime as dt
import math

import numpy as np
import pytest
import tifffile

from src.twin_core import DigitalTwin
from src.twin_core.field_state import FieldGrid
from src.twin_core.flight_replay import (ImagePose, find_flights, grid_for_points, image_tiles, offset_latlon,
                                         read_pose, replay)
from src.twin_core.paddy_model import PaddyParams, ndvi_potential, daily_gdd

LAT, LON = 10.79, 78.71


def _xmp(lat, lon, alt, yaw):
    return (f'<x:xmpmeta><rdf:Description drone-dji:GpsLatitude="{lat:+.8f}" '
            f'drone-dji:GpsLongitude="{lon:+.8f}" drone-dji:RelativeAltitude="+{alt:.2f}" '
            f'drone-dji:GimbalYawDegree="{yaw:+.1f}"/></x:xmpmeta>').encode()


def _write_set(folder, name, ndvi, lat, lon, alt=30.0, yaw=0.0, shape=(48, 64)):
    red = np.full(shape, 800.0)
    nir = red * (1 + ndvi) / (1 - ndvi)
    tag = _xmp(lat, lon, alt, yaw)
    for band, arr in (("R", red), ("NIR", nir)):
        tifffile.imwrite(folder / f"{name}_MS_{band}.TIF", arr.astype(np.uint16),
                         extratags=[(700, 1, len(tag), tag, True)])


WEATHER = [{"temperature": 28.0, "humidity": 80.0, "precipitation": 5.0}]


def flat_weather(start, end):
    return WEATHER * (end - start).days


def make_dataset(root, vigor=0.85, seed=3):
    """
    A nursery flight, then field flights over a "real" field simulated by the
    paddy model itself with a hidden crop vigor (0.85 = grows 15% less than
    modelled), plus 0.02 image noise.
    """
    from src.twin_core.paddy_model import ndvi_potential, stage_for_gdd
    rng = np.random.default_rng(seed)
    real = DigitalTwin("real", grid=FieldGrid.centered_on(LAT, LON), storage_dir=None, cross_layer=False)
    real.state.scalars["gdd"] = 0.0
    real.state.growth_stage = stage_for_gdd(real.params, 0.0)
    real.state.mean["ndvi"][:] = ndvi_potential(real.params, 0.0)
    real.state.aux["vigor_factor"] = np.full(real.state.grid.shape, vigor, dtype=np.float32)
    transplant = dt.date(2025, 7, 19)
    flights = [("01", dt.date(2025, 7, 15), "Nursery"), ("02", transplant, "Nursery"),
               ("03", dt.date(2025, 8, 9), "Vegetative"), ("04", dt.date(2025, 8, 23), "Vegetative"),
               ("05", dt.date(2025, 9, 6), "Booting")]
    day = transplant
    for seq, date, stage in flights:
        if date > day:
            real.step((date - day).days, flat_weather(day, date))
            day = date
        d = root / "Field001" / f"F1_{seq}_{date:%Y_%m_%d}_1000_{stage}_Stage"
        d.mkdir(parents=True)
        true = float(real.state.mean["ndvi"].mean())
        k = 0
        for north in (-30.0, 0.0, 30.0):           # a 3 x 3 lawn-mower of image positions
            for east in (-30.0, 0.0, 30.0):
                lat, lon = offset_latlon(LAT, LON, north, east)
                _write_set(d, f"DJI_{seq}{k:04d}", true + rng.normal(0, 0.02), lat, lon)
                k += 1
    return root


def test_reads_dji_pose_and_tiles_point_the_right_way(tmp_path):
    _write_set(tmp_path, "IMG", 0.6, LAT, LON, alt=40.0, yaw=90.0)
    pose = read_pose(tmp_path / "IMG_MS_R.TIF")
    assert pose.lat == pytest.approx(LAT) and pose.lon == pytest.approx(LON)
    assert pose.alt_m == pytest.approx(40.0) and pose.yaw_deg == pytest.approx(90.0)
    red = np.full((30, 30), 100.0)
    nir = red.copy()
    nir[:10] = 300.0                                     # top third of the frame is greener
    tiles = image_tiles(red, nir, ImagePose(LAT, LON, 30.0, 90.0), tiles=3)
    assert len(tiles) == 9
    top, top_left = tiles[1], tiles[0]                   # row-major: top-middle, top-left
    assert top[2] > tiles[4][2]
    # Facing east (yaw 90): the top of the frame is east of the drone, its left is north.
    assert top_left[0] > LAT
    assert top[1] > LON and abs(top[0] - LAT) < 1e-5
    ground_h = 2 * 30.0 * math.tan(math.radians(48.1) / 2)
    east_m = (top[1] - LON) * math.radians(1) * 6_371_000 * math.cos(math.radians(LAT))
    assert east_m == pytest.approx(ground_h / 3, rel=0.02)


def test_grid_covers_flight():
    pts = [offset_latlon(LAT, LON, n, e) for n in (-50, 50) for e in (-40, 40)]
    g = grid_for_points(pts)
    assert all(g.latlon_to_cell(*p) is not None for p in pts)
    assert g.cell_size_m == 2.5 and g.rows >= 48


def test_replay_learns_the_real_field(tmp_path):
    flights = find_flights(make_dataset(tmp_path))
    assert [f.stage for f in flights] == ["Nursery", "Nursery", "Vegetative", "Vegetative", "Booting"]
    assert all(len(f.prefixes) == 9 for f in flights)
    seen = []
    res = replay(flights, lambda g: DigitalTwin("r", grid=g, storage_dir=None), flat_weather,
                 on_flight=lambda row, twin, before, measured: seen.append(measured is not None))
    rows = res["flights"]
    assert res["transplant"] == "2025-07-19" and len(rows) == 3 and all(seen)
    assert rows[0]["tiles"] == 81 and rows[0]["cells_observed"] > 300
    # The model alone expects a greener field; the twin learns the hidden vigor
    # and predicts the later flights better.
    later = rows[1:]
    assert all(r["model_only_ndvi"] > r["observed_ndvi"] + 0.03 for r in later), later
    assert all(r["twin_rmse"] < 0.75 * r["model_only_rmse"] for r in later), later
    vigor = float(res["twin"].state.aux["vigor_factor"][res["twin"].state.grid.rows // 2].mean())
    assert 0.8 < vigor < 0.95
    assert float(res["model_only"].state.aux["vigor_factor"].mean()) == pytest.approx(1.0, abs=0.02)


def test_cli_writes_report_maps_and_twin(tmp_path, monkeypatch, capsys):
    import json
    from scripts import replay_flights
    from src.config.config import settings
    monkeypatch.setattr(settings, "twin_storage_dir", tmp_path / "store")
    make_dataset(tmp_path / "raw")
    rows = replay_flights.main(["--raw", str(tmp_path / "raw"), "--offline", "--save-twin",
                                "--out", str(tmp_path / "out"), "--params", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert "Predicting the next flight" in out and "% better" in out
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert len(rep["flights"]) == len(rows) == 3
    assert len(list((tmp_path / "out").glob("flight_F1_*.png"))) == 3
    twin = DigitalTwin.load_or_create("real_F1", tmp_path / "store")
    assert twin.state.sim_day > 40
