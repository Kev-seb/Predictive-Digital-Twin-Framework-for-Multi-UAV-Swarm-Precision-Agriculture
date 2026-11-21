"""Cross-layer correction (src/twin_core/ensemble.py)."""

import numpy as np

from src.twin_core import DigitalTwin, FieldGrid
from src.twin_core.virtual_field import VirtualField


def _twin(**kw):
    return DigitalTwin("x", grid=FieldGrid.centered_on(10.0, 78.0), **kw)


def test_ensemble_only_where_it_is_useful():
    twin = _twin()
    twin.step(1)
    assert twin.ensemble is not None and twin.ensemble.n == 24
    assert twin.fork().cross_layer is False                     # forecasts don't carry it
    assert VirtualField(twin.state.grid).truth.cross_layer is False   # the truth has no filter
    assert _twin(crop_model="legacy").cross_layer is False
    off = _twin(cross_layer=False)
    off.step(1)
    assert off.ensemble is None


def test_moisture_reading_also_corrects_leak_rate_and_reports_it():
    twin = _twin()
    twin.step(5)                                                # let correlations build
    perc0 = twin.state.aux["percolation_mm_day"].copy()
    lat, lon = twin.state.grid.cell_center_latlon(20, 20)
    dry = float(twin.state.mean["moisture"][20, 20]) - 0.25    # much drier than the twin thinks
    res = twin.observe_point(lat, lon, {"moisture": dry}, source="soil_probe")
    assert res[0]["cross_layer"]["percolation"] > 0
    # Drier than expected -> the soil there drains faster than assumed.
    assert twin.state.aux["percolation_mm_day"][20, 20] > perc0[20, 20] * 1.2
    far = twin.state.aux["percolation_mm_day"][2, 2]
    assert abs(far - perc0[2, 2]) < 1e-3                       # only near the reading
    # An NDVI reading corrects only the crop-vigor parameter, never other layers (CROSS_FROM).
    moist = twin.state.mean["moisture"].copy()
    res = twin.observe_point(lat, lon, {"ndvi": 0.3}, source="drone_rgb")
    assert set(res[0].get("cross_layer", {})) <= {"vigor"}
    assert np.array_equal(twin.state.mean["moisture"], moist)
    assert twin.state.aux["vigor_factor"][20, 20] < 1.0              # much browner than expected


def test_osse_cross_layer_learns_the_leaky_patch():
    from scripts.validate_cross_layer import run
    r = run(seed=1, days=21, drone_every=3, survey_every=7, horizon=7)
    per, cross = r["per_layer"], r["cross_layer"]
    assert cross["rmse"]["moisture"] < per["rmse"]["moisture"]
    assert cross["leak_rate_factor_error"] < 0.6 * per["leak_rate_factor_error"]
    assert cross["forecast_moisture_rmse_leaky"] < 0.8 * per["forecast_moisture_rmse_leaky"]
    assert cross["coverage_95"]["moisture"] > 0.9               # still honest about uncertainty
