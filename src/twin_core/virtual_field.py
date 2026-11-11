"""
virtual_field.py
----------------
The simulated "physical" paddy field and its sensors.

The project is simulation-only, so something has to play the role of the
real field the twin is trying to track. ``VirtualField`` is that hidden
ground truth: it runs the same agronomic simulator, but

  * starts from a different state (e.g. a fungal outbreak the twin has not
    been told about, a drier corner),
  * experiences the *actual* weather, which differs from the forecast the
    twin is given.

Sensors read the truth with realistic noise and produce the same products
the real pipeline does (RGB AI result dicts, 4-band reflectance), so they
flow through ``src.twin_core.ingest`` exactly like real data would.

``twin_error()`` scores the twin against the truth (RMSE and whether the
twin's own uncertainty is honest). This is an observing-system simulation
experiment (OSSE), the standard way to validate a data-assimilation system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.twin_core.field_state import LAYERS, FieldGrid, FieldState
from src.twin_core.ingest import DEFAULT_FRAME_WH
from src.twin_core.twin import DigitalTwin

NIR_REFLECTANCE = 0.45


class VirtualField:
    def __init__(
        self,
        grid: FieldGrid,
        seed: int = 42,
        outbreaks: Optional[List[Dict[str, float]]] = None,
        dry_patch: Optional[Dict[str, float]] = None,
        weather_noise: Dict[str, float] = None,
        propagation_model: str = "pde",
        zone_layout: tuple = (4, 4),
    ):
        self.rng = np.random.default_rng(seed)
        state = FieldState.initial(grid, seed=seed + 1)
        rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]

        if outbreaks is None:
            outbreaks = [{"row": grid.rows * 0.3, "col": grid.cols * 0.65,
                          "radius": grid.rows * 0.12, "load": 0.55}]
        for ob in outbreaks:
            d = np.hypot(rr - ob["row"], cc - ob["col"])
            bump = ob["load"] * np.exp(-(d / max(ob["radius"], 1e-6)) ** 2)
            state.mean["fungus"] = np.clip(state.mean["fungus"] + bump, 0.0, 1.0).astype(np.float32)
            state.mean["ndvi"] = np.clip(state.mean["ndvi"] - 0.5 * bump, -1.0, 1.0).astype(np.float32)

        if dry_patch is None:
            dry_patch = {"row": grid.rows * 0.8, "col": grid.cols * 0.2,
                         "radius": grid.rows * 0.15, "deficit": 0.3}
        if dry_patch:
            d = np.hypot(rr - dry_patch["row"], cc - dry_patch["col"])
            dip = dry_patch["deficit"] * np.exp(-(d / max(dry_patch["radius"], 1e-6)) ** 2)
            state.mean["moisture"] = np.clip(state.mean["moisture"] - dip, 0.0, 1.0).astype(np.float32)

        state.mean["stress"] = np.clip((1.0 - state.mean["ndvi"]) / 2.0
                                       + 0.35 * state.mean["fungus"], 0.0, 1.0).astype(np.float32)
        for name in LAYERS:
            state.var[name][:] = 0.0  # truth has no uncertainty

        self.truth = DigitalTwin(field_id="virtual_field", state=state, storage_dir=None,
                                 propagation_model=propagation_model, zone_layout=zone_layout)
        self.weather_noise = weather_noise or {"temperature": 1.0, "humidity": 4.0,
                                               "precipitation": 0.35}

    # ------------------------------------------------------------ dynamics
    @property
    def grid(self) -> FieldGrid:
        return self.truth.state.grid

    @property
    def sim_day(self) -> int:
        return self.truth.state.sim_day

    def actual_weather(self, forecast: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """What really happens: the forecast plus error (rain error is multiplicative)."""
        out = []
        n = self.weather_noise
        for w in forecast:
            a = dict(w)
            a["temperature"] = float(w.get("temperature", 29.0) + self.rng.normal(0, n["temperature"]))
            a["humidity"] = float(np.clip(w.get("humidity", 78.0)
                                          + self.rng.normal(0, n["humidity"]), 30.0, 100.0))
            a["precipitation"] = float(max(0.0, w.get("precipitation", 0.0)
                                           * self.rng.lognormal(0.0, n["precipitation"])))
            out.append(a)
        return out

    def advance(self, days: int, forecast: List[Dict[str, float]],
                interventions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, float]]:
        actual = self.actual_weather(forecast[:days])
        self.truth.step(days=days, weather=actual, interventions=interventions)
        for name in LAYERS:
            self.truth.state.var[name][:] = 0.0
        return actual

    # ------------------------------------------------------------- sensors
    def _true_at(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        return self.grid.latlon_to_cell(lat, lon)

    def drone_camera(self, lat: float, lon: float, alt: float = 10.0,
                     frame_wh=DEFAULT_FRAME_WH) -> Optional[Dict[str, Any]]:
        """
        Simulated nadir RGB frame run through the RGB AI. Returns a dict shaped
        like ``RGBInferenceResult.to_dict()`` or None when outside the field.
        The reading averages the truth over the camera footprint.
        """
        if self._true_at(lat, lon) is None:
            return None
        from src.twin_core.ingest import footprint_radius_m

        radius = footprint_radius_m(alt)
        dist = self.grid.cell_distances_m(lat, lon)
        mask = dist <= max(radius, self.grid.cell_size_m * 0.75)
        stress_true = float(self.truth.state.mean["stress"][mask].mean())
        fungus_true = float(self.truth.state.mean["fungus"][mask].mean())

        stress = float(np.clip(stress_true + self.rng.normal(0, 0.06), 0.0, 1.0))
        coverage = float(np.clip(fungus_true + self.rng.normal(0, 0.08), 0.0, 1.0))
        detections = []
        if coverage > 0.08:
            # Split the diseased area into a few boxes covering `coverage` of the frame.
            w, h = frame_wh
            k = int(self.rng.integers(1, 4))
            side = np.sqrt(coverage * w * h / k)
            for _ in range(k):
                x1 = float(self.rng.uniform(0, max(1.0, w - side)))
                y1 = float(self.rng.uniform(0, max(1.0, h - side)))
                detections.append({"x1": x1, "y1": y1, "x2": x1 + side, "y2": y1 + side,
                                   "label": "Leaf Blast", "confidence": 1.0})
        return {
            "crop_stress_score": stress,
            "stress_label": "Severe" if stress > 0.6 else "Moderate" if stress > 0.4
                            else "Mild" if stress > 0.25 else "Healthy",
            "disease_detections": detections,
            "model_confidence": float(np.clip(self.rng.normal(0.85, 0.05), 0.5, 1.0)),
            "quality_score": float(np.clip(self.rng.normal(0.9, 0.05), 0.0, 1.0)),
        }

    def multispectral_survey(self, noise: float = 0.01, coverage: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Simulated 4-band reflectance over the field grid, consistent with the
        truth's NDVI / moisture / nitrogen through the index formulas in
        src/indices/indices.py. ``coverage`` < 1 leaves random cells as NaN
        (cloud shadow, missed images).
        """
        m = self.truth.state.mean
        nir = np.full(self.grid.shape, NIR_REFLECTANCE, dtype=np.float32)
        ndvi = np.clip(m["ndvi"], -0.95, 0.95)
        moist = np.clip(m["moisture"], 0.02, 0.98)
        cire = np.clip(m["nitrogen"] * 5.0, 0.05, 9.0)
        bands = {
            "nir": nir,
            "red": nir * (1.0 - ndvi) / (1.0 + ndvi),
            "green": nir * moist / (1.0 - moist),
            "red_edge": nir / (cire + 1.0),
        }
        for k, v in bands.items():
            bands[k] = np.clip(v * (1.0 + self.rng.normal(0, noise, v.shape)), 1e-4, None).astype(np.float32)
        if coverage < 1.0:
            missing = self.rng.random(self.grid.shape) > coverage
            for v in bands.values():
                v[missing] = np.nan
        return bands

    # ------------------------------------------------------------- scoring
    def twin_error(self, twin: DigitalTwin) -> Dict[str, Dict[str, float]]:
        """
        Per layer: RMSE of the twin mean vs truth, and the fraction of cells
        where the truth lies inside the twin's 95% interval (should be ~0.95
        if the twin's uncertainty is well calibrated).
        """
        out = {}
        for name in LAYERS:
            err = twin.state.mean[name] - self.truth.state.mean[name]
            std = np.sqrt(twin.state.var[name])
            out[name] = {
                "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
                "bias": round(float(err.mean()), 4),
                "coverage_95": round(float(np.mean(np.abs(err) <= 1.96 * std)), 3),
            }
        return out


def serpentine_waypoints(grid: FieldGrid, lane_spacing_m: float = 10.0,
                         step_m: float = 5.0) -> List[Tuple[float, float]]:
    """Lawn-mower survey path over the field as (lat, lon) waypoints."""
    width = grid.cols * grid.cell_size_m
    height = grid.rows * grid.cell_size_m
    pts: List[Tuple[float, float]] = []
    lanes = np.arange(lane_spacing_m / 2.0, height, lane_spacing_m)
    xs = np.arange(step_m / 2.0, width, step_m)
    for i, south in enumerate(lanes):
        for east in (xs if i % 2 == 0 else xs[::-1]):
            row = south / grid.cell_size_m - 0.5
            col = east / grid.cell_size_m - 0.5
            lat_r, lon_r = grid.cell_center_latlon(0, 0)
            lat_d, lon_d = grid.cell_center_latlon(1, 1)
            pts.append((lat_r + row * (lat_d - lat_r), lon_r + col * (lon_d - lon_r)))
    return pts
