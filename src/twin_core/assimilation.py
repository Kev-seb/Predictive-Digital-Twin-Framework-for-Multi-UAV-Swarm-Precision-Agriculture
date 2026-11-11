"""
assimilation.py
---------------
Per-cell Kalman filtering that keeps the twin consistent with observations.

Each grid cell and layer is treated as an independent scalar state:

    predict:   P <- P + Q                       (model uncertainty grows daily)
    update:    K  = P / (P + R)
               x <- x + K (z - x)
               P <- (1 - K) P

``Q`` is the per-day process noise of the agronomic simulator and ``R`` the
observation noise of a sensor. A point observation (phone RGB scout, drone
camera footprint) also updates nearby cells, with ``R`` inflated by distance,
so a single reading informs its neighbourhood without overriding it.

This is the Phase-2 baseline; it can be replaced by an ensemble Kalman filter
over ``AgronomicDigitalTwinSimulator`` without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.twin_core.field_state import LAYER_BOUNDS, LAYERS, FieldState

# Per-day growth of variance in each layer when the model runs unobserved.
PROCESS_NOISE_PER_DAY: Dict[str, float] = {
    "ndvi": 0.0015,
    "stress": 0.0015,
    "moisture": 0.004,
    "nitrogen": 0.002,
    "fungus": 0.003,
}

# Typical observation variance per sensor source and layer.
SENSOR_NOISE: Dict[str, Dict[str, float]] = {
    "multispectral": {"ndvi": 0.0009, "stress": 0.0025, "moisture": 0.01, "nitrogen": 0.01},
    "phone_rgb": {"stress": 0.01, "ndvi": 0.01, "fungus": 0.02},
    "drone_rgb": {"stress": 0.006, "ndvi": 0.006, "fungus": 0.015},
    "soil_probe": {"moisture": 0.0025, "nitrogen": 0.004},
    "ue5": {"ndvi": 0.004, "stress": 0.004, "moisture": 0.01},
}
DEFAULT_OBS_VAR = 0.01


@dataclass
class AssimilationResult:
    layer: str
    cells_updated: int
    mean_innovation: float      # average (z - x) before the update
    mean_gain: float            # average Kalman gain applied
    variance_before: float
    variance_after: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "layer": self.layer,
            "cells_updated": self.cells_updated,
            "mean_innovation": round(self.mean_innovation, 5),
            "mean_gain": round(self.mean_gain, 5),
            "variance_before": round(self.variance_before, 6),
            "variance_after": round(self.variance_after, 6),
        }


def observation_variance(source: str, layer: str, override: Optional[float] = None) -> float:
    if override is not None:
        return float(override)
    return SENSOR_NOISE.get(source, {}).get(layer, DEFAULT_OBS_VAR)


def inflate_process_noise(state: FieldState, days: float = 1.0) -> None:
    """Prediction step for the covariance: uncertainty grows while unobserved."""
    for name in LAYERS:
        state.var[name] += np.float32(PROCESS_NOISE_PER_DAY[name] * days)
    state.clip()


def kalman_update(
    state: FieldState,
    layer: str,
    z: np.ndarray,
    r: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> AssimilationResult:
    """
    Scalar Kalman update of ``layer`` with observation ``z`` and variance ``r``
    (both broadcastable to the grid). Cells where ``mask`` is False, ``z`` is NaN
    or ``r`` is not finite are left untouched.
    """
    if layer not in state.mean:
        raise KeyError(f"Unknown layer '{layer}'. Expected one of {LAYERS}.")
    x = state.mean[layer]
    p = state.var[layer]
    z = np.broadcast_to(np.asarray(z, dtype=np.float32), x.shape)
    r = np.broadcast_to(np.asarray(r, dtype=np.float32), x.shape)

    valid = np.isfinite(z) & np.isfinite(r) & (r > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    n = int(valid.sum())
    if n == 0:
        return AssimilationResult(layer, 0, 0.0, 0.0, float(p.mean()), float(p.mean()))

    p_before = float(p[valid].mean())
    innovation = z[valid] - x[valid]
    gain = p[valid] / (p[valid] + r[valid])

    x[valid] = x[valid] + gain * innovation
    p[valid] = (1.0 - gain) * p[valid]
    lo, hi = LAYER_BOUNDS[layer]
    np.clip(x, lo, hi, out=x)
    np.clip(p, 1e-6, 1.0, out=p)

    return AssimilationResult(
        layer=layer,
        cells_updated=n,
        mean_innovation=float(innovation.mean()),
        mean_gain=float(gain.mean()),
        variance_before=p_before,
        variance_after=float(p[valid].mean()),
    )


def assimilate_grid(
    state: FieldState,
    layer: str,
    observed: np.ndarray,
    source: str = "multispectral",
    obs_var: Optional[float] = None,
) -> AssimilationResult:
    """
    Assimilate a full raster observation. If its shape differs from the twin
    grid it is area-resampled (nearest for upsampling) onto the grid first.
    NaN pixels are treated as missing.
    """
    observed = np.asarray(observed, dtype=np.float32)
    if observed.shape != state.grid.shape:
        observed = _resample_to(observed, state.grid.shape)
    r = observation_variance(source, layer, obs_var)
    return kalman_update(state, layer, observed, np.float32(r))


def assimilate_point(
    state: FieldState,
    layer: str,
    value: float,
    lat: float,
    lon: float,
    source: str = "phone_rgb",
    obs_var: Optional[float] = None,
    footprint_m: float = 3.0,
    influence_radius_m: float = 8.0,
) -> AssimilationResult:
    """
    Assimilate one geotagged reading. Cells within ``footprint_m`` get the
    nominal observation variance; beyond that it grows quadratically with
    distance and cells past ``influence_radius_m`` are not touched.
    """
    r0 = observation_variance(source, layer, obs_var)
    dist = state.grid.cell_distances_m(lat, lon)
    excess = np.clip(dist - footprint_m, 0.0, None)
    scale = max(influence_radius_m - footprint_m, 1e-6)
    r = r0 * (1.0 + 9.0 * (excess / scale) ** 2)
    mask = dist <= influence_radius_m
    return kalman_update(state, layer, np.float32(value), r.astype(np.float32), mask=mask)


def _resample_to(arr: np.ndarray, shape) -> np.ndarray:
    rows, cols = shape
    h, w = arr.shape
    r_idx = np.arange(h) * rows // h
    c_idx = np.arange(w) * cols // w
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    finite = np.isfinite(arr)
    np.add.at(total, (r_idx[:, None], c_idx[None, :]), np.where(finite, arr, 0.0))
    np.add.at(count, (r_idx[:, None], c_idx[None, :]), finite.astype(np.float64))
    out = np.full(shape, np.nan, dtype=np.float32)
    has = count > 0
    out[has] = (total[has] / count[has]).astype(np.float32)
    # Upsampling: fill target cells that received no source pixel.
    empty = ~has
    if np.any(empty):
        rr, cc = np.nonzero(empty)
        out[rr, cc] = arr[rr * h // rows, cc * w // cols]
    return out
