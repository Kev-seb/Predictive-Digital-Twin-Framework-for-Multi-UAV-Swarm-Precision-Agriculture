"""
ensemble.py
-----------
Cross-layer correction for the twin: a small ensemble of the paddy model
supplies the correlations between layers that the per-cell Kalman filter
(assimilation.py) cannot see on its own.

The twin's filter treats each layer separately, so a drone NDVI reading only
corrects NDVI. In the crop model the layers are linked: a leaky patch loses
its standing water, the root zone dries, and the canopy greens less. So low
NDVI is also evidence of low moisture and of a high percolation rate.

This is a hybrid ensemble Kalman filter:

  * ``N`` members run the same paddy model as the twin, each with its own
    rain (forecast error) and its own soil percolation rate (the twin
    doesn't know the soil exactly), plus the filter's process noise.
  * Each day the members are re-centred on the twin's mean and scaled to
    its variance. The twin's tuned mean and variance stay authoritative; the
    ensemble contributes only the *shape* of the uncertainty: correlations
    between layers in each cell, smoothed over a 3 x 3 neighbourhood to
    tame sampling noise.
  * When layer ``a`` (one of ``CROSS_FROM``) is observed with innovation ``d = z - x_a``, every
    correlated layer ``b`` gets the multivariate Kalman update

        C_ab = rho_ab sqrt(P_a P_b),   K_b = C_ab / (P_a + R)
        x_b += K_b d,                  P_b -= K_b C_ab

    and the estimated parameters (``PARAMS``: soil percolation, crop vigor)
    get the same update, so the twin learns where the soil is leaky and
    where the crop grows better or worse than modelled, and forecasts that.

Correlations below ``RHO_FLOOR`` are treated as sampling noise and ignored.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from src.twin_core.assimilation import PROCESS_NOISE_PER_DAY
from src.twin_core.field_state import LAYER_BOUNDS

CROSS_LAYERS = ("ndvi", "moisture", "nitrogen", "fungus")
# Observed layers that correct the others. NDVI is left out: in the OSSE
# (scripts/validate_cross_layer.py) the hidden leaky patch dries without its
# NDVI dropping for weeks, so NDVI-driven corrections only moved moisture
# the wrong way (+8% moisture error, drone-only). Soil readings
# (moisture, nitrogen) and disease readings carry the useful links.
CROSS_FROM = ("moisture", "nitrogen", "fungus")
RHO_FLOOR = 0.45         # |rho| below this is sampling noise for 24 members x 9 cells
RHO_CAP = 0.95
RAIN_LOG_SD = 0.35       # forecast rain error, as in the virtual field's weather

# Estimated per-cell parameters (log scale): aux key in the twin's state,
# prior spread, daily random walk (so they keep adapting), and which observed
# layers may correct them (None = the twin's cross_from).
#   percolation: how leaky the soil is (x0.2 .. x6 at 95%).
#   vigor:       how well the crop grows relative to the model (variety,
#                management, anything the model lacks), +-25% at 95%. Only
#                NDVI informs it, and it only scales the canopy target, so
#                a field that keeps greening more (or less) than modelled is
#                forecast that way from then on.
PARAMS = {
    "percolation": {"aux": "percolation_mm_day", "prior_sd": 0.9, "walk_sd": 0.03, "from": None},
    "vigor": {"aux": "vigor_factor", "prior_sd": 0.12, "walk_sd": 0.01, "from": ("ndvi",)},
}
PERC_LOG_SD = PARAMS["percolation"]["prior_sd"]


def _smooth_noise(rng, shape, coarse=8):
    """Unit-variance noise, correlated over ~1/coarse of the field."""
    c = rng.normal(0.0, 1.0, (coarse, coarse))
    f = zoom(c, (shape[0] / coarse, shape[1] / coarse), order=1)[:shape[0], :shape[1]]
    return (f - f.mean()) / max(f.std(), 1e-6)


class CrossLayerEnsemble:
    def __init__(self, twin, n: int = 24, seed: int = 0, cross_from=None):
        from src.twin_core.paddy_model import PaddyFieldSimulator

        self.n = n
        self.cross_from = tuple(cross_from or CROSS_FROM)
        self.rng = np.random.default_rng(seed)
        st = twin.state
        shape = st.grid.shape
        self.shape = shape
        self.params: Dict[str, np.ndarray] = {}      # name -> (n, rows, cols) log values
        for name, spec in PARAMS.items():
            base = st.aux.get(spec["aux"])
            base = np.ones(shape) if base is None else base.astype(np.float64)
            self.params[name] = np.log(np.maximum(base, 1e-3))[None] + spec["prior_sd"] * self._noise()
        self.members: List = []
        for i in range(n):
            ms = st.copy()
            for k in CROSS_LAYERS:
                noise = self.rng.normal(0.0, 1.0, shape) * np.sqrt(st.var[k])
                ms.mean[k] = np.clip(ms.mean[k] + noise, *LAYER_BOUNDS[k]).astype(np.float32)
            self.members.append((ms, PaddyFieldSimulator(ms, twin.zone_labels, twin.zone_names, twin.params)))
        self._corr: Optional[Dict] = None

    def _noise(self):
        """(n, rows, cols) smooth unit noise with zero mean and unit spread in every cell."""
        noise = np.stack([_smooth_noise(self.rng, self.shape) for _ in range(self.n)])
        return (noise - noise.mean(0)[None]) / np.maximum(noise.std(0)[None], 1e-6)

    def estimate(self, name: str) -> np.ndarray:
        """The twin's estimate of a parameter: geometric mean of the members."""
        return np.exp(self.params[name].mean(0)).astype(np.float32)

    def write_params(self, state) -> None:
        for name, spec in PARAMS.items():
            state.aux[spec["aux"]] = self.estimate(name)

    # ------------------------------------------------------------ forecast
    def step_day(self, weather: Dict, interventions: List[Dict]) -> None:
        rain = float(weather.get("precipitation", 0.0))
        for name, spec in PARAMS.items():
            if spec["walk_sd"]:
                self.params[name] = self.params[name] + spec["walk_sd"] * self._noise()
        for i, (ms, model) in enumerate(self.members):
            for name, spec in PARAMS.items():
                ms.aux[spec["aux"]] = np.exp(self.params[name][i]).astype(np.float32)
            w = dict(weather)
            w["precipitation"] = rain * float(np.exp(self.rng.normal(0.0, RAIN_LOG_SD)))
            model.sync_from_moisture()
            model.simulate_day(weather=w, interventions=interventions)
            for k in CROSS_LAYERS:
                q = np.sqrt(PROCESS_NOISE_PER_DAY[k])
                ms.mean[k] = np.clip(ms.mean[k] + self.rng.normal(0.0, q, ms.mean[k].shape),
                                     *LAYER_BOUNDS[k]).astype(np.float32)
        self._corr = None

    def recenter(self, state) -> None:
        """Move the members onto the twin's mean and variance, keeping their correlations."""
        for k in CROSS_LAYERS:
            stack = np.stack([ms.mean[k] for ms, _ in self.members]).astype(np.float64)
            mu, sd = stack.mean(0), stack.std(0)
            scale = np.minimum(np.sqrt(state.var[k]) / np.maximum(sd, 1e-6), 10.0)
            stack = state.mean[k][None] + (stack - mu[None]) * scale[None]
            lo, hi = LAYER_BOUNDS[k]
            for (ms, _), row in zip(self.members, stack):
                ms.mean[k] = np.clip(row, lo, hi).astype(np.float32)
        self._corr = None

    # -------------------------------------------------------- correlations
    def _stats(self):
        if self._corr is None:
            anoms = {}
            for k in CROSS_LAYERS:
                stack = np.stack([ms.mean[k] for ms, _ in self.members]).astype(np.float64)
                anoms[k] = stack - stack.mean(0)[None]
            for name, stack in self.params.items():
                anoms[name] = stack - stack.mean(0)[None]
            var = {k: uniform_filter((a ** 2).mean(0), size=3, mode="nearest") for k, a in anoms.items()}
            self._corr = {"anoms": anoms, "var": var, "pairs": {}}
        return self._corr

    def corr(self, a: str, b: str) -> np.ndarray:
        """Smoothed per-cell correlation of layers a and b, noise floor removed."""
        s = self._stats()
        key = tuple(sorted((a, b)))
        if key not in s["pairs"]:
            cov = uniform_filter((s["anoms"][a] * s["anoms"][b]).mean(0), size=3, mode="nearest")
            rho = cov / np.sqrt(np.maximum(s["var"][a] * s["var"][b], 1e-12))
            rho = np.clip(rho, -RHO_CAP, RHO_CAP)
            s["pairs"][key] = np.where(np.abs(rho) < RHO_FLOOR, 0.0, rho)
        return s["pairs"][key]

    def percolation(self) -> np.ndarray:
        """The twin's estimate of soil percolation (mm/day)."""
        return self.estimate("percolation")

    # --------------------------------------------------------------- update
    def cross_update(self, state, layer: str, valid: np.ndarray, innovation: np.ndarray,
                     p_obs: np.ndarray, r: np.ndarray) -> Dict[str, float]:
        """
        Apply an observation of ``layer`` (at the ``valid`` cells, with the
        filter's innovation, prior variance and observation variance there)
        to the other layers and to the percolation parameter. Returns the
        mean absolute change per corrected layer.
        """
        if layer not in CROSS_LAYERS or not np.any(valid):
            return {}
        changes = {}
        denom = p_obs + r
        if layer in self.cross_from:
            for b in CROSS_LAYERS:
                if b == layer:
                    continue
                rho = self.corr(layer, b)[valid]
                if not np.any(rho):
                    continue
                pb = state.var[b][valid].astype(np.float64)
                c = rho * np.sqrt(p_obs * pb)
                k = c / denom
                delta = k * innovation
                state.mean[b][valid] = np.clip(state.mean[b][valid] + delta, *LAYER_BOUNDS[b])
                state.var[b][valid] = np.maximum(pb - k * c, 1e-6)
                changes[b] = float(np.abs(delta).mean())
        for name, spec in PARAMS.items():
            if layer not in (spec["from"] or self.cross_from):
                continue
            rho = self.corr(layer, name)[valid]
            if not np.any(rho):
                continue
            sd_p = np.sqrt(self._stats()["var"][name][valid])
            k = rho * sd_p * np.sqrt(p_obs) / denom
            shift = np.zeros(self.shape)
            shift[valid] = k * innovation
            shrink = np.ones(self.shape)
            shrink[valid] = np.sqrt(np.clip(1.0 - rho ** 2 * p_obs / denom, 0.05, 1.0))
            stack = self.params[name]
            mu = stack.mean(0)
            self.params[name] = mu[None] + shift[None] + (stack - mu[None]) * shrink[None]
            self._corr = None
            changes[name] = float(np.abs(shift[valid]).mean())
        if any(n in changes for n in PARAMS):
            self.write_params(state)
        return changes
