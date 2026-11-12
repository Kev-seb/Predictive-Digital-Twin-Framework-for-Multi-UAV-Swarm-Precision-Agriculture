"""
field_state.py
--------------
Georeferenced, uncertainty-aware state of a single paddy field.

The field is a regular grid of square cells. Every physical layer stores a
mean and a variance per cell, so the twin always knows how confident it is
about each part of the field. That variance is what the assimilation step
uses to weigh model predictions against new observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_000.0

# Physical layers tracked by the twin. Units are normalised to [0, 1] except
# NDVI, which keeps its natural [-1, 1] range.
LAYERS: Tuple[str, ...] = ("ndvi", "stress", "moisture", "nitrogen", "fungus")

# Valid value range per layer (used to clip after every update).
LAYER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ndvi": (-1.0, 1.0),
    "stress": (0.0, 1.0),
    "moisture": (0.0, 1.0),
    "nitrogen": (0.0, 1.0),
    "fungus": (0.0, 1.0),
}

# Prior variance of a freshly initialised layer (no observation yet).
# Tuned together with PROCESS_NOISE_PER_DAY (see assimilation.py).
DEFAULT_PRIOR_VAR: Dict[str, float] = {
    "ndvi": 0.00104,
    "stress": 0.0012,
    "moisture": 0.003,
    "nitrogen": 0.00063,
    "fungus": 0.00198,
}

# Healthy mid-vegetative paddy defaults used when no survey has been ingested.
PADDY_DEFAULTS: Dict[str, float] = {
    "ndvi": 0.72,
    "stress": 0.05,
    "moisture": 0.9,    # ~45 mm standing water (see paddy_model.moisture_from_water)
    "nitrogen": 0.55,
    "fungus": 0.01,     # latent inoculum
}


@dataclass(frozen=True)
class FieldGrid:
    """
    Regular grid anchored at the field's north-west corner.

    Row 0 is the northern edge and column 0 is the western edge, which
    matches the row/col convention used by the UE5 HISM grid.
    """

    origin_lat: float
    origin_lon: float
    rows: int = 40
    cols: int = 40
    cell_size_m: float = 2.5

    @classmethod
    def centered_on(cls, lat: float, lon: float, rows: int = 40, cols: int = 40,
                    cell_size_m: float = 2.5) -> "FieldGrid":
        half_h = rows * cell_size_m / 2.0
        half_w = cols * cell_size_m / 2.0
        dlat = math.degrees(half_h / EARTH_RADIUS_M)
        dlon = math.degrees(half_w / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
        return cls(origin_lat=lat + dlat, origin_lon=lon - dlon,
                   rows=rows, cols=cols, cell_size_m=cell_size_m)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def area_m2(self) -> float:
        return self.rows * self.cols * self.cell_size_m ** 2

    def _m_per_deg_lon(self) -> float:
        return math.radians(1.0) * EARTH_RADIUS_M * math.cos(math.radians(self.origin_lat))

    @staticmethod
    def _m_per_deg_lat() -> float:
        return math.radians(1.0) * EARTH_RADIUS_M

    def latlon_to_local_m(self, lat: float, lon: float) -> Tuple[float, float]:
        """Return (east_m, south_m) offset from the grid origin."""
        east = (lon - self.origin_lon) * self._m_per_deg_lon()
        south = (self.origin_lat - lat) * self._m_per_deg_lat()
        return east, south

    def latlon_to_cell(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """Return (row, col) of the cell containing the point, or None if outside."""
        east, south = self.latlon_to_local_m(lat, lon)
        row = int(math.floor(south / self.cell_size_m))
        col = int(math.floor(east / self.cell_size_m))
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cell_center_latlon(self, row: int, col: int) -> Tuple[float, float]:
        south = (row + 0.5) * self.cell_size_m
        east = (col + 0.5) * self.cell_size_m
        lat = self.origin_lat - south / self._m_per_deg_lat()
        lon = self.origin_lon + east / self._m_per_deg_lon()
        return lat, lon

    def center_latlon(self) -> Tuple[float, float]:
        lat = self.origin_lat - (self.rows * self.cell_size_m / 2.0) / self._m_per_deg_lat()
        lon = self.origin_lon + (self.cols * self.cell_size_m / 2.0) / self._m_per_deg_lon()
        return lat, lon

    def cell_distances_m(self, lat: float, lon: float) -> np.ndarray:
        """Distance (m) from a point to every cell centre, shape (rows, cols)."""
        east, south = self.latlon_to_local_m(lat, lon)
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        dy = (rr + 0.5) * self.cell_size_m - south
        dx = (cc + 0.5) * self.cell_size_m - east
        return np.sqrt(dx * dx + dy * dy)

    def block_zones(self, zone_rows: int = 4, zone_cols: int = 4) -> Tuple[np.ndarray, Dict[int, str]]:
        """Split the grid into management zones (used by the agronomic simulator)."""
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        zr = np.minimum(rr * zone_rows // self.rows, zone_rows - 1)
        zc = np.minimum(cc * zone_cols // self.cols, zone_cols - 1)
        labels = (zr * zone_cols + zc).astype(np.int32)
        names = {int(z): f"Zone {chr(65 + z // zone_cols)}{z % zone_cols + 1}"
                 for z in range(zone_rows * zone_cols)}
        return labels, names

    def to_dict(self) -> Dict[str, float]:
        return {
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "rows": self.rows,
            "cols": self.cols,
            "cell_size_m": self.cell_size_m,
        }


@dataclass
class FieldState:
    """Mean + variance per layer on a :class:`FieldGrid`, at one point in twin time."""

    grid: FieldGrid
    mean: Dict[str, np.ndarray]
    var: Dict[str, np.ndarray]
    sim_day: int = 0
    growth_stage: str = "Vegetative"
    last_weather: Dict[str, float] = field(default_factory=dict)
    # Process-model state that is not assimilated directly (water depth,
    # fungicide protection, ...) and field-wide scalars (thermal time, ...).
    aux: Dict[str, np.ndarray] = field(default_factory=dict)
    scalars: Dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- build
    @classmethod
    def initial(cls, grid: FieldGrid, values: Optional[Dict[str, float]] = None,
                growth_stage: str = "Vegetative", seed: Optional[int] = 0) -> "FieldState":
        """
        Build a prior state. With ``seed`` set, small spatial noise is added so
        the field is not perfectly uniform (a uniform field never develops
        interesting spatial dynamics in the simulator).
        """
        values = {**PADDY_DEFAULTS, **(values or {})}
        rng = np.random.default_rng(seed) if seed is not None else None
        mean, var = {}, {}
        for name in LAYERS:
            arr = np.full(grid.shape, values[name], dtype=np.float32)
            if rng is not None:
                arr += rng.normal(0.0, 0.02, grid.shape).astype(np.float32)
            lo, hi = LAYER_BOUNDS[name]
            mean[name] = np.clip(arr, lo, hi)
            var[name] = np.full(grid.shape, DEFAULT_PRIOR_VAR[name], dtype=np.float32)
        return cls(grid=grid, mean=mean, var=var, growth_stage=growth_stage)

    def copy(self) -> "FieldState":
        return FieldState(
            grid=self.grid,
            mean={k: v.copy() for k, v in self.mean.items()},
            var={k: v.copy() for k, v in self.var.items()},
            sim_day=self.sim_day,
            growth_stage=self.growth_stage,
            last_weather=dict(self.last_weather),
            aux={k: v.copy() for k, v in self.aux.items()},
            scalars=dict(self.scalars),
        )

    def clip(self) -> None:
        for name, arr in self.mean.items():
            lo, hi = LAYER_BOUNDS[name]
            np.clip(arr, lo, hi, out=arr)
        for arr in self.var.values():
            np.clip(arr, 1e-6, 1.0, out=arr)

    # ------------------------------------------------------------- queries
    def resample(self, layer: str, rows: int, cols: int, which: str = "mean") -> np.ndarray:
        """
        Area-average a layer onto a coarser (rows x cols) grid, e.g. the
        20x20 HISM grid used by UE5. Works for any target size; cells are
        assigned to the target block their centre falls into.
        """
        src = (self.mean if which == "mean" else self.var)[layer]
        r_idx = (np.arange(self.grid.rows) * rows // self.grid.rows)
        c_idx = (np.arange(self.grid.cols) * cols // self.grid.cols)
        out = np.zeros((rows, cols), dtype=np.float64)
        cnt = np.zeros((rows, cols), dtype=np.float64)
        np.add.at(out, (r_idx[:, None], c_idx[None, :]), src)
        np.add.at(cnt, (r_idx[:, None], c_idx[None, :]), 1.0)
        # Target cells with no source cell (target finer than source) fall back
        # to nearest-neighbour sampling.
        empty = cnt == 0
        if np.any(empty):
            rr, cc = np.nonzero(empty)
            out[rr, cc] = src[rr * self.grid.rows // rows, cc * self.grid.cols // cols]
            cnt[rr, cc] = 1.0
        return (out / cnt).astype(np.float32)

    def resample_aux(self, name: str, rows: int, cols: int) -> np.ndarray:
        """Area-average an auxiliary process layer (e.g. water_mm) like resample()."""
        tmp = FieldState(grid=self.grid, mean={"_": self.aux[name]}, var={"_": self.aux[name]})
        return tmp.resample("_", rows, cols)

    def summary(self) -> Dict[str, object]:
        layers = {}
        for name in LAYERS:
            m, v = self.mean[name], self.var[name]
            layers[name] = {
                "mean": round(float(m.mean()), 4),
                "min": round(float(m.min()), 4),
                "max": round(float(m.max()), 4),
                "mean_std": round(float(np.sqrt(v).mean()), 4),
            }
        stress = self.mean["stress"]
        return {
            "sim_day": self.sim_day,
            "growth_stage": self.growth_stage,
            "grid": self.grid.to_dict(),
            "area_ha": round(self.grid.area_m2 / 10_000.0, 3),
            "layers": layers,
            "pct_stressed": round(float((stress > 0.5).mean() * 100.0), 2),
            "pct_healthy": round(float((stress < 0.2).mean() * 100.0), 2),
            "last_weather": self.last_weather,
            "process": {
                **{k: round(float(v), 2) for k, v in self.scalars.items()},
                **({"water_mm_mean": round(float(self.aux["water_mm"].mean()), 1),
                    "pct_no_standing_water": round(float((self.aux["water_mm"] <= 0).mean() * 100), 1)}
                   if "water_mm" in self.aux else {}),
            },
        }

    # --------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"mean__{k}": v for k, v in self.mean.items()}
        arrays.update({f"var__{k}": v for k, v in self.var.items()})
        arrays.update({f"aux__{k}": v for k, v in self.aux.items()})
        meta = np.array([repr({
            "grid": self.grid.to_dict(),
            "sim_day": self.sim_day,
            "growth_stage": self.growth_stage,
            "last_weather": self.last_weather,
            "scalars": {k: float(v) for k, v in self.scalars.items()},
        })])
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, __meta__=meta, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "FieldState":
        import ast

        with np.load(Path(path), allow_pickle=False) as data:
            meta = ast.literal_eval(str(data["__meta__"][0]))
            mean = {k.split("__", 1)[1]: data[k].astype(np.float32)
                    for k in data.files if k.startswith("mean__")}
            var = {k.split("__", 1)[1]: data[k].astype(np.float32)
                   for k in data.files if k.startswith("var__")}
            aux = {k.split("__", 1)[1]: data[k].astype(np.float32)
                   for k in data.files if k.startswith("aux__")}
        return cls(
            grid=FieldGrid(**meta["grid"]),
            mean=mean,
            var=var,
            sim_day=int(meta["sim_day"]),
            growth_stage=str(meta["growth_stage"]),
            last_weather=dict(meta.get("last_weather", {})),
            aux=aux,
            scalars=dict(meta.get("scalars", {})),
        )


def iter_cells(grid: FieldGrid) -> Iterable[Tuple[int, int]]:
    for r in range(grid.rows):
        for c in range(grid.cols):
            yield r, c
