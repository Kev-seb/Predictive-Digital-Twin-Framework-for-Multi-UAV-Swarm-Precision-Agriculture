"""
flight_replay.py
----------------
Replay real multispectral drone flights through the twin.

For each flight (oldest first) the twin is run forward to the flight date
on the weather of those days, then every image's NDVI is assimilated at the
spot on the ground the image saw. Before assimilating, the twin's NDVI at
those spots is its *prediction* of the flight, so each flight scores the
twin: how far ahead it could see, and how much better it does than the
model alone (a twin that never assimilates).

Images are found as ``<prefix>_MS_R.TIF`` + ``<prefix>_MS_NIR.TIF`` (DJI
Mavic 3M naming). Position comes from the image's DJI XMP
(``drone-dji:GpsLatitude``, ``RelativeAltitude``, ``GimbalYawDegree``), or
else its EXIF GPS. With ``tiles > 1`` each image is split into tiles x tiles
patches, each placed at its own ground position (nadir camera, rotated by
the gimbal yaw), so one image informs several twin cells.

NDVI from uncalibrated digital numbers is approximate; the calibrated
paddy parameters (scripts/calibrate_paddy.py) are fitted to the same NDVI,
so the model and the observations share one scale.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.twin_core.field_state import EARTH_RADIUS_M, FieldGrid

FLIGHT_RE = re.compile(r"^(F\d+)_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\(1\))?_([A-Za-z]+)_[Ss]tage$")
_XMP = {k: re.compile(rf'drone-dji:{k}\s*=\s*"([-+0-9.eE]+)"'.encode()) for k in
        ("GpsLatitude", "GpsLongitude", "GpsLongtitude", "RelativeAltitude", "GimbalYawDegree", "FlightYawDegree")}

# DJI Mavic 3M multispectral camera field of view (degrees).
HFOV_DEG = 61.2
VFOV_DEG = 48.1


@dataclass
class ImagePose:
    lat: float
    lon: float
    alt_m: float = 30.0
    yaw_deg: float = 0.0


@dataclass
class FlightImages:
    flight_id: str
    date: dt.date
    stage: str
    path: Path
    prefixes: List[Path] = field(default_factory=list)   # directory/prefix of each R+NIR pair


# ----------------------------------------------------------------- reading
def read_band(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(str(path)).astype(np.float32)


def _xmp_bytes(path: Path) -> bytes:
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            tag = tif.pages[0].tags.get(700)
            if tag is not None:
                v = tag.value
                return v if isinstance(v, bytes) else str(v).encode()
    except Exception:
        pass
    with open(path, "rb") as f:
        return f.read(512 * 1024)


def _exif_gps(path: Path) -> Optional[Tuple[float, float, Optional[float]]]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v):
        d, m, s = (float(x) for x in v)
        return d + m / 60.0 + s / 3600.0
    lat = deg(gps[2]) * (-1 if gps.get(1, "N") in ("S", b"S") else 1)
    lon = deg(gps[4]) * (-1 if gps.get(3, "E") in ("W", b"W") else 1)
    alt = float(gps[6]) if 6 in gps else None
    return lat, lon, alt


def read_pose(path: Path, default_alt_m: float = 30.0) -> Optional[ImagePose]:
    """Where an image was taken: DJI XMP first, then EXIF GPS. None if unknown."""
    xmp = _xmp_bytes(path)
    vals = {}
    for k, rx in _XMP.items():
        m = rx.search(xmp)
        if m:
            vals[k] = float(m.group(1))
    lon = vals.get("GpsLongitude", vals.get("GpsLongtitude"))      # DJI's own typo exists in the wild
    if "GpsLatitude" in vals and lon is not None:
        yaw = vals.get("GimbalYawDegree", vals.get("FlightYawDegree", 0.0))
        return ImagePose(vals["GpsLatitude"], lon, abs(vals.get("RelativeAltitude", default_alt_m)) or default_alt_m,
                         yaw)
    gps = _exif_gps(path)
    if gps is None:
        return None
    # EXIF altitude is above sea level, not above the field: use the default.
    return ImagePose(gps[0], gps[1], default_alt_m, 0.0)


def find_flights(raw: Path, field_code: Optional[str] = None) -> List[FlightImages]:
    """Flight directories under ``raw`` (named like F1_03_2025_08_16_1609_Vegetative_Stage), oldest first."""
    out = []
    for d in sorted(p for p in Path(raw).rglob("*") if p.is_dir() and FLIGHT_RE.match(p.name)):
        if "(1)" in d.name:
            continue    # duplicate directory in the dataset
        code, seq, yr, mo, da, _, stage = FLIGHT_RE.match(d.name).groups()
        if field_code and code != field_code:
            continue
        prefixes = sorted({f.with_name(f.name[:-len("_MS_R.TIF")]) for f in d.rglob("*_MS_R.TIF")})
        out.append(FlightImages(f"{code}_{seq}", dt.date(int(yr), int(mo), int(da)), stage.capitalize(), d,
                                [p for p in prefixes if p.with_name(p.name + "_MS_NIR.TIF").exists()]))
    return sorted(out, key=lambda f: f.date)


# --------------------------------------------------------------- geometry
def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def image_tiles(red: np.ndarray, nir: np.ndarray, pose: ImagePose, tiles: int = 3,
                hfov_deg: float = HFOV_DEG, vfov_deg: float = VFOV_DEG,
                min_valid: int = 100) -> List[Tuple[float, float, float, float]]:
    """
    Median NDVI of each of ``tiles`` x ``tiles`` patches with its ground
    position: [(lat, lon, ndvi, patch_size_m)]. Image +x is to the camera's
    right, image row 0 is forward (top of the frame), the camera looks
    straight down and the frame's top points along ``yaw_deg`` from north.
    """
    h, w = red.shape
    ground_w = 2.0 * pose.alt_m * math.tan(math.radians(hfov_deg) / 2.0)
    ground_h = 2.0 * pose.alt_m * math.tan(math.radians(vfov_deg) / 2.0)
    yaw = math.radians(pose.yaw_deg)
    out = []
    for i in range(tiles):
        for j in range(tiles):
            r0, r1 = i * h // tiles, (i + 1) * h // tiles
            c0, c1 = j * w // tiles, (j + 1) * w // tiles
            rr, nn = red[r0:r1, c0:c1], nir[r0:r1, c0:c1]
            valid = (rr > 0) & (nn > 0)
            if valid.sum() < min_valid:
                continue
            ndvi = float(np.median((nn[valid] - rr[valid]) / (nn[valid] + rr[valid])))
            fwd = (0.5 - (i + 0.5) / tiles) * ground_h        # metres ahead of the drone
            right = ((j + 0.5) / tiles - 0.5) * ground_w      # metres to its right
            north = fwd * math.cos(yaw) - right * math.sin(yaw)
            east = fwd * math.sin(yaw) + right * math.cos(yaw)
            lat, lon = offset_latlon(pose.lat, pose.lon, north, east)
            out.append((lat, lon, ndvi, min(ground_w, ground_h) / tiles))
    return out


def grid_for_points(points: Iterable[Tuple[float, float]], cell_size_m: float = 2.5,
                    margin_m: float = 10.0, max_cells: int = 80) -> FieldGrid:
    """A FieldGrid covering the given (lat, lon) points plus a margin."""
    pts = np.asarray(list(points), dtype=np.float64)
    lat_c, lon_c = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    north = np.radians(pts[:, 0] - lat_c) * EARTH_RADIUS_M
    east = np.radians(pts[:, 1] - lon_c) * EARTH_RADIUS_M * math.cos(math.radians(lat_c))
    height = 2.0 * (np.abs(north).max() + margin_m)
    width = 2.0 * (np.abs(east).max() + margin_m)
    cell = max(cell_size_m, max(height, width) / max_cells)
    rows, cols = int(math.ceil(height / cell)), int(math.ceil(width / cell))
    return FieldGrid.centered_on(lat_c, lon_c, rows=rows, cols=cols, cell_size_m=cell)


def rasterize(grid: FieldGrid, obs) -> np.ndarray:
    """Average image-tile NDVI over the twin cells each tile covers (NaN = not seen)."""
    total = np.zeros(grid.shape)
    count = np.zeros(grid.shape)
    for lat, lon, ndvi, size_m in obs:
        east, south = grid.latlon_to_local_m(lat, lon)
        half = size_m / 2.0
        r0 = max(int(math.floor((south - half) / grid.cell_size_m)), 0)
        r1 = min(int(math.ceil((south + half) / grid.cell_size_m)), grid.rows)
        c0 = max(int(math.floor((east - half) / grid.cell_size_m)), 0)
        c1 = min(int(math.ceil((east + half) / grid.cell_size_m)), grid.cols)
        if r0 < r1 and c0 < c1:
            total[r0:r1, c0:c1] += ndvi
            count[r0:r1, c0:c1] += 1
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    seen = count > 0
    out[seen] = (total[seen] / count[seen]).astype(np.float32)
    return out


# ----------------------------------------------------------------- replay
WeatherFn = Callable[[dt.date, dt.date], List[Dict[str, float]]]


def archive_weather(lat: float, lon: float) -> WeatherFn:
    """Daily weather from the Open-Meteo archive for (start, end] (needs network)."""
    def fn(start: dt.date, end: dt.date) -> List[Dict[str, float]]:
        from src.twin_core.weather_feed import openmeteo_daily_to_twin
        from src.weather.openmeteo_client import fetch_historical_daily
        return openmeteo_daily_to_twin(fetch_historical_daily(lat, lon, start + dt.timedelta(days=1), end))
    return fn


def default_weather(start: dt.date, end: dt.date) -> List[Dict[str, float]]:
    from src.twin_core.weather_feed import synthetic_weather
    return synthetic_weather((end - start).days, seed=start.toordinal())


def flight_observations(flight: FlightImages, tiles: int = 3, max_images: Optional[int] = None,
                        default_alt_m: float = 30.0, seed: int = 0):
    """(lat, lon, ndvi, patch_m) for every usable image tile of a flight, and counts."""
    prefixes = list(flight.prefixes)
    if max_images and len(prefixes) > max_images:
        rng = np.random.default_rng(seed)
        prefixes = [prefixes[i] for i in sorted(rng.choice(len(prefixes), max_images, replace=False))]
    obs, no_pose = [], 0
    for pre in prefixes:
        red_path = pre.with_name(pre.name + "_MS_R.TIF")
        pose = read_pose(red_path, default_alt_m)
        if pose is None:
            no_pose += 1
            continue
        red, nir = read_band(red_path), read_band(pre.with_name(pre.name + "_MS_NIR.TIF"))
        if red.shape != nir.shape:
            continue
        obs.extend(image_tiles(red, nir, pose, tiles))
    return obs, {"images": len(prefixes), "no_position": no_pose, "tiles": len(obs)}


def replay(flights: Sequence[FlightImages], make_twin: Callable[[FieldGrid], "DigitalTwin"],
           weather: WeatherFn = default_weather, transplant: Optional[dt.date] = None,
           tiles: int = 3, max_images: Optional[int] = None, obs_var: float = 0.0025,
           on_flight: Optional[Callable] = None) -> Dict:
    """
    Run the twin through ``flights``. Two twins start alike: one assimilates
    each flight (the digital twin), the other never does (model only). Each
    flight is first scored as a prediction for both, then assimilated.
    ``on_flight(row, twin, ndvi_before, ndvi_measured)`` is called after
    each flight (``ndvi_measured`` is None when it had no usable images).
    Nursery flights are skipped (the crop isn't in the field yet); the
    season starts at ``transplant`` (default: the last nursery flight, else
    25 days before the first field flight).
    """
    nursery = [f for f in flights if f.stage == "Nursery"]
    field_flights = [f for f in flights if f.stage != "Nursery"]
    if not field_flights:
        raise ValueError("no field-stage flights to replay")
    if transplant is None:
        transplant = max(f.date for f in nursery) if nursery else \
            field_flights[0].date - dt.timedelta(days=25)

    per_flight, all_obs = [], {}
    for f in field_flights:
        all_obs[f.flight_id] = flight_observations(f, tiles, max_images)
    pts = [(o[0], o[1]) for obs, _ in all_obs.values() for o in obs]
    if not pts:
        raise ValueError("no image had a usable position (GPS) and R/NIR bands")
    grid = grid_for_points(pts)
    twin, model_only = make_twin(grid), make_twin(grid)
    from src.twin_core.paddy_model import ndvi_potential, stage_for_gdd
    for t in (twin, model_only):
        # Transplanting day: thermal time starts at zero, the canopy is young seedlings.
        t.state.scalars["gdd"] = 0.0
        t.state.growth_stage = stage_for_gdd(t.params, 0.0)
        t.state.mean["ndvi"][:] = ndvi_potential(t.params, 0.0)
        t._refresh_stress()
    day = transplant
    for f in field_flights:
        obs, counts = all_obs[f.flight_id]
        gap = (f.date - day).days
        if gap > 0:
            w = weather(day, f.date)
            twin.step(gap, w)
            model_only.step(gap, w)
            day = f.date
        row = {"flight_id": f.flight_id, "date": f.date.isoformat(), "stage_labelled": f.stage,
               "days_since_last": gap, **counts}
        before, raster = twin.state.mean["ndvi"].copy(), None
        if obs:
            z = np.array([o[2] for o in obs])
            cells = [grid.latlon_to_cell(o[0], o[1]) for o in obs]
            inside = [i for i, c in enumerate(cells) if c is not None]
            if inside:
                zi = z[inside]
                pred = np.array([twin.state.mean["ndvi"][cells[i]] for i in inside])
                pred0 = np.array([model_only.state.mean["ndvi"][cells[i]] for i in inside])
                row.update({
                    "observed_ndvi": round(float(np.median(zi)), 4),
                    "twin_predicted_ndvi": round(float(np.median(pred)), 4),
                    "model_only_ndvi": round(float(np.median(pred0)), 4),
                    "twin_rmse": round(float(np.sqrt(np.mean((pred - zi) ** 2))), 4),
                    "model_only_rmse": round(float(np.sqrt(np.mean((pred0 - zi) ** 2))), 4),
                    "observed_spread": round(float(np.std(zi)), 4),
                })
            raster = rasterize(grid, obs)
            row["cells_observed"] = int(np.isfinite(raster).sum())
            twin.observe_grid("ndvi", raster, source="multispectral", obs_var=obs_var)
        row["twin_growth_stage"] = twin.state.growth_stage
        per_flight.append(row)
        if on_flight:
            on_flight(row, twin, before, raster)
    return {"grid": grid.to_dict(), "transplant": transplant.isoformat(), "flights": per_flight,
            "twin": twin, "model_only": model_only}
