"""
ingest.py
---------
Adapters that turn sensor products into twin observations.

* RGB AI result + telemetry  -> geotagged point observation
  (crop_stress_score -> stress, disease box coverage -> fungus)
* Multispectral bands        -> raster observations
  (NDVI -> ndvi, NDWI -> moisture, CIre -> nitrogen)

The mappings mirror the ones AgronomicDigitalTwinSimulator uses to initialise
its layers from spectral indices, so observations and model live in the same
units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.twin_core.twin import DigitalTwin

# Downward camera horizontal field of view used to size the ground footprint.
CAMERA_HFOV_DEG = 70.0
MIN_FOOTPRINT_M = 1.5          # hand-held phone scouting at ~1.5 m
MIN_QUALITY = 0.3              # frames below this quality are not assimilated
DEFAULT_FRAME_WH = (640, 480)


@dataclass
class PointObservation:
    lat: float
    lon: float
    values: Dict[str, float]
    obs_var: Dict[str, float]
    footprint_m: float
    influence_radius_m: float
    source: str

    def apply(self, twin: DigitalTwin) -> List[Dict[str, Any]]:
        return twin.observe_point(self.lat, self.lon, self.values, self.source,
                                  self.obs_var, self.footprint_m, self.influence_radius_m)


def footprint_radius_m(altitude_m: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Ground radius seen by a nadir camera at ``altitude_m`` above canopy."""
    return max(MIN_FOOTPRINT_M, altitude_m * math.tan(math.radians(hfov_deg) / 2.0))


def disease_coverage(detections: List[Dict[str, Any]], frame_wh=DEFAULT_FRAME_WH) -> float:
    """Confidence-weighted fraction of the frame covered by disease boxes."""
    w, h = frame_wh
    area = float(w * h) or 1.0
    covered = 0.0
    for d in detections or []:
        bw = max(0.0, float(d["x2"]) - float(d["x1"]))
        bh = max(0.0, float(d["y2"]) - float(d["y1"]))
        covered += bw * bh * float(d.get("confidence", 1.0))
    return float(np.clip(covered / area, 0.0, 1.0))


def rgb_result_to_observation(
    result: Dict[str, Any],
    telemetry: Dict[str, Any],
    source: str = "drone_rgb",
    frame_wh=DEFAULT_FRAME_WH,
) -> Optional[PointObservation]:
    """
    Build a point observation from ``RGBInferenceResult.to_dict()`` output and
    fused telemetry (``lat``, ``lon``, ``alt`` = height above canopy in m).
    Returns None when the frame should not be assimilated (no GPS fix, poor
    image quality).
    """
    lat, lon = telemetry.get("lat"), telemetry.get("lon")
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        return None
    quality = float(result.get("quality_score", 1.0))
    if quality < MIN_QUALITY:
        return None
    confidence = float(np.clip(result.get("model_confidence", 1.0), 0.05, 1.0))

    from src.twin_core.assimilation import observation_variance

    # Less confident / lower quality inferences count for less.
    inflate = 1.0 / (confidence * max(quality, MIN_QUALITY))
    values = {"stress": float(np.clip(result.get("crop_stress_score", 0.0), 0.0, 1.0))}
    obs_var = {"stress": observation_variance(source, "stress") * inflate}

    detections = result.get("disease_detections")
    if detections is not None:
        values["fungus"] = disease_coverage(detections, frame_wh)
        obs_var["fungus"] = observation_variance(source, "fungus") * inflate

    alt = float(telemetry.get("alt", 0.0) or 0.0)
    footprint = footprint_radius_m(alt)
    return PointObservation(
        lat=float(lat), lon=float(lon), values=values, obs_var=obs_var,
        footprint_m=footprint, influence_radius_m=max(2.5 * footprint, footprint + 4.0),
        source=source,
    )


def ingest_rgb_result(twin: DigitalTwin, result: Dict[str, Any], telemetry: Dict[str, Any],
                      source: str = "drone_rgb", frame_wh=DEFAULT_FRAME_WH) -> Dict[str, Any]:
    obs = rgb_result_to_observation(result, telemetry, source, frame_wh)
    if obs is None:
        return {"assimilated": False, "reason": "no GPS fix or low image quality"}
    try:
        results = obs.apply(twin)
    except ValueError as e:
        return {"assimilated": False, "reason": str(e)}
    return {"assimilated": True, "footprint_m": round(obs.footprint_m, 2),
            "values": obs.values, "results": results}


def bands_to_layers(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Green/Red/RedEdge/NIR reflectance -> twin layer observations."""
    from src.indices.indices import compute_cire, compute_ndvi, compute_ndwi

    g = np.asarray(bands["green"], dtype=np.float32)
    r = np.asarray(bands["red"], dtype=np.float32)
    re = np.asarray(bands["red_edge"], dtype=np.float32)
    nir = np.asarray(bands["nir"], dtype=np.float32)
    return {
        "ndvi": compute_ndvi(nir, r),
        "moisture": np.clip((compute_ndwi(g, nir) + 1.0) / 2.0, 0.0, 1.0),
        "nitrogen": np.clip(compute_cire(nir, re) / 5.0, 0.0, 1.0),
    }


def ingest_multispectral(twin: DigitalTwin, bands: Dict[str, np.ndarray],
                         source: str = "multispectral") -> List[Dict[str, Any]]:
    """Assimilate a multispectral survey already resampled onto the field extent."""
    results: List[Dict[str, Any]] = []
    for layer, arr in bands_to_layers(bands).items():
        results.extend(twin.observe_grid(layer, arr, source=source))
    return results
