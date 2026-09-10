"""
mission_object.py
-----------------
Full Mission dataclasses for Live Field Mode.

A Mission is the top-level container for everything recorded during
a single field walk. It carries:
  - GPS track
  - Per-frame observations (image + AI results + telemetry)
  - Real-time alerts
  - Weather log
  - Satellite snapshot context
  - Treatment plan
  - Digital Twin state snapshot
  - Mission statistics
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── GPS ────────────────────────────────────────────────────────────────

@dataclass
class GPSPoint:
    lat: float
    lon: float
    alt: float = 0.0
    accuracy_m: float = 5.0
    timestamp: float = 0.0


# ── Fused telemetry (GPS + IMU) ────────────────────────────────────────

@dataclass
class FusedTelemetry:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    heading: float = 0.0      # degrees 0–360
    pitch: float = 0.0        # degrees
    roll: float = 0.0         # degrees
    speed: float = 0.0        # m/s (dead-reckoning fused)
    battery: float = 100.0    # %
    accuracy_m: float = 5.0
    signal_strength: int = 0  # 0–4 bars
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat, "lon": self.lon, "alt": self.alt,
            "heading": self.heading, "pitch": self.pitch, "roll": self.roll,
            "speed": self.speed, "battery": self.battery,
            "accuracy_m": self.accuracy_m, "signal_strength": self.signal_strength,
            "timestamp": self.timestamp,
        }


# ── AI inference results ───────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int; y1: int; x2: int; y2: int
    label: str
    confidence: float
    color: str = "#ef4444"


@dataclass
class InferenceMetadata:
    model_name: str = "RGB Stress Estimator"
    model_version: str = "v1.0"
    inference_time_ms: float = 0.0
    confidence: float = 1.0
    quality_score: float = 1.0
    low_confidence_reason: Optional[str] = None


@dataclass
class RGBInferenceResult:
    # Core outputs
    crop_stress_score: float = 0.0         # 0–1
    stress_label: str = "Healthy"          # Healthy | Mild | Moderate | Severe | Critical
    disease_detections: List[BoundingBox] = field(default_factory=list)
    crop_stage: str = "Unknown"
    weed_detections: List[BoundingBox] = field(default_factory=list)
    weed_coverage_pct: float = 0.0

    # RGB vegetation indices (not multispectral — clearly labelled)
    grvi: float = 0.0    # Green-Red Vegetation Index
    vari: float = 0.0    # Visible Atmospherically Resistant Index
    exg: float = 0.0     # Excess Green Index

    # Spatial maps (may be None for lightweight mode)
    heatmap_rgb: Optional[Any] = None       # np.ndarray HxWx3
    segmentation_mask: Optional[Any] = None # np.ndarray HxW uint8

    # Metadata
    meta: InferenceMetadata = field(default_factory=InferenceMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop_stress_score": round(self.crop_stress_score, 4),
            "stress_label": self.stress_label,
            "crop_stage": self.crop_stage,
            "weed_coverage_pct": round(self.weed_coverage_pct, 2),
            "grvi": round(self.grvi, 4),
            "vari": round(self.vari, 4),
            "exg": round(self.exg, 4),
            "disease_detections": [
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                 "label": b.label, "confidence": round(b.confidence, 4)}
                for b in self.disease_detections
            ],
            "weed_detections": [
                {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                 "label": b.label, "confidence": round(b.confidence, 4)}
                for b in self.weed_detections
            ],
            "model_name": self.meta.model_name,
            "model_version": self.meta.model_version,
            "inference_time_ms": round(self.meta.inference_time_ms, 2),
            "model_confidence": round(self.meta.confidence, 4),
            "quality_score": round(self.meta.quality_score, 4),
            "low_confidence_reason": self.meta.low_confidence_reason,
        }


# ── Image quality ──────────────────────────────────────────────────────

@dataclass
class ImageQualityReport:
    passed: bool = True
    quality_score: float = 1.0   # 0–1
    blur_score: float = 1.0      # Laplacian variance normalised
    brightness: float = 128.0    # mean pixel value 0–255
    is_blurred: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    is_obstructed: bool = False
    rejection_reason: Optional[str] = None


# ── Per-frame observation ──────────────────────────────────────────────

@dataclass
class Observation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_id: str = ""
    timestamp: float = 0.0
    sequence_num: int = 0

    # Spatial
    telemetry: FusedTelemetry = field(default_factory=FusedTelemetry)

    # AI
    inference: Optional[RGBInferenceResult] = None
    quality: Optional[ImageQualityReport] = None

    # Storage paths (relative to mission dir)
    frame_original_path: Optional[str] = None
    frame_annotated_path: Optional[str] = None
    inference_json_path: Optional[str] = None

    # Weather snapshot at this moment
    weather: Optional[Dict[str, Any]] = None


# ── Alert ──────────────────────────────────────────────────────────────

@dataclass
class Alert:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_id: str = ""
    timestamp: float = 0.0
    severity: str = "warning"       # warning | critical
    category: str = "stress"        # stress | disease | battery | coverage
    message: str = ""
    lat: float = 0.0
    lon: float = 0.0
    acknowledged: bool = False


# ── Mission statistics ─────────────────────────────────────────────────

@dataclass
class MissionStatistics:
    total_frames: int = 0
    frames_accepted: int = 0
    frames_rejected: int = 0
    total_distance_m: float = 0.0
    duration_seconds: float = 0.0
    mean_stress_score: float = 0.0
    max_stress_score: float = 0.0
    disease_detections_total: int = 0
    weed_coverage_mean_pct: float = 0.0
    alerts_warning: int = 0
    alerts_critical: int = 0
    area_covered_m2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "frames_accepted": self.frames_accepted,
            "frames_rejected": self.frames_rejected,
            "total_distance_m": round(self.total_distance_m, 2),
            "duration_seconds": round(self.duration_seconds, 1),
            "mean_stress_score": round(self.mean_stress_score, 4),
            "max_stress_score": round(self.max_stress_score, 4),
            "disease_detections_total": self.disease_detections_total,
            "weed_coverage_mean_pct": round(self.weed_coverage_mean_pct, 2),
            "alerts_warning": self.alerts_warning,
            "alerts_critical": self.alerts_critical,
            "area_covered_m2": round(self.area_covered_m2, 1),
        }


# ── Full Mission Object ────────────────────────────────────────────────

@dataclass
class Mission:
    """
    Top-level container for a single field walk session.
    Everything produced during one walk belongs to this object.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str = "active"              # active | paused | completed | exported

    # Field location
    field_lat: float = 0.0
    field_lon: float = 0.0
    field_name: str = "Field Alpha"

    # Data
    gps_track: List[GPSPoint] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    weather_log: List[Dict[str, Any]] = field(default_factory=list)

    # Context snapshots
    satellite_snapshot: Optional[Dict[str, Any]] = None
    treatment_plan: Optional[Dict[str, Any]] = None
    digital_twin_state: Optional[Dict[str, Any]] = None

    # Statistics (updated incrementally)
    statistics: MissionStatistics = field(default_factory=MissionStatistics)

    # Output paths
    report_pdf_path: Optional[str] = None
    kml_path: Optional[str] = None
    json_export_path: Optional[str] = None

    # Sensor source for this mission
    sensor_source: str = "phone_rgb"    # phone_rgb | uav_rgb | uav_ms | tiff

    # Camera calibration profile
    camera_profile: str = "default"

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "field_lat": self.field_lat,
            "field_lon": self.field_lon,
            "sensor_source": self.sensor_source,
            "statistics": self.statistics.to_dict(),
        }
