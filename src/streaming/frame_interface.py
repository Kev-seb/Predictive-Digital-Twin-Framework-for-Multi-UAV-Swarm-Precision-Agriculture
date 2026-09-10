"""
frame_interface.py
------------------
Unified Frame Interface — the sensor-agnostic core of the Live Field pipeline.

Every image entering the AI pipeline is wrapped in a UnifiedFrame regardless
of its origin:
    - phone_rgb      → Android smartphone camera
    - uav_rgb        → UAV RGB camera
    - uav_ms         → UAV multispectral camera (future)
    - tiff_upload    → Uploaded GeoTIFF (existing Research Mode)
    - usb_camera     → Wired USB camera (future)

The AI engine, Digital Twin, GIS, and report generator NEVER inspect the
`source` field — they only see the standardised UnifiedFrame.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.mission.mission_object import FusedTelemetry, ImageQualityReport


@dataclass
class UnifiedFrame:
    """
    Canonical, source-agnostic frame that travels through the entire pipeline.

    Every component downstream receives this and nothing else. Adding a new
    sensor only requires implementing a new adapter that produces UnifiedFrame.
    """

    # ── Identity ───────────────────────────────────────────────────
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_id: str = ""
    source: str = "phone_rgb"       # See module docstring for valid values
    sequence_num: int = 0
    timestamp: float = field(default_factory=time.time)

    # ── Image data (always HxWx3 uint8 RGB) ───────────────────────
    image_rgb: Optional[np.ndarray] = None   # H×W×3 uint8

    # ── Spatial context ────────────────────────────────────────────
    gps: FusedTelemetry = field(default_factory=FusedTelemetry)
    heading: float = 0.0            # degrees 0–360

    # ── Environmental snapshot at capture time ─────────────────────
    weather_snapshot: Optional[Dict[str, Any]] = None

    # ── Quality assessment (filled by IQA stage) ──────────────────
    quality: Optional[ImageQualityReport] = None

    # ── Camera calibration profile applied ────────────────────────
    camera_profile: str = "default"

    # ── Computed properties ────────────────────────────────────────
    @property
    def is_valid(self) -> bool:
        """True if the frame carries actual image data and passed QA."""
        if self.image_rgb is None:
            return False
        if self.quality is not None and not self.quality.passed:
            return False
        return True

    @property
    def shape(self):
        return self.image_rgb.shape if self.image_rgb is not None else None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Lightweight metadata dict — without the image array."""
        return {
            "frame_id": self.frame_id,
            "mission_id": self.mission_id,
            "source": self.source,
            "sequence_num": self.sequence_num,
            "timestamp": self.timestamp,
            "heading": self.heading,
            "camera_profile": self.camera_profile,
            "gps": self.gps.to_dict() if self.gps else {},
            "weather": self.weather_snapshot,
            "quality_score": self.quality.quality_score if self.quality else None,
            "quality_passed": self.quality.passed if self.quality else True,
        }


class FrameSync:
    """
    Sequence counter and timestamp synchronizer.

    Ensures every frame across a mission has a globally monotonic
    sequence number and a consistent timestamp baseline.
    """

    def __init__(self, mission_id: str, source: str = "phone_rgb"):
        self.mission_id = mission_id
        self.source = source
        self._counter = 0
        self._start_time = time.time()

    def stamp(self, image_rgb: np.ndarray, gps: FusedTelemetry,
              heading: float = 0.0,
              weather_snapshot: Optional[Dict[str, Any]] = None,
              camera_profile: str = "default") -> UnifiedFrame:
        """
        Wrap a raw image into a fully-stamped UnifiedFrame.
        """
        self._counter += 1
        return UnifiedFrame(
            frame_id=str(uuid.uuid4()),
            mission_id=self.mission_id,
            source=self.source,
            sequence_num=self._counter,
            timestamp=time.time(),
            image_rgb=image_rgb,
            gps=gps,
            heading=heading,
            weather_snapshot=weather_snapshot,
            camera_profile=camera_profile,
        )

    @property
    def frames_stamped(self) -> int:
        return self._counter

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time
