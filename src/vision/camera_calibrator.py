"""
camera_calibrator.py
---------------------
Per-device camera color calibration for Live Field Mode.

Different Android phones produce very different RGB outputs due to:
    - Auto white balance
    - HDR processing
    - Manufacturer color tuning
    - Different sensor characteristics

Calibration corrects the color distribution to a consistent reference
so RGB vegetation indices (GRVI, VARI, ExG) are comparable across devices.

Calibration procedure:
    1. User holds phone over a calibration reference (grey card or grass patch)
    2. System captures N frames and computes per-channel gain + offset
    3. Profile is saved to outputs/calibration/<device_id>.json
    4. Profile is applied to all subsequent frames from that device

Calibration modes:
    - grey_card     : 18% grey reference (most accurate)
    - grass_patch   : calibrate against a known-healthy green reference
    - auto_white    : normalize channels to equal means (basic correction)
    - default       : no-op (identity transform)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


CALIBRATION_DIR = Path("outputs/calibration")
DEFAULT_PROFILE_ID = "default"


class CalibrationProfile:
    """Color calibration profile: per-channel gain and offset."""

    def __init__(self, profile_id: str = "default",
                 gains: Optional[np.ndarray] = None,
                 offsets: Optional[np.ndarray] = None,
                 method: str = "default"):
        self.profile_id = profile_id
        self.method = method
        self.created_at = time.time()
        # gains and offsets are [R, G, B] arrays
        self.gains   = gains   if gains   is not None else np.array([1.0, 1.0, 1.0])
        self.offsets = offsets if offsets is not None else np.array([0.0, 0.0, 0.0])

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply gain + offset correction. Returns uint8 RGB."""
        img = image_rgb.astype(np.float32)
        img[:, :, 0] = img[:, :, 0] * self.gains[0] + self.offsets[0]
        img[:, :, 1] = img[:, :, 1] * self.gains[1] + self.offsets[1]
        img[:, :, 2] = img[:, :, 2] * self.gains[2] + self.offsets[2]
        return np.clip(img, 0, 255).astype(np.uint8)

    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "method": self.method,
            "created_at": self.created_at,
            "gains": self.gains.tolist(),
            "offsets": self.offsets.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibrationProfile":
        return cls(
            profile_id=d["profile_id"],
            gains=np.array(d["gains"]),
            offsets=np.array(d["offsets"]),
            method=d.get("method", "default"),
        )

    def save(self) -> None:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIBRATION_DIR / f"{self.profile_id}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, profile_id: str) -> "CalibrationProfile":
        path = CALIBRATION_DIR / f"{profile_id}.json"
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls(profile_id=DEFAULT_PROFILE_ID)  # identity


# ── In-memory cache ────────────────────────────────────────────────────
_profile_cache: Dict[str, CalibrationProfile] = {}


def get_profile(profile_id: str) -> CalibrationProfile:
    """Get a cached or loaded calibration profile."""
    if profile_id not in _profile_cache:
        _profile_cache[profile_id] = CalibrationProfile.load(profile_id)
    return _profile_cache[profile_id]


class CameraCalibrator:
    """
    Calibrate a phone camera by collecting reference frames and
    computing per-channel correction gains.
    """

    def __init__(self, device_id: str, method: str = "auto_white"):
        self.device_id = device_id
        self.method = method
        self._frames: list = []

    def add_reference_frame(self, image_rgb: np.ndarray) -> None:
        """Add a reference frame for calibration computation."""
        self._frames.append(image_rgb.copy())

    def compute_profile(self) -> CalibrationProfile:
        """
        Compute the calibration profile from collected reference frames.
        """
        if not self._frames:
            return CalibrationProfile(self.device_id)

        # Stack frames and compute mean per channel
        stack = np.stack(self._frames, axis=0).astype(np.float32)
        mean_r = float(stack[:, :, :, 0].mean())
        mean_g = float(stack[:, :, :, 1].mean())
        mean_b = float(stack[:, :, :, 2].mean())

        if self.method == "auto_white":
            # Normalize so all channels have the same mean (grey-world assumption)
            target = (mean_r + mean_g + mean_b) / 3.0
            gains = np.array([
                target / max(mean_r, 1e-5),
                target / max(mean_g, 1e-5),
                target / max(mean_b, 1e-5),
            ])
            offsets = np.array([0.0, 0.0, 0.0])

        elif self.method == "grey_card":
            # Target: 18% grey = pixel value ~46 (0–255)
            target = 46.0
            gains = np.array([
                target / max(mean_r, 1e-5),
                target / max(mean_g, 1e-5),
                target / max(mean_b, 1e-5),
            ])
            offsets = np.array([0.0, 0.0, 0.0])

        elif self.method == "grass_patch":
            # Calibrate green channel to 160, reduce R and B relative to G
            # Empirical targets for healthy vegetation in RGB
            target_r, target_g, target_b = 60.0, 140.0, 55.0
            gains = np.array([
                target_r / max(mean_r, 1e-5),
                target_g / max(mean_g, 1e-5),
                target_b / max(mean_b, 1e-5),
            ])
            offsets = np.array([0.0, 0.0, 0.0])

        else:
            # Default: identity
            gains   = np.array([1.0, 1.0, 1.0])
            offsets = np.array([0.0, 0.0, 0.0])

        # Clamp gains to reasonable range
        gains = np.clip(gains, 0.3, 3.0)

        profile = CalibrationProfile(
            profile_id=self.device_id,
            gains=gains,
            offsets=offsets,
            method=self.method,
        )
        profile.save()
        _profile_cache[self.device_id] = profile
        return profile

    def frames_collected(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()
