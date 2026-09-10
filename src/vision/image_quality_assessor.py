"""
image_quality_assessor.py
--------------------------
Image Quality Assessment (IQA) module for Live Field Mode.

Runs BEFORE AI inference to reject or flag poor-quality frames.
Checks performed:
    1. Blur (Laplacian variance)
    2. Brightness (underexposure / overexposure)
    3. Edge density (lens obstruction / rain drops)
    4. Resolution check
    5. Saturation check (camera failure / lens cap)

A quality score (0–1) is computed and a pass/fail decision is returned.
Only frames with quality.passed == True reach the AI engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.mission.mission_object import ImageQualityReport


# ── Thresholds ─────────────────────────────────────────────────────────
BLUR_THRESHOLD          = 80.0    # Laplacian variance — below = blurred
BRIGHTNESS_MIN          = 35.0    # Mean pixel value — below = underexposed
BRIGHTNESS_MAX          = 220.0   # Mean pixel value — above = overexposed
EDGE_DENSITY_MIN        = 0.01    # Below = obstructed / rain-covered lens
SATURATION_MIN          = 5.0     # HSV S-channel mean — below = grey/failure
MIN_WIDTH               = 240
MIN_HEIGHT              = 240


class ImageQualityAssessor:
    """
    Stateless image quality assessor.

    Usage:
        assessor = ImageQualityAssessor()
        report = assessor.assess(frame_rgb)
        if report.passed:
            # send to AI
    """

    def __init__(
        self,
        blur_threshold: float = BLUR_THRESHOLD,
        brightness_min: float = BRIGHTNESS_MIN,
        brightness_max: float = BRIGHTNESS_MAX,
        edge_density_min: float = EDGE_DENSITY_MIN,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.edge_density_min = edge_density_min

    # ── Public interface ───────────────────────────────────────────

    def assess(self, image_rgb: np.ndarray) -> ImageQualityReport:
        """
        Assess image quality and return a structured report.
        """
        if image_rgb is None or image_rgb.size == 0:
            return ImageQualityReport(
                passed=False, quality_score=0.0,
                rejection_reason="empty_frame"
            )

        h, w = image_rgb.shape[:2]

        # Resolution check
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return ImageQualityReport(
                passed=False, quality_score=0.0,
                rejection_reason=f"resolution_too_low_{w}x{h}"
            )

        # Convert to grayscale for blur + edge analysis
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # ── 1. Blur detection ──────────────────────────────────────
        blur_score = self._laplacian_variance(gray)
        is_blurred = blur_score < self.blur_threshold

        # ── 2. Brightness ─────────────────────────────────────────
        brightness = float(np.mean(gray))
        is_underexposed = brightness < self.brightness_min
        is_overexposed  = brightness > self.brightness_max

        # ── 3. Edge density (obstruction / rain) ──────────────────
        edges = cv2.Canny(gray, 30, 100)
        edge_density = float(np.sum(edges > 0)) / (h * w)
        is_obstructed = edge_density < self.edge_density_min

        # ── 4. Saturation check ───────────────────────────────────
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        is_desaturated = saturation_mean < SATURATION_MIN

        # ── Compute composite quality score ───────────────────────
        blur_norm     = min(blur_score / (self.blur_threshold * 3), 1.0)
        brightness_norm = 1.0 - abs(brightness - 128.0) / 128.0
        edge_norm     = min(edge_density / 0.05, 1.0)
        sat_norm      = min(saturation_mean / 50.0, 1.0)
        quality_score = float(0.40 * blur_norm + 0.25 * brightness_norm +
                              0.20 * edge_norm + 0.15 * sat_norm)
        quality_score = max(0.0, min(1.0, quality_score))

        # ── Determine pass/fail ────────────────────────────────────
        passed = True
        rejection_reason = None

        if is_blurred:
            passed = False
            rejection_reason = "motion_blur"
        elif is_underexposed:
            passed = False
            rejection_reason = "underexposed"
        elif is_overexposed:
            passed = False
            rejection_reason = "overexposed"
        elif is_obstructed:
            passed = False
            rejection_reason = "lens_obstruction"
        elif is_desaturated:
            passed = False
            rejection_reason = "camera_failure"

        return ImageQualityReport(
            passed=passed,
            quality_score=round(quality_score, 4),
            blur_score=round(min(blur_score / self.blur_threshold, 2.0), 4),
            brightness=round(brightness, 2),
            is_blurred=is_blurred,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            is_obstructed=is_obstructed,
            rejection_reason=rejection_reason,
        )

    def get_rejection_label(self, reason: str) -> str:
        labels = {
            "motion_blur":     "Motion Blur — Hold device steady",
            "underexposed":    "Underexposed — Move to better lighting",
            "overexposed":     "Overexposed — Avoid direct sunlight",
            "lens_obstruction":"Lens Obstructed — Clean camera lens",
            "camera_failure":  "Camera Error — Check camera feed",
            "resolution_too_low": "Resolution Too Low",
            "empty_frame":     "Empty Frame",
        }
        for k, v in labels.items():
            if reason and reason.startswith(k):
                return v
        return reason or "Unknown"

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Higher = sharper image. Threshold ~80 for acceptable focus."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
