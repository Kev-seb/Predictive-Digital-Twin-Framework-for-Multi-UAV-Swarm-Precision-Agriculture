"""
rgb_disease_detector.py
------------------------
RGB-based crop disease detection using colour thresholding +
morphological analysis + contour filtering.

Detects common crop disease signatures visible in RGB:
    - Brown spot / lesions   (Reddish-brown discolouration)
    - Leaf yellowing         (Nitrogen deficiency / mosaic virus)
    - Blight patches         (Dark necrotic tissue)
    - Rust infection         (Reddish-orange pustules)
    - Powdery mildew         (Whitish surface coating)

Returns bounding boxes + labels + confidence scores.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.mission.mission_object import BoundingBox


class RGBDiseaseDetector:
    """
    Detects disease signatures in RGB images using OpenCV colour analysis.
    """

    def __init__(self, min_area_px: int = 400, confidence_min: float = 0.35):
        self.min_area_px   = min_area_px
        self.confidence_min = confidence_min

    def detect(self, image_rgb: np.ndarray) -> list:
        """
        Detect disease regions.

        Returns:
            List[BoundingBox] with label and confidence for each detection.
        """
        detections: list[BoundingBox] = []

        # Convert to HSV for robust colour matching
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # ── 1. Brown spot / Leaf blight ────────────────────────────
        # Brown: H 5–25, S 60–255, V 50–200
        mask_brown = cv2.inRange(hsv,
                                  np.array([5,  60,  40]),
                                  np.array([20, 255, 200]))
        self._extract_boxes(mask_brown, image_rgb.shape, "Brown Spot",
                             0.70, "#c2410c", detections)

        # ── 2. Leaf yellowing ─────────────────────────────────────
        # Yellow: H 20–40, S 80–255, V 100–255
        mask_yellow = cv2.inRange(hsv,
                                   np.array([20, 80,  100]),
                                   np.array([40, 255, 255]))
        self._extract_boxes(mask_yellow, image_rgb.shape, "Leaf Yellowing",
                             0.65, "#eab308", detections)

        # ── 3. Blight / Necrotic tissue ───────────────────────────
        # Very dark: V < 50, any H
        mask_dark = cv2.inRange(hsv,
                                 np.array([0, 0,   0]),
                                 np.array([180, 255, 55]))
        # But exclude soil (H 5-20 + low sat or already in brown mask)
        mask_dark = cv2.bitwise_and(mask_dark, cv2.bitwise_not(mask_brown))
        self._extract_boxes(mask_dark, image_rgb.shape, "Leaf Blight",
                             0.55, "#7c2d12", detections)

        # ── 4. Rust infection ─────────────────────────────────────
        # Rust orange: H 5–15, S > 150, V > 100
        mask_rust = cv2.inRange(hsv,
                                 np.array([5,  150, 80]),
                                 np.array([15, 255, 220]))
        self._extract_boxes(mask_rust, image_rgb.shape, "Rust Infection",
                             0.60, "#ea580c", detections)

        # ── 5. Powdery mildew ─────────────────────────────────────
        # Whitish: S < 30, V > 180
        mask_white = cv2.inRange(hsv,
                                  np.array([0,  0,  180]),
                                  np.array([180, 30, 255]))
        self._extract_boxes(mask_white, image_rgb.shape, "Powdery Mildew",
                             0.50, "#e2e8f0", detections)

        return detections

    def _extract_boxes(self, mask: np.ndarray, img_shape: tuple,
                        label: str, base_confidence: float,
                        color: str, out: list) -> None:
        """Extract bounding boxes from a binary mask via contour analysis."""
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_shape[:2]
        img_area = h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px:
                continue

            # Area fraction as confidence modifier
            area_frac = area / img_area
            # Larger patches = higher confidence up to a point
            conf_mod = min(1.0, area_frac * 50.0)
            confidence = round(base_confidence * (0.7 + 0.3 * conf_mod), 3)

            if confidence < self.confidence_min:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            out.append(BoundingBox(
                x1=x, y1=y, x2=x + bw, y2=y + bh,
                label=label,
                confidence=confidence,
                color=color,
            ))
