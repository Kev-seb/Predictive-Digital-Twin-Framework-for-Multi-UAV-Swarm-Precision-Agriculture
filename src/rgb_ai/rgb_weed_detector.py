"""
rgb_weed_detector.py
---------------------
RGB weed detection using colour + morphological segmentation.

Strategy:
    1. Segment vegetation from background using ExG (Excess Green)
    2. Within vegetation pixels, discriminate weeds from crops using:
       - Texture (weeds have more irregular texture than row crops)
       - Colour deviation (weeds often slightly different green tone)
       - Shape (weeds have less regular leaf geometry)
    3. Cluster weed pixels into bounding boxes

Returns:
    - weed_mask: pixel-level weed probability map
    - detections: list of BoundingBox
    - weed_coverage_pct: percentage of vegetation area occupied by weeds
"""

from __future__ import annotations

import cv2
import numpy as np

from src.mission.mission_object import BoundingBox


class RGBWeedDetector:
    """
    Detects weed patches from RGB imagery using texture + colour analysis.
    """

    def __init__(self, weed_confidence_min: float = 0.40):
        self.weed_confidence_min = weed_confidence_min

    def detect(self, image_rgb: np.ndarray) -> tuple:
        """
        Detect weed regions.

        Returns:
            (detections: List[BoundingBox], weed_mask: np.ndarray, coverage_pct: float)
        """
        img = image_rgb.astype(np.float32) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # ── Step 1: Vegetation mask via ExG ────────────────────────
        exg = 2 * G - R - B
        veg_mask = (exg > 0.05).astype(np.uint8)

        # Morphological cleanup
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        veg_mask = cv2.morphologyEx(veg_mask * 255, cv2.MORPH_CLOSE, kernel5)
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel5)
        veg_binary = (veg_mask > 127).astype(np.uint8)

        if veg_binary.sum() < 1000:
            # No significant vegetation — nothing to detect
            return [], np.zeros(image_rgb.shape[:2], np.uint8), 0.0

        # ── Step 2: Texture variance as weed discriminator ────────
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Local variance in a small window
        blur = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32)
        diff = (gray.astype(np.float32) - blur) ** 2
        local_var = cv2.GaussianBlur(diff, (15, 15), 0)

        # Normalise local variance
        lv_max = local_var.max()
        if lv_max > 1e-5:
            local_var_norm = local_var / lv_max
        else:
            local_var_norm = local_var

        # ── Step 3: Colour irregularity within vegetation ─────────
        # Crops tend to be uniformly green; weeds deviate in hue
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_channel = hsv[:, :, 0]  # 0–180 in OpenCV
        # Mean hue of vegetation pixels
        veg_hues = h_channel[veg_binary == 1]
        mean_hue = float(veg_hues.mean()) if len(veg_hues) > 0 else 60.0

        # Hue deviation from dominant crop hue
        hue_dev = np.abs(h_channel - mean_hue)
        hue_dev_norm = np.clip(hue_dev / 40.0, 0.0, 1.0)

        # ── Step 4: Weed probability map ─────────────────────────
        # High texture variance + high hue deviation within vegetation = weed
        weed_prob = (0.6 * local_var_norm + 0.4 * hue_dev_norm) * veg_binary.astype(np.float32)

        # ── Step 5: Threshold + contour extraction ────────────────
        weed_thresh = (weed_prob > 0.55).astype(np.uint8) * 255
        kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        weed_thresh = cv2.morphologyEx(weed_thresh, cv2.MORPH_CLOSE, kernel9)
        weed_thresh = cv2.morphologyEx(weed_thresh, cv2.MORPH_OPEN, kernel9)

        detections: list[BoundingBox] = []
        contours, _ = cv2.findContours(weed_thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        h, w = image_rgb.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600:
                continue

            # Confidence = mean weed_prob inside the bounding box
            x, y, bw, bh = cv2.boundingRect(cnt)
            roi_prob = weed_prob[y:y+bh, x:x+bw]
            confidence = round(float(roi_prob.mean()) * 1.5, 3)
            confidence = max(0.0, min(1.0, confidence))

            if confidence < self.weed_confidence_min:
                continue

            detections.append(BoundingBox(
                x1=x, y1=y, x2=x + bw, y2=y + bh,
                label="Weed",
                confidence=confidence,
                color="#84cc16",
            ))

        # ── Coverage percentage ────────────────────────────────────
        veg_pixels = int(veg_binary.sum())
        weed_pixels = int((weed_thresh > 127).sum())
        coverage_pct = round(100.0 * weed_pixels / max(veg_pixels, 1), 2)

        weed_mask_out = (weed_prob * 255).astype(np.uint8)
        return detections, weed_mask_out, coverage_pct
