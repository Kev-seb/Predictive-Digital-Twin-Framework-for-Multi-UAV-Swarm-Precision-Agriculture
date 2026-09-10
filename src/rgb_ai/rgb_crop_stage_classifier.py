"""
rgb_crop_stage_classifier.py
-----------------------------
Crop growth stage classification from RGB images.

Uses a rule-based system on colour + texture features:
    - Vegetation fraction (ExG-based)
    - Mean green intensity
    - Texture complexity (edge density)
    - Vegetation colour maturity (green vs yellowing)

Stages output:
    Bare Soil   → no/minimal vegetation
    Seedling    → sparse, light-green, small plants
    Vegetative  → dense, dark green canopy
    Tillering   → intermediate stage for paddy/wheat
    Flowering   → yellow-green, lower intensity
    Grain Fill  → yellowish-brown tones dominate
    Harvest     → predominantly yellow/brown

Note: These are ESTIMATES from RGB. Ground truth requires
multispectral imagery or manual scouting.
"""

from __future__ import annotations

import cv2
import numpy as np


STAGE_ORDER = [
    "Bare Soil",
    "Seedling",
    "Early Vegetative",
    "Vegetative",
    "Tillering / Stem Extension",
    "Flowering / Booting",
    "Grain Fill",
    "Harvest-Ready",
]


class RGBCropStageClassifier:
    """
    Estimates crop growth stage from RGB colour + texture features.
    """

    def classify(self, image_rgb: np.ndarray) -> tuple:
        """
        Classify crop growth stage.

        Returns:
            (stage: str, confidence: float, feature_dict: dict)
        """
        img = image_rgb.astype(np.float32) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # ── Feature extraction ─────────────────────────────────────

        # 1. Vegetation fraction using ExG
        exg = 2 * G - R - B
        veg_frac = float(np.mean(exg > 0.05))

        # 2. Mean green intensity within vegetation pixels
        veg_mask = exg > 0.05
        green_intensity = float(np.mean(G[veg_mask])) if veg_mask.sum() > 0 else 0.0

        # 3. Texture complexity (edge density as proxy for canopy density)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.sum(edges > 0)) / (image_rgb.shape[0] * image_rgb.shape[1])

        # 4. Yellowing index (yellow = R+G high, B low, low ExG)
        yellow_mask = (R > 0.55) & (G > 0.45) & (B < 0.30) & (exg < 0.05)
        yellow_frac = float(np.mean(yellow_mask))

        # 5. Brown / senescent fraction
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        brown_mask = (h > 5) & (h < 25) & (s > 50)
        brown_frac = float(np.mean(brown_mask))

        # ── Rule-based classification ──────────────────────────────
        features = {
            "veg_frac": round(veg_frac, 3),
            "green_intensity": round(green_intensity, 3),
            "edge_density": round(edge_density, 4),
            "yellow_frac": round(yellow_frac, 3),
            "brown_frac": round(brown_frac, 3),
        }

        stage, confidence = self._rules(veg_frac, green_intensity,
                                         edge_density, yellow_frac, brown_frac)
        return stage, confidence, features

    @staticmethod
    def _rules(veg_frac: float, green_int: float, edge_den: float,
               yellow_frac: float, brown_frac: float) -> tuple:

        # Harvest / senescent: dominated by brown/yellow
        if brown_frac > 0.30 or yellow_frac > 0.35:
            if brown_frac > 0.25:
                return "Harvest-Ready", min(0.88, 0.60 + brown_frac)
            return "Grain Fill", min(0.85, 0.55 + yellow_frac)

        # Bare soil: minimal vegetation
        if veg_frac < 0.10:
            conf = max(0.70, 0.95 - veg_frac * 5)
            return "Bare Soil", round(conf, 2)

        # Seedling: low vegetation, light green
        if veg_frac < 0.30 and green_int < 0.45:
            return "Seedling", round(0.65 + veg_frac, 2)

        # Early vegetative: growing coverage
        if veg_frac < 0.50 and green_int < 0.55:
            return "Early Vegetative", round(0.65 + veg_frac * 0.3, 2)

        # Tillering: dense, moderate green
        if veg_frac > 0.50 and edge_den > 0.04 and green_int < 0.65:
            return "Tillering / Stem Extension", 0.72

        # Flowering: dense but greens slightly muted + some yellow
        if veg_frac > 0.60 and yellow_frac > 0.05:
            return "Flowering / Booting", 0.68

        # Fully vegetative: dense, dark green
        if veg_frac > 0.60 and green_int > 0.55:
            return "Vegetative", round(0.75 + veg_frac * 0.15, 2)

        # Default
        return "Vegetative", 0.55
