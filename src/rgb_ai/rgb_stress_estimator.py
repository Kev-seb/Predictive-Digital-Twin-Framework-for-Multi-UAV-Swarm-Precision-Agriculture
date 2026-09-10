"""
rgb_stress_estimator.py
------------------------
RGB-only vegetation stress estimation using color-space analysis.

Computes:
    GRVI  — Green-Red Vegetation Index  (Gobron et al. 1999)
    VARI  — Visible Atmospherically Resistant Index  (Gitelson et al. 2002)
    ExG   — Excess Green Index  (Woebbecke et al. 1995)
    GLI   — Green Leaf Index
    MGRVI — Modified Green-Red Vegetation Index

These are the physically correct alternatives to NDVI/NDRE/NDWI for RGB
cameras (which CANNOT compute NIR-based indices).

Clearly labelled as "RGB AI Estimates" — NOT multispectral indices.
"""

from __future__ import annotations

import cv2
import numpy as np


_EPSILON = 1e-7


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > _EPSILON, a / b, 0.0)


class RGBStressEstimator:
    """
    Estimates crop stress from RGB images using vegetation colour indices.

    All outputs in [-1, 1] or [0, 1] range matching convention.
    """

    def compute_indices(self, image_rgb: np.ndarray) -> dict:
        """
        Compute all RGB vegetation indices from a uint8 HxWx3 RGB array.

        Returns a dict of 2D float32 arrays.
        """
        img = image_rgb.astype(np.float32) / 255.0
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        # Green-Red Vegetation Index
        grvi = _safe_div(G - R, G + R)

        # Visible Atmospherically Resistant Index
        vari = _safe_div(G - R, G + R - B)

        # Excess Green Index (normalised to -1..1)
        exg_raw = 2 * G - R - B
        exg = np.clip(exg_raw / 2.0, -1.0, 1.0)

        # Green Leaf Index
        gli = _safe_div(2 * G - R - B, 2 * G + R + B)

        # Modified GRVI (quadratic)
        mgrvi = _safe_div(G**2 - R**2, G**2 + R**2)

        # VGDVI (visible green difference vegetation index)
        vgdvi = _safe_div(G - R - B, G + R + B)

        return {
            "grvi":  grvi.astype(np.float32),
            "vari":  np.clip(vari, -1, 1).astype(np.float32),
            "exg":   exg.astype(np.float32),
            "gli":   gli.astype(np.float32),
            "mgrvi": mgrvi.astype(np.float32),
            "vgdvi": vgdvi.astype(np.float32),
        }

    def estimate_stress(self, image_rgb: np.ndarray) -> tuple:
        """
        Estimate crop stress score from RGB.

        Returns:
            (stress_score: float 0-1, stress_label: str, indices: dict)
        """
        indices = self.compute_indices(image_rgb)

        # Composite stress: low GRVI + low VARI + low ExG → high stress
        grvi_mean = float(np.nanmean(indices["grvi"]))
        vari_mean = float(np.nanmean(indices["vari"]))
        exg_mean  = float(np.nanmean(indices["exg"]))

        # Normalise to 0–1 stress (higher = more stressed)
        # Healthy vegetation → GRVI~0.3, VARI~0.2, ExG~0.2
        # Stressed vegetation → GRVI~0, VARI~0, ExG~0 or negative
        s_grvi = max(0.0, min(1.0, (0.35 - grvi_mean) / 0.50))
        s_vari = max(0.0, min(1.0, (0.25 - vari_mean) / 0.40))
        s_exg  = max(0.0, min(1.0, (0.15 - exg_mean) / 0.35))

        stress_score = float(0.40 * s_grvi + 0.35 * s_vari + 0.25 * s_exg)
        stress_score = max(0.0, min(1.0, stress_score))

        if stress_score < 0.20:
            label = "Healthy"
        elif stress_score < 0.40:
            label = "Mild Stress"
        elif stress_score < 0.60:
            label = "Moderate Stress"
        elif stress_score < 0.80:
            label = "Severe Stress"
        else:
            label = "Critical Stress"

        return stress_score, label, indices

    def generate_stress_heatmap(self, image_rgb: np.ndarray,
                                 indices: dict) -> np.ndarray:
        """
        Generate a colour-coded stress heatmap overlay.
        Red = stressed, Green = healthy.
        Returns HxWx3 uint8 RGB.
        """
        # Use GRVI as the primary spatial indicator
        grvi = indices["grvi"]
        # Map: GRVI -0.2..0.5 → stress 1..0
        stress_map = np.clip((0.4 - grvi) / 0.6, 0.0, 1.0)

        # Apply RdYlGn colormap (via cv2)
        stress_uint8 = (stress_map * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(stress_uint8, cv2.COLORMAP_RdYlGn)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image
        alpha = 0.55
        blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
        return blended.astype(np.uint8)

    def confidence(self, image_rgb: np.ndarray, indices: dict) -> float:
        """
        Estimate model confidence based on image greenness and contrast.
        Higher green content and higher contrast → higher confidence.
        """
        img = image_rgb.astype(np.float32) / 255.0
        green_mean = float(np.mean(img[:, :, 1]))
        contrast = float(np.std(indices["grvi"]))

        # Scale: moderate green + good contrast = high confidence
        conf = min(1.0, green_mean * 1.8) * min(1.0, contrast * 8.0)
        return round(max(0.3, min(1.0, conf)), 3)
