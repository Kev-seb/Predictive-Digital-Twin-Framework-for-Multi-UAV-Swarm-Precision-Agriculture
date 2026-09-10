"""
rgb_inference_engine.py
------------------------
Master RGB AI inference engine for Live Field Mode.

Orchestrates the full hybrid pipeline:
    OpenCV preprocessing
        → Image Quality Assessment
        → Camera Calibration
        → RGB Stress Estimator (OpenCV colour analysis)
        → RGB Disease Detector (OpenCV morphology)
        → RGB Crop Stage Classifier
        → RGB Weed Detector
        → Confidence Monitoring
        → Result assembly

Every output clearly labels itself as "RGB AI Estimate" to distinguish
from multispectral Research Mode outputs.

The inference result carries:
    - All AI detections
    - Per-prediction confidence + model version
    - Inference timing
    - Quality score of the frame that produced it

This module is the ONLY thing the rest of the platform talks to.
It is sensor-agnostic by accepting UnifiedFrame objects.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.streaming.frame_interface import UnifiedFrame
from src.mission.mission_object import (
    RGBInferenceResult, InferenceMetadata, BoundingBox
)
from src.vision.image_quality_assessor import ImageQualityAssessor
from src.vision.camera_calibrator import get_profile
from src.rgb_ai.rgb_stress_estimator import RGBStressEstimator
from src.rgb_ai.rgb_disease_detector import RGBDiseaseDetector
from src.rgb_ai.rgb_weed_detector import RGBWeedDetector
from src.rgb_ai.rgb_crop_stage_classifier import RGBCropStageClassifier


MODEL_NAME    = "RGB Field AI"
MODEL_VERSION = "v1.0"


class RGBInferenceEngine:
    """
    Master inference engine for the RGB Live Field Mode pipeline.

    Usage:
        engine = RGBInferenceEngine()
        result = engine.infer(unified_frame)
    """

    def __init__(self, run_iqa: bool = True,
                 run_calibration: bool = True,
                 generate_heatmap: bool = True):
        self.run_iqa          = run_iqa
        self.run_calibration  = run_calibration
        self.generate_heatmap = generate_heatmap

        # Sub-engines
        self._iqa       = ImageQualityAssessor()
        self._stress    = RGBStressEstimator()
        self._disease   = RGBDiseaseDetector()
        self._weed      = RGBWeedDetector()
        self._stage     = RGBCropStageClassifier()

        # Frame counter for lightweight mode switching
        self._frame_count = 0

    def infer(self, frame: UnifiedFrame) -> RGBInferenceResult:
        """
        Run the full inference pipeline on a UnifiedFrame.

        Returns RGBInferenceResult even if inference is degraded.
        On empty/invalid frames returns a zero-confidence result.
        """
        self._frame_count += 1
        t_start = time.perf_counter()

        # ── 0. Validate input ──────────────────────────────────────
        if frame.image_rgb is None or frame.image_rgb.size == 0:
            return self._empty_result(0.0, "empty_frame")

        image = frame.image_rgb.copy()

        # ── 1. Image Quality Assessment ────────────────────────────
        quality = frame.quality
        if self.run_iqa and quality is None:
            quality = self._iqa.assess(image)

        if quality is not None and not quality.passed:
            return self._empty_result(
                quality.quality_score,
                quality.rejection_reason or "quality_fail"
            )

        quality_score = quality.quality_score if quality else 1.0

        # ── 2. Camera calibration ─────────────────────────────────
        if self.run_calibration and frame.camera_profile != "default":
            profile = get_profile(frame.camera_profile)
            image = profile.apply(image)

        # ── 3. OpenCV preprocessing ───────────────────────────────
        # Resize to processing resolution (max 640px wide for speed)
        proc_img = self._preprocess(image)

        # ── 4. Stress estimation ───────────────────────────────────
        stress_score, stress_label, rgb_indices = self._stress.estimate_stress(proc_img)
        stress_confidence = self._stress.confidence(proc_img, rgb_indices)

        # ── 5. Disease detection ───────────────────────────────────
        disease_boxes = self._disease.detect(proc_img)

        # ── 6. Crop stage ─────────────────────────────────────────
        crop_stage, stage_conf, _ = self._stage.classify(proc_img)

        # ── 7. Weed detection ─────────────────────────────────────
        weed_boxes, weed_mask, weed_coverage = self._weed.detect(proc_img)

        # ── 8. Stress heatmap ─────────────────────────────────────
        heatmap_rgb = None
        if self.generate_heatmap:
            heatmap_rgb = self._stress.generate_stress_heatmap(proc_img, rgb_indices)

        # ── 9. Composite confidence ───────────────────────────────
        composite_conf = round(
            0.50 * stress_confidence +
            0.25 * stage_conf +
            0.25 * quality_score,
            3
        )

        # Low confidence reason
        low_conf_reason = None
        if composite_conf < 0.50:
            if quality_score < 0.50:
                low_conf_reason = self._iqa.get_rejection_label(
                    quality.rejection_reason if quality and quality.rejection_reason else ""
                )
            elif stress_confidence < 0.40:
                low_conf_reason = "Low vegetation content — check viewpoint"

        # ── 10. Timing ────────────────────────────────────────────
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        meta = InferenceMetadata(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inference_time_ms=round(inference_ms, 2),
            confidence=composite_conf,
            quality_score=round(quality_score, 4),
            low_confidence_reason=low_conf_reason,
        )

        return RGBInferenceResult(
            crop_stress_score=round(stress_score, 4),
            stress_label=stress_label,
            disease_detections=disease_boxes,
            crop_stage=crop_stage,
            weed_detections=weed_boxes,
            weed_coverage_pct=round(weed_coverage, 2),
            grvi=round(float(np.nanmean(rgb_indices["grvi"])), 4),
            vari=round(float(np.nanmean(rgb_indices["vari"])), 4),
            exg=round(float(np.nanmean(rgb_indices["exg"])), 4),
            heatmap_rgb=heatmap_rgb,
            segmentation_mask=weed_mask,
            meta=meta,
        )

    def draw_detections(self, image_rgb: np.ndarray,
                         result: RGBInferenceResult,
                         show_disease: bool = True,
                         show_weed: bool = True,
                         show_heatmap: bool = False) -> np.ndarray:
        """
        Draw bounding boxes and overlay on the image.

        Returns HxWx3 uint8 annotated image.
        """
        if show_heatmap and result.heatmap_rgb is not None:
            # Use heatmap as base
            h, w = image_rgb.shape[:2]
            hm = cv2.resize(result.heatmap_rgb, (w, h))
            annotated = hm.copy()
        else:
            annotated = image_rgb.copy()

        h, w = annotated.shape[:2]

        def _scale_box(box: BoundingBox, src_w: int, src_h: int):
            """Scale box from processing resolution back to display resolution."""
            proc_w = min(640, src_w)
            proc_h = int(src_h * proc_w / src_w)
            sx = w / proc_w
            sy = h / proc_h
            return (int(box.x1 * sx), int(box.y1 * sy),
                    int(box.x2 * sx), int(box.y2 * sy))

        # Infer processing dimensions
        src_w = min(640, w)

        if show_disease:
            for box in result.disease_detections:
                x1, y1, x2, y2 = _scale_box(box, w, h)
                color_bgr = self._hex_to_bgr(box.color)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
                label = f"{box.label} {box.confidence:.0%}"
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        if show_weed:
            for box in result.weed_detections:
                x1, y1, x2, y2 = _scale_box(box, w, h)
                color_bgr = self._hex_to_bgr(box.color)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
                label = f"Weed {box.confidence:.0%}"
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        # Stress indicator in corner
        stress_colors = {
            "Healthy":        (34, 197, 94),
            "Mild Stress":    (234, 179, 8),
            "Moderate Stress":(249, 115, 22),
            "Severe Stress":  (239, 68, 68),
            "Critical Stress":(127, 29, 29),
        }
        sc = stress_colors.get(result.stress_label, (148, 163, 184))
        cv2.rectangle(annotated, (0, 0), (280, 40), (15, 23, 42), -1)
        cv2.putText(annotated,
                    f"Stress: {result.stress_label}  [{result.meta.confidence:.0%}]",
                    (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1)

        return annotated

    @staticmethod
    def _preprocess(image_rgb: np.ndarray, max_width: int = 640) -> np.ndarray:
        """Resize + denoise for consistent processing resolution."""
        h, w = image_rgb.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image_rgb

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)

    @staticmethod
    def _empty_result(quality_score: float, reason: str) -> RGBInferenceResult:
        return RGBInferenceResult(
            crop_stress_score=0.0,
            stress_label="Unknown",
            meta=InferenceMetadata(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                confidence=0.0,
                quality_score=quality_score,
                low_confidence_reason=reason,
            )
        )
