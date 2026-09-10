"""
mission_recorder.py
--------------------
Background thread that consumes inference results from the DetectionQueue
and writes them to the mission database + saves frame images.

Storage layout:
    outputs/live_missions/
        <mission_id>/
            frames/
                original/
                    <seq>_<frame_id>.jpg     ← original RGB frame
                annotated/
                    <seq>_<frame_id>.jpg     ← annotated with bounding boxes
            inference/
                <seq>_<frame_id>.json        ← full inference JSON
            missions.db                      ← SQLite database (shared)

Every observation is stored with:
    - Original RGB frame
    - Annotated frame (with detections drawn)
    - Full inference JSON (for model retraining / audit)
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.mission.mission_object import (
    Mission, Observation, FusedTelemetry, RGBInferenceResult
)
from src.mission import mission_database as db
from src.rgb_ai.rgb_inference_engine import RGBInferenceEngine
from src.streaming.frame_interface import UnifiedFrame


MISSION_DIR = Path("outputs/live_missions")


class MissionRecorder:
    """
    Background worker that saves frames + inference results for a mission.
    """

    def __init__(self, mission_id: str, engine: Optional[RGBInferenceEngine] = None):
        self.mission_id = mission_id
        self._engine = engine
        self._seq = 0

        # Create mission directory structure
        self._mission_dir = MISSION_DIR / mission_id
        self._frames_orig_dir = self._mission_dir / "frames" / "original"
        self._frames_ann_dir  = self._mission_dir / "frames" / "annotated"
        self._inf_dir         = self._mission_dir / "inference"

        for d in [self._frames_orig_dir, self._frames_ann_dir, self._inf_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Distance tracking
        self._last_lat: Optional[float] = None
        self._last_lon: Optional[float] = None
        self._total_dist = 0.0

        # Stats accumulators
        self._stress_accum  = 0.0
        self._max_stress    = 0.0
        self._frames_total  = 0
        self._frames_ok     = 0
        self._disease_count = 0
        self._start_time    = time.time()

    # ── Main recording method ──────────────────────────────────────

    def record(
        self,
        frame: UnifiedFrame,
        result: RGBInferenceResult,
        weather: Optional[dict] = None,
    ) -> Observation:
        """
        Save a frame + inference result to disk and database.
        Returns the created Observation.
        """
        self._seq += 1
        self._frames_total += 1
        obs_id = str(uuid.uuid4())
        stem = f"{self._seq:06d}_{frame.frame_id[:8]}"

        # ── 1. Save original frame ─────────────────────────────────
        orig_path: Optional[str] = None
        if frame.image_rgb is not None:
            orig_rel = f"frames/original/{stem}.jpg"
            orig_abs = self._mission_dir / orig_rel
            img_bgr = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(orig_abs), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            orig_path = str(orig_rel)

        # ── 2. Save annotated frame ────────────────────────────────
        ann_path: Optional[str] = None
        if frame.image_rgb is not None and self._engine is not None:
            annotated = self._engine.draw_detections(
                frame.image_rgb, result,
                show_disease=True, show_weed=True, show_heatmap=False,
            )
            ann_rel = f"frames/annotated/{stem}.jpg"
            ann_abs = self._mission_dir / ann_rel
            ann_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(ann_abs), ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ann_path = str(ann_rel)

        # ── 3. Save inference JSON ─────────────────────────────────
        inf_path: Optional[str] = None
        inf_dict = result.to_dict()
        inf_dict.update(frame.to_metadata_dict())
        inf_dict["weather"] = weather
        inf_rel = f"inference/{stem}.json"
        inf_abs = self._mission_dir / inf_rel
        with open(inf_abs, "w") as f:
            json.dump(inf_dict, f, indent=2)
        inf_path = str(inf_rel)

        # ── 4. Update distance accumulator ────────────────────────
        tele = frame.gps
        if tele and tele.lat and tele.lon:
            if self._last_lat is not None:
                self._total_dist += _haversine_m(
                    self._last_lat, self._last_lon, tele.lat, tele.lon
                )
            self._last_lat = tele.lat
            self._last_lon = tele.lon

            # GPS track
            db.insert_gps_point(
                mission_id=self.mission_id,
                timestamp=tele.timestamp or time.time(),
                lat=tele.lat, lon=tele.lon,
                alt=tele.alt, accuracy_m=tele.accuracy_m,
            )

        # ── 5. Update stats ────────────────────────────────────────
        passed = result.meta.quality_score > 0.3 and result.meta.confidence > 0.1
        if passed:
            self._frames_ok += 1
            self._stress_accum += result.crop_stress_score
            self._max_stress = max(self._max_stress, result.crop_stress_score)
            self._disease_count += len(result.disease_detections)

        # ── 6. Insert observation to DB ───────────────────────────
        tele_dict = tele.to_dict() if tele else {}
        quality_dict = {
            "passed": passed,
            "quality_score": result.meta.quality_score,
            "rejection_reason": result.meta.low_confidence_reason,
        }
        db.insert_observation(
            obs_id=obs_id,
            mission_id=self.mission_id,
            timestamp=frame.timestamp,
            seq=self._seq,
            tele=tele_dict,
            inference=inf_dict,
            quality=quality_dict,
            weather=weather,
            frame_original_path=orig_path,
            frame_annotated_path=ann_path,
            inference_json_path=inf_path,
        )

        # ── 7. Update mission statistics ───────────────────────────
        elapsed = time.time() - self._start_time
        mean_stress = (self._stress_accum / max(self._frames_ok, 1))
        db.update_mission_stats(self.mission_id, {
            "total_frames":           self._frames_total,
            "frames_accepted":        self._frames_ok,
            "frames_rejected":        self._frames_total - self._frames_ok,
            "total_distance_m":       self._total_dist,
            "duration_seconds":       elapsed,
            "mean_stress_score":      mean_stress,
            "max_stress_score":       self._max_stress,
            "disease_detections_total": self._disease_count,
        })

        return Observation(
            id=obs_id,
            mission_id=self.mission_id,
            timestamp=frame.timestamp,
            sequence_num=self._seq,
            telemetry=tele or FusedTelemetry(),
            inference=result,
            frame_original_path=orig_path,
            frame_annotated_path=ann_path,
            inference_json_path=inf_path,
            weather=weather,
        )

    @property
    def stats(self) -> dict:
        elapsed = time.time() - self._start_time
        mean_stress = self._stress_accum / max(self._frames_ok, 1)
        return {
            "total_frames": self._frames_total,
            "frames_accepted": self._frames_ok,
            "total_distance_m": round(self._total_dist, 2),
            "duration_seconds": round(elapsed, 1),
            "mean_stress": round(mean_stress, 4),
            "max_stress": round(self._max_stress, 4),
        }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
