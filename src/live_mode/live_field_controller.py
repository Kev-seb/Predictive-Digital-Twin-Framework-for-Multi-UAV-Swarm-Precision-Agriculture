"""
live_field_controller.py
------------------------
Master session orchestrator for Live Field Mode.

Manages the lifecycle of one field walk session:
    1. Start mission → create DB record + recorder + alert engine
    2. Process frames → IQA → AI → record → alert check → broadcast
    3. Stop mission → update DB + generate report
    4. Replay mode → load observations from DB and replay at speed

Exposes a thread-safe interface used by both:
    - FastAPI backend (receives phone frames / telemetry)
    - Streamlit dashboard (reads current state for display)
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.streaming.frame_interface import UnifiedFrame, FrameSync
from src.streaming.frame_queue import (
    FrameQueue, DetectionQueue, TelemetryQueue, AlertQueue
)
from src.mission.mission_object import (
    Mission, FusedTelemetry, RGBInferenceResult, Alert
)
from src.mission import mission_database as db
from src.mission.mission_recorder import MissionRecorder
from src.live_mode.alert_engine import AlertEngine
from src.rgb_ai.rgb_inference_engine import RGBInferenceEngine
from src.vision.image_quality_assessor import ImageQualityAssessor
from src.vision.camera_calibrator import get_profile
from src.telemetry.phone_telemetry_adapter import PhoneTelemetryAdapter


class LiveFieldController:
    """
    Singleton-ish session controller for Live Field Mode.
    One instance per Streamlit session (via st.cache_resource).
    """

    def __init__(self):
        # Session state
        self._mission_id: Optional[str] = None
        self._active: bool = False
        self._paused: bool = False
        self._start_time: float = 0.0
        self._lock = threading.Lock()

        # Pipeline components
        self._engine      = RGBInferenceEngine(run_iqa=True, run_calibration=True)
        self._iqa         = ImageQualityAssessor()
        self._adapter     = PhoneTelemetryAdapter()
        self._recorder:   Optional[MissionRecorder] = None
        self._alerts:     Optional[AlertEngine] = None
        self._sync:       Optional[FrameSync] = None

        # Queues
        self.frame_q      = FrameQueue()
        self.detection_q  = DetectionQueue()
        self.telemetry_q  = TelemetryQueue()
        self.alert_q      = AlertQueue()

        # Latest state (thread-safe reads)
        self._latest_tele: FusedTelemetry = FusedTelemetry()
        self._latest_result: Optional[RGBInferenceResult] = None
        self._latest_frame_rgb: Optional[np.ndarray] = None
        self._latest_annotated_rgb: Optional[np.ndarray] = None
        self._latest_alerts: List[Dict] = []
        self._current_weather: Optional[Dict] = None

        # QoS metrics
        self._frames_total = 0
        self._frames_accepted = 0
        self._inference_times: List[float] = []
        self._last_frame_time: float = 0.0
        self._phone_connected: bool = False

        # Replay state
        self._replay_active: bool = False
        self._replay_observations: List[Dict] = []
        self._replay_index: int = 0
        self._replay_speed: float = 1.0
        self._replay_mission_id: Optional[str] = None

        # Processing thread
        self._proc_thread: Optional[threading.Thread] = None

    # ── Mission lifecycle ──────────────────────────────────────────

    def start_mission(
        self,
        name: str = "",
        field_lat: float = 0.0,
        field_lon: float = 0.0,
        field_name: str = "Field Alpha",
        camera_profile: str = "default",
    ) -> str:
        with self._lock:
            if self._active:
                return self._mission_id or ""

            mission_id = str(uuid.uuid4())
            self._mission_id = mission_id
            self._active = True
            self._paused = False
            self._start_time = time.time()

            # Create database record
            db.create_mission(
                mission_id=mission_id,
                name=name or f"Mission {time.strftime('%Y-%m-%d %H:%M')}",
                field_lat=field_lat,
                field_lon=field_lon,
                sensor_source="phone_rgb",
                camera_profile=camera_profile,
                field_name=field_name,
            )

            # Initialise components
            self._sync     = FrameSync(mission_id, source="phone_rgb")
            self._recorder = MissionRecorder(mission_id, engine=self._engine)
            self._alerts   = AlertEngine(mission_id)
            self._adapter.reset()

            # Reset queues
            self.frame_q.drain()
            self.detection_q.drain()
            self.telemetry_q.drain()
            self.alert_q.drain()
            self._latest_alerts.clear()
            self._inference_times.clear()
            self._frames_total = 0
            self._frames_accepted = 0

            # Start processing thread
            if self._proc_thread is None or not self._proc_thread.is_alive():
                self._proc_thread = threading.Thread(
                    target=self._processing_loop, daemon=True
                )
                self._proc_thread.start()

        return mission_id

    def stop_mission(self) -> Optional[str]:
        """Stop the active mission and return its ID."""
        with self._lock:
            if not self._active:
                return None
            mid = self._mission_id
            self._active = False
            db.complete_mission(mid)
        return mid

    def pause_mission(self) -> None:
        with self._lock:
            self._paused = True

    def resume_mission(self) -> None:
        with self._lock:
            self._paused = False

    # ── Data ingestion (called by FastAPI handlers) ────────────────

    def ingest_jpeg_frame(self, jpeg_bytes: bytes,
                           camera_profile: str = "default") -> bool:
        """Decode a JPEG frame and push it to the frame queue."""
        if not self._active or self._paused:
            return False
        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return False
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            tele = self._latest_tele
            frame = self._sync.stamp(
                image_rgb=img_rgb,
                gps=tele,
                heading=tele.heading,
                weather_snapshot=self._current_weather,
                camera_profile=camera_profile,
            )
            self._latest_frame_rgb = img_rgb
            self.frame_q.put(frame)
            self._last_frame_time = time.time()
            self._phone_connected = True
            return True
        except Exception:
            return False

    def ingest_telemetry(self, packet: Dict[str, Any]) -> FusedTelemetry:
        """Process a telemetry JSON packet from the phone."""
        tele = self._adapter.ingest(packet)
        self._latest_tele = tele
        self.telemetry_q.put(tele)

        # Mirror to existing MAVLINK_TELEMETRY format if shared_state available
        return tele

    def ingest_webrtc_frame(self, img_rgb: np.ndarray) -> bool:
        """Push a WebRTC-decoded frame directly to the queue."""
        if not self._active or self._paused:
            return False
        tele = self._latest_tele
        frame = self._sync.stamp(
            image_rgb=img_rgb,
            gps=tele,
            heading=tele.heading,
            weather_snapshot=self._current_weather,
        )
        self._latest_frame_rgb = img_rgb
        self.frame_q.put(frame)
        self._last_frame_time = time.time()
        self._phone_connected = True
        return True

    # ── Processing loop (background thread) ───────────────────────

    def _processing_loop(self) -> None:
        """Background thread: dequeue frames, run AI, store results."""
        while self._active or not self.frame_q.empty():
            frame: Optional[UnifiedFrame] = self.frame_q.get(timeout=0.3)
            if frame is None:
                continue

            if self._paused:
                continue

            try:
                # IQA
                quality = self._iqa.assess(frame.image_rgb)
                frame.quality = quality
                self._frames_total += 1

                # Run AI
                result = self._engine.infer(frame)

                if quality.passed:
                    self._frames_accepted += 1
                    # Track inference timing
                    self._inference_times.append(result.meta.inference_time_ms)
                    if len(self._inference_times) > 100:
                        self._inference_times.pop(0)

                    # Annotated frame
                    annotated = self._engine.draw_detections(
                        frame.image_rgb, result,
                        show_disease=True, show_weed=True, show_heatmap=False,
                    )
                    self._latest_annotated_rgb = annotated

                # Store result
                self._latest_result = result
                self.detection_q.put((frame, result))

                # Record to disk + DB
                if self._recorder:
                    self._recorder.record(frame, result, self._current_weather)

                # Alert check
                if self._alerts and self._latest_tele:
                    new_alerts = self._alerts.evaluate(result, self._latest_tele)
                    for a in new_alerts:
                        self._latest_alerts.append({
                            "id": a.id,
                            "severity": a.severity,
                            "category": a.category,
                            "message": a.message,
                            "timestamp": a.timestamp,
                        })
                        self.alert_q.put(a)
                    # Keep only last 50
                    if len(self._latest_alerts) > 50:
                        self._latest_alerts = self._latest_alerts[-50:]

            except Exception as e:
                pass  # Never crash the processing thread

    # ── Replay mode ───────────────────────────────────────────────

    def start_replay(self, mission_id: str, speed: float = 1.0) -> bool:
        """Load a completed mission and start replay mode."""
        observations = db.get_observations(mission_id, limit=5000)
        if not observations:
            return False

        self._replay_observations = observations
        self._replay_index = 0
        self._replay_active = True
        self._replay_speed = max(0.1, min(10.0, speed))
        self._replay_mission_id = mission_id

        thread = threading.Thread(target=self._replay_loop, daemon=True)
        thread.start()
        return True

    def stop_replay(self) -> None:
        self._replay_active = False

    def _replay_loop(self) -> None:
        """Replay observations at the specified speed."""
        observations = self._replay_observations
        if not observations:
            return

        for i in range(len(observations)):
            if not self._replay_active:
                break
            self._replay_index = i
            obs = observations[i]

            # Load annotated frame if it exists
            ann_path = obs.get("frame_annotated_path")
            if ann_path:
                full_path = Path("outputs/live_missions") / self._replay_mission_id / ann_path
                if full_path.exists():
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        self._latest_annotated_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Reconstruct telemetry
            self._latest_tele = FusedTelemetry(
                lat=obs.get("lat", 0.0),
                lon=obs.get("lon", 0.0),
                alt=obs.get("alt", 0.0),
                heading=obs.get("heading", 0.0),
                pitch=obs.get("pitch", 0.0),
                roll=obs.get("roll", 0.0),
                speed=obs.get("speed", 0.0),
                battery=obs.get("battery", 100.0),
                timestamp=obs.get("timestamp", time.time()),
            )

            # Reconstruct inference result
            inf_json = obs.get("json_inference_full")
            if inf_json:
                try:
                    inf_dict = json.loads(inf_json)
                    from src.mission.mission_object import InferenceMetadata
                    self._latest_result = RGBInferenceResult(
                        crop_stress_score=inf_dict.get("crop_stress_score", 0.0),
                        stress_label=inf_dict.get("stress_label", "Unknown"),
                        crop_stage=inf_dict.get("crop_stage", "Unknown"),
                        weed_coverage_pct=inf_dict.get("weed_coverage_pct", 0.0),
                        grvi=inf_dict.get("grvi", 0.0),
                        vari=inf_dict.get("vari", 0.0),
                        exg=inf_dict.get("exg", 0.0),
                        meta=InferenceMetadata(
                            model_name=inf_dict.get("model_name", "RGB Field AI"),
                            model_version=inf_dict.get("model_version", "v1.0"),
                            inference_time_ms=inf_dict.get("inference_time_ms", 0.0),
                            confidence=inf_dict.get("model_confidence", 0.0),
                        ),
                    )
                except Exception:
                    pass

            # Replay delay adjusted by speed
            time.sleep(0.5 / self._replay_speed)

        self._replay_active = False

    # ── State accessors (Streamlit reads these) ────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_replay_active(self) -> bool:
        return self._replay_active

    @property
    def replay_progress(self) -> float:
        if not self._replay_observations:
            return 0.0
        return self._replay_index / max(len(self._replay_observations) - 1, 1)

    @property
    def phone_connected(self) -> bool:
        if not self._phone_connected:
            return False
        return (time.time() - self._last_frame_time) < 5.0

    @property
    def latest_telemetry(self) -> FusedTelemetry:
        return self._latest_tele

    @property
    def latest_result(self) -> Optional[RGBInferenceResult]:
        return self._latest_result

    @property
    def latest_annotated_frame(self) -> Optional[np.ndarray]:
        return self._latest_annotated_rgb

    @property
    def latest_alerts(self) -> List[Dict]:
        return list(self._latest_alerts)

    @property
    def mission_id(self) -> Optional[str]:
        return self._mission_id

    @property
    def elapsed_seconds(self) -> float:
        if not self._active:
            return 0.0
        return time.time() - self._start_time

    @property
    def qos_metrics(self) -> Dict[str, Any]:
        inf_times = self._inference_times
        avg_inf_ms = sum(inf_times) / max(len(inf_times), 1)
        fps = self._frames_accepted / max(self.elapsed_seconds, 1.0)
        return {
            "frames_total":    self._frames_total,
            "frames_accepted": self._frames_accepted,
            "fps":             round(fps, 2),
            "avg_inference_ms": round(avg_inf_ms, 1),
            "phone_connected": self.phone_connected,
            "elapsed_s":       round(self.elapsed_seconds, 1),
        }

    def set_weather(self, weather: Dict[str, Any]) -> None:
        self._current_weather = weather
