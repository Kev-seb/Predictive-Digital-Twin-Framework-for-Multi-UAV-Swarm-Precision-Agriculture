"""
alert_engine.py
---------------
Real-time alert generation for Live Field Mode.

Alert rules:
    CRITICAL:
        - Stress score > 0.75
        - Disease confidence > 0.80
        - Battery < 10%
        - 3+ critical stress observations within 50m radius

    WARNING:
        - Stress score > 0.55
        - Disease detected (any confidence > threshold)
        - Battery < 25%
        - Weed coverage > 30%
        - Low AI confidence for 10+ consecutive frames

Alerts are stored in the database and displayed in the dashboard.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from typing import List, Optional

from src.mission.mission_object import Alert, RGBInferenceResult, FusedTelemetry
from src.mission import mission_database as db


# ── Thresholds ─────────────────────────────────────────────────────────
STRESS_CRITICAL    = 0.75
STRESS_WARNING     = 0.55
DISEASE_CRITICAL   = 0.80
DISEASE_WARNING    = 0.45
WEED_WARNING       = 30.0   # coverage %
BATTERY_CRITICAL   = 10.0
BATTERY_WARNING    = 25.0
LOW_CONFIDENCE     = 0.40
LOW_CONF_FRAMES    = 10     # consecutive low-confidence frames before alert
HOTSPOT_RADIUS_M   = 50.0
HOTSPOT_COUNT      = 3


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class AlertEngine:
    """
    Analyses each inference result against all alert rules and
    generates Alert objects when thresholds are exceeded.
    """

    def __init__(self, mission_id: str):
        self.mission_id = mission_id
        self._battery_alerted_critical = False
        self._battery_alerted_warning  = False
        self._low_conf_counter         = 0
        self._stress_hotspots: deque   = deque(maxlen=50)
        self._last_alert_time: dict    = {}   # category → last timestamp

    def evaluate(
        self,
        inference: RGBInferenceResult,
        telemetry: FusedTelemetry,
    ) -> List[Alert]:
        """
        Evaluate an inference result against all alert rules.

        Returns a list of new alerts (may be empty).
        """
        alerts: List[Alert] = []

        lat = telemetry.lat
        lon = telemetry.lon
        ts  = time.time()

        # ── 1. Stress alerts ───────────────────────────────────────
        if inference.crop_stress_score >= STRESS_CRITICAL:
            if self._cooldown_ok("stress_critical", 30.0):
                a = Alert(
                    id=str(uuid.uuid4()),
                    mission_id=self.mission_id,
                    timestamp=ts,
                    severity="critical",
                    category="stress",
                    message=(f"⚠ CRITICAL STRESS: {inference.stress_label} "
                             f"(Score: {inference.crop_stress_score:.2f}) detected at "
                             f"GPS {lat:.5f}, {lon:.5f}"),
                    lat=lat, lon=lon,
                )
                alerts.append(a)
                self._stress_hotspots.append((lat, lon, ts))

        elif inference.crop_stress_score >= STRESS_WARNING:
            if self._cooldown_ok("stress_warning", 60.0):
                a = Alert(
                    id=str(uuid.uuid4()),
                    mission_id=self.mission_id,
                    timestamp=ts,
                    severity="warning",
                    category="stress",
                    message=(f"⚡ Elevated Stress: {inference.stress_label} "
                             f"(Score: {inference.crop_stress_score:.2f})"),
                    lat=lat, lon=lon,
                )
                alerts.append(a)

        # ── 2. Disease alerts ──────────────────────────────────────
        for det in inference.disease_detections:
            if det.confidence >= DISEASE_CRITICAL:
                if self._cooldown_ok(f"disease_critical_{det.label}", 45.0):
                    a = Alert(
                        id=str(uuid.uuid4()),
                        mission_id=self.mission_id,
                        timestamp=ts,
                        severity="critical",
                        category="disease",
                        message=(f"🔴 DISEASE DETECTED: {det.label} — "
                                  f"Confidence {det.confidence:.0%}. Immediate inspection recommended."),
                        lat=lat, lon=lon,
                    )
                    alerts.append(a)

            elif det.confidence >= DISEASE_WARNING:
                if self._cooldown_ok(f"disease_warning_{det.label}", 90.0):
                    a = Alert(
                        id=str(uuid.uuid4()),
                        mission_id=self.mission_id,
                        timestamp=ts,
                        severity="warning",
                        category="disease",
                        message=(f"🟡 Possible {det.label} — Confidence {det.confidence:.0%}. "
                                  f"Monitor area closely."),
                        lat=lat, lon=lon,
                    )
                    alerts.append(a)

        # ── 3. Weed coverage alert ─────────────────────────────────
        if inference.weed_coverage_pct >= WEED_WARNING:
            if self._cooldown_ok("weed_coverage", 120.0):
                a = Alert(
                    id=str(uuid.uuid4()),
                    mission_id=self.mission_id,
                    timestamp=ts,
                    severity="warning",
                    category="weed",
                    message=(f"🌿 High Weed Coverage: {inference.weed_coverage_pct:.1f}% "
                              f"detected. Herbicide treatment recommended."),
                    lat=lat, lon=lon,
                )
                alerts.append(a)

        # ── 4. Battery alerts ─────────────────────────────────────
        if telemetry.battery <= BATTERY_CRITICAL and not self._battery_alerted_critical:
            a = Alert(
                id=str(uuid.uuid4()),
                mission_id=self.mission_id,
                timestamp=ts,
                severity="critical",
                category="battery",
                message=f"🔋 CRITICAL BATTERY: {telemetry.battery:.0f}% — End mission immediately!",
                lat=lat, lon=lon,
            )
            alerts.append(a)
            self._battery_alerted_critical = True

        elif telemetry.battery <= BATTERY_WARNING and not self._battery_alerted_warning:
            a = Alert(
                id=str(uuid.uuid4()),
                mission_id=self.mission_id,
                timestamp=ts,
                severity="warning",
                category="battery",
                message=f"🔋 Low Battery: {telemetry.battery:.0f}% — Consider ending mission soon.",
                lat=lat, lon=lon,
            )
            alerts.append(a)
            self._battery_alerted_warning = True

        # Reset battery alert when charged (for multi-session safety)
        if telemetry.battery > BATTERY_WARNING + 5:
            self._battery_alerted_warning = False
        if telemetry.battery > BATTERY_CRITICAL + 5:
            self._battery_alerted_critical = False

        # ── 5. Low confidence alert ────────────────────────────────
        if inference.meta.confidence < LOW_CONFIDENCE:
            self._low_conf_counter += 1
        else:
            self._low_conf_counter = 0

        if self._low_conf_counter >= LOW_CONF_FRAMES:
            if self._cooldown_ok("low_confidence", 30.0):
                reason = inference.meta.low_confidence_reason or "Check lighting and camera angle"
                a = Alert(
                    id=str(uuid.uuid4()),
                    mission_id=self.mission_id,
                    timestamp=ts,
                    severity="warning",
                    category="quality",
                    message=f"📷 Low AI Confidence ({inference.meta.confidence:.0%}) — {reason}",
                    lat=lat, lon=lon,
                )
                alerts.append(a)
                self._low_conf_counter = 0

        # ── 6. Stress hotspot cluster alert ───────────────────────
        if len(self._stress_hotspots) >= HOTSPOT_COUNT:
            recent = [(lat2, lon2) for lat2, lon2, t2 in self._stress_hotspots
                      if (ts - t2) < 120.0]
            if len(recent) >= HOTSPOT_COUNT:
                # Check if they are all within HOTSPOT_RADIUS_M of each other
                cluster = self._is_cluster(recent)
                if cluster and self._cooldown_ok("hotspot_cluster", 120.0):
                    a = Alert(
                        id=str(uuid.uuid4()),
                        mission_id=self.mission_id,
                        timestamp=ts,
                        severity="critical",
                        category="hotspot",
                        message=(f"🎯 STRESS CLUSTER: {len(recent)} critical stress zones detected "
                                  f"within {HOTSPOT_RADIUS_M:.0f}m. Intervention strongly recommended."),
                        lat=lat, lon=lon,
                    )
                    alerts.append(a)

        # ── Persist to database ────────────────────────────────────
        for alert in alerts:
            db.insert_alert(
                alert_id=alert.id,
                mission_id=alert.mission_id,
                timestamp=alert.timestamp,
                severity=alert.severity,
                category=alert.category,
                message=alert.message,
                lat=alert.lat,
                lon=alert.lon,
            )

        return alerts

    def _cooldown_ok(self, key: str, seconds: float) -> bool:
        """Return True if enough time has passed since the last alert of this type."""
        now = time.time()
        last = self._last_alert_time.get(key, 0.0)
        if (now - last) >= seconds:
            self._last_alert_time[key] = now
            return True
        return False

    @staticmethod
    def _is_cluster(points: list) -> bool:
        """Check if all points are within HOTSPOT_RADIUS_M of the centroid."""
        if len(points) < 2:
            return False
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        clat, clon = sum(lats)/len(lats), sum(lons)/len(lons)
        return all(
            _haversine_m(p[0], p[1], clat, clon) <= HOTSPOT_RADIUS_M
            for p in points
        )
