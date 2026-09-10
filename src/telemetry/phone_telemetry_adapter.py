"""
phone_telemetry_adapter.py
--------------------------
Normalizes phone sensor JSON payloads into the MAVLink-compatible
FusedTelemetry structure used throughout the platform.

The phone PWA sends JSON messages over WebSocket in this format:
{
    "type": "telemetry",
    "ts":   1720000000.123,         # Unix epoch float
    "gps":  { "lat": 11.0, "lon": 79.0, "alt": 0.0, "acc": 5.0 },
    "imu":  { "ax": 0.1, "ay": 0.2, "az": 9.8,      # accelerometer m/s²
               "gx": 0.0, "gy": 0.0, "gz": 0.01 },  # gyroscope rad/s
    "compass": 45.3,        # degrees
    "battery":  82.0,       # %
    "signal":   3,          # bars 0-4
}

This adapter also feeds the SensorFusion engine and returns the result.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.mission.mission_object import FusedTelemetry
from src.telemetry.sensor_fusion import SensorFusion


class PhoneTelemetryAdapter:
    """
    Converts incoming phone JSON packets into FusedTelemetry using
    the SensorFusion engine.

    One instance per session / per phone connection.
    """

    def __init__(self):
        self._fusion = SensorFusion()
        self._last_packet_time = 0.0
        self._packets_received = 0

    def ingest(self, packet: Dict[str, Any]) -> FusedTelemetry:
        """
        Parse a raw phone telemetry packet and feed sensor fusion.

        Returns the latest FusedTelemetry after fusion.
        """
        self._last_packet_time = time.time()
        self._packets_received += 1

        # ── GPS ────────────────────────────────────────────────────
        gps = packet.get("gps", {})
        if gps:
            self._fusion.update_gps(
                lat=float(gps.get("lat", 0.0)),
                lon=float(gps.get("lon", 0.0)),
                alt=float(gps.get("alt", 0.0)),
                accuracy_m=float(gps.get("acc", 10.0)),
            )

        # ── IMU ────────────────────────────────────────────────────
        imu = packet.get("imu", {})
        if imu:
            self._fusion.update_imu(
                acc_x=float(imu.get("ax", 0.0)),
                acc_y=float(imu.get("ay", 0.0)),
                acc_z=float(imu.get("az", 9.81)),
                gyr_x=float(imu.get("gx", 0.0)),
                gyr_y=float(imu.get("gy", 0.0)),
                gyr_z=float(imu.get("gz", 0.0)),
            )

        # ── Compass ────────────────────────────────────────────────
        compass = packet.get("compass")
        if compass is not None:
            self._fusion.update_compass(float(compass))

        # ── Battery ────────────────────────────────────────────────
        battery = packet.get("battery")
        if battery is not None:
            self._fusion.update_battery(float(battery))

        # ── Signal ────────────────────────────────────────────────
        signal = packet.get("signal")
        if signal is not None:
            self._fusion.update_signal(int(signal))

        return self._fusion.get_fused()

    def to_mavlink_dict(self, tele: FusedTelemetry) -> Dict[str, Any]:
        """
        Convert FusedTelemetry to the MAVLINK_TELEMETRY dict format
        already used by the existing dashboard for drop-in compatibility.
        """
        return {
            "lat": tele.lat,
            "lon": tele.lon,
            "alt": tele.alt,
            "pitch": tele.pitch,
            "roll": tele.roll,
            "yaw": tele.heading,
            "battery": tele.battery,
            "speed": tele.speed,
            "connected": True,
            "is_spraying": False,
            "payload_mass": 0.0,
            "autopilot_mode": "live_field",
            "source": "phone",
            "accuracy_m": tele.accuracy_m,
        }

    @property
    def is_stale(self) -> bool:
        """True if no packet received in the last 5 seconds."""
        return (time.time() - self._last_packet_time) > 5.0

    @property
    def packets_received(self) -> int:
        return self._packets_received

    def reset(self) -> None:
        self._fusion.reset()
        self._last_packet_time = 0.0
        self._packets_received = 0
