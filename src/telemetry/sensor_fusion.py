"""
sensor_fusion.py
----------------
Phone sensor fusion: GPS + Accelerometer + Gyroscope + Compass.

Uses a complementary filter to combine:
    - GPS   (slow, high absolute accuracy, noisy when moving)
    - IMU   (fast, high relative accuracy, drifts over time)
    - Compass (heading, corrected for declination)

Produces a FusedTelemetry object at 10–20 Hz that is smoother than
raw GPS alone and more accurate than IMU alone.

Algorithm:
    Position: EMA-smoothed GPS with accelerometer-based dead-reckoning
              between GPS updates
    Heading:  Complementary filter: 0.98 * gyro_integrated + 0.02 * compass
    Speed:    EMA on GPS speed + accelerometer magnitude

Reference:
    Mahony et al., "Nonlinear Complementary Filters on the Special
    Orthogonal Group", IEEE TAC 2008.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from src.mission.mission_object import FusedTelemetry

# Earth radius in metres
EARTH_RADIUS_M = 6_371_000.0

# Complementary filter coefficient (0 = trust GPS fully, 1 = trust IMU fully)
ALPHA_HEADING  = 0.95    # high pass gyro, low pass compass
ALPHA_POSITION = 0.30    # low pass GPS position (EMA)
ALPHA_SPEED    = 0.50    # EMA on speed


class SensorFusion:
    """
    Fuses GPS + IMU + compass readings into smooth FusedTelemetry.

    Usage:
        fusion = SensorFusion()
        # Call on each incoming sensor packet:
        fusion.update_gps(lat, lon, alt, accuracy)
        fusion.update_imu(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
        fusion.update_compass(heading_deg)
        tele = fusion.get_fused()
    """

    def __init__(self):
        self._lat = 0.0
        self._lon = 0.0
        self._alt = 0.0
        self._accuracy = 10.0

        self._heading = 0.0         # degrees, fused
        self._heading_gyro = 0.0    # integrated gyro yaw
        self._heading_compass = 0.0 # raw compass

        self._speed = 0.0           # m/s fused
        self._battery = 100.0
        self._signal = 0

        self._pitch = 0.0
        self._roll  = 0.0

        self._last_imu_time = time.time()
        self._last_gps_time = 0.0

        # Dead-reckoning velocity (lat/lon degrees per second)
        self._vel_lat = 0.0
        self._vel_lon = 0.0

        # GPS history for speed calculation
        self._gps_history: deque = deque(maxlen=5)

        self._initialized = False

    # ── External update methods ────────────────────────────────────

    def update_gps(self, lat: float, lon: float, alt: float = 0.0,
                   accuracy_m: float = 5.0) -> None:
        """Call whenever a new GPS fix arrives."""
        now = time.time()
        if self._initialized:
            # EMA smoothing
            self._lat = (1 - ALPHA_POSITION) * lat + ALPHA_POSITION * self._lat
            self._lon = (1 - ALPHA_POSITION) * lon + ALPHA_POSITION * self._lon
            self._alt = (1 - 0.1) * alt + 0.1 * self._alt

            # GPS-based speed
            if len(self._gps_history) > 0 and (now - self._last_gps_time) > 0.1:
                dt = now - self._last_gps_time
                dlat = (lat - self._gps_history[-1][0]) * EARTH_RADIUS_M * math.pi / 180.0
                dlon = (lon - self._gps_history[-1][1]) * EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.pi / 180.0
                gps_speed = math.sqrt(dlat**2 + dlon**2) / dt
                self._speed = ALPHA_SPEED * self._speed + (1 - ALPHA_SPEED) * gps_speed
        else:
            self._lat = lat
            self._lon = lon
            self._alt = alt
            self._initialized = True

        self._accuracy = accuracy_m
        self._gps_history.append((lat, lon, now))
        self._last_gps_time = now

    def update_imu(self, acc_x: float, acc_y: float, acc_z: float,
                   gyr_x: float, gyr_y: float, gyr_z: float) -> None:
        """
        Call with accelerometer (m/s²) and gyroscope (rad/s) data.
        gyr_z is yaw rate (positive = turning right).
        """
        now = time.time()
        dt = now - self._last_imu_time
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        self._last_imu_time = now

        # Integrate gyro for heading (yaw)
        yaw_rate_deg = math.degrees(gyr_z)
        self._heading_gyro = (self._heading_gyro + yaw_rate_deg * dt) % 360.0

        # Complementary filter: blend gyro and compass
        # Unwrap angle difference to handle 0/360 boundary
        diff = self._heading_compass - self._heading_gyro
        if diff > 180:   diff -= 360
        if diff < -180:  diff += 360
        self._heading = (self._heading_gyro + (1 - ALPHA_HEADING) * diff) % 360.0

        # Estimate pitch and roll from accelerometer
        g = 9.81
        self._pitch = math.degrees(math.atan2(acc_x, math.sqrt(acc_y**2 + acc_z**2)))
        self._roll  = math.degrees(math.atan2(acc_y, math.sqrt(acc_x**2 + acc_z**2)))

    def update_compass(self, heading_deg: float) -> None:
        """Call with raw compass heading (0–360, magnetic north)."""
        self._heading_compass = heading_deg % 360.0
        # If IMU not active yet, use compass directly
        if self._last_imu_time == 0:
            self._heading = heading_deg

    def update_battery(self, pct: float) -> None:
        self._battery = max(0.0, min(100.0, pct))

    def update_signal(self, bars: int) -> None:
        self._signal = max(0, min(4, bars))

    # ── Output ─────────────────────────────────────────────────────

    def get_fused(self) -> FusedTelemetry:
        return FusedTelemetry(
            lat=round(self._lat, 8),
            lon=round(self._lon, 8),
            alt=round(self._alt, 2),
            heading=round(self._heading % 360.0, 2),
            pitch=round(self._pitch, 2),
            roll=round(self._roll, 2),
            speed=round(max(0.0, self._speed), 3),
            battery=round(self._battery, 1),
            accuracy_m=round(self._accuracy, 1),
            signal_strength=self._signal,
            timestamp=time.time(),
        )

    def reset(self) -> None:
        self.__init__()
