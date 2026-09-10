"""
mission_database.py
-------------------
SQLite persistence layer for Live Field Mode missions.

Design principles:
  - PostgreSQL-ready: uses only standard SQL (no SQLite-specific extensions)
  - UUID primary keys (TEXT in SQLite, UUID in PostgreSQL)
  - JSONB stored as TEXT in SQLite (native JSONB type in PostgreSQL)
  - ISO-8601 timestamps (TEXT in SQLite, TIMESTAMPTZ in PostgreSQL)
  - Cascade rules on foreign keys

To migrate to PostgreSQL:
  1. Replace sqlite3 with psycopg2 / asyncpg
  2. Change TEXT → UUID for PK/FK columns
  3. Change TEXT → TIMESTAMPTZ for timestamp columns
  4. Change TEXT → JSONB for json_* columns
  5. Enable cascade constraints (already defined in schema)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("outputs/live_missions/missions.db")

# Thread-local connections for multi-thread safety
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        -- ── missions ────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS missions (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            started_at      TEXT,
            ended_at        TEXT,
            status          TEXT NOT NULL DEFAULT 'active',
            field_lat       REAL NOT NULL DEFAULT 0.0,
            field_lon       REAL NOT NULL DEFAULT 0.0,
            field_name      TEXT NOT NULL DEFAULT 'Field Alpha',
            sensor_source   TEXT NOT NULL DEFAULT 'phone_rgb',
            camera_profile  TEXT NOT NULL DEFAULT 'default',
            total_frames    INTEGER NOT NULL DEFAULT 0,
            frames_accepted INTEGER NOT NULL DEFAULT 0,
            frames_rejected INTEGER NOT NULL DEFAULT 0,
            total_distance_m    REAL NOT NULL DEFAULT 0.0,
            duration_seconds    REAL NOT NULL DEFAULT 0.0,
            mean_stress_score   REAL NOT NULL DEFAULT 0.0,
            max_stress_score    REAL NOT NULL DEFAULT 0.0,
            disease_count       INTEGER NOT NULL DEFAULT 0,
            alerts_warning      INTEGER NOT NULL DEFAULT 0,
            alerts_critical     INTEGER NOT NULL DEFAULT 0,
            area_covered_m2     REAL NOT NULL DEFAULT 0.0,
            json_satellite      TEXT,
            json_treatment      TEXT,
            json_twin_state     TEXT,
            report_pdf_path     TEXT,
            kml_path            TEXT,
            json_export_path    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- ── observations ────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS observations (
            id              TEXT PRIMARY KEY,
            mission_id      TEXT NOT NULL,
            timestamp       REAL NOT NULL,
            sequence_num    INTEGER NOT NULL DEFAULT 0,

            -- Fused telemetry
            lat             REAL NOT NULL DEFAULT 0.0,
            lon             REAL NOT NULL DEFAULT 0.0,
            alt             REAL NOT NULL DEFAULT 0.0,
            heading         REAL NOT NULL DEFAULT 0.0,
            pitch           REAL NOT NULL DEFAULT 0.0,
            roll            REAL NOT NULL DEFAULT 0.0,
            speed           REAL NOT NULL DEFAULT 0.0,
            battery         REAL NOT NULL DEFAULT 100.0,
            accuracy_m      REAL NOT NULL DEFAULT 5.0,
            signal_strength INTEGER NOT NULL DEFAULT 0,

            -- AI results
            crop_stress_score   REAL,
            stress_label        TEXT,
            crop_stage          TEXT,
            weed_coverage_pct   REAL,
            grvi                REAL,
            vari                REAL,
            exg                 REAL,
            model_name          TEXT,
            model_version       TEXT,
            model_confidence    REAL,
            inference_time_ms   REAL,

            -- Image quality
            quality_score       REAL,
            quality_passed      INTEGER,
            rejection_reason    TEXT,

            -- JSON blobs (JSONB in PostgreSQL)
            json_disease_detections TEXT,
            json_weed_detections    TEXT,
            json_weather            TEXT,
            json_inference_full     TEXT,

            -- File paths
            frame_original_path  TEXT,
            frame_annotated_path TEXT,
            inference_json_path  TEXT,

            FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_obs_mission_id ON observations(mission_id);
        CREATE INDEX IF NOT EXISTS idx_obs_timestamp  ON observations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_obs_stress     ON observations(crop_stress_score);

        -- ── gps_track ───────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS gps_track (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            lat         REAL NOT NULL,
            lon         REAL NOT NULL,
            alt         REAL NOT NULL DEFAULT 0.0,
            accuracy_m  REAL NOT NULL DEFAULT 5.0,
            FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_gps_mission ON gps_track(mission_id);

        -- ── alerts ──────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS alerts (
            id              TEXT PRIMARY KEY,
            mission_id      TEXT NOT NULL,
            timestamp       REAL NOT NULL,
            severity        TEXT NOT NULL DEFAULT 'warning',
            category        TEXT NOT NULL DEFAULT 'stress',
            message         TEXT NOT NULL,
            lat             REAL NOT NULL DEFAULT 0.0,
            lon             REAL NOT NULL DEFAULT 0.0,
            acknowledged    INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_mission ON alerts(mission_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

        -- ── weather_log ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS weather_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            temperature REAL,
            humidity    REAL,
            wind_speed  REAL,
            wind_dir    REAL,
            precipitation REAL,
            cloud_cover REAL,
            json_full   TEXT,
            FOREIGN KEY (mission_id) REFERENCES missions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_weather_mission ON weather_log(mission_id);
    """)
    conn.commit()


# ── Mission CRUD ───────────────────────────────────────────────────────

def create_mission(mission_id: str, name: str, field_lat: float, field_lon: float,
                   sensor_source: str = "phone_rgb", camera_profile: str = "default",
                   field_name: str = "Field Alpha") -> None:
    conn = _get_connection()
    conn.execute("""
        INSERT INTO missions (id, name, started_at, status, field_lat, field_lon,
                              field_name, sensor_source, camera_profile)
        VALUES (?, ?, datetime('now'), 'active', ?, ?, ?, ?, ?)
    """, (mission_id, name, field_lat, field_lon, field_name, sensor_source, camera_profile))
    conn.commit()


def complete_mission(mission_id: str) -> None:
    conn = _get_connection()
    conn.execute("""
        UPDATE missions
        SET status = 'completed', ended_at = datetime('now'), updated_at = datetime('now')
        WHERE id = ?
    """, (mission_id,))
    conn.commit()


def update_mission_stats(mission_id: str, stats: Dict[str, Any]) -> None:
    conn = _get_connection()
    conn.execute("""
        UPDATE missions SET
            total_frames = ?,
            frames_accepted = ?,
            frames_rejected = ?,
            total_distance_m = ?,
            duration_seconds = ?,
            mean_stress_score = ?,
            max_stress_score = ?,
            disease_count = ?,
            alerts_warning = ?,
            alerts_critical = ?,
            area_covered_m2 = ?,
            updated_at = datetime('now')
        WHERE id = ?
    """, (
        stats.get("total_frames", 0),
        stats.get("frames_accepted", 0),
        stats.get("frames_rejected", 0),
        stats.get("total_distance_m", 0.0),
        stats.get("duration_seconds", 0.0),
        stats.get("mean_stress_score", 0.0),
        stats.get("max_stress_score", 0.0),
        stats.get("disease_detections_total", 0),
        stats.get("alerts_warning", 0),
        stats.get("alerts_critical", 0),
        stats.get("area_covered_m2", 0.0),
        mission_id,
    ))
    conn.commit()


def get_mission(mission_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    row = conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
    return dict(row) if row else None


def list_missions(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM missions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_mission(mission_id: str) -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
    conn.commit()


# ── Observation CRUD ───────────────────────────────────────────────────

def insert_observation(obs_id: str, mission_id: str, timestamp: float,
                       seq: int, tele: Dict, inference: Optional[Dict],
                       quality: Optional[Dict], weather: Optional[Dict],
                       frame_original_path: Optional[str] = None,
                       frame_annotated_path: Optional[str] = None,
                       inference_json_path: Optional[str] = None) -> None:
    conn = _get_connection()
    inf = inference or {}
    q = quality or {}
    conn.execute("""
        INSERT INTO observations (
            id, mission_id, timestamp, sequence_num,
            lat, lon, alt, heading, pitch, roll, speed, battery, accuracy_m, signal_strength,
            crop_stress_score, stress_label, crop_stage, weed_coverage_pct,
            grvi, vari, exg, model_name, model_version, model_confidence, inference_time_ms,
            quality_score, quality_passed, rejection_reason,
            json_disease_detections, json_weed_detections, json_weather, json_inference_full,
            frame_original_path, frame_annotated_path, inference_json_path
        ) VALUES (
            ?,?,?,?,  ?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,  ?,?,?,?,?,?,?,  ?,?,?,
            ?,?,?,?,  ?,?,?
        )
    """, (
        obs_id, mission_id, timestamp, seq,
        tele.get("lat", 0.0), tele.get("lon", 0.0), tele.get("alt", 0.0),
        tele.get("heading", 0.0), tele.get("pitch", 0.0), tele.get("roll", 0.0),
        tele.get("speed", 0.0), tele.get("battery", 100.0),
        tele.get("accuracy_m", 5.0), tele.get("signal_strength", 0),
        inf.get("crop_stress_score"), inf.get("stress_label"),
        inf.get("crop_stage"), inf.get("weed_coverage_pct"),
        inf.get("grvi"), inf.get("vari"), inf.get("exg"),
        inf.get("model_name"), inf.get("model_version"),
        inf.get("model_confidence"), inf.get("inference_time_ms"),
        q.get("quality_score"), int(q.get("passed", True)),
        q.get("rejection_reason"),
        json.dumps(inf.get("disease_detections", [])),
        json.dumps(inf.get("weed_detections", [])),
        json.dumps(weather) if weather else None,
        json.dumps(inf) if inf else None,
        frame_original_path, frame_annotated_path, inference_json_path,
    ))
    conn.commit()


def get_observations(mission_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM observations WHERE mission_id = ? ORDER BY sequence_num ASC LIMIT ?",
        (mission_id, limit)
    ).fetchall()
    return [dict(r) for r in rows]


# ── GPS track ──────────────────────────────────────────────────────────

def insert_gps_point(mission_id: str, timestamp: float, lat: float, lon: float,
                     alt: float = 0.0, accuracy_m: float = 5.0) -> None:
    conn = _get_connection()
    conn.execute(
        "INSERT INTO gps_track (mission_id, timestamp, lat, lon, alt, accuracy_m) VALUES (?,?,?,?,?,?)",
        (mission_id, timestamp, lat, lon, alt, accuracy_m)
    )
    conn.commit()


def get_gps_track(mission_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM gps_track WHERE mission_id = ? ORDER BY timestamp ASC",
        (mission_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Alerts ─────────────────────────────────────────────────────────────

def insert_alert(alert_id: str, mission_id: str, timestamp: float,
                 severity: str, category: str, message: str,
                 lat: float = 0.0, lon: float = 0.0) -> None:
    conn = _get_connection()
    conn.execute("""
        INSERT INTO alerts (id, mission_id, timestamp, severity, category, message, lat, lon)
        VALUES (?,?,?,?,?,?,?,?)
    """, (alert_id, mission_id, timestamp, severity, category, message, lat, lon))
    conn.commit()


def get_alerts(mission_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM alerts WHERE mission_id = ? ORDER BY timestamp ASC",
        (mission_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Weather log ────────────────────────────────────────────────────────

def insert_weather(mission_id: str, timestamp: float, weather: Dict[str, Any]) -> None:
    conn = _get_connection()
    conn.execute("""
        INSERT INTO weather_log
            (mission_id, timestamp, temperature, humidity, wind_speed, wind_dir,
             precipitation, cloud_cover, json_full)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        mission_id, timestamp,
        weather.get("temperature"), weather.get("humidity"),
        weather.get("wind_speed"), weather.get("wind_direction"),
        weather.get("precipitation"), weather.get("cloud_cover"),
        json.dumps(weather),
    ))
    conn.commit()


def get_weather_log(mission_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM weather_log WHERE mission_id = ? ORDER BY timestamp ASC",
        (mission_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# Initialise on import
init_db()
