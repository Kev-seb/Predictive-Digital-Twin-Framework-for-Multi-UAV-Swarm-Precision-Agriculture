"""
field_store.py - fields, flights and tasks saved in SQLite (outputs/paddytwin.db), so a browser refresh
never loses work and a field can be reopened later. Each flight's GeoTIFF is copied to
outputs/fields/<field_id>/.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "outputs" / "paddytwin.db"
FILES = ROOT / "outputs" / "fields"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fields (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created REAL,
    lat REAL, lon REAL, width_m REAL, height_m REAL);
CREATE TABLE IF NOT EXISTS flights (id INTEGER PRIMARY KEY, field_id INTEGER NOT NULL, name TEXT, loaded_at REAL,
    path TEXT, ndvi_mean REAL, stress_mean REAL, units TEXT,
    FOREIGN KEY(field_id) REFERENCES fields(id));
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, field_id INTEGER NOT NULL, zone TEXT, action TEXT,
    dose TEXT, planned_day TEXT, priority INTEGER, status TEXT, done_on TEXT, note TEXT, updated REAL,
    FOREIGN KEY(field_id) REFERENCES fields(id));
"""


def _db(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or DB
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def create_field(name: str, lat=None, lon=None, width_m=None, height_m=None) -> int:
    with _db() as con:
        cur = con.execute("INSERT INTO fields(name, created, lat, lon, width_m, height_m) VALUES (?,?,?,?,?,?)",
                          (name, time.time(), lat, lon, width_m, height_m))
        return int(cur.lastrowid)


def add_flight(field_id: int, name: str, source_path, ndvi_mean: float, stress_mean: float, units: str = "") -> int:
    with _db() as con:
        n = con.execute("SELECT COUNT(*) FROM flights WHERE field_id=?", (field_id,)).fetchone()[0] + 1
        dest = FILES / str(field_id) / f"flight{n:02d}_{Path(name).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source_path and Path(source_path).exists():
            shutil.copyfile(source_path, dest)
        cur = con.execute("INSERT INTO flights(field_id, name, loaded_at, path, ndvi_mean, stress_mean, units) "
                          "VALUES (?,?,?,?,?,?,?)", (field_id, name, time.time(), str(dest), ndvi_mean, stress_mean,
                                                     units))
        return int(cur.lastrowid)


def list_fields() -> List[Dict[str, Any]]:
    with _db() as con:
        rows = con.execute("SELECT f.*, COUNT(l.id) AS flights, MAX(l.loaded_at) AS last_flight FROM fields f "
                           "LEFT JOIN flights l ON l.field_id = f.id GROUP BY f.id ORDER BY last_flight DESC").fetchall()
        return [dict(r) for r in rows]


def flights(field_id: int) -> List[Dict[str, Any]]:
    with _db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM flights WHERE field_id=? ORDER BY loaded_at", (field_id,))]


def save_tasks(field_id: int, rows: List[Dict[str, Any]]) -> None:
    with _db() as con:
        con.execute("DELETE FROM tasks WHERE field_id=?", (field_id,))
        con.executemany("INSERT INTO tasks(field_id, zone, action, dose, planned_day, priority, status, done_on, note, "
                        "updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [(field_id, r.get("Zone"), r.get("Action"), r.get("Dose"), r.get("Planned day"),
                          int(r.get("Priority") or 0), r.get("Status"), str(r["Done on"]) if r.get("Done on") else None,
                          r.get("Note") or "", time.time()) for r in rows])


def load_tasks(field_id: int) -> List[Dict[str, Any]]:
    with _db() as con:
        rows = con.execute("SELECT * FROM tasks WHERE field_id=? ORDER BY id", (field_id,)).fetchall()
    return [{"Zone": r["zone"], "Action": r["action"], "Dose": r["dose"], "Planned day": r["planned_day"],
             "Priority": r["priority"], "Status": r["status"], "Done on": r["done_on"], "Note": r["note"]} for r in rows]


def delete_field(field_id: int) -> None:
    with _db() as con:
        con.execute("DELETE FROM tasks WHERE field_id=?", (field_id,))
        con.execute("DELETE FROM flights WHERE field_id=?", (field_id,))
        con.execute("DELETE FROM fields WHERE id=?", (field_id,))
    shutil.rmtree(FILES / str(field_id), ignore_errors=True)
