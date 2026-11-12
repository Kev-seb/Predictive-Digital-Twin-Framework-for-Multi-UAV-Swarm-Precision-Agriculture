"""
swarm_guidance.py
-----------------
Fleet bookkeeping and per-drone flight patterns for the dashboard's swarm
simulation (any number of drones).

Drone 0 ("drone_alpha") is the primary: it mirrors MAVLINK_TELEMETRY and is
the one that flies uploaded waypoint missions. Every pattern is written for
drone i of n and reproduces the original two-drone (alpha / beta) paths
exactly when n = 2.

Local frame: metres east (x) / north (y) of home, z up.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
          "iota", "kappa", "lambda", "mu")
# alpha / beta keep their original colours; the rest are distinct on a dark background.
_COLORS = ("#38bdf8", "#ec4899", "#a3e635", "#f59e0b", "#a78bfa", "#2dd4bf",
           "#f87171", "#facc15", "#60a5fa", "#fb923c", "#c084fc", "#4ade80")

MAX_DRONES = 12
FIELD_HALF_M = 27.5
SPRAY_NDVI_THRESHOLD = 0.35

Target = Tuple[np.ndarray, float, str]   # (target_pos xyz, target_yaw, autopilot_mode)


def drone_id(i: int) -> str:
    return f"drone_{_GREEK[i]}" if i < len(_GREEK) else f"drone_{i + 1:02d}"


def drone_ids(n: int) -> List[str]:
    return [drone_id(i) for i in range(max(1, min(n, MAX_DRONES)))]


def drone_color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def drone_label(i: int) -> str:
    name = _GREEK[i].capitalize() if i < len(_GREEK) else f"{i + 1:02d}"
    role = "Lead" if i == 0 else f"Sector {chr(65 + i)}" if i < 26 else "Support"
    return f"Drone {name} ({'Sector A' if i == 0 else role})"


def make_drone_record(i: int) -> Dict[str, Any]:
    return {"id": drone_id(i), "label": drone_label(i), "lat": 37.7749, "lon": -122.4194,
            "alt": 10.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0, "battery": 100.0,
            "speed": 0.0, "is_spraying": False, "color": drone_color(i)}


def spawn_position(i: int, n: int) -> np.ndarray:
    """Evenly on a 21.2 m circle starting south-west; n=2 gives (-15,-15) and (15,15)."""
    ang = math.radians(225.0) + 2.0 * math.pi * i / max(n, 1)
    r = 15.0 * math.sqrt(2.0)
    return np.array([r * math.cos(ang), r * math.sin(ang), 10.0])


def ensure_fleet(shared_state: Dict[str, Any], n: int, simulator_factory) -> List[str]:
    """
    Make PHYSICS_SIMULATORS and MULTIPLAYER_DRONES hold exactly ``n`` drones,
    adding (``simulator_factory(drone_id)``) or removing followers. The primary
    drone is never removed. Returns the ordered drone ids.
    """
    ids = drone_ids(n)
    sims = shared_state.setdefault("PHYSICS_SIMULATORS", {})
    records = shared_state.setdefault("MULTIPLAYER_DRONES", {})
    for i, d_id in enumerate(ids):
        if d_id not in sims:
            sim = simulator_factory(d_id)
            sim.pos = spawn_position(i, len(ids))
            sim.target_pos = sim.pos.copy()
            sims[d_id] = sim
        if d_id not in records:
            records[d_id] = make_drone_record(i)
    keep = set(ids)
    for d_id in [d for d in list(sims) if d not in keep]:
        del sims[d_id]
    for d_id in [d for d in list(records) if d not in keep]:
        del records[d_id]
    shared_state["PHYSICS_SIMULATOR"] = sims[ids[0]]
    return ids


# ======================================================================= patterns
def coordinated_spraying(i: int, n: int, t: float, sweep_time: float = 60.0,
                         cycles: float = 8.0, alt: float = 8.0) -> Target:
    """
    Continuous full-width serpentine sweeps. Even drones sweep north->south,
    odd drones south->north; pairs beyond the first start half a sweep (etc.)
    later so they do not fly in formation.
    """
    pairs = max(1, math.ceil(n / 2))
    phase = (i // 2) / pairs
    t_norm = ((t % sweep_time) / sweep_time + phase) % 1.0
    southward = i % 2 == 0
    y = 25.0 - t_norm * 50.0 if southward else -25.0 + t_norm * 50.0
    x = 25.0 * math.sin(t_norm * math.pi * 2.0 * cycles)
    dx = 25.0 * math.pi * 2.0 * cycles * math.cos(t_norm * math.pi * 2.0 * cycles)
    yaw = math.atan2(-50.0 if southward else 50.0, dx)
    return np.array([x, y, alt]), yaw, "terrain_follow"


def synchronized_scouting(i: int, n: int, t: float, alt: float = 12.0) -> Target:
    """Parallel sector scans: sector centres spread west->east across the field."""
    centre = -20.0 + (40.0 * i / (n - 1) if n > 1 else 20.0)
    amp = 15.0 if n <= 2 else min(15.0, 20.0 / (n - 1))
    if i % 2 == 0:
        x, y = centre + amp * math.sin(t * 0.2), 20.0 * math.cos(t * 0.05)
    else:
        x, y = centre + amp * math.cos(t * 0.2), 20.0 * math.sin(t * 0.05)
    return np.array([x, y, alt]), math.atan2(y, x), "terrain_follow"


def orbit_avoidance_test(i: int, n: int, t: float, alt: float = 10.0) -> Target:
    """Overlapping 12 m orbits around centres 8 m from the field centre, phases spread evenly."""
    phase = 2.0 * math.pi * i / max(n, 1)
    centre_ang = math.pi + phase
    cx, cy = 8.0 * math.cos(centre_ang), 8.0 * math.sin(centre_ang)
    x = cx + 12.0 * math.cos(t * 0.3 + phase)
    y = cy + 12.0 * math.sin(t * 0.3 + phase)
    return np.array([x, y, alt]), t * 0.3 + math.pi / 2 + phase, "stabilized"


def holding_orbit(i: int, n: int, t: float, alt: float = 10.0) -> Target:
    """18 m holding orbit, drones evenly spaced around it."""
    phase = 2.0 * math.pi * i / max(n, 1)
    x, y = 18.0 * math.sin(t * 0.4 + phase), 18.0 * math.cos(t * 0.4 + phase)
    return np.array([x, y, alt]), math.atan2(y, x), "stabilized"


PATTERNS = {
    "coordinated_spraying": coordinated_spraying,
    "synchronized_scouting": synchronized_scouting,
    "orbit_avoidance_test": orbit_avoidance_test,
}


def target_for(mode: str, i: int, n: int, t: float) -> Target:
    return PATTERNS.get(mode, holding_orbit)(i, n, t)


def spray_dose(ndvi_map: Optional[np.ndarray], x: float, y: float,
               field_half: float = FIELD_HALF_M, threshold: float = SPRAY_NDVI_THRESHOLD) -> float:
    """Stress-proportional dose (0 = off) from the NDVI under a drone at local (x, y)."""
    if ndvi_map is None or ndvi_map.size == 0:
        return 0.0
    h, w = ndvi_map.shape
    u = float(np.clip((x + field_half) / (2.0 * field_half), 0.0, 1.0))
    v = float(np.clip(1.0 - (y + field_half) / (2.0 * field_half), 0.0, 1.0))
    ndvi = float(ndvi_map[int(v * (h - 1)), int(u * (w - 1))])
    if ndvi >= threshold:
        return 0.0
    return float(np.clip((threshold - ndvi) / threshold, 0.0, 1.0))


def synthetic_ndvi_map(size: int = 64) -> np.ndarray:
    """Demo NDVI map with three stressed patches (used before any survey is processed)."""
    sy = np.full((size, size), 0.60, dtype=np.float32)
    sy[4:18, 4:20] = 0.15
    sy[24:38, 36:56] = 0.22
    sy[48:62, 8:28] = 0.18
    return sy
