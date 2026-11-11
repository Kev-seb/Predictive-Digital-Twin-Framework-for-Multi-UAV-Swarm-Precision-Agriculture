"""
swarm.py
--------
N-drone fleet for the paddy digital twin: task generation, allocation and
physics-based execution.

* ``Fleet``           any number of drones, each a UAVFlightDynamicsSimulator
                      sharing one ``shared_state`` (so the existing
                      potential-field collision avoidance works across the fleet)
* ``tasks_from_twin`` scouting tasks (one per zone) or spray tasks (zones
                      whose fungal load exceeds a threshold, dose ∝ load)
* ``allocate``        greedy min-makespan assignment that respects battery
                      reserve and spray-tank capacity; tasks no drone can
                      take are returned as ``unassigned``
* ``Fleet.execute``   flies the plan in 6-DOF physics at 20 Hz; scouting legs
                      trigger camera captures every ``capture_spacing_m``,
                      completed spray tasks become simulator interventions

Local frame: metres east (x) / north (y) of the fleet's launch pad, z up,
matching flight_physics.py and the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.twin_core.field_state import EARTH_RADIUS_M, FieldGrid

# Energy model distilled from UAVFlightDynamicsSimulator.step():
#   drain %/s = 0.04 + 0.065 * thrust_ratio (+0.045 while spraying); thrust_ratio ~ 1
DRAIN_FLY_PCT_S = 0.105
DRAIN_SPRAY_PCT_S = 0.045
SPRAY_RATE_KG_S = 0.12          # payload depletion while spraying
BATTERY_RESERVE_PCT = 35.0      # physics evaluates RTL below 30 %
PLANNING_SPEED_M_S = 4.0        # effective ground speed incl. turns (calibrated in tests)


@dataclass
class DroneSpec:
    tank_kg: float = 10.0
    scout_alt_m: float = 10.0
    spray_alt_m: float = 3.0
    spray_swath_m: float = 5.0
    scout_lane_m: float = 10.0
    capture_spacing_m: float = 5.0


@dataclass
class Task:
    task_id: str
    kind: str                       # "scout" | "spray"
    zone_id: int
    legs: List[Tuple[float, float]]  # local (x, y) path through the zone
    priority: float = 0.0
    payload_kg: float = 0.0
    alt_m: float = 10.0

    @property
    def work_length_m(self) -> float:
        return sum(_dist(a, b) for a, b in zip(self.legs, self.legs[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "kind": self.kind, "zone_id": self.zone_id,
                "priority": round(self.priority, 4), "payload_kg": round(self.payload_kg, 3),
                "alt_m": self.alt_m, "work_length_m": round(self.work_length_m, 1),
                "n_legs": len(self.legs)}


@dataclass
class Assignment:
    drone_id: str
    tasks: List[Task] = field(default_factory=list)
    est_time_s: float = 0.0
    est_battery_pct: float = 0.0
    payload_kg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"drone_id": self.drone_id, "tasks": [t.task_id for t in self.tasks],
                "est_time_s": round(self.est_time_s, 1),
                "est_battery_pct": round(self.est_battery_pct, 1),
                "payload_kg": round(self.payload_kg, 3)}


@dataclass
class Plan:
    assignments: Dict[str, Assignment]
    unassigned: List[Task]

    @property
    def makespan_s(self) -> float:
        return max((a.est_time_s for a in self.assignments.values()), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"makespan_s": round(self.makespan_s, 1),
                "assignments": {k: a.to_dict() for k, a in self.assignments.items()},
                "unassigned": [t.to_dict() for t in self.unassigned]}


# ====================================================================== geometry
def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LocalFrame:
    """lat/lon <-> metres east/north of a home point."""

    def __init__(self, home_lat: float, home_lon: float):
        self.home_lat, self.home_lon = home_lat, home_lon
        self.m_lat = math.radians(1.0) * EARTH_RADIUS_M
        self.m_lon = self.m_lat * math.cos(math.radians(home_lat))

    def to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        return (lon - self.home_lon) * self.m_lon, (lat - self.home_lat) * self.m_lat

    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        return self.home_lat + y / self.m_lat, self.home_lon + x / self.m_lon


def zone_bounds_local(grid: FieldGrid, zone_labels: np.ndarray, zone_id: int,
                      frame: LocalFrame) -> Tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of a zone in the local frame."""
    rows, cols = np.nonzero(zone_labels == zone_id)
    nw = grid.cell_center_latlon(int(rows.min()), int(cols.min()))
    se = grid.cell_center_latlon(int(rows.max()), int(cols.max()))
    half = grid.cell_size_m / 2.0
    x0, y1 = frame.to_local(*nw)
    x1, y0 = frame.to_local(*se)
    return x0 - half, x1 + half, y0 - half, y1 + half


def lawnmower(bounds: Tuple[float, float, float, float], spacing: float) -> List[Tuple[float, float]]:
    """Lane endpoints covering a rectangle, lanes running east-west."""
    x0, x1, y0, y1 = bounds
    n = max(1, int(round((y1 - y0) / spacing)))
    ys = [y1 - (i + 0.5) * (y1 - y0) / n for i in range(n)]
    pts: List[Tuple[float, float]] = []
    for i, y in enumerate(ys):
        a, b = (x0 + spacing / 2, x1 - spacing / 2)
        pts.extend([(a, y), (b, y)] if i % 2 == 0 else [(b, y), (a, y)])
    return pts


# ========================================================================= tasks
def tasks_from_twin(twin, kind: str, frame: LocalFrame, spec: DroneSpec = DroneSpec(),
                    fungus_threshold: float = 0.2, dose_l_per_ha: float = 25.0,
                    zones: Optional[List[int]] = None, min_infected_frac: float = 0.1) -> List[Task]:
    """
    ``scout``: one lawn-mower task per zone (priority = mean uncertainty, so
    the least-known zones are visited first).
    ``spray``: zones where at least ``min_infected_frac`` of the area has a
    fungal load above ``fungus_threshold`` (an action threshold on infected
    area, so a hotspot is not averaged away by healthy surroundings). Dose
    scales with the infected cells' load (1 L ≈ 1 kg); priority = infected
    fraction x load.
    """
    if kind not in ("scout", "spray"):
        raise ValueError("kind must be 'scout' or 'spray'")
    grid, labels = twin.state.grid, twin.zone_labels
    fungus = twin.state.mean["fungus"]
    uncertainty = np.sqrt(twin.state.var["stress"])
    cell_ha = grid.cell_size_m ** 2 / 10_000.0
    tasks: List[Task] = []
    for z_id in sorted(twin.zone_names):
        if zones is not None and z_id not in zones:
            continue
        mask = labels == z_id
        if not np.any(mask):
            continue
        bounds = zone_bounds_local(grid, labels, z_id, frame)
        if kind == "scout":
            tasks.append(Task(f"scout-z{z_id}", "scout", z_id,
                              lawnmower(bounds, spec.scout_lane_m),
                              priority=float(uncertainty[mask].mean()), alt_m=spec.scout_alt_m))
        else:
            zone_f = fungus[mask]
            infected = zone_f > fungus_threshold
            frac = float(infected.mean())
            if frac < min_infected_frac:
                continue
            load = float(zone_f[infected].mean())
            area_ha = float(mask.sum()) * cell_ha
            dose = dose_l_per_ha * float(np.clip(load / max(fungus_threshold, 1e-6), 0.5, 2.0))
            tasks.append(Task(f"spray-z{z_id}", "spray", z_id,
                              lawnmower(bounds, spec.spray_swath_m), priority=frac * load,
                              payload_kg=round(area_ha * dose, 3), alt_m=spec.spray_alt_m))
    return tasks


# ==================================================================== allocation
def _estimate(start: Tuple[float, float], tasks: List[Task], home: Tuple[float, float],
              speed: float = PLANNING_SPEED_M_S) -> Tuple[float, float]:
    """(time_s, battery_pct) for flying ``tasks`` in order from ``start`` and back home."""
    pos, t, spray_t = start, 0.0, 0.0
    for task in tasks:
        t += _dist(pos, task.legs[0]) / speed
        work = task.work_length_m / speed
        t += work
        if task.kind == "spray":
            spray_t += work
        pos = task.legs[-1]
    t += _dist(pos, home) / speed
    return t, t * DRAIN_FLY_PCT_S + spray_t * DRAIN_SPRAY_PCT_S


def allocate(tasks: List[Task], drones: Dict[str, Dict[str, Any]],
             speed: float = PLANNING_SPEED_M_S) -> Plan:
    """
    Greedy min-makespan allocation. ``drones`` maps id -> {"home": (x, y),
    "battery": pct, "tank_kg": capacity}. Tasks are taken in priority order
    and given to the drone whose finish time grows least, among drones that
    keep their battery reserve and tank capacity.
    """
    assignments = {d: Assignment(d) for d in drones}
    unassigned: List[Task] = []
    for task in sorted(tasks, key=lambda t: -t.priority):
        best, best_time = None, math.inf
        for d_id, info in drones.items():
            a = assignments[d_id]
            if a.payload_kg + task.payload_kg > info.get("tank_kg", 10.0) + 1e-9:
                continue
            t, pct = _estimate(info["home"], a.tasks + [task], info["home"], speed)
            if info.get("battery", 100.0) - pct < BATTERY_RESERVE_PCT:
                continue
            if t < best_time:
                best, best_time = d_id, t
        if best is None:
            unassigned.append(task)
            continue
        a = assignments[best]
        a.tasks.append(task)
        a.payload_kg += task.payload_kg
        a.est_time_s, a.est_battery_pct = _estimate(drones[best]["home"], a.tasks,
                                                    drones[best]["home"], speed)
    return Plan(assignments, unassigned)


# ========================================================================= fleet
@dataclass
class _Leg:
    target: Tuple[float, float, float]
    spray: bool = False
    capture: bool = False
    task: Optional[Task] = None
    last_of_task: bool = False


class Fleet:
    def __init__(self, home_lat: float, home_lon: float, spec: DroneSpec = DroneSpec(),
                 pad_spacing_m: float = 8.0):
        self.frame = LocalFrame(home_lat, home_lon)
        self.spec = spec
        self.pad_spacing_m = pad_spacing_m
        self.shared_state: Dict[str, Any] = {"PHYSICS_SIMULATORS": {}, "SWARM_WARNINGS": [],
                                             "CURRENT_ENV": {"wind_speed": 0.0, "wind_direction": 0.0}}
        self.drones: Dict[str, Any] = {}
        self.pads: Dict[str, Tuple[float, float]] = {}

    @classmethod
    def for_field(cls, grid: FieldGrid, n_drones: int = 3, spec: DroneSpec = DroneSpec()) -> "Fleet":
        """Launch pad 6 m south of the field's southern edge, centred."""
        lat_c, lon_c = grid.center_latlon()
        south_offset = grid.rows * grid.cell_size_m / 2.0 + 6.0
        pad_lat = lat_c - south_offset / (math.radians(1.0) * EARTH_RADIUS_M)
        fleet = cls(pad_lat, lon_c, spec)
        for i in range(n_drones):
            fleet.add_drone(f"drone_{i + 1:02d}")
        return fleet

    # ---------------------------------------------------------------- registry
    def add_drone(self, drone_id: str) -> None:
        from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator

        if drone_id in self.drones:
            raise ValueError(f"Drone '{drone_id}' already exists")
        # Pads alternate either side of the centre: 0, +s, -s, +2s, -2s ...
        used = set(self.pads.values())
        i = 0
        while True:
            k = (i + 1) // 2
            pad = (k * self.pad_spacing_m * (1 if i % 2 else -1), 0.0)
            if pad not in used:
                break
            i += 1
        sim = UAVFlightDynamicsSimulator(self.shared_state, self.frame.home_lat,
                                         self.frame.home_lon, drone_id=drone_id)
        sim.pos = np.array([pad[0], pad[1], 0.5])
        sim.target_pos = sim.pos.copy()
        self.drones[drone_id] = sim
        self.pads[drone_id] = pad
        self.shared_state["PHYSICS_SIMULATORS"][drone_id] = sim

    def remove_drone(self, drone_id: str) -> None:
        if drone_id not in self.drones:
            raise KeyError(drone_id)
        del self.drones[drone_id]
        del self.pads[drone_id]
        del self.shared_state["PHYSICS_SIMULATORS"][drone_id]

    def service(self) -> None:
        """Battery swap and tank refill on the pad before a mission."""
        for d_id, sim in self.drones.items():
            sim.battery = 100.0
            sim.payload_mass = self.spec.tank_kg
            sim.autopilot_mode = "stabilized"
            sim.is_spraying = False

    def drone_info(self) -> Dict[str, Dict[str, Any]]:
        return {d: {"home": self.pads[d], "battery": float(s.battery),
                    "tank_kg": self.spec.tank_kg} for d, s in self.drones.items()}

    def snapshot(self) -> Dict[str, Any]:
        """Fleet positions in the same shape as the dashboard's /swarm endpoint."""
        out = {}
        for d_id, s in self.drones.items():
            lat, lon = self.frame.to_latlon(float(s.pos[0]), float(s.pos[1]))
            out[d_id] = {"id": d_id, "lat": lat, "lon": lon, "alt": float(s.pos[2]),
                         "yaw": float(s.attitude[2]), "pitch": float(s.attitude[1]),
                         "roll": float(s.attitude[0]), "battery": round(float(s.battery), 2),
                         "speed": round(float(np.linalg.norm(s.vel)), 2),
                         "is_spraying": bool(s.is_spraying),
                         "payload_kg": round(float(s.payload_mass), 3),
                         "mode": s.autopilot_mode}
        return {"home_lat": self.frame.home_lat, "home_lon": self.frame.home_lon, "drones": out}

    # --------------------------------------------------------------- execution
    def _legs_for(self, drone_id: str, tasks: List[Task], transit_alt: float) -> List[_Leg]:
        legs: List[_Leg] = []
        if not tasks:
            return legs  # idle drones stay on the pad
        pad = self.pads[drone_id]
        legs.append(_Leg((pad[0], pad[1], transit_alt)))
        for task in tasks:
            first = task.legs[0]
            legs.append(_Leg((first[0], first[1], transit_alt)))
            legs.append(_Leg((first[0], first[1], task.alt_m)))
            for j, pt in enumerate(task.legs[1:], start=1):
                legs.append(_Leg((pt[0], pt[1], task.alt_m), spray=task.kind == "spray",
                                 capture=task.kind == "scout", task=task,
                                 last_of_task=(j == len(task.legs) - 1)))
            if len(task.legs) == 1:
                legs[-1].task, legs[-1].last_of_task = task, True
            last = task.legs[-1]
            legs.append(_Leg((last[0], last[1], transit_alt)))
        legs.append(_Leg((pad[0], pad[1], transit_alt)))
        legs.append(_Leg((pad[0], pad[1], 0.5)))
        return legs

    def execute(
        self,
        plan: Plan,
        on_capture: Optional[Callable[[str, float, float, float], None]] = None,
        dt: float = 0.05,
        max_sim_s: float = 3600.0,
        arrive_m: float = 2.0,
        track_every_s: float = 1.0,
        wind: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Fly ``plan`` to completion (or ``max_sim_s``). Returns a mission report."""
        self.service()
        if wind:
            # Weather wind is km/h; flight physics expects m/s.
            self.shared_state["CURRENT_ENV"] = {
                "wind_speed": float(wind.get("wind_speed", 0.0)) / 3.6,
                "wind_direction": float(wind.get("wind_direction", 0.0))}
        self.shared_state["SWARM_WARNINGS"].clear()

        ids = [d for d in plan.assignments if d in self.drones]
        legs = {d: self._legs_for(d, plan.assignments[d].tasks,
                                  transit_alt=14.0 + 2.0 * k) for k, d in enumerate(ids)}
        idx = {d: 0 for d in ids}
        stats = {d: {"distance_m": 0.0, "flight_s": 0.0, "captures": 0, "tasks_done": [],
                     "aborted": False, "battery_start": float(self.drones[d].battery),
                     "payload_start": float(self.drones[d].payload_mass)} for d in ids}
        since_capture = {d: 0.0 for d in ids}
        # Variable-rate spraying: the physics sprays at a fixed 0.12 kg/s, so
        # flow is duty-cycled to apply each task's planned dose over its path.
        duty = {task.task_id: min(1.0, task.payload_kg / max(
                    task.work_length_m / PLANNING_SPEED_M_S * SPRAY_RATE_KG_S, 1e-6))
                for a in plan.assignments.values() for task in a.tasks if task.kind == "spray"}
        applied: Dict[str, float] = {k: 0.0 for k in duty}
        sprayed_time: Dict[str, float] = {k: 0.0 for k in duty}
        track: Dict[str, List[List[float]]] = {d: [] for d in ids}
        completed_sprays: List[Dict[str, Any]] = []
        captures: List[List[Any]] = []          # [t_s, drone_id, lat, lon, alt]
        proximity_events = 0
        t, next_track = 0.0, 0.0

        while t < max_sim_s and any(idx[d] < len(legs[d]) for d in ids):
            for d in ids:
                sim = self.drones[d]
                if idx[d] >= len(legs[d]):
                    sim.is_spraying = False
                    continue
                if sim.autopilot_mode in ("rtl", "landing") and not stats[d]["aborted"]:
                    stats[d]["aborted"] = True        # battery failsafe took over
                    pad = self.pads[d]
                    legs[d] = legs[d][:idx[d]] + [_Leg((pad[0], pad[1], 12.0)),
                                                  _Leg((pad[0], pad[1], 0.5))]
                    sim.autopilot_mode = "stabilized"  # let the planner fly it home
                leg = legs[d][idx[d]]
                sim.target_pos = np.array(leg.target, dtype=float)
                dx, dy = leg.target[0] - sim.pos[0], leg.target[1] - sim.pos[1]
                if math.hypot(dx, dy) > 1.0:
                    sim.target_yaw = math.atan2(dy, dx)
                spraying = False
                if leg.spray and sim.payload_mass > 0.0 and leg.task is not None:
                    tid = leg.task.task_id
                    sprayed_time[tid] += dt
                    planned = leg.task.payload_kg
                    spraying = (applied[tid] < duty[tid] * SPRAY_RATE_KG_S * sprayed_time[tid]
                                and applied[tid] < planned)
                sim.is_spraying = spraying
                before = sim.pos[:2].copy()
                payload_before = float(sim.payload_mass)
                sim.step(dt)
                if spraying:
                    applied[leg.task.task_id] += payload_before - float(sim.payload_mass)
                moved = float(np.linalg.norm(sim.pos[:2] - before))
                stats[d]["distance_m"] += moved
                stats[d]["flight_s"] += dt
                if leg.capture:
                    since_capture[d] += moved
                    if since_capture[d] >= self.spec.capture_spacing_m:
                        since_capture[d] = 0.0
                        stats[d]["captures"] += 1
                        lat, lon = self.frame.to_latlon(float(sim.pos[0]), float(sim.pos[1]))
                        captures.append([round(t, 2), d, lat, lon, round(float(sim.pos[2]), 2)])
                        if on_capture is not None:
                            on_capture(d, lat, lon, float(sim.pos[2]))
                landing = idx[d] == len(legs[d]) - 1
                tol = 0.6 if landing else arrive_m      # touch-down needs to be precise
                if np.linalg.norm(np.array(leg.target) - sim.pos) < tol:
                    if leg.last_of_task and leg.task is not None:
                        stats[d]["tasks_done"].append(leg.task.task_id)
                        if leg.task.kind == "spray":
                            tid = leg.task.task_id
                            completed_sprays.append({
                                "zone_id": leg.task.zone_id, "type": "Fungicide Spray",
                                "drone_id": d, "planned_kg": round(leg.task.payload_kg, 3),
                                "applied_kg": round(applied[tid], 3)})
                    idx[d] += 1
            proximity_events += len(self.shared_state["SWARM_WARNINGS"])
            self.shared_state["SWARM_WARNINGS"].clear()
            if t >= next_track:
                for d in ids:
                    s = self.drones[d]
                    lat, lon = self.frame.to_latlon(float(s.pos[0]), float(s.pos[1]))
                    track[d].append([round(t, 1), lat, lon, round(float(s.pos[2]), 2),
                                     int(bool(s.is_spraying))])
                next_track += track_every_s
            t += dt

        for d in ids:
            sim = self.drones[d]
            sim.is_spraying = False
            s = stats[d]
            s["battery_used_pct"] = round(s.pop("battery_start") - float(sim.battery), 2)
            s["spray_used_kg"] = round(s.pop("payload_start") - float(sim.payload_mass), 3)
            s["distance_m"] = round(s["distance_m"], 1)
            s["flight_s"] = round(s["flight_s"], 1)
            s["est_time_s"] = round(plan.assignments[d].est_time_s, 1)
            s["completed"] = idx[d] >= len(legs[d]) and not s["aborted"]

        return {
            "sim_seconds": round(t, 1),
            "timed_out": t >= max_sim_s,
            "drones": stats,
            "completed_sprays": completed_sprays,
            "unassigned": [task.task_id for task in plan.unassigned],
            "proximity_events": proximity_events,
            "track": track,
            "captures": captures,
        }
