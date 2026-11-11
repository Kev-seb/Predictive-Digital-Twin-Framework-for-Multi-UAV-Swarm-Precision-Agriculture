"""
mavlink_mission.py
------------------
Execute fleet plans on MAVLink vehicles: ArduCopter SITL, the in-repo
emulator (sim_vehicle.py) or real aircraft.

* ``compile_mission``   Assignment -> MAVLink mission items
                        (takeoff, waypoints, DO_SET_SERVO spray on/off,
                        DO_CHANGE_SPEED for variable-rate dosing, RTL)
* ``to_qgc_wpl``        the same mission as a QGC WPL 110 / Mission Planner file
* ``VehicleLink``       GCS side of the MAVLink mission protocol: upload,
                        GUIDED + arm + takeoff, AUTO, monitor until landed
* ``execute_plan_mavlink``  fly a whole fleet Plan in parallel and return a
                        report in the same shape as ``Fleet.execute``

Variable-rate spraying: a mission can only switch the pump on/off, so the
planned dose is set by ground speed instead. With a fixed pump flow F (kg/s),
lanes of total length L, n lanes and a per-lane-end settling time t_v, a
dose D needs  D / F = L / v + n t_v,  i.e.  v = L / (D / F - n t_v), clamped to
the airframe's range. The pump runs along the lanes only and is switched off
on the headland turns between them, as on real spray drones. The realised dose is measured from telemetry (pump-on
time x flow) and reported next to the plan as ``applied_kg``.

Instance i of ArduPilot SITL (``sim_vehicle.py -v ArduCopter -I i``) listens on
tcp:127.0.0.1:5760 + 10 * i, which is the default connection here.
"""

from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")   # ArduPilot speaks MAVLink 2 (servo9+, mission_type)

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pymavlink import mavutil

from src.twin_core.swarm import (
    PLANNING_SPEED_M_S, SPRAY_RATE_KG_S, Assignment, LocalFrame, Plan, Task,
)

mavlink = mavutil.mavlink

SPRAY_SERVO = 9          # ArduPilot SERVOn_FUNCTION = 22 (SprayerPump) on this channel
SPRAY_PWM_ON = 1900
SPRAY_PWM_OFF = 1100
CRUISE_SPEED_M_S = 5.0
MIN_SPRAY_SPEED_M_S = 1.0
MAX_SPRAY_SPEED_M_S = 8.0
# Pump-on time lost at each spray-path vertex while the autopilot settles on the
# waypoint (tracking lag ~ speed / position gain). Fitted on the emulator's
# flight physics; re-fit for ArduPilot from the planned vs applied_kg the
# mission report gives.
VERTEX_DWELL_S = 1.6

COPTER_MODES = {v: k for k, v in mavutil.mode_mapping_acm.items()}   # "AUTO" -> 3


def default_connection(instance: int) -> str:
    return f"tcp:127.0.0.1:{5760 + 10 * instance}"


@dataclass
class MissionItem:
    command: int
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    params: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    frame: int = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    task_id: Optional[str] = None      # which plan task this item belongs to
    kind: Optional[str] = None         # "scout" | "spray" while inside a task's work legs
    last_of_task: bool = False


@dataclass
class CompiledMission:
    drone_id: str
    items: List[MissionItem]           # items[0] is the home placeholder (ArduPilot convention)
    spray_speed: Dict[str, float] = field(default_factory=dict)
    spray_span: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # first pump-on .. last pump-off seq

    def task_ranges(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        for seq, it in enumerate(self.items):
            if it.task_id and it.kind:
                lo, hi = out.get(it.task_id, (seq, seq))
                out[it.task_id] = (min(lo, seq), max(hi, seq))
        return out


def spray_lanes(task: Task) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Lawn-mower legs come in (lane start, lane end) pairs; the legs between are turns."""
    return [(task.legs[k], task.legs[k + 1]) for k in range(0, len(task.legs) - 1, 2)]


def spray_speed_for(task: Task, flow_kg_s: float = SPRAY_RATE_KG_S,
                    vertex_dwell_s: float = VERTEX_DWELL_S) -> float:
    """Ground speed that applies task.payload_kg along the task's spray lanes."""
    lanes = spray_lanes(task)
    if task.payload_kg <= 0 or not lanes:
        return CRUISE_SPEED_M_S
    lane_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in lanes)
    moving_s = task.payload_kg / flow_kg_s - len(lanes) * vertex_dwell_s
    if moving_s <= lane_m / MAX_SPRAY_SPEED_M_S:
        return MAX_SPRAY_SPEED_M_S   # dose too small to reach even at top speed: least overspray
    v = lane_m / moving_s
    return float(min(MAX_SPRAY_SPEED_M_S, max(MIN_SPRAY_SPEED_M_S, v)))


def compile_mission(assignment: Assignment, frame: LocalFrame, transit_alt: float = 14.0,
                    home: Optional[Tuple[float, float]] = None) -> CompiledMission:
    """Assignment -> mission items in the drone's own home frame."""
    home_lat, home_lon = home if home else (frame.home_lat, frame.home_lon)
    C = mavlink
    items: List[MissionItem] = [MissionItem(C.MAV_CMD_NAV_WAYPOINT, home_lat, home_lon, 0.0)]
    items.append(MissionItem(C.MAV_CMD_NAV_TAKEOFF, home_lat, home_lon, transit_alt))
    speeds: Dict[str, float] = {}
    spans: Dict[str, Tuple[int, int]] = {}
    pump = lambda pwm: MissionItem(C.MAV_CMD_DO_SET_SERVO, params=(SPRAY_SERVO, pwm, 0, 0))  # noqa: E731

    def wp(x: float, y: float, alt: float, **kw) -> MissionItem:
        lat, lon = frame.to_latlon(x, y)
        return MissionItem(C.MAV_CMD_NAV_WAYPOINT, lat, lon, alt, **kw)

    for task in assignment.tasks:
        first = task.legs[0]
        items.append(wp(first[0], first[1], transit_alt))
        items.append(wp(first[0], first[1], task.alt_m))
        spraying = task.kind == "spray"
        if spraying:
            v = spray_speed_for(task)
            speeds[task.task_id] = v
            items.append(MissionItem(C.MAV_CMD_DO_CHANGE_SPEED, params=(1.0, v, -1.0, 0.0)))
            span_lo = len(items)
            items.append(pump(SPRAY_PWM_ON))              # first lane starts at legs[0]
        for j, pt in enumerate(task.legs[1:], start=1):
            items.append(wp(pt[0], pt[1], task.alt_m, task_id=task.task_id, kind=task.kind,
                            last_of_task=(j == len(task.legs) - 1)))
            if spraying:
                lane_end = j % 2 == 1                     # legs[1], legs[3], ... end a lane
                if lane_end:
                    items.append(pump(SPRAY_PWM_OFF))     # headland turn: pump off
                elif j < len(task.legs) - 1:
                    items.append(pump(SPRAY_PWM_ON))      # next lane starts here
        if spraying:
            if items[-1].command != C.MAV_CMD_DO_SET_SERVO or items[-1].params[1] != SPRAY_PWM_OFF:
                items.append(pump(SPRAY_PWM_OFF))
            spans[task.task_id] = (span_lo, len(items) - 1)
            items.append(MissionItem(C.MAV_CMD_DO_CHANGE_SPEED, params=(1.0, CRUISE_SPEED_M_S, -1.0, 0.0)))
        last = task.legs[-1]
        items.append(wp(last[0], last[1], transit_alt))
    items.append(MissionItem(C.MAV_CMD_NAV_RETURN_TO_LAUNCH))
    return CompiledMission(assignment.drone_id, items, speeds, spans)


def to_qgc_wpl(mission: CompiledMission) -> str:
    """QGC WPL 110 text (QGroundControl / Mission Planner .waypoints)."""
    lines = ["QGC WPL 110"]
    for seq, it in enumerate(mission.items):
        frame = 0 if seq == 0 else 3          # 0 = global (home), 3 = relative altitude
        p = it.params
        lines.append("\t".join(str(v) for v in (
            seq, 1 if seq == 0 else 0, frame, it.command, p[0], p[1], p[2], p[3],
            f"{it.lat:.8f}", f"{it.lon:.8f}", f"{it.alt:.2f}", 1)))
    return "\n".join(lines) + "\n"


# =================================================================== GCS link
class MavlinkError(RuntimeError):
    pass


class VehicleLink:
    """Ground-control side of one MAVLink vehicle."""

    def __init__(self, connection: str, timeout_s: float = 30.0, source_system: int = 255):
        self.connection = connection
        self.timeout_s = timeout_s
        self.master = mavutil.mavlink_connection(connection, source_system=source_system,
                                                 autoreconnect=False)

    # ------------------------------------------------------------- basics
    def wait_heartbeat(self) -> None:
        if self.master.wait_heartbeat(timeout=self.timeout_s) is None:
            raise MavlinkError(f"No heartbeat on {self.connection}")

    def _recv(self, types, timeout: Optional[float] = None, cond: Optional[Callable] = None):
        deadline = time.monotonic() + (timeout or self.timeout_s)
        while time.monotonic() < deadline:
            msg = self.master.recv_match(type=types, blocking=True, timeout=0.5)
            if msg is not None and (cond is None or cond(msg)):
                return msg
        return None

    def command(self, cmd: int, *params: float, timeout: Optional[float] = None) -> None:
        p = list(params) + [0.0] * (7 - len(params))
        self.master.mav.command_long_send(self.master.target_system, self.master.target_component,
                                          cmd, 0, *p)
        ack = self._recv("COMMAND_ACK", timeout, cond=lambda m: m.command == cmd)
        if ack is None or ack.result != mavlink.MAV_RESULT_ACCEPTED:
            raise MavlinkError(f"Command {cmd} not accepted ({getattr(ack, 'result', 'timeout')})")

    def arm(self, retry_s: float = 120.0) -> None:
        """
        Arm, retrying while pre-arm checks fail: a freshly started ArduPilot SITL
        rejects arming until GPS and the EKF have settled (~30-60 s).
        """
        deadline = time.monotonic() + retry_s
        last = None
        while time.monotonic() < deadline:
            try:
                self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, timeout=5.0)
                return
            except MavlinkError as e:
                last = e
                time.sleep(2.0)
        raise MavlinkError(f"Arming refused for {retry_s:.0f} s: {last}")

    def set_mode(self, name: str) -> None:
        mode_id = COPTER_MODES[name]
        self.master.set_mode(mode_id)
        hb = self._recv("HEARTBEAT", cond=lambda m: m.custom_mode == mode_id)
        if hb is None:
            raise MavlinkError(f"Vehicle did not enter {name}")

    # ----------------------------------------------------------- mission
    def upload(self, mission: CompiledMission) -> None:
        m = self.master
        items = mission.items
        m.mav.mission_count_send(m.target_system, m.target_component, len(items),
                                 mavlink.MAV_MISSION_TYPE_MISSION)
        while True:
            msg = self._recv(["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"])
            if msg is None:
                raise MavlinkError("Mission upload timed out")
            if msg.get_type() == "MISSION_ACK":
                if msg.type != mavlink.MAV_MISSION_ACCEPTED:
                    raise MavlinkError(f"Mission rejected (MAV_MISSION_RESULT {msg.type})")
                return
            it = items[msg.seq]
            m.mav.mission_item_int_send(
                m.target_system, m.target_component, msg.seq, it.frame, it.command,
                0, 1, *it.params, int(round(it.lat * 1e7)), int(round(it.lon * 1e7)), float(it.alt),
                mavlink.MAV_MISSION_TYPE_MISSION)

    def fly(self, mission: CompiledMission, on_position: Optional[Callable[..., None]] = None,
            takeoff_alt: Optional[float] = None, max_flight_s: float = 3600.0) -> Dict[str, Any]:
        """
        GUIDED -> arm -> takeoff -> AUTO, then monitor until the vehicle has
        landed and disarmed after RTL (or ``max_flight_s`` wall seconds).
        ``on_position(lat, lon, rel_alt_m, seq, spraying, t_s)`` is called for
        every GLOBAL_POSITION_INT; ``t_s`` is the vehicle's boot time, so tracks
        keep simulated time even when SITL runs faster than real time.
        """
        alt = takeoff_alt or mission.items[1].alt
        self.set_mode("GUIDED")
        self.arm()
        self.command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"), 0, 0, alt)
        self._recv("GLOBAL_POSITION_INT", timeout=120.0,
                   cond=lambda p: p.relative_alt / 1000.0 >= 0.9 * alt)
        self.set_mode("AUTO")

        reached: List[int] = []
        seq, spraying, armed, airborne_seen = 1, False, True, False
        n_pos = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_flight_s:
            msg = self.master.recv_match(type=["GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED",
                                               "MISSION_CURRENT", "SERVO_OUTPUT_RAW", "HEARTBEAT"],
                                         blocking=True, timeout=1.0)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "MISSION_ITEM_REACHED":
                if msg.seq not in reached:
                    reached.append(msg.seq)
            elif t == "MISSION_CURRENT":
                seq = msg.seq
            elif t == "SERVO_OUTPUT_RAW":
                spraying = getattr(msg, f"servo{SPRAY_SERVO}_raw", 0) > 1500
            elif t == "HEARTBEAT" and msg.get_srcSystem() == self.master.target_system:
                armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if not armed and airborne_seen:
                    break
            elif t == "GLOBAL_POSITION_INT":
                rel = msg.relative_alt / 1000.0
                airborne_seen = airborne_seen or rel > 1.0
                n_pos += 1
                if on_position is not None:
                    on_position(msg.lat / 1e7, msg.lon / 1e7, rel, seq, spraying, msg.time_boot_ms / 1000.0)
        return {"reached": reached, "landed": not armed, "positions": n_pos,
                "wall_s": round(time.monotonic() - t0, 2)}

    def close(self) -> None:
        try:
            self.master.close()
        except Exception:
            pass


# ======================================================= fleet over MAVLink
def execute_plan_mavlink(
    plan: Plan,
    frame: LocalFrame,
    connections: Dict[str, str],
    homes: Optional[Dict[str, Tuple[float, float]]] = None,
    on_capture: Optional[Callable[[str, float, float, float], None]] = None,
    capture_spacing_m: float = 5.0,
    timeout_s: float = 60.0,
    max_flight_s: float = 1800.0,
) -> Dict[str, Any]:
    """
    Fly every assignment of ``plan`` on its MAVLink vehicle in parallel.
    Returns the same report shape as ``Fleet.execute`` (drones, completed_sprays,
    unassigned, track) plus per-drone ``mission_items`` and ``connection``.
    """
    lock = threading.Lock()
    report: Dict[str, Any] = {"drones": {}, "completed_sprays": [], "track": {},
                              "unassigned": [t.task_id for t in plan.unassigned],
                              "backend": "mavlink", "errors": {}}

    def run(drone_id: str, assignment: Assignment, k: int) -> None:
        mission = compile_mission(assignment, frame, transit_alt=14.0 + 2.0 * k,
                                  home=(homes or {}).get(drone_id))
        ranges = mission.task_ranges()
        kind_of = {seq: (it.task_id, it.kind) for seq, it in enumerate(mission.items) if it.kind}
        spray_span = mission.spray_span   # pump time is attributed to the task flying it
        state = {"last": None, "since": 0.0, "dist": 0.0, "captures": 0, "spray_s": {},
                 "t_last": None, "track": []}
        link = None
        try:
            link = VehicleLink(connections[drone_id], timeout_s=timeout_s)
            link.wait_heartbeat()
            link.upload(mission)

            def on_position(lat, lon, alt, seq, spraying, now):
                x, y = frame.to_local(lat, lon)
                if state["last"] is not None:
                    step = math.hypot(x - state["last"][0], y - state["last"][1])
                    state["dist"] += step
                    task_id, kind = kind_of.get(seq, (None, None))
                    if kind == "scout":
                        state["since"] += step
                        if state["since"] >= capture_spacing_m:
                            state["since"] = 0.0
                            state["captures"] += 1
                            if on_capture is not None:
                                on_capture(drone_id, lat, lon, alt)
                if spraying and state["t_last"] is not None:
                    for tid, (lo, hi) in spray_span.items():
                        if lo <= seq <= hi:
                            state["spray_s"][tid] = state["spray_s"].get(tid, 0.0) + (now - state["t_last"])
                state["t_last"] = now
                state["last"] = (x, y)
                state["track"].append([round(now, 2), lat, lon, round(alt, 2), int(bool(spraying))])

            flown = link.fly(mission, on_position, max_flight_s=max_flight_s)
            done = [tid for tid, (_, hi) in ranges.items() if hi in flown["reached"]]
            sprays = []
            for task in assignment.tasks:
                if task.kind == "spray" and task.task_id in done:
                    sprays.append({"zone_id": task.zone_id, "type": "Fungicide Spray", "drone_id": drone_id,
                                   "planned_kg": round(task.payload_kg, 3),
                                   # measured from telemetry: pump-on time x pump flow
                                   "applied_kg": round(state["spray_s"].get(task.task_id, 0.0)
                                                       * SPRAY_RATE_KG_S, 3),
                                   "spray_speed_m_s": round(mission.spray_speed[task.task_id], 2)})
            t0 = state["track"][0][0] if state["track"] else 0.0
            with lock:
                report["drones"][drone_id] = {
                    "connection": connections[drone_id], "mission_items": len(mission.items),
                    "tasks_done": done, "completed": flown["landed"] and len(done) == len(ranges),
                    "distance_m": round(state["dist"], 1), "captures": state["captures"],
                    "positions": flown["positions"], "wall_s": flown["wall_s"],
                    "est_time_s": round(assignment.est_time_s, 1)}
                report["completed_sprays"].extend(sprays)
                report["track"][drone_id] = [[round(p[0] - t0, 2)] + p[1:] for p in state["track"]]
        except Exception as e:
            with lock:
                report["errors"][drone_id] = f"{type(e).__name__}: {e}"
        finally:
            if link is not None:
                link.close()

    threads = []
    for k, (d_id, a) in enumerate(plan.assignments.items()):
        if not a.tasks:
            continue
        if d_id not in connections:
            report["errors"][d_id] = "no MAVLink connection configured"
            continue
        th = threading.Thread(target=run, args=(d_id, a, k), name=f"mavlink-{d_id}", daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return report
