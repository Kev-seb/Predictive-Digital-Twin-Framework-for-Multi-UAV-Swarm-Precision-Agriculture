"""
sim_vehicle.py
--------------
A MAVLink vehicle emulator that stands in for ArduCopter SITL, driven by the
repo's 6-DOF flight physics (UAVFlightDynamicsSimulator).

It implements the subset of ArduCopter behaviour the twin's ground-station
code relies on, so the whole MAVLink path (mission upload, arming, takeoff,
AUTO, telemetry) can be tested without building ArduPilot:

  * TCP server on 127.0.0.1:<port> (ArduPilot SITL instance i uses 5760 + 10 i)
  * HEARTBEAT as an ArduPilot quadrotor with copter custom modes
  * mission upload protocol (MISSION_COUNT / MISSION_REQUEST_INT / MISSION_ITEM_INT / MISSION_ACK)
  * SET_MODE, COMMAND_LONG: ARM_DISARM, NAV_TAKEOFF, DO_SET_MODE, MISSION_START
  * AUTO: NAV_TAKEOFF, NAV_WAYPOINT (carrot following at the commanded speed),
    DO_CHANGE_SPEED, DO_SET_SERVO (sprayer pump on servo 9), NAV_RETURN_TO_LAUNCH
  * RTL -> LAND -> automatic disarm on touchdown, like ArduCopter
  * GLOBAL_POSITION_INT 10 Hz, MISSION_CURRENT, MISSION_ITEM_REACHED,
    SERVO_OUTPUT_RAW, SYS_STATUS, COMMAND_ACK

``time_scale`` > 1 runs simulated time faster than wall time; all timestamps
the vehicle sends are simulated time.

It is not a flight controller. Use real SITL (sim_vehicle.py -v ArduCopter)
to validate against ArduPilot itself; the ground-station code is identical.
"""

from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")

import math
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymavlink import mavutil

from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator
from src.twin_core.swarm import LocalFrame

mavlink = mavutil.mavlink
MODES = {v: k for k, v in mavutil.mode_mapping_acm.items()}
SPRAY_SERVO = 9


class EmulatedCopter:
    def __init__(self, port: int, home_lat: float, home_lon: float, sysid: int = 1,
                 time_scale: float = 1.0, dt: float = 0.05, wind_mps: float = 0.0,
                 wind_dir_deg: float = 0.0, host: str = "127.0.0.1"):
        self.port, self.sysid, self.time_scale, self.dt = port, sysid, max(time_scale, 0.01), dt
        self.frame = LocalFrame(home_lat, home_lon)
        self.conn = mavutil.mavlink_connection(f"tcpin:{host}:{port}", source_system=sysid,
                                               source_component=1)
        shared = {"PHYSICS_SIMULATORS": {}, "SWARM_WARNINGS": [],
                  "CURRENT_ENV": {"wind_speed": wind_mps, "wind_direction": wind_dir_deg}}
        self.sim = UAVFlightDynamicsSimulator(shared, home_lat, home_lon, drone_id=f"sitl_{sysid}")
        self.sim.pos = np.array([0.0, 0.0, 0.0])
        self.sim.target_pos = self.sim.pos.copy()
        self.sim.battery = 100.0

        self.armed = False
        self.mode = "STABILIZE"
        self.t = 0.0                       # simulated seconds since boot
        self.mission: List = []
        self.cur = 0
        self._upload: Optional[List] = None
        self._gcs = (255, 0)
        self.speed = 5.0
        self.servo = {SPRAY_SERVO: 1100}
        self.carrot: Optional[np.ndarray] = None
        self.guided_target: Optional[np.ndarray] = None
        self.rtl_phase = None
        self._next = {"hb": 0.0, "pos": 0.0, "slow": 0.0, "peek": 0.0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reached_log: List[int] = []

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "EmulatedCopter":
        self._thread = threading.Thread(target=self._run, name=f"emulated-copter-{self.port}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self.conn.close()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            t_wall = time.monotonic()
            self._drain()
            self._control()
            if self.armed:
                self.sim.step(self.dt)
                if self.sim.pos[2] <= 0.0 and self.mode in ("LAND", "RTL") and self.rtl_phase == "land":
                    self._disarm()
            self.t += self.dt
            self._telemetry()
            sleep = self.dt / self.time_scale - (time.monotonic() - t_wall)
            if sleep > 0:
                time.sleep(sleep)

    # ---------------------------------------------------------------- input
    def _drain(self) -> None:
        if self.t >= self._next["peek"]:
            self._next["peek"] = self.t + 1.0
            self._drop_closed_client()
        while True:
            try:
                msg = self.conn.recv_match(blocking=False)
            except Exception:
                return
            if msg is None:
                return
            self._handle(msg)

    def _drop_closed_client(self) -> None:
        port = getattr(self.conn, "port", None)
        if port is None:
            return
        try:
            if port.recv(1, socket.MSG_PEEK) == b"":
                port.close()
                self.conn.port = None
                self.conn.fd = self.conn.listen.fileno()
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self.conn.port = None
            self.conn.fd = self.conn.listen.fileno()

    def _ack(self, command: int, result: int = mavlink.MAV_RESULT_ACCEPTED) -> None:
        self.conn.mav.command_ack_send(command, result)

    def _handle(self, msg) -> None:
        t = msg.get_type()
        if t == "BAD_DATA":
            return
        self._gcs = (msg.get_srcSystem(), msg.get_srcComponent())
        if t == "MISSION_COUNT":
            self._upload = [None] * msg.count
            if msg.count == 0:
                self.mission, self.cur = [], 0
                self.conn.mav.mission_ack_send(*self._gcs, mavlink.MAV_MISSION_ACCEPTED)
            else:
                self.conn.mav.mission_request_int_send(*self._gcs, 0)
        elif t in ("MISSION_ITEM_INT", "MISSION_ITEM") and self._upload is not None:
            if msg.seq < len(self._upload):
                scale = 1e7 if t == "MISSION_ITEM_INT" else 1.0
                self._upload[msg.seq] = (msg.command, msg.x / scale, msg.y / scale, msg.z,
                                         (msg.param1, msg.param2, msg.param3, msg.param4))
                nxt = next((i for i, it in enumerate(self._upload) if it is None), None)
                if nxt is None:
                    self.mission, self._upload, self.cur = self._upload, None, 1
                    self.conn.mav.mission_ack_send(*self._gcs, mavlink.MAV_MISSION_ACCEPTED)
                else:
                    self.conn.mav.mission_request_int_send(*self._gcs, nxt)
        elif t == "SET_MODE":
            self._set_mode(mavutil.mode_mapping_acm.get(msg.custom_mode))
        elif t == "COMMAND_LONG":
            self._command(msg)

    def _command(self, msg) -> None:
        c = msg.command
        if c == mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if msg.param1 >= 0.5:
                self.armed = True
                self.sim.payload_mass = self.sim.max_payload_mass
            else:
                self._disarm()
            self._ack(c)
        elif c == mavlink.MAV_CMD_NAV_TAKEOFF:
            if self.armed and self.mode == "GUIDED":
                self.guided_target = np.array([self.sim.pos[0], self.sim.pos[1], float(msg.param7)])
                self._ack(c)
            else:
                self._ack(c, mavlink.MAV_RESULT_FAILED)
        elif c == mavlink.MAV_CMD_DO_SET_MODE:
            ok = self._set_mode(mavutil.mode_mapping_acm.get(int(msg.param2)))
            self._ack(c, mavlink.MAV_RESULT_ACCEPTED if ok else mavlink.MAV_RESULT_FAILED)
        elif c == mavlink.MAV_CMD_MISSION_START:
            ok = self._set_mode("AUTO")
            self._ack(c, mavlink.MAV_RESULT_ACCEPTED if ok else mavlink.MAV_RESULT_FAILED)
        elif c == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            self._ack(c)
        else:
            self._ack(c, mavlink.MAV_RESULT_UNSUPPORTED)

    def _set_mode(self, name: Optional[str]) -> bool:
        if name not in ("STABILIZE", "GUIDED", "AUTO", "RTL", "LAND", "LOITER"):
            return False
        if name == "AUTO" and not self.mission:
            return False
        self.mode = name
        self.carrot = None
        self.rtl_phase = "return" if name == "RTL" else ("land" if name == "LAND" else None)
        if name == "AUTO" and self.cur < 1:
            self.cur = 1
        return True

    def _disarm(self) -> None:
        self.armed = False
        self.servo[SPRAY_SERVO] = 1100
        self.sim.is_spraying = False
        self.sim.vel = np.zeros(3)
        self.sim.pos[2] = 0.0
        self.mode = "STABILIZE" if self.mode not in ("LAND", "RTL") else self.mode

    # -------------------------------------------------------------- control
    def _carrot_to(self, goal: np.ndarray, speed: float) -> bool:
        """Advance the carrot toward goal; returns True once the vehicle is there."""
        if self.carrot is None:
            self.carrot = self.sim.pos.copy()
        d = goal - self.carrot
        dist = float(np.linalg.norm(d))
        step = speed * self.dt
        self.carrot = goal.copy() if dist <= step else self.carrot + d / dist * step
        self.sim.target_pos = self.carrot.copy()
        h = goal[:2] - self.sim.pos[:2]
        if float(np.linalg.norm(h)) > 1.0:
            self.sim.target_yaw = math.atan2(h[1], h[0])
        return dist <= step and float(np.linalg.norm(goal - self.sim.pos)) < 1.5

    def _control(self) -> None:
        self.sim.autopilot_mode = "stabilized"
        self.sim.is_spraying = self.armed and self.servo.get(SPRAY_SERVO, 1100) > 1500 \
            and self.sim.payload_mass > 0.0
        if not self.armed:
            return
        if self.mode == "GUIDED" and self.guided_target is not None:
            self._carrot_to(self.guided_target, 2.5)
        elif self.mode == "AUTO":
            self._auto()
        elif self.mode in ("RTL", "LAND"):
            self._rtl()
        else:
            self.sim.target_pos = self.sim.pos.copy()

    def _item_done(self) -> None:
        self.conn.mav.mission_item_reached_send(self.cur)
        self.reached_log.append(self.cur)
        self.cur += 1
        self.carrot = self.sim.pos.copy()

    def _auto(self) -> None:
        # Process any number of instantaneous DO_ items in one tick.
        for _ in range(64):
            if self.cur >= len(self.mission):
                self.sim.target_pos = self.sim.pos.copy()   # end of mission: hold
                return
            cmd, lat, lon, alt, p = self.mission[self.cur]
            if cmd == mavlink.MAV_CMD_DO_SET_SERVO:
                self.servo[int(p[0])] = int(p[1])
                self.cur += 1
                continue
            if cmd == mavlink.MAV_CMD_DO_CHANGE_SPEED:
                if p[1] > 0:
                    self.speed = float(p[1])
                self.cur += 1
                continue
            break
        if cmd == mavlink.MAV_CMD_NAV_TAKEOFF:
            goal = np.array([self.sim.pos[0], self.sim.pos[1], alt])
            self._carrot_to(goal, 2.5)
            if self.sim.pos[2] >= 0.9 * alt:
                self._item_done()
        elif cmd == mavlink.MAV_CMD_NAV_WAYPOINT:
            x, y = self.frame.to_local(lat, lon)
            if self._carrot_to(np.array([x, y, alt]), self.speed if self.cur > 1 else 5.0):
                self._item_done()
        elif cmd == mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            self.conn.mav.mission_item_reached_send(self.cur)
            self.reached_log.append(self.cur)
            self.cur += 1
            self._set_mode("RTL")
        else:
            self.cur += 1   # unsupported NAV command: skip, as ArduPilot would refuse it at upload

    def _rtl(self) -> None:
        z = max(float(self.sim.pos[2]), 2.0)
        if self.rtl_phase == "return":
            if self._carrot_to(np.array([0.0, 0.0, z]), 5.0):
                self.rtl_phase = "land"
                self.mode = "LAND"
                self.carrot = None
        if self.rtl_phase == "land":
            self._carrot_to(np.array([0.0, 0.0, -1.0]), 1.0)

    # ------------------------------------------------------------ telemetry
    def _telemetry(self) -> None:
        mav, now = self.conn.mav, self.t
        if now >= self._next["hb"]:
            self._next["hb"] = now + 1.0
            base = mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (
                mavlink.MAV_MODE_FLAG_SAFETY_ARMED if self.armed else 0)
            mav.heartbeat_send(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, base,
                               MODES[self.mode],
                               mavlink.MAV_STATE_ACTIVE if self.armed else mavlink.MAV_STATE_STANDBY)
        if now >= self._next["pos"]:
            self._next["pos"] = now + 0.1
            x, y, z = (float(v) for v in self.sim.pos)
            lat, lon = self.frame.to_latlon(x, y)
            vx, vy, vz = (float(v) for v in self.sim.vel)
            hdg = int((math.degrees(math.atan2(vx, vy)) % 360) * 100) if abs(vx) + abs(vy) > 0.1 else 0
            mav.global_position_int_send(int(now * 1000), int(lat * 1e7), int(lon * 1e7),
                                         int(z * 1000), int(z * 1000),
                                         int(vy * 100), int(vx * 100), int(-vz * 100), hdg)
        if now >= self._next["slow"]:
            self._next["slow"] = now + 0.5
            if self.mission:
                mav.mission_current_send(min(self.cur, len(self.mission) - 1))
            sv = [1100] * 16
            sv[SPRAY_SERVO - 1] = self.servo.get(SPRAY_SERVO, 1100)
            mav.servo_output_raw_send(int(now * 1e6), 0, *sv)
            mav.sys_status_send(0, 0, 0, 500, 16000, -1, int(self.sim.battery), 0, 0, 0, 0, 0, 0)


class EmulatedFleet:
    """One EmulatedCopter per drone on SITL-style ports base_port + 10 * i."""

    def __init__(self, homes: Dict[str, Tuple[float, float]], base_port: int = 5760,
                 time_scale: float = 1.0, wind_mps: float = 0.0, wind_dir_deg: float = 0.0):
        self.vehicles: Dict[str, EmulatedCopter] = {}
        self.connections: Dict[str, str] = {}
        for i, (d_id, (lat, lon)) in enumerate(homes.items()):
            port = base_port + 10 * i
            self.vehicles[d_id] = EmulatedCopter(port, lat, lon, sysid=i + 1, time_scale=time_scale,
                                                 wind_mps=wind_mps, wind_dir_deg=wind_dir_deg).start()
            self.connections[d_id] = f"tcp:127.0.0.1:{port}"

    def stop(self) -> None:
        for v in self.vehicles.values():
            v.stop()
