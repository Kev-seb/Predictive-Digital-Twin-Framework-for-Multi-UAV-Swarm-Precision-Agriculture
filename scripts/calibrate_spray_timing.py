"""
calibrate_spray_timing.py
-------------------------
Fit the spray lane-time model  t_lane = L / v_eff + t0  for a MAVLink target
by flying a calibration mission: serpentine lanes of 10, 20, 40 and 60 m
(twice each) with 5 m headland turns, the pump on along each lane, at the
mission spray speed, i.e. the same geometry as a spray task.

    python scripts/calibrate_spray_timing.py --mode sitl        # ArduCopter SITL instance 0
    python scripts/calibrate_spray_timing.py --mode emulated    # in-repo emulator

Writes outputs/calibration/spray_timing_<mode>.json, which the twin API uses
to set pump rates (src/twin_core/mavlink_mission.py, LaneTimeModel).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.config import settings  # noqa: E402
from src.twin_core import FieldGrid  # noqa: E402
from src.twin_core import mavlink_mission as mm  # noqa: E402
from src.twin_core.swarm import Assignment, Fleet, Task  # noqa: E402

LANES_M = (10.0, 20.0, 40.0, 60.0, 60.0, 40.0, 20.0, 10.0)


def calibration_mission(frame, spacing_m: float = 5.0, alt_m: float = 3.0):
    """Serpentine lanes with swath-wide headland turns, like a real spray mission."""
    legs, y, x = [], 10.0, 0.0
    for i, length in enumerate(LANES_M):
        x_end = x + length if i % 2 == 0 else x - length
        legs += [(x, y), (x_end, y)]
        x, y = x_end, y + spacing_m
    task = Task("calib", "spray", -1, legs, payload_kg=1e6, alt_m=alt_m)   # huge dose -> full pump
    mission = mm.compile_mission(Assignment("calib", [task]), frame, transit_alt=10.0)
    on = [s for s, it in enumerate(mission.items)
          if it.command == mm.mavlink.MAV_CMD_DO_SET_SERVO and it.params[1] > mm.SPRAY_PWM_OFF]
    off = [s for s, it in enumerate(mission.items)
           if it.command == mm.mavlink.MAV_CMD_DO_SET_SERVO and it.params[1] == mm.SPRAY_PWM_OFF]
    return mission, list(zip(on, off))


def fit(lengths, times):
    A = np.column_stack([lengths, np.ones(len(lengths))])
    (inv_v, t0), *_ = np.linalg.lstsq(A, np.asarray(times), rcond=None)
    pred = A @ np.array([inv_v, t0])
    return 1.0 / inv_v, float(t0), float(np.sqrt(np.mean((pred - times) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sitl", "emulated"], default=settings.twin_mavlink_mode)
    ap.add_argument("--connection", default=None, help="default: instance 0 (sitl) or a local emulator")
    ap.add_argument("--time-scale", type=float, default=20.0, help="emulator speed-up")
    ap.add_argument("--out", type=Path, default=settings.twin_params_file.parent)
    args = ap.parse_args()

    fleet = Fleet.for_field(FieldGrid.centered_on(settings.field_center_lat, settings.field_center_lon), 1)
    home = fleet.frame.to_latlon(*fleet.pads["drone_01"])
    emu = None
    conn = args.connection
    if args.mode == "emulated" and conn is None:
        from src.twin_core.sim_vehicle import EmulatedCopter
        emu = EmulatedCopter(25760, *home, time_scale=args.time_scale).start()
        conn = "tcp:127.0.0.1:25760"
    conn = conn or mm.default_connection(0)

    mission, spans = calibration_mission(fleet.frame)
    pump_s = [0.0] * len(spans)
    last_t = {"t": None}

    def on_position(lat, lon, alt, seq, frac, t):
        if frac > 0 and last_t["t"] is not None:
            for i, (lo, hi) in enumerate(spans):
                if lo <= seq <= hi:
                    pump_s[i] += t - last_t["t"]
        last_t["t"] = t

    link = mm.VehicleLink(conn, timeout_s=30)
    try:
        link.wait_heartbeat()
        link.upload(mission)
        flown = link.fly(mission, on_position, max_flight_s=900)
    finally:
        link.close()
        if emu:
            emu.stop()

    v, t0, rmse = fit(LANES_M, pump_s)
    print(f"{args.mode} via {conn}: landed={flown['landed']}")
    for length, t in zip(LANES_M, pump_s):
        print(f"  lane {length:5.1f} m  pump-on {t:5.2f} s  model {length / v + t0:5.2f} s")
    print(f"fit: v_eff = {v:.2f} m/s, t0 = {t0:.2f} s, RMSE {rmse:.2f} s "
          f"(commanded spray speed {mm.SPRAY_SPEED_M_S} m/s)")
    model = mm.LaneTimeModel(round(v, 3), round(t0, 3),
                             f"{args.mode}: fitted on lanes {list(LANES_M)} m, RMSE {rmse:.2f} s")
    path = args.out / f"spray_timing_{args.mode}.json"
    model.to_json(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
