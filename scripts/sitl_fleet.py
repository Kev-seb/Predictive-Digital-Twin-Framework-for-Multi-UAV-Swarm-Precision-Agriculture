"""
sitl_fleet.py
-------------
Start one ArduCopter SITL instance per twin drone, each at its launch pad,
on the ports the twin API expects (tcp:127.0.0.1:5760 + 10 * i).

Prerequisites (on your machine; see docs/sitl.md):
    ArduPilot checked out and built for SITL, Tools/autotest on PATH
    (so `sim_vehicle.py` runs), and the twin API running with
    UAV_TWIN_MAVLINK_MODE=sitl.

    python scripts/sitl_fleet.py              # print the commands
    python scripts/sitl_fleet.py --launch     # start them (Ctrl+C stops all)
    python scripts/sitl_fleet.py --launch --speedup 5
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8008")
    ap.add_argument("--launch", action="store_true", help="start the instances instead of printing")
    ap.add_argument("--speedup", type=float, default=1.0, help="SITL --speedup factor")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.api}/swarm/sitl-homes", timeout=10) as r:
        homes = json.load(r)
    if homes["mode"] != "sitl":
        print("Note: the twin API is not in SITL mode. Restart it with UAV_TWIN_MAVLINK_MODE=sitl.",
              file=sys.stderr)

    commands = []
    for inst in homes["instances"]:
        cmd = inst["command"] + (f" --speedup {args.speedup:g}" if args.speedup != 1.0 else "")
        commands.append((inst, cmd))
        print(f"# {inst['drone_id']} -> {inst['connection']}\n{cmd}\n")
    if not args.launch:
        return

    procs = []
    try:
        for inst, cmd in commands:
            log = open(f"sitl_{inst['drone_id']}.log", "w")
            procs.append(subprocess.Popen(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT))
            time.sleep(2.0)   # stagger start-up; each instance builds its own eeprom/logs
        print(f"Started {len(procs)} SITL instances (logs: sitl_<drone>.log). "
              "Allow ~60 s for GPS/EKF before flying. Ctrl+C to stop.")
        while all(p.poll() is None for p in procs):
            time.sleep(1.0)
        print("A SITL instance exited; check its log.", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
