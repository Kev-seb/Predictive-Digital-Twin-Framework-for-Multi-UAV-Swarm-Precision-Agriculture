"""
sitl_fleet.py
-------------
Start one ArduCopter SITL instance per twin drone, each at its launch pad,
on the ports the twin API expects (tcp:127.0.0.1:5760 + 10 * i).

Runs the built SITL binary directly (not sim_vehicle.py), so each instance
gets its own working directory and log and the launcher supervises the
vehicles themselves. Prerequisites (see docs/sitl.md): an ArduPilot checkout
built with `./waf configure --board sitl && ./waf copter`, and the twin API
running with UAV_TWIN_MAVLINK_MODE=sitl.

    python scripts/sitl_fleet.py --ardupilot ~/ardupilot              # print the commands
    python scripts/sitl_fleet.py --ardupilot ~/ardupilot --launch     # start them (Ctrl+C stops all)
    python scripts/sitl_fleet.py --ardupilot ~/ardupilot --launch --speedup 5
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def commands(api: str, ardupilot: Path, speedup: float, workdir: Path):
    with urllib.request.urlopen(f"{api}/swarm/sitl-homes", timeout=10) as r:
        homes = json.load(r)
    if homes["mode"] != "sitl":
        print("Note: the twin API is not in SITL mode. Restart it with UAV_TWIN_MAVLINK_MODE=sitl.",
              file=sys.stderr)
    binary = ardupilot / "build" / "sitl" / "bin" / "arducopter"
    defaults = ardupilot / "Tools" / "autotest" / "default_params" / "copter.parm"
    out = []
    for inst in homes["instances"]:
        i = inst["instance"]
        cmd = [str(binary), "--model", "+", "--speedup", f"{speedup:g}", "--defaults", str(defaults),
               "-I", str(i), "--sysid", str(i + 1),
               "--home", f"{inst['home_lat']},{inst['home_lon']},0,0", "-w"]
        out.append((inst, cmd, workdir / f"sitl_{inst['drone_id']}"))
    return out, binary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8008")
    ap.add_argument("--ardupilot", type=Path, default=Path(os.environ.get("ARDUPILOT_HOME", "~/ardupilot")),
                    help="ArduPilot checkout (default $ARDUPILOT_HOME or ~/ardupilot)")
    ap.add_argument("--launch", action="store_true", help="start the instances instead of printing")
    ap.add_argument("--speedup", type=float, default=1.0, help="SITL --speedup factor")
    ap.add_argument("--workdir", type=Path, default=Path("sitl_fleet"), help="per-instance dirs and logs")
    args = ap.parse_args()

    cmds, binary = commands(args.api, args.ardupilot.expanduser(), args.speedup, args.workdir)
    for inst, cmd, wd in cmds:
        print(f"# {inst['drone_id']} -> {inst['connection']}  (cwd {wd})\n{shlex.join(cmd)}\n")
    if not args.launch:
        return
    if not binary.exists():
        sys.exit(f"{binary} not found: build SITL first (./waf configure --board sitl && ./waf copter).")

    procs = []
    try:
        for inst, cmd, wd in cmds:
            wd.mkdir(parents=True, exist_ok=True)
            log = open(wd / "arducopter.log", "w")
            procs.append((inst["drone_id"], subprocess.Popen(cmd, cwd=wd, stdout=log, stderr=subprocess.STDOUT)))
        print(f"Started {len(procs)} ArduCopter SITL instances (logs in {args.workdir}/). "
              "Allow ~60 s for GPS/EKF before flying. Ctrl+C to stop.", flush=True)
        while True:
            dead = [(d, p.returncode) for d, p in procs if p.poll() is not None]
            if dead:
                print(f"SITL instance(s) exited: {dead}; see their arducopter.log.", file=sys.stderr)
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
