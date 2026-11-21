"""
One-click start for the UE5 digital twin demo.

    python scripts/start_demo.py                 # start the twin, print the UE5 steps
    python scripts/start_demo.py --ue-project "D:/UE/AgriTwin/AgriTwin.uproject"
                                                 # ...and open UE5 with twin_live.py running
    python scripts/start_demo.py --demo          # ...and start the demo story at once

On Windows, double-click start_demo.bat in the repository root instead (set
UE_PROJECT in it once). The twin API keeps running until you press Ctrl+C or
close the window; an already-running twin is reused.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "ue5_integration" / "python" / "twin_live.py"
REQUIRED = ["fastapi", "uvicorn", "numpy", "scipy", "pydantic"]
STARTED = []          # API processes this script started (for tests / callers)


def missing_packages():
    import importlib.util
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def api_up(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url + "/", timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def post(url, path, body=None, timeout=10.0):
    req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_api(url, wait_s=120.0, log_path=None):
    """Start digital_twin_api.py in the background; return the process once /health answers."""
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, str(ROOT / "digital_twin_api.py")], cwd=str(ROOT),
                            stdout=log, stderr=subprocess.STDOUT)
    end = time.time() + wait_s
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"the twin API exited (code {proc.returncode}); see {log_path}")
        if api_up(url):
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"the twin API did not answer at {url} within {wait_s:.0f} s; see {log_path}")


def find_unreal_editor():
    """Newest UnrealEditor.exe in the default Epic Games folder (Windows), else None."""
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files", r"D:\Program Files"]
    hits = []
    for r in dict.fromkeys(roots):
        hits += glob.glob(os.path.join(r, "Epic Games", "UE_5*", "Engine", "Binaries", "Win64", "UnrealEditor.exe"))
    return sorted(hits)[-1] if hits else None


def ue_command(editor, project, script=SCRIPT):
    """UnrealEditor command line that opens the project and runs twin_live.py once loaded.

    A single string (Windows' native form), so the quotes reach UE as written.
    """
    return f'"{editor}" "{project}" -ExecutePythonScript="{Path(script)}"'



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008, help="twin API port (twin_live.py expects 8008)")
    ap.add_argument("--ue-project", default=os.environ.get("UE_PROJECT") or None,
                    help="your .uproject; opens UE5 with twin_live.py running")
    ap.add_argument("--ue-editor", default=os.environ.get("UE_EDITOR") or None,
                    help="path to UnrealEditor.exe (default: newest in Program Files/Epic Games)")
    ap.add_argument("--demo", action="store_true", help="start the demo story right away")
    ap.add_argument("--fresh", action="store_true", help="reset the field to day 0 with a new outbreak")
    ap.add_argument("--no-wait", action="store_true", help="exit after starting (leaves the API running)")
    args = ap.parse_args(argv)
    url = f"http://127.0.0.1:{args.port}"

    print("=== Paddy digital twin: one-click start ===\n")
    gone = missing_packages()
    if gone:
        print(f"Missing Python packages: {', '.join(gone)}\nInstall them once with:\n"
              f'  "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"')
        return 2

    proc = None
    if api_up(url):
        print(f"[1/3] Twin API already running at {url}, using it")
    else:
        print(f"[1/3] Starting the twin API at {url} ...", flush=True)
        log_path = ROOT / "logs" / "twin_api.log"
        log_path.parent.mkdir(exist_ok=True)
        os.environ["UAV_TWIN_API_PORT"] = str(args.port)
        try:
            proc = start_api(url, log_path=log_path)
            STARTED.append(proc)
        except RuntimeError as e:
            print(f"      FAILED: {e}")
            return 1
        print(f"      ready (log: {log_path})")
    if args.fresh:
        post(url, "/sim/reset", {})
        print("      field reset to day 0")

    if args.ue_project:
        editor = args.ue_editor or find_unreal_editor()
        if not editor or not Path(editor).exists():
            print("[2/3] Could not find UnrealEditor.exe; pass --ue-editor or set UE_EDITOR")
        else:
            print(f"[2/3] Opening {Path(args.ue_project).name} in UE5 with twin_live.py "
                  "(the fields appear once the editor has loaded)")
            subprocess.Popen(ue_command(editor, args.ue_project), shell=os.name != "nt")
    else:
        print("[2/3] In UE5: File > Execute Python Script... and pick\n"
              f"      {SCRIPT}\n"
              "      (or pass --ue-project / set UE_PROJECT to open UE5 with it running)")

    if args.demo:
        post(url, "/demo/start", {})
        print("[3/3] Demo story started")
    else:
        print("[3/3] Run the show from UE5's Digital Twin menu (Start demo, Next day, Scout, Spray,\n"
              "      Inject outbreak), or rerun this with --demo")

    if args.no_wait or proc is None:
        if proc is None:
            print("\nThe twin was already running; nothing more to do here.")
        return 0
    print("\nThe twin is running. Keep this window open; press Ctrl+C to stop it.")
    try:
        while proc.poll() is None:
            time.sleep(1.0)
        print(f"The twin API stopped (code {proc.returncode}).")
        return proc.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping the twin ...")
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
