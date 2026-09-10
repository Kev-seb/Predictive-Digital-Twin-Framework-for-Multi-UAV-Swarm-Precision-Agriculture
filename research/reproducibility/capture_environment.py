"""
capture_environment.py
Captures and saves the complete runtime environment for reproducibility.
Writes research/reproducibility/environment.json
"""
import json
import sys
import os
import platform
import subprocess
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, cwd=str(ROOT),
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"

def capture():
    env = {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": {
            "commit": _run("git rev-parse HEAD"),
            "branch": _run("git rev-parse --abbrev-ref HEAD"),
            "status": _run("git status --short"),
            "tag": _run("git describe --tags --abbrev=0"),
        },
        "hardware": {
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": {},
    }

    # Capture installed packages
    pkgs_raw = _run(f"{sys.executable} -m pip list --format=json")
    try:
        pkgs = json.loads(pkgs_raw)
        env["packages"] = {p["name"]: p["version"] for p in pkgs}
    except Exception:
        env["packages"] = {"error": pkgs_raw}

    # PyTorch specific
    try:
        import torch
        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "num_threads": torch.get_num_threads(),
        }
    except ImportError:
        env["torch"] = {"error": "not installed"}

    out_path = ROOT / "research" / "reproducibility" / "environment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"[ENV] Saved environment to {out_path}")
    return env

if __name__ == "__main__":
    e = capture()
    print(f"  Git commit : {e['git']['commit'][:12]}")
    print(f"  Python     : {e['python']['version'].split()[0]}")
    print(f"  PyTorch    : {e.get('torch', {}).get('version', 'N/A')}")
    print(f"  CUDA       : {e.get('torch', {}).get('cuda_available', False)}")
