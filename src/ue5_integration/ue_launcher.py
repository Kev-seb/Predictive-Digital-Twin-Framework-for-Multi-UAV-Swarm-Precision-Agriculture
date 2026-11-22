"""
ue_launcher.py - open the UE5 PaddyTwin level from outside the editor (used by the dashboard's
"Open in Unreal Engine" button). Windows only; the dashboard must run on the same PC as Unreal.

  * Unreal already open  -> bring its window to the front and start the paddy twin in it
                            (PythonScriptPlugin remote execution, enabled in the project).
  * Unreal not open      -> launch UnrealEditor with the project; init_paddy.py sees
                            PADDYTWIN_AUTOSTART=1 and starts the twin once the level has loaded.

Paths: UE_EDITOR / UE_PROJECT environment variables win; otherwise the newest UE 5 install
from the Epic Games Launcher, and the first project under Documents/Unreal Projects that
contains /Game/TwinDemo.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START_TWIN = ("import builtins, twin_paddy\n"
              "print('PADDY_ALREADY' if getattr(builtins, '_twin_live', None) is not None else "
              "('PADDY_STARTED', twin_paddy.start())[0])")


# ------------------------------------------------------------------ discovery
def find_editor() -> str | None:
    env = os.environ.get("UE_EDITOR")
    if env and Path(env).is_file():
        return env
    hits = []
    dat = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Epic" / "UnrealEngineLauncher" / "LauncherInstalled.dat"
    try:
        for item in json.loads(dat.read_text(encoding="utf-8")).get("InstallationList", []):
            if str(item.get("AppName", "")).startswith("UE_5"):
                exe = Path(item["InstallLocation"]) / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"
                if exe.is_file():
                    hits.append(str(exe))
    except Exception:
        pass
    for root in dict.fromkeys([os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files", r"D:\Program Files"]):
        hits += glob.glob(os.path.join(root, "Epic Games", "UE_5*", "Engine", "Binaries", "Win64", "UnrealEditor.exe"))
    return sorted(set(hits))[-1] if hits else None


def find_project() -> str | None:
    env = os.environ.get("UE_PROJECT")
    if env and Path(env).is_file():
        return env
    home = Path.home()
    roots = [home / "Documents" / "Unreal Projects", home / "OneDrive" / "Documents" / "Unreal Projects"]
    for root in roots:
        for up in sorted(root.glob("*/*.uproject")):
            if (up.parent / "Content" / "TwinDemo").is_dir():
                return str(up)
    return None


# ------------------------------------------------------------------ editor state
def editor_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq UnrealEditor.exe", "/NH"], capture_output=True,
                             text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return "UnrealEditor.exe" in out
    except Exception:
        return False


def bring_to_front() -> None:
    """Best effort: Windows may only flash the taskbar button instead of switching windows."""
    ps = ("$p = Get-Process UnrealEditor -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle } | "
          "Select-Object -First 1; if ($p) { $w = New-Object -ComObject WScript.Shell; "
          "[void]$w.AppActivate($p.Id) }")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=10,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        pass


def _remote_exec(editor: str, code: str, wait_s: float = 8.0) -> str:
    """Run Python in the open editor (PythonScriptPlugin remote execution). Returns its printed output."""
    rx_dir = Path(editor).parents[2] / "Plugins" / "Experimental" / "PythonScriptPlugin" / "Content" / "Python"
    if str(rx_dir) not in sys.path:
        sys.path.insert(0, str(rx_dir))
    import remote_execution as rx
    r = rx.RemoteExecution(rx.RemoteExecutionConfig())
    r.start()
    try:
        deadline = time.time() + wait_s
        while not r.remote_nodes and time.time() < deadline:
            time.sleep(0.2)
        if not r.remote_nodes:
            raise RuntimeError("the editor did not answer (is remote Python execution enabled in the project?)")
        r.open_command_connection(r.remote_nodes[0]["node_id"])
        res = r.run_command(code, unattended=True, exec_mode=rx.MODE_EXEC_FILE, raise_on_failure=False)
        out = "\n".join(o.get("output", "") for o in res.get("output", []))
        if not res.get("success"):
            raise RuntimeError(f"{res.get('result')}\n{out}".strip())
        return out
    finally:
        r.stop()


# ------------------------------------------------------------------ the button
def open_paddy_twin() -> tuple[bool, str]:
    """(ok, message) for the dashboard."""
    editor = find_editor()
    if editor is None:
        return False, "Unreal Engine 5 was not found. Set UE_EDITOR to the path of UnrealEditor.exe."
    if editor_running():
        bring_to_front()
        try:
            out = _remote_exec(editor, START_TWIN)
        except Exception as e:
            return False, f"Unreal is open, but the paddy twin could not be started from here: {e}"
        if "PADDY_ALREADY" in out:
            return True, "Unreal is open and the paddy twin is already running. Switch to the Unreal window."
        return True, "The paddy twin is now running in Unreal. Switch to the Unreal window."
    project = find_project()
    if project is None:
        return False, "No Unreal project with the PaddyTwin level was found. Set UE_PROJECT to its .uproject file."
    env = dict(os.environ, PADDYTWIN_AUTOSTART="1")
    subprocess.Popen([editor, project], env=env, close_fds=True,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return True, (f"Opening {Path(project).stem} in Unreal Engine. It takes about a minute; the paddy twin "
                  "starts by itself once the level has loaded.")
