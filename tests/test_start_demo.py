"""scripts/start_demo.py: starts the twin, reuses a running one, builds the UE5 command."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start_demo", ROOT / "scripts" / "start_demo.py")
start_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(start_demo)


def test_ue_command_opens_project_and_runs_script():
    cmd = start_demo.ue_command(r"C:\Program Files\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor.exe",
                                r"D:\My Projects\AgriTwin\AgriTwin.uproject", Path("C:/repo/twin_live.py"))
    assert cmd.startswith('"C:\\Program Files\\Epic Games\\UE_5.4')
    assert '"D:\\My Projects\\AgriTwin\\AgriTwin.uproject"' in cmd
    assert "-ExecutePythonScript=" in cmd and "twin_live.py" in cmd


def test_dependencies_present_and_api_detection():
    assert start_demo.missing_packages() == []
    assert start_demo.api_up("http://127.0.0.1:9", timeout=0.3) is False


def test_starts_api_runs_demo_and_reuses_it(capsys):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    try:
        assert start_demo.main(["--port", str(port), "--fresh", "--demo", "--no-wait"]) == 0
        assert start_demo.api_up(url)
        out = capsys.readouterr().out
        assert "Starting the twin API" in out and "Demo story started" in out
        assert start_demo.main(["--port", str(port), "--no-wait"]) == 0
        assert "already running" in capsys.readouterr().out
    finally:
        for proc in start_demo.STARTED:
            proc.terminate()
            proc.wait(10)
