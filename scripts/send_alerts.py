"""
send_alerts.py - check the live twin and send farmer alerts (for Windows Task Scheduler / cron).

    python scripts/send_alerts.py            # send to the recipients saved in the dashboard
    python scripts/send_alerts.py --dry-run  # print what would be sent

Schedule it every morning, e.g. Windows:
    schtasks /Create /SC DAILY /ST 06:30 /TN PaddyTwinAlerts /TR "<path>\\.venv\\Scripts\\python.exe <path>\\scripts\\send_alerts.py"
Channels and credentials: see src/alerts/alert_engine.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alerts import alert_engine as ae  # noqa: E402

API = f"http://{os.getenv('UAV_TWIN_API_HOST', '127.0.0.1')}:{os.getenv('UAV_TWIN_API_PORT', '8008')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        frame = json.load(urllib.request.urlopen(f"{API}/ue5/frame?since=-1", timeout=30))
    except Exception as e:
        print(f"The twin API is not reachable at {API}: {e}")
        return 1
    alerts = ae.evaluate(frame)
    name = (frame.get("field_source") or {}).get("name", "your field")
    recipients = ae.load_recipients()
    if not alerts:
        print("No alerts.")
        return 0
    for al in alerts:
        print(f"[{al['level']}] {ae.message(al, 'en')}")
    if a.dry_run or not recipients:
        print("(dry run)" if a.dry_run else "No recipients saved: add them on the dashboard's Farm home page.")
        return 0
    for r in ae.send(alerts, recipients, field_name=name):
        print(f"  {r['to']}: {r['alert']} -> {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
