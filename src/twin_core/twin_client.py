"""
twin_client.py
--------------
Non-blocking forwarder from other processes (mobile API / live field
controller) to the twin API, which is the twin's single owner.

Inference results are queued and POSTed to ``/twin/ingest/rgb`` from a
daemon thread, rate-limited, and dropped (not retried) if the twin API is
down, so the live pipeline is never slowed by the twin.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional


class TwinForwarder:
    def __init__(
        self,
        base_url: str,
        source: str = "phone_rgb",
        height_above_canopy_m: Optional[float] = 1.5,
        min_interval_s: float = 1.0,
        timeout_s: float = 2.0,
        max_queue: int = 64,
    ):
        """
        ``height_above_canopy_m`` replaces the telemetry altitude, because a
        phone's GPS altitude is above sea level, not above the crop. Pass None
        for sources whose ``alt`` already is height above canopy (drones with
        terrain following).
        """
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.height = height_above_canopy_m
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._last_sent = 0.0
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="twin-forwarder", daemon=True)
        self._thread.start()

    def __call__(self, result: Dict[str, Any], telemetry: Dict[str, Any]) -> bool:
        """Queue one observation. Returns False if it was rate-limited or dropped."""
        now = time.monotonic()
        if now - self._last_sent < self.min_interval_s:
            return False
        tele = dict(telemetry)
        if self.height is not None:
            tele["alt"] = self.height
        payload = {"result": _strip_arrays(result), "telemetry": tele, "source": self.source}
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            self.dropped += 1
            return False
        self._last_sent = now
        return True

    def _run(self) -> None:
        import requests

        while True:
            payload = self._q.get()
            try:
                r = requests.post(f"{self.base_url}/twin/ingest/rgb", json=payload,
                                  timeout=self.timeout_s)
                r.raise_for_status()
                self.sent += 1
            except Exception as e:  # twin API down or rejected: drop, never block
                self.failed += 1
                self.last_error = str(e)[:200]

    def stats(self) -> Dict[str, Any]:
        return {"sent": self.sent, "failed": self.failed, "dropped": self.dropped,
                "queued": self._q.qsize(), "last_error": self.last_error}


def _strip_arrays(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only JSON-friendly scalar/list fields the twin needs."""
    keep = ("crop_stress_score", "stress_label", "crop_stage", "disease_detections",
            "model_confidence", "quality_score", "weed_coverage_pct")
    return {k: result[k] for k in keep if k in result}
