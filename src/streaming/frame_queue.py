"""
frame_queue.py
--------------
Thread-safe queues for the Live Field Mode pipeline.

Three separate queues flow through the system:
    FrameQueue     — raw UnifiedFrame objects waiting for IQA + AI
    DetectionQueue — RGBInferenceResult objects ready for dashboard display
    TelemetryQueue — FusedTelemetry snapshots for map/telemetry panel
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Optional

from src.streaming.frame_interface import UnifiedFrame
from src.mission.mission_object import RGBInferenceResult, FusedTelemetry


class _BoundedQueue:
    """Thread-safe deque with a maximum capacity (oldest items dropped)."""

    def __init__(self, maxlen: int = 30):
        self._q: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

    def put(self, item: Any) -> None:
        with self._lock:
            self._q.append(item)
        self._not_empty.set()

    def get(self, timeout: float = 0.5) -> Optional[Any]:
        if self._not_empty.wait(timeout=timeout):
            with self._lock:
                if self._q:
                    item = self._q.popleft()
                    if not self._q:
                        self._not_empty.clear()
                    return item
        return None

    def get_latest(self) -> Optional[Any]:
        """Return newest item without blocking, or None if empty."""
        with self._lock:
            return self._q[-1] if self._q else None

    def drain(self) -> list:
        """Return all items and clear the queue."""
        with self._lock:
            items = list(self._q)
            self._q.clear()
            self._not_empty.clear()
        return items

    def qsize(self) -> int:
        with self._lock:
            return len(self._q)

    def empty(self) -> bool:
        with self._lock:
            return len(self._q) == 0


class FrameQueue(_BoundedQueue):
    """Queue of UnifiedFrame objects (raw, pre-inference). Max 10 buffered frames."""
    def __init__(self):
        super().__init__(maxlen=10)

    def put(self, frame: UnifiedFrame) -> None:
        super().put(frame)

    def get(self, timeout: float = 0.5) -> Optional[UnifiedFrame]:
        return super().get(timeout=timeout)


class DetectionQueue(_BoundedQueue):
    """Queue of (UnifiedFrame, RGBInferenceResult) pairs. Max 100 buffered results."""
    def __init__(self):
        super().__init__(maxlen=100)


class TelemetryQueue(_BoundedQueue):
    """Queue of FusedTelemetry snapshots. Max 200 buffered."""
    def __init__(self):
        super().__init__(maxlen=200)

    def put(self, tele: FusedTelemetry) -> None:
        super().put(tele)

    def get_latest(self) -> Optional[FusedTelemetry]:
        return super().get_latest()


class AlertQueue(_BoundedQueue):
    """Queue of alert dicts for real-time display. Max 50 buffered."""
    def __init__(self):
        super().__init__(maxlen=50)
