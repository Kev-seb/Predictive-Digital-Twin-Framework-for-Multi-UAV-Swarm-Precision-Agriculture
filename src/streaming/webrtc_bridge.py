"""
webrtc_bridge.py
-----------------
WebRTC peer connection manager using aiortc.

Handles:
    - SDP offer/answer exchange
    - ICE candidate management
    - Video track reception from phone
    - Frame extraction → OpenCV ndarray → LiveFieldController

Architecture:
    Phone Browser (WebRTC) ──── SDP/ICE via /ws/signal ────► Server
    Phone Camera Track ─────────── RTP Video ──────────────► aiortc
                                                              │
                                                           VideoTrack
                                                              │
                                                         frame_callback
                                                              │
                                                    LiveFieldController.ingest_webrtc_frame()
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import threading
from typing import Dict, Optional

import av
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import — aiortc is optional (falls back to JPEG mode if not installed)
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaBlackhole
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("aiortc not available — WebRTC video disabled. Falling back to JPEG upload mode.")


class WebRTCBridge:
    """
    Singleton WebRTC bridge that manages peer connections.
    One RTCPeerConnection per phone session.
    """
    _instance: Optional["WebRTCBridge"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._peer_connections: Dict[str, "RTCPeerConnection"] = {}
        self._controller = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> "WebRTCBridge":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_controller(self, controller) -> None:
        self._controller = controller

    async def handle_offer(self, sdp: str, session_id: str) -> Optional[str]:
        """
        Process an SDP offer from the phone and return an SDP answer.
        """
        if not AIORTC_AVAILABLE:
            logger.warning("WebRTC not available — returning None answer")
            return None

        try:
            pc = RTCPeerConnection()
            self._peer_connections[session_id] = pc

            @pc.on("connectionstatechange")
            async def on_state():
                logger.info(f"WebRTC [{session_id}] state: {pc.connectionState}")
                if pc.connectionState in ("failed", "closed"):
                    self._peer_connections.pop(session_id, None)

            @pc.on("track")
            def on_track(track):
                logger.info(f"WebRTC track received: {track.kind}")
                if track.kind == "video":
                    # Spawn async task to consume frames
                    asyncio.ensure_future(self._consume_video(track, session_id))

            # Set remote description (phone's offer)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type="offer")
            )

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return pc.localDescription.sdp

        except Exception as e:
            logger.error(f"WebRTC offer handling failed: {e}")
            return None

    async def add_ice_candidate(self, session_id: str, candidate: dict) -> None:
        """Add an ICE candidate from the phone to the peer connection."""
        if not AIORTC_AVAILABLE:
            return

        pc = self._peer_connections.get(session_id)
        if pc is None:
            return

        try:
            from aiortc import RTCIceCandidate
            ice = RTCIceCandidate(
                component=candidate.get("component", 1),
                foundation=candidate.get("foundation", ""),
                ip=candidate.get("ip", ""),
                port=candidate.get("port", 0),
                priority=candidate.get("priority", 0),
                protocol=candidate.get("protocol", "udp"),
                type=candidate.get("type", "host"),
                sdpMid=candidate.get("sdpMid"),
                sdpMLineIndex=candidate.get("sdpMLineIndex"),
            )
            await pc.addIceCandidate(ice)
        except Exception as e:
            logger.debug(f"ICE candidate error: {e}")

    async def _consume_video(self, track, session_id: str) -> None:
        """
        Consume video frames from the WebRTC track and forward to controller.
        """
        try:
            while True:
                frame = await track.recv()

                if self._controller is None:
                    continue

                try:
                    # Convert av.VideoFrame → numpy RGB
                    img = frame.to_ndarray(format="rgb24")
                    self._controller.ingest_webrtc_frame(img)
                except Exception:
                    pass

        except Exception:
            pass  # Track closed
        finally:
            self._peer_connections.pop(session_id, None)

    async def close_all(self) -> None:
        for pc in list(self._peer_connections.values()):
            await pc.close()
        self._peer_connections.clear()

    @property
    def active_connections(self) -> int:
        return len(self._peer_connections)

    @property
    def webrtc_available(self) -> bool:
        return AIORTC_AVAILABLE
