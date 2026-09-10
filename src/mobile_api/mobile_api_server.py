"""
mobile_api_server.py
---------------------
FastAPI backend for the Live Field Mode companion app.

Runs on port 8080 (configurable) as a daemon thread alongside Streamlit.

Endpoints:
    GET  /                  → Health check
    GET  /mobile            → Serve Android PWA HTML
    GET  /mobile/{file}     → Serve PWA assets
    POST /api/frame         → JPEG frame upload (fallback mode)
    POST /api/frames/bulk   → Bulk offline reconnect upload
    WS   /ws/telemetry      → Real-time telemetry stream
    WS   /ws/signal         → WebRTC signaling channel
    GET  /api/mission/start → Start mission
    GET  /api/mission/stop  → Stop mission
    GET  /api/mission/status → Mission status
    GET  /api/missions      → List past missions
    GET  /api/detections    → Latest AI detections
    GET  /api/qr            → QR code for phone connection
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Controller is injected at startup — avoids circular imports
_controller = None
MOBILE_APP_DIR = Path(__file__).parent.parent / "mobile" / "companion_app"
API_PORT = 8080


def set_controller(controller) -> None:
    """Inject the LiveFieldController instance."""
    global _controller
    _controller = controller


# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(
    title="Live Field Mode API",
    description="Mobile backend for smartphone-powered UAV Ground Station",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket clients
_ws_telemetry_clients: set = set()
_ws_signal_clients: dict = {}   # session_id → websocket
_active_drones_telemetry: dict = {} # drone_id → telemetry_packet


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok", "service": "Live Field Mode API", "version": "1.0.0"}


# ── Serve PWA ─────────────────────────────────────────────────────────

@app.get("/mobile", response_class=HTMLResponse)
async def serve_pwa():
    """Serve the Android companion PWA."""
    index_path = MOBILE_APP_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PWA not built yet. Run: python setup.py install</h1>")


@app.get("/mobile/{filename}")
async def serve_pwa_asset(filename: str):
    """Serve PWA JS/CSS/manifest assets."""
    asset_path = MOBILE_APP_DIR / filename
    if asset_path.exists() and asset_path.is_file():
        content_types = {
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".json": "application/json",
            ".png":  "image/png",
            ".ico":  "image/x-icon",
        }
        ct = content_types.get(asset_path.suffix, "application/octet-stream")
        return Response(content=asset_path.read_bytes(), media_type=ct)
    raise HTTPException(status_code=404, detail=f"Asset not found: {filename}")


# ── Frame upload (JPEG fallback) ────────────────────────────────────────

@app.post("/api/frame")
async def upload_frame(
    file: UploadFile = File(...),
    camera_profile: str = "default",
    telemetry: Optional[str] = Form(None),
):
    """
    Receive a single JPEG frame from the phone (fallback transport).
    Used when WebRTC is unavailable.
    """
    if _controller is None:
        raise HTTPException(503, "Controller not ready")

    # Ingest telemetry if provided (e.g. over HTTP fallback)
    if telemetry:
        try:
            packet = json.loads(telemetry)
            _controller.ingest_telemetry(packet)
        except Exception:
            pass

    jpeg_bytes = await file.read()
    success = _controller.ingest_jpeg_frame(jpeg_bytes, camera_profile=camera_profile)
    return {"accepted": success, "timestamp": time.time()}


@app.post("/api/frame/stereo")
async def upload_stereo_frame(
    left_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
    metadata: str = Form(...),
):
    """
    Ingest a left/right stereo image pair from Unity,
    execute SGBM disparity mapping, extract canopy height model,
    and update the digital twin state.
    """
    if _controller is None:
        raise HTTPException(503, "Controller not ready")
        
    import numpy as np
    import cv2
        
    try:
        meta = json.loads(metadata)
        drone_id = meta.get("drone_id", "unknown")
        frame_id = meta.get("frame_id", 0)
        baseline = float(meta.get("baseline", 0.5))
        focal_length = float(meta.get("focal_length", 1200.0))
        position = meta.get("position", [0.0, 0.0, 30.0])
        altitude = float(position[2]) if len(position) > 2 else 30.0
    except Exception as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}")
        
    # Read files
    left_bytes = await left_file.read()
    right_bytes = await right_file.read()
    
    # Decode image buffers
    left_arr = np.frombuffer(left_bytes, np.uint8)
    right_arr = np.frombuffer(right_bytes, np.uint8)
    
    left_img = cv2.imdecode(left_arr, cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(right_arr, cv2.IMREAD_COLOR)
    
    if left_img is None or right_img is None:
        raise HTTPException(400, "Left or Right image decoding failed")
        
    # Run reconstruction disparity map using UAVSpatialReconstruction
    from src.spatial.reconstruction import UAVSpatialReconstruction
    reconstructor = UAVSpatialReconstruction(
        focal_length_px=focal_length,
        baseline_meters=baseline,
        uav_altitude_meters=altitude
    )
    
    try:
        dsm = reconstructor.generate_dsm(left_img, right_img)
        chm, dtm = reconstructor.generate_chm(dsm)
        max_height = float(np.max(chm))
    except Exception as e:
        # Fallback if SGBM fails
        max_height = 0.0
        
    # Feed the Left image into the core deep learning pipeline to run stress classification
    left_jpg_bytes = cv2.imencode(".jpg", left_img)[1].tobytes()
    accepted = _controller.ingest_jpeg_frame(left_jpg_bytes)
    
    # Update Digital Twin memory state
    if not hasattr(_controller, "digital_twin_state") or _controller.digital_twin_state is None:
        _controller.digital_twin_state = {}
    _controller.digital_twin_state["canopy_max_height"] = max_height
    _controller.digital_twin_state["last_active_drone"] = drone_id
    
    return {
        "accepted": accepted,
        "frame_id": frame_id,
        "reconstructed": True,
        "max_height": round(max_height, 2),
        "timestamp": time.time()
    }


@app.get("/api/field/state")
async def get_field_state(rows: int = 20, columns: int = 20, test_fixture: Optional[str] = None):
    """
    Return a grid of NDVI and stress values for each crop position.
    Validates input ranges and returns:
      - A deterministic layout if test_fixture == "deterministic"
      - Actual active digital twin spatial states from historical survey records if available
      - A simulated fallback state with a center stress hotspot if no real surveys exist
    """
    if _controller is None:
        raise HTTPException(503, "Controller not ready")
        
    if rows <= 0 or columns <= 0 or rows > 100 or columns > 100:
        raise HTTPException(400, "Rows and columns must be between 1 and 100")
        
    import numpy as np
    import cv2
    from src.digital_twin.twin import FieldDigitalTwin
    
    # 1. Deterministic test fixture validation
    if test_fixture == "deterministic":
        ndvi = np.full((rows, columns), 0.75, dtype=np.float32)
        stress = np.full((rows, columns), 0.10, dtype=np.float32)
        
        # Healthy corner (top-left)
        ndvi[:min(5, rows), :min(5, columns)] = 0.85
        stress[:min(5, rows), :min(5, columns)] = 0.05
        
        # Mild-stress region (bottom-right)
        ndvi[max(0, rows - 5):, max(0, columns - 5):] = 0.50
        stress[max(0, rows - 5):, max(0, columns - 5):] = 0.35
        
        # Severe-stress center
        center_r = rows // 2
        center_c = columns // 2
        ndvi[max(0, center_r - 2):min(rows, center_r + 3), max(0, center_c - 2):min(columns, center_c + 3)] = 0.15
        stress[max(0, center_r - 2):min(rows, center_r + 3), max(0, center_c - 2):min(columns, center_c + 3)] = 0.85
        
        return {
            "rows": rows,
            "columns": columns,
            "ndvi": ndvi.flatten().tolist(),
            "stress": stress.flatten().tolist(),
            "is_simulation_fallback": True,
            "is_test_fixture": True,
            "timestamp": time.time()
        }

    # 2. Try to load actual digital-twin spatial state
    try:
        twin = FieldDigitalTwin()
        if twin.state.get("surveys_logged"):
            # Load the latest survey
            latest_date = twin.state["surveys_logged"][-1]
            npz_path = twin.history_dir / f"survey_{latest_date}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                ndvi_orig = data["ndvi"].astype(np.float32)
                stress_orig = data["stress"].astype(np.float32)
                
                # Check dimensions and resample to match requested grid size
                if ndvi_orig.ndim == 1:
                    # Reshape to square if flat
                    side = int(np.sqrt(ndvi_orig.size))
                    ndvi_orig = ndvi_orig.reshape((side, side))
                    stress_orig = stress_orig.reshape((side, side))
                
                ndvi_resampled = cv2.resize(ndvi_orig, (columns, rows), interpolation=cv2.INTER_LINEAR)
                stress_resampled = cv2.resize(stress_orig, (columns, rows), interpolation=cv2.INTER_LINEAR)
                
                return {
                    "rows": rows,
                    "columns": columns,
                    "ndvi": ndvi_resampled.flatten().tolist(),
                    "stress": stress_resampled.flatten().tolist(),
                    "is_simulation_fallback": False,
                    "is_test_fixture": False,
                    "timestamp": time.time()
                }
    except Exception as e:
        # Silently log and drop through to fallback
        pass

    # 3. Simulation fallback state
    # Start with a base healthy state (mean NDVI=0.72, stress=0.12)
    ndvi = np.random.normal(0.72, 0.05, (rows, columns))
    
    # Read mean stress if available from live observations to scale our baseline
    live_mean_stress = 0.12
    if hasattr(_controller, "digital_twin_state") and _controller.digital_twin_state:
        live_mean_stress = _controller.digital_twin_state.get("cumulative_stress_index", 0.12)
    
    # Adjust base values
    stress = np.clip((1.0 - ndvi) / 2.0 + (live_mean_stress - 0.12), 0.0, 1.0)
    
    # Inject a simulated circular pathogen hotspot to demonstrate visual state sync
    center_r = rows // 2
    center_c = columns // 2
    for r in range(rows):
        for c in range(columns):
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            if dist < 4:
                # Severity increases towards the center of the hotspot
                intensity = 1.0 - (dist / 4.0)
                ndvi[r, c] = np.clip(ndvi[r, c] - 0.35 * intensity, -1.0, 1.0)
                stress[r, c] = np.clip(stress[r, c] + 0.50 * intensity, 0.0, 1.0)
                
    # Flatten arrays to list to match JsonUtility compatibility contract
    ndvi_list = ndvi.flatten().tolist()
    stress_list = stress.flatten().tolist()
    
    return {
        "rows": rows,
        "columns": columns,
        "ndvi": ndvi_list,
        "stress": stress_list,
        "is_simulation_fallback": True,
        "is_test_fixture": False,
        "timestamp": time.time()
    }


@app.get("/api/field/scenario/state")
async def get_field_scenario_state(day: int = 1, mode: str = "SIMULATION", rows: int = 20, columns: int = 20):
    """
    Timeline state query endpoint for Unity.
    - If mode == "REAL", returns the historical survey corresponding to the day offset index.
    - If mode == "SIMULATION", returns a deterministic spatiotemporal simulation progression.
    """
    if _controller is None:
        raise HTTPException(503, "Controller not ready")
        
    if rows <= 0 or columns <= 0 or rows > 100 or columns > 100:
        raise HTTPException(400, "Rows and columns must be between 1 and 100")
        
    import numpy as np
    import cv2
    from src.digital_twin.twin import FieldDigitalTwin
    
    is_sim = (mode.upper() == "SIMULATION")
    
    if not is_sim:
        # Load historical survey state by day index (1-based)
        try:
            twin = FieldDigitalTwin()
            surveys = twin.state.get("surveys_logged", [])
            if surveys:
                idx = max(0, min(day - 1, len(surveys) - 1))
                latest_date = surveys[idx]
                npz_path = twin.history_dir / f"survey_{latest_date}.npz"
                if npz_path.exists():
                    data = np.load(npz_path)
                    ndvi_orig = data["ndvi"].astype(np.float32)
                    stress_orig = data["stress"].astype(np.float32)
                    
                    if ndvi_orig.ndim == 1:
                        side = int(np.sqrt(ndvi_orig.size))
                        ndvi_orig = ndvi_orig.reshape((side, side))
                        stress_orig = stress_orig.reshape((side, side))
                        
                    ndvi_res = cv2.resize(ndvi_orig, (columns, rows), interpolation=cv2.INTER_LINEAR)
                    stress_res = cv2.resize(stress_orig, (columns, rows), interpolation=cv2.INTER_LINEAR)
                    
                    return {
                        "rows": rows,
                        "columns": columns,
                        "ndvi": ndvi_res.flatten().tolist(),
                        "stress": stress_res.flatten().tolist(),
                        "is_simulation": False,
                        "timestamp": time.time()
                    }
        except Exception:
            pass
            
        # Fallback if no historical data is found
        is_sim = True
        
    # Generate deterministic simulation scenario progression
    ndvi = np.full((rows, columns), 0.75, dtype=np.float32)
    stress = np.full((rows, columns), 0.10, dtype=np.float32)
    
    center_r, center_c = rows // 2, columns // 2
    
    if day <= 1:
        # Day 1: Healthy
        pass
    elif day <= 5:
        # Day 5: Mild Stress
        for r in range(rows):
            for c in range(columns):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 6:
                    ndvi[r, c] = 0.58
                    stress[r, c] = 0.28
    elif day <= 10:
        # Day 10: Disease Hotspot
        for r in range(rows):
            for c in range(columns):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 4:
                    ndvi[r, c] = 0.15
                    stress[r, c] = 0.85
                elif dist < 8:
                    ndvi[r, c] = 0.48
                    stress[r, c] = 0.38
    elif day <= 15:
        # Day 15: Intervention
        for r in range(rows):
            for c in range(columns):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 4:
                    ndvi[r, c] = 0.42
                    stress[r, c] = 0.48
                elif dist < 8:
                    ndvi[r, c] = 0.65
                    stress[r, c] = 0.20
    else:
        # Day 20: Recovery
        for r in range(rows):
            for c in range(columns):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 4:
                    ndvi[r, c] = 0.70
                    stress[r, c] = 0.15
                
    return {
        "rows": rows,
        "columns": columns,
        "ndvi": ndvi.flatten().tolist(),
        "stress": stress.flatten().tolist(),
        "is_simulation": True,
        "timestamp": time.time()
    }


@app.post("/api/frames/bulk")
async def upload_bulk_frames(frames: list):
    """
    Receive a batch of offline-cached frames when phone reconnects.
    Each frame is: { "jpeg_b64": str, "timestamp": float, "telemetry": dict }
    """
    if _controller is None:
        raise HTTPException(503, "Controller not ready")

    accepted = 0
    for frame_data in frames:
        try:
            jpeg_b64 = frame_data.get("jpeg_b64", "")
            jpeg_bytes = base64.b64decode(jpeg_b64)

            # Replay telemetry for this frame
            tele = frame_data.get("telemetry", {})
            if tele:
                _controller.ingest_telemetry(tele)

            ok = _controller.ingest_jpeg_frame(jpeg_bytes)
            if ok:
                accepted += 1
        except Exception:
            pass

    return {"frames_received": len(frames), "frames_accepted": accepted}


# ── WebSocket: Telemetry ───────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """
    Bidirectional telemetry WebSocket.
    Phone → Server: telemetry JSON packets
    Server → Phone: acknowledgement + mission status
    """
    await websocket.accept()
    _ws_telemetry_clients.add(websocket)

    try:
        async for message in websocket.iter_text():
            try:
                packet = json.loads(message)
                msg_type = packet.get("type", "telemetry")

                if msg_type == "telemetry" and _controller:
                    tele = _controller.ingest_telemetry(packet)
                    
                    # Compute potential-field velocity command if drone_id is present
                    drone_id = packet.get("drone_id")
                    if drone_id:
                        import numpy as np
                        # 1. Update active tracker dict
                        _active_drones_telemetry[drone_id] = packet
                        
                        # 2. Extract state vectors
                        pos = np.array(packet.get("position", [0.0, 0.0, 10.0]))
                        vel = np.array(packet.get("velocity", [0.0, 0.0, 0.0]))
                        seq = packet.get("telemetry_seq", 0)
                        
                        # 3. Target zone (pull toward origin center [0, 0, 10])
                        target_pos = np.array([0.0, 0.0, 10.0])
                        
                        # 4. Attractive force
                        k_att = 0.5
                        F_att = -k_att * (pos - target_pos)
                        
                        # 5. Repulsive force (collision avoidance safety boundary d0 = 6m)
                        F_rep = np.zeros(3)
                        k_rep = 35.0
                        d0 = 6.0
                        
                        for other_id, other_packet in _active_drones_telemetry.items():
                            if other_id != drone_id:
                                other_pos = np.array(other_packet.get("position", [0.0, 0.0, 10.0]))
                                diff = pos - other_pos
                                dist = np.linalg.norm(diff)
                                if 0.05 < dist < d0:
                                    direction = diff / dist
                                    rep_mag = k_rep * (1.0 / dist - 1.0 / d0) * (1.0 / (dist ** 2))
                                    F_rep += direction * rep_mag
                        
                        # 6. Combined acceleration command
                        dt = 0.05
                        total_acc = F_att + F_rep
                        vel_cmd = vel + total_acc * dt
                        
                        # Clamp velocity to a safe speed limit (5 m/s)
                        speed_limit = 5.0
                        cmd_speed = np.linalg.norm(vel_cmd)
                        if cmd_speed > speed_limit:
                            vel_cmd = (vel_cmd / cmd_speed) * speed_limit
                        
                        await websocket.send_json({
                            "type": "velocity_command",
                            "drone_id": drone_id,
                            "telemetry_seq": seq,
                            "velocity": vel_cmd.tolist(),
                            "acceleration": total_acc.tolist(),
                            "control_mode": "potential_field",
                            "timestamp": time.time(),
                        })
                    else:
                        # Standard mobile phone response: ack with server timestamp
                        await websocket.send_json({
                            "type": "ack",
                            "server_ts": time.time(),
                            "mission_active": _controller.is_active,
                        })

                elif msg_type == "heartbeat":
                    await websocket.send_json({
                        "type": "heartbeat_ack",
                        "server_ts": time.time(),
                    })

                elif msg_type == "frame":
                    # Phone can also send frames via WS as base64 (low latency alternative)
                    jpeg_b64 = packet.get("jpeg_b64", "")
                    if jpeg_b64 and _controller:
                        jpeg_bytes = base64.b64decode(jpeg_b64)
                        _controller.ingest_jpeg_frame(jpeg_bytes)

            except Exception:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        _ws_telemetry_clients.discard(websocket)


# ── WebSocket: WebRTC signaling ────────────────────────────────────────

@app.websocket("/ws/signal")
async def ws_signal(websocket: WebSocket):
    """
    WebRTC signaling channel.
    Relays SDP offers/answers and ICE candidates between phone and server.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    _ws_signal_clients[session_id] = websocket

    try:
        async for message in websocket.iter_text():
            try:
                msg = json.loads(message)
                msg_type = msg.get("type", "")

                if msg_type == "offer":
                    # Phone sends SDP offer — forward to WebRTC bridge
                    sdp = msg.get("sdp", "")
                    from src.streaming.webrtc_bridge import WebRTCBridge
                    bridge = WebRTCBridge.get_instance()
                    answer_sdp = await bridge.handle_offer(sdp, session_id)
                    if answer_sdp:
                        await websocket.send_json({
                            "type": "answer",
                            "sdp": answer_sdp,
                        })

                elif msg_type == "ice_candidate":
                    # ICE candidate from phone
                    candidate = msg.get("candidate", {})
                    from src.streaming.webrtc_bridge import WebRTCBridge
                    bridge = WebRTCBridge.get_instance()
                    await bridge.add_ice_candidate(session_id, candidate)

            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _ws_signal_clients.pop(session_id, None)


# ── Mission endpoints ─────────────────────────────────────────────────

@app.get("/api/mission/start")
async def start_mission(
    name: str = "",
    lat: float = 0.0,
    lon: float = 0.0,
    field_name: str = "Field Alpha",
):
    if _controller is None:
        raise HTTPException(503, "Controller not ready")

    mission_id = _controller.start_mission(
        name=name, field_lat=lat, field_lon=lon, field_name=field_name
    )
    return {"mission_id": mission_id, "started": True}


@app.get("/api/mission/stop")
async def stop_mission():
    if _controller is None:
        raise HTTPException(503, "Controller not ready")
    mid = _controller.stop_mission()
    return {"mission_id": mid, "stopped": True}


@app.get("/api/mission/status")
async def mission_status():
    if _controller is None:
        return {"active": False}
    return {
        "active": _controller.is_active,
        "mission_id": _controller.mission_id,
        "elapsed_s": _controller.elapsed_seconds,
        "qos": _controller.qos_metrics,
    }


@app.get("/api/missions")
async def list_missions():
    from src.mission import mission_database as db
    return {"missions": db.list_missions(limit=20)}


@app.get("/api/detections")
async def get_detections():
    """Return the latest AI inference result."""
    if _controller is None:
        raise HTTPException(503, "Controller not ready")
    result = _controller.latest_result
    if result is None:
        return {"detections": None}
    return {"detections": result.to_dict(), "telemetry": _controller.latest_telemetry.to_dict()}


@app.get("/api/qr")
async def get_qr_code():
    """
    Return a QR code image encoding the server's local WiFi URL.
    Phone scans this to auto-connect.
    """
    try:
        import qrcode
        local_ip = _get_local_ip()
        url = f"http://{local_ip}:{API_PORT}/mobile"
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"url": url, "qr_png_b64": b64}
    except ImportError:
        local_ip = _get_local_ip()
        return {"url": f"http://{local_ip}:{API_PORT}/mobile", "qr_png_b64": None}


def _get_local_ip() -> str:
    """Get the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Server startup (called as daemon thread) ───────────────────────────

def run_server(port: int = API_PORT) -> None:
    """Start the uvicorn server. Meant to run in a daemon thread."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def start_mobile_api_server(controller, port: int = API_PORT) -> int:
    """
    Start the FastAPI mobile API server in a daemon thread.
    Returns the port it's listening on.
    """
    set_controller(controller)
    
    # Initialize public tunnel URL attribute dynamically
    controller.public_tunnel_url = None

    # Try port range
    actual_port = port
    for p in range(port, port + 10):
        try:
            import socket as _sock
            s = _sock.socket()
            s.bind(("0.0.0.0", p))
            s.close()
            actual_port = p
            break
        except OSError:
            continue

    thread = threading.Thread(
        target=run_server, args=(actual_port,), daemon=True
    )
    thread.start()
    time.sleep(0.5)  # Let the server start
    print(f"Live Field Mode API server started on http://0.0.0.0:{actual_port}")

    # Launch automatic localhost.run tunnel in background
    _start_tunnel_process(controller, actual_port)

    return actual_port


def _start_tunnel_process(controller, port: int) -> None:
    """Run localtunnel in a background thread and parse the URL."""
    import subprocess
    import re
    import shutil
    import sys

    # Find lt or npx
    lt_path = shutil.which("lt")
    npx_path = shutil.which("npx")
    if not lt_path and not npx_path:
        print("[Tunnel] localtunnel/npx not found on system. Auto-tunneling disabled.")
        return

    cmd = [lt_path] if lt_path else [npx_path, "localtunnel"]
    cmd += ["--port", str(port)]

    # Add shell=True on Windows if calling via npm package
    is_windows = sys.platform.startswith("win")

    def run_tunnel():
        while True:
            try:
                print(f"[Tunnel] Starting localtunnel on port {port}...")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=is_windows
                )
                
                # Read stdout line by line
                for line in iter(proc.stdout.readline, ""):
                    # Look for URL like https://xyz.loca.lt
                    match = re.search(r"https://[a-zA-Z0-9-.]+\.loca\.lt", line)
                    if match:
                        url = match.group(0)
                        controller.public_tunnel_url = url
                        print(f"[Tunnel] Active URL auto-detected: {url}")
                
                # Process ended
                proc.wait()
                print("[Tunnel] Localtunnel process exited. Restarting in 5 seconds...")
            except Exception as e:
                print(f"[Tunnel] Error running localtunnel: {e}")
            time.sleep(5)

    thread = threading.Thread(target=run_tunnel, daemon=True)
    thread.start()

