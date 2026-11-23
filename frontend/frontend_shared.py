"""
dashboard.py
------------
AI-Powered UAV Crop Stress Intelligence & Temporal Precision Agriculture Platform
Research-Grade Streamlit Dashboard

Run with:
    streamlit run dashboard.py

Tabs:
    1. Upload & Process      — load multispectral TIFF, compute indices
    2. Vegetation Analytics  — NDVI/NDRE/NDWI/EVI heatmaps + stats
    3. Stress Intelligence   — stress segmentation, severity map, GradCAM
    4. Temporal Analytics    — cross-stage NDVI/stress progression
    5. Field Zoning (GIS)    — management zones, stress regions, grid map
    6. Weather & Risk        — weather-aware stress inference
    7. AI Report             — auto-generated precision agriculture report
"""

import io
import os
import sys
import warnings
from pathlib import Path

# Add project root to Python path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ── MAVLink Telemetry Backend Service ──────────────────────────
import threading
import json
import time
import math
import asyncio
import websockets
from http.server import HTTPServer, BaseHTTPRequestHandler

from src.digital_twin.camera_feed import get_live_camera_frame
from src.digital_twin.flight_physics import UAVFlightDynamicsSimulator

# Set up shared global telemetry data state with st.cache_resource
@st.cache_resource
def get_shared_state():
    return {
        "MAVLINK_TELEMETRY": {
            "lat": 11.0,
            "lon": 79.0,
            "alt": 10.0,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "battery": 100.0,
            "speed": 0.0,
            "connected": False,
            "is_spraying": False,
            "payload_mass": 10.0,
            "autopilot_mode": "stabilized"
        },
        "CURRENT_ENV": {
            "wind_speed": 8.0,
            "wind_direction": 45.0,
            "season": "Spring Green",
            "active_agent": "Dynamic Smart Tracking"
        },
        "TELEMETRY_BUFFER": [],
        "REPLAY_STATE": {
            "is_replaying": False,
            "current_index": 0,
            "speed": 1,
            "paused": False,
            "total_frames": 0
        },
        "GPU_ENGINE": None,
        "MAVLINK_CONNECTION_STRING": "udp:127.0.0.1:14550",
        "MAVLINK_PROTOCOL": "Generic MAVLink",
        "CAMERA_SOURCE": "Simulated UAV Camera",
        "HOME_LAT": 11.0,
        "HOME_LON": 79.0,
        "CHM": None,
        "NDVI_MAP": None,
        "PHYSICS_SIMULATOR": None,
        "PHYSICS_SIMULATORS": {},
        "SWARM_WARNINGS": [],
        "ACTIVE_PARTICLES": np.zeros((0, 3), dtype=np.float32),
        "WS_CLIENTS": set(),
        "TELEMETRY_PORT": None,
        "WEBSOCKET_PORT": None,
        "MULTIPLAYER_DRONES": {
            "drone_alpha": {
                "id": "drone_alpha",
                "label": "Drone Alpha (Sector A)",
                "lat": 37.7749,
                "lon": -122.4194,
                "alt": 10.0,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "battery": 100.0,
                "speed": 0.0,
                "is_spraying": False,
                "color": "#38bdf8"
            },
            "drone_beta": {
                "id": "drone_beta",
                "label": "Drone Beta (Sector B)",
                "lat": 37.7749,
                "lon": -122.4194,
                "alt": 10.0,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "battery": 100.0,
                "speed": 0.0,
                "is_spraying": False,
                "color": "#ec4899"
            }
        },
        "COLLABORATIVE_ANNOTATIONS": [],
        "WS_CLIENT_ROLES": {},
        "WS_CLIENT_CAMERAS": {}
    }

shared_state = get_shared_state()
MAVLINK_TELEMETRY = shared_state["MAVLINK_TELEMETRY"]
CURRENT_ENV = shared_state["CURRENT_ENV"]
TELEMETRY_BUFFER = shared_state["TELEMETRY_BUFFER"]
REPLAY_STATE = shared_state["REPLAY_STATE"]
WS_CLIENTS = shared_state["WS_CLIENTS"]
MULTIPLAYER_DRONES = shared_state["MULTIPLAYER_DRONES"]
COLLABORATIVE_ANNOTATIONS = shared_state["COLLABORATIVE_ANNOTATIONS"]
WS_CLIENT_ROLES = shared_state["WS_CLIENT_ROLES"]
WS_CLIENT_CAMERAS = shared_state["WS_CLIENT_CAMERAS"]

MAX_BUFFER_SIZE = 5000
buffer_lock = threading.Lock()

def append_to_buffer(state):
    with buffer_lock:
        TELEMETRY_BUFFER.append({
            "timestamp": time.time(),
            "lat": state["lat"],
            "lon": state["lon"],
            "alt": state["alt"],
            "pitch": state["pitch"],
            "roll": state["roll"],
            "yaw": state["yaw"],
            "battery": state["battery"],
            "speed": state["speed"],
            "is_spraying": state["is_spraying"],
            "connected": state["connected"]
        })
        if len(TELEMETRY_BUFFER) > MAX_BUFFER_SIZE:
            TELEMETRY_BUFFER.pop(0)


class TelemetryRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(MAVLINK_TELEMETRY).encode('utf-8'))
        elif self.path == '/camera':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            while True:
                try:
                    frame_bytes = get_live_camera_frame(shared_state, MAVLINK_TELEMETRY)
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.08)  # ~12 FPS
                except Exception as e:
                    break
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_http_server():
    global TELEMETRY_PORT
    for port in range(8021, 8042):
        try:
            server_address = ('127.0.0.1', port)
            httpd = HTTPServer(server_address, TelemetryRequestHandler)
            TELEMETRY_PORT = port
            shared_state["TELEMETRY_PORT"] = port
            print(f"MAVLink Telemetry API server listening on http://127.0.0.1:{port}/telemetry")
            httpd.serve_forever()
            return
        except OSError:
            continue

def run_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def handler(websocket, path=None):
        WS_CLIENTS.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "telemetry_update":
                        drone_id = data.get("drone_id", "drone_alpha")
                        if drone_id in MULTIPLAYER_DRONES:
                            drone = MULTIPLAYER_DRONES[drone_id]
                            drone["lat"] = data.get("lat", drone["lat"])
                            drone["lon"] = data.get("lon", drone["lon"])
                            drone["alt"] = data.get("alt", drone["alt"])
                            drone["pitch"] = data.get("pitch", drone["pitch"])
                            drone["roll"] = data.get("roll", drone["roll"])
                            drone["yaw"] = data.get("yaw", drone["yaw"])
                            drone["battery"] = data.get("battery", drone["battery"])
                            drone["speed"] = data.get("speed", drone["speed"])
                            drone["is_spraying"] = data.get("is_spraying", drone["is_spraying"])

                        # Only update legacy drone if not in active SITL connection or replay mode
                        if drone_id == "drone_alpha" and not MAVLINK_TELEMETRY.get("connected") and not REPLAY_STATE.get("is_replaying"):
                            MAVLINK_TELEMETRY["lat"] = data.get("lat", MAVLINK_TELEMETRY["lat"])
                            MAVLINK_TELEMETRY["lon"] = data.get("lon", MAVLINK_TELEMETRY["lon"])
                            MAVLINK_TELEMETRY["alt"] = data.get("alt", MAVLINK_TELEMETRY["alt"])
                            MAVLINK_TELEMETRY["pitch"] = data.get("pitch", MAVLINK_TELEMETRY["pitch"])
                            MAVLINK_TELEMETRY["roll"] = data.get("roll", MAVLINK_TELEMETRY["roll"])
                            MAVLINK_TELEMETRY["yaw"] = data.get("yaw", MAVLINK_TELEMETRY["yaw"])
                            MAVLINK_TELEMETRY["battery"] = data.get("battery", MAVLINK_TELEMETRY["battery"])
                            MAVLINK_TELEMETRY["speed"] = data.get("speed", MAVLINK_TELEMETRY["speed"])
                            MAVLINK_TELEMETRY["is_spraying"] = data.get("is_spraying", MAVLINK_TELEMETRY["is_spraying"])
                            append_to_buffer(MAVLINK_TELEMETRY)
                    elif data.get("type") == "client_update":
                        role = data.get("role")
                        if role:
                            WS_CLIENT_ROLES[websocket] = role
                        camera = data.get("camera")
                        if camera:
                            WS_CLIENT_CAMERAS[websocket] = camera
                    elif data.get("type") == "add_annotation":
                        annotation = data.get("annotation")
                        if annotation:
                            COLLABORATIVE_ANNOTATIONS.append(annotation)
                            if len(COLLABORATIVE_ANNOTATIONS) > 50:
                                COLLABORATIVE_ANNOTATIONS.pop(0)
                    elif data.get("type") == "delete_annotation":
                        ann_id = data.get("id")
                        for idx, ann in enumerate(COLLABORATIVE_ANNOTATIONS):
                            if ann.get("id") == ann_id:
                                COLLABORATIVE_ANNOTATIONS.pop(idx)
                                break
                    elif data.get("type") == "clear_annotations":
                        COLLABORATIVE_ANNOTATIONS.clear()
                    elif data.get("type") == "environment_update":
                        CURRENT_ENV["wind_speed"] = data.get("wind_speed", CURRENT_ENV["wind_speed"])
                        CURRENT_ENV["wind_direction"] = data.get("wind_direction", CURRENT_ENV["wind_direction"])
                        CURRENT_ENV["season"] = data.get("season", CURRENT_ENV["season"])
                        CURRENT_ENV["active_agent"] = data.get("active_agent", CURRENT_ENV["active_agent"])
                except Exception as e:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            WS_CLIENTS.discard(websocket)
            if websocket in WS_CLIENT_ROLES:
                del WS_CLIENT_ROLES[websocket]
            if websocket in WS_CLIENT_CAMERAS:
                del WS_CLIENT_CAMERAS[websocket]

    async def broadcast_loop():
        while True:
            if WS_CLIENTS:
                if MAVLINK_TELEMETRY.get("connected") or REPLAY_STATE.get("is_replaying"):
                    for key in ["lat", "lon", "alt", "pitch", "roll", "yaw", "battery", "speed", "is_spraying"]:
                        MULTIPLAYER_DRONES["drone_alpha"][key] = MAVLINK_TELEMETRY[key]

                supervisor_ws = next((ws for ws, role in WS_CLIENT_ROLES.items() if role == "Supervisor"), None)
                sync_camera = WS_CLIENT_CAMERAS.get(supervisor_ws) if supervisor_ws else None

                payload = {
                    "telemetry": MAVLINK_TELEMETRY,
                    "drones": MULTIPLAYER_DRONES,
                    "annotations": COLLABORATIVE_ANNOTATIONS,
                    "sync_camera": sync_camera,
                    "clients": {
                        "total": len(WS_CLIENTS),
                        "roles": list(WS_CLIENT_ROLES.values())
                    },
                    "replay": {
                        "is_replaying": REPLAY_STATE["is_replaying"],
                        "current_index": REPLAY_STATE["current_index"],
                        "paused": REPLAY_STATE["paused"],
                        "total_frames": len(TELEMETRY_BUFFER)
                    },
                    "environment": CURRENT_ENV
                }
                msg = json.dumps(payload)
                await asyncio.gather(*[client.send(msg) for client in WS_CLIENTS], return_exceptions=True)
            await asyncio.sleep(0.05)

    async def main():
        global WEBSOCKET_PORT
        server = None
        for port in range(8786, 8807):
            try:
                server = await websockets.serve(handler, "127.0.0.1", port)
                WEBSOCKET_PORT = port
                shared_state["WEBSOCKET_PORT"] = port
                print(f"WebSocket Sync Server listening on ws://127.0.0.1:{port}")
                break
            except OSError:
                continue
        if server is None:
            print("Failed to bind WebSocket server on any port")
            return
        await broadcast_loop()

    loop.run_until_complete(main())

def start_mavlink_listener():
    active_conn_string = shared_state.get("MAVLINK_CONNECTION_STRING", "udp:127.0.0.1:14550")
    last_packet_time = 0
    connection = None

    try:
        from pymavlink import mavutil
        connection = mavutil.mavlink_connection(active_conn_string)
    except Exception as e:
        connection = None
        print(f"MAVLink Connection initiation failed: {e}")

    # Instantiate flight physics simulators for the swarm
    home_lat = shared_state.get("HOME_LAT", 11.0)
    home_lon = shared_state.get("HOME_LON", 79.0)
    sim_alpha = UAVFlightDynamicsSimulator(shared_state, home_lat=home_lat, home_lon=home_lon, drone_id="drone_alpha")
    sim_beta = UAVFlightDynamicsSimulator(shared_state, home_lat=home_lat, home_lon=home_lon, drone_id="drone_beta")

    # Position them slightly offset from center
    sim_alpha.pos = np.array([-15.0, -15.0, 10.0])
    sim_beta.pos = np.array([15.0, 15.0, 10.0])

    shared_state["PHYSICS_SIMULATOR"] = sim_alpha
    shared_state["PHYSICS_SIMULATORS"] = {
        "drone_alpha": sim_alpha,
        "drone_beta": sim_beta
    }

    # Mock generator state variables
    mock_t = 0.0
    mock_radius = 18.0

    from src.digital_twin.gpu_physics import GPUPhysicsEngine
    try:
        shared_state["GPU_ENGINE"] = GPUPhysicsEngine(max_particles=50000)
    except Exception as e:
        print(f"Failed to load GPU engine: {e}")

    while True:
        # Dynamic Connection Check
        latest_conn_string = shared_state.get("MAVLINK_CONNECTION_STRING", "udp:127.0.0.1:14550")
        if latest_conn_string != active_conn_string:
            print(f"MAVLink Connection string changed from {active_conn_string} to {latest_conn_string}. Re-connecting...")
            active_conn_string = latest_conn_string
            if connection is not None:
                try:
                    connection.close()
                except:
                    pass
            try:
                from pymavlink import mavutil
                connection = mavutil.mavlink_connection(active_conn_string)
            except Exception as e:
                connection = None
                print(f"Re-connection failed: {e}")

        # Step GPU Engine
        if shared_state.get("GPU_ENGINE") is not None:
            engine = shared_state["GPU_ENGINE"]
            # Emit particles for all active spraying swarm drones
            for d_id, drone in MULTIPLAYER_DRONES.items():
                if drone.get("is_spraying"):
                    engine.emit_particles(
                        count=150,
                        source_pos=(drone["lon"], drone["lat"], drone["alt"] - 1.0),
                        initial_velocity=(0.0, 0.0, -4.0),
                        spread=1.5
                    )

            # Emit particles for the primary drone (used in Digital Twin tab)
            if MAVLINK_TELEMETRY.get("is_spraying") and MAVLINK_TELEMETRY.get("connected"):
                engine.emit_particles(
                    count=150,
                    source_pos=(MAVLINK_TELEMETRY["lon"], MAVLINK_TELEMETRY["lat"], MAVLINK_TELEMETRY["alt"] - 1.0),
                    initial_velocity=(0.0, 0.0, -4.0),
                    spread=1.5
                )

            # Use environmental wind if available
            wind_speed = CURRENT_ENV.get("wind_speed", 0.0)
            wind_dir = math.radians(CURRENT_ENV.get("wind_direction", 0.0))
            wx = math.cos(wind_dir) * wind_speed * 9e-6
            wy = math.sin(wind_dir) * wind_speed * 9e-6

            engine.update_particles(dt=0.05, wind_vector=(wx, wy, 0.0))
            shared_state["ACTIVE_PARTICLES"] = engine.get_active_particles_numpy()

        # Check Replay override
        if REPLAY_STATE.get("is_replaying"):
            if not REPLAY_STATE.get("paused") and len(TELEMETRY_BUFFER) > 0:
                idx = int(REPLAY_STATE.get("current_index", 0))
                with buffer_lock:
                    if idx < len(TELEMETRY_BUFFER):
                        frame = TELEMETRY_BUFFER[idx]
                        MAVLINK_TELEMETRY["lat"] = frame["lat"]
                        MAVLINK_TELEMETRY["lon"] = frame["lon"]
                        MAVLINK_TELEMETRY["alt"] = frame["alt"]
                        MAVLINK_TELEMETRY["pitch"] = frame["pitch"]
                        MAVLINK_TELEMETRY["roll"] = frame["roll"]
                        MAVLINK_TELEMETRY["yaw"] = frame["yaw"]
                        MAVLINK_TELEMETRY["battery"] = frame["battery"]
                        MAVLINK_TELEMETRY["speed"] = frame["speed"]
                        MAVLINK_TELEMETRY["is_spraying"] = frame["is_spraying"]
                        MAVLINK_TELEMETRY["connected"] = True

                        next_idx = idx + max(1, int(REPLAY_STATE["speed"]))
                        if next_idx >= len(TELEMETRY_BUFFER):
                            next_idx = 0
                        REPLAY_STATE["current_index"] = next_idx
            time.sleep(0.05)
            continue

        packet_received = False

        if connection is not None:
            try:
                # Ask for telemetry, attitude, status, and actuator/servo outputs
                msg = connection.recv_match(
                    type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'SYS_STATUS', 'VFR_HUD', 'SERVO_OUTPUT_RAW', 'ACTUATOR_OUTPUTS'],
                    blocking=False
                )
                if msg is not None:
                    packet_received = True
                    last_packet_time = time.time()
                    MAVLINK_TELEMETRY["connected"] = True

                    msg_type = msg.get_type()
                    if msg_type == 'GLOBAL_POSITION_INT':
                        MAVLINK_TELEMETRY["lat"] = msg.lat / 1e7
                        MAVLINK_TELEMETRY["lon"] = msg.lon / 1e7
                        MAVLINK_TELEMETRY["alt"] = msg.relative_alt / 1000.0
                    elif msg_type == 'ATTITUDE':
                        MAVLINK_TELEMETRY["pitch"] = msg.pitch
                        MAVLINK_TELEMETRY["roll"] = msg.roll
                        MAVLINK_TELEMETRY["yaw"] = msg.yaw
                    elif msg_type == 'SYS_STATUS':
                        MAVLINK_TELEMETRY["battery"] = msg.battery_remaining / 10.0 if msg.battery_remaining != -1 else 100.0
                    elif msg_type == 'VFR_HUD':
                        MAVLINK_TELEMETRY["speed"] = msg.groundspeed
                    elif msg_type == 'SERVO_OUTPUT_RAW':
                        # ArduPilot spraying output channel detection (e.g. Servo 5 is standard spray output)
                        servo5 = getattr(msg, 'servo5_raw', 1000)
                        MAVLINK_TELEMETRY["is_spraying"] = (servo5 > 1600)
                    elif msg_type == 'ACTUATOR_OUTPUTS':
                        # PX4 spraying output channel detection (often aux channels like channel 4/5)
                        outputs = getattr(msg, 'output', [])
                        if len(outputs) > 4:
                            # normalized range typically -1 to 1, or pwm values
                            MAVLINK_TELEMETRY["is_spraying"] = (outputs[4] > 1500 or outputs[4] > 0.5)
            except Exception as e:
                pass

        # Fallback to high-fidelity 6-DOF simulator if offline
        if not packet_received and (time.time() - last_packet_time) > 4.0:
            MAVLINK_TELEMETRY["connected"] = False

            sim_alpha = shared_state["PHYSICS_SIMULATORS"].get("drone_alpha")
            sim_beta = shared_state["PHYSICS_SIMULATORS"].get("drone_beta")

            # Dynamic home coordinates update from UI
            home_lat = shared_state.get("HOME_LAT", 11.0)
            home_lon = shared_state.get("HOME_LON", 79.0)
            if sim_alpha:
                sim_alpha.home_lat = home_lat
                sim_alpha.home_lon = home_lon
            if sim_beta:
                sim_beta.home_lat = home_lat
                sim_beta.home_lon = home_lon

            swarm_mode = shared_state.get("SWARM_MISSION_TYPE", "coordinated_spraying")
            mock_t += 0.05

            # Determine flight targets and spray commands based on selected swarm mission type
            if swarm_mode == "coordinated_spraying":
                # Coordinated Spraying: Alpha sprays Sector A (West), Beta sprays Sector B (East)
                alpha_y = 25.0 * math.sin(mock_t * 0.15)
                sim_alpha.target_pos = np.array([-15.0, alpha_y, 8.0])
                sim_alpha.target_yaw = math.pi/2 if math.cos(mock_t * 0.15) > 0 else -math.pi/2
                sim_alpha.autopilot_mode = "terrain_follow"
                sim_alpha.is_spraying = (abs(alpha_y) < 15.0)

                beta_y = 25.0 * math.cos(mock_t * 0.15)
                sim_beta.target_pos = np.array([15.0, beta_y, 8.0])
                sim_beta.target_yaw = 0.0 if math.sin(mock_t * 0.15) > 0 else math.pi
                sim_beta.autopilot_mode = "terrain_follow"
                sim_beta.is_spraying = (abs(beta_y) < 15.0)

            elif swarm_mode == "synchronized_scouting":
                # Synchronized Scouting: Parallel scanning sweeps of the field sectors
                alpha_x = -20.0 + 15.0 * math.sin(mock_t * 0.2)
                alpha_y = 20.0 * math.cos(mock_t * 0.05)
                sim_alpha.target_pos = np.array([alpha_x, alpha_y, 12.0])
                sim_alpha.target_yaw = math.atan2(alpha_y, alpha_x)
                sim_alpha.autopilot_mode = "terrain_follow"
                sim_alpha.is_spraying = False

                beta_x = 20.0 + 15.0 * math.cos(mock_t * 0.2)
                beta_y = 20.0 * math.sin(mock_t * 0.05)
                sim_beta.target_pos = np.array([beta_x, beta_y, 12.0])
                sim_beta.target_yaw = math.atan2(beta_y, beta_x)
                sim_beta.autopilot_mode = "terrain_follow"
                sim_beta.is_spraying = False

            elif swarm_mode == "orbit_avoidance_test":
                # Orbit Avoidance: Overlapping circular paths intersecting at center to test Potential Field collision avoidance
                alpha_x = -8.0 + 12.0 * math.cos(mock_t * 0.3)
                alpha_y = 0.0 + 12.0 * math.sin(mock_t * 0.3)
                sim_alpha.target_pos = np.array([alpha_x, alpha_y, 10.0])
                sim_alpha.target_yaw = mock_t * 0.3 + math.pi/2
                sim_alpha.autopilot_mode = "stabilized"
                sim_alpha.is_spraying = False

                beta_x = 8.0 + 12.0 * math.cos(mock_t * 0.3 + math.pi)
                beta_y = 0.0 + 12.0 * math.sin(mock_t * 0.3 + math.pi)
                sim_beta.target_pos = np.array([beta_x, beta_y, 10.0])
                sim_beta.target_yaw = mock_t * 0.3 + math.pi/2 + math.pi
                sim_beta.autopilot_mode = "stabilized"
                sim_beta.is_spraying = False
            else:
                # Single Waypoint mission fallback (Alpha follows mission, Beta orbits at distance)
                waypoints = shared_state.get("MISSION_WAYPOINTS")
                if waypoints and len(waypoints) > 0:
                    wp_idx = shared_state.get("CURRENT_WP_INDEX", 0)
                    if wp_idx >= len(waypoints):
                        wp_idx = 0
                        shared_state["CURRENT_WP_INDEX"] = 0

                    target_wp = waypoints[wp_idx]
                    lat_deg_per_meter = 1.0 / 111320.0
                    lon_deg_per_meter = 1.0 / (111320.0 * math.cos(math.radians(home_lat)))

                    target_y = (target_wp[0] - home_lat) / lat_deg_per_meter
                    target_x = (target_wp[1] - home_lon) / lon_deg_per_meter
                    target_z = target_wp[2]

                    sim_alpha.target_pos = np.array([target_x, target_y, target_z])
                    sim_alpha.autopilot_mode = shared_state.get("AUTOPILOT_MODE", "terrain_follow")

                    dist_to_wp = np.linalg.norm(sim_alpha.pos - sim_alpha.target_pos)
                    if dist_to_wp < 2.5:
                        wp_idx = (wp_idx + 1) % len(waypoints)
                        shared_state["CURRENT_WP_INDEX"] = wp_idx
                        spray_triggers = shared_state.get("MISSION_SPRAY_TRIGGERS")
                        if spray_triggers and wp_idx < len(spray_triggers):
                            sim_alpha.is_spraying = spray_triggers[wp_idx]
                else:
                    # Default orbit
                    alpha_x = 18.0 * math.sin(mock_t * 0.4)
                    alpha_y = 18.0 * math.cos(mock_t * 0.4)
                    sim_alpha.target_pos = np.array([alpha_x, alpha_y, 10.0])
                    sim_alpha.target_yaw = math.atan2(alpha_y, alpha_x)
                    sim_alpha.autopilot_mode = "stabilized"

                # Beta fallback orbit
                beta_x = 18.0 * math.sin(mock_t * 0.4 + math.pi)
                beta_y = 18.0 * math.cos(mock_t * 0.4 + math.pi)
                sim_beta.target_pos = np.array([beta_x, beta_y, 10.0])
                sim_beta.target_yaw = math.atan2(beta_y, beta_x)
                sim_beta.autopilot_mode = "stabilized"

            # Step both flight dynamics simulators
            if sim_alpha:
                sim_alpha.step(dt=0.05)
            if sim_beta:
                sim_beta.step(dt=0.05)

            # Map positions to global coordinates
            lat_deg_per_meter = 1.0 / 111320.0
            lon_deg_per_meter = 1.0 / (111320.0 * math.cos(math.radians(home_lat)))

            # Map main telemetry (Legacy/Alpha support)
            MAVLINK_TELEMETRY["lat"] = home_lat + (sim_alpha.pos[1] * lat_deg_per_meter)
            MAVLINK_TELEMETRY["lon"] = home_lon + (sim_alpha.pos[0] * lon_deg_per_meter)
            MAVLINK_TELEMETRY["alt"] = sim_alpha.pos[2]
            MAVLINK_TELEMETRY["speed"] = np.linalg.norm(sim_alpha.vel)
            MAVLINK_TELEMETRY["yaw"] = sim_alpha.attitude[2]
            MAVLINK_TELEMETRY["pitch"] = sim_alpha.attitude[1]
            MAVLINK_TELEMETRY["roll"] = sim_alpha.attitude[0]
            MAVLINK_TELEMETRY["battery"] = sim_alpha.battery
            MAVLINK_TELEMETRY["is_spraying"] = sim_alpha.is_spraying
            MAVLINK_TELEMETRY["payload_mass"] = sim_alpha.payload_mass
            MAVLINK_TELEMETRY["autopilot_mode"] = sim_alpha.autopilot_mode

            # Sync both drones to MULTIPLAYER_DRONES dictionary
            if "drone_alpha" in MULTIPLAYER_DRONES and sim_alpha:
                d_alpha = MULTIPLAYER_DRONES["drone_alpha"]
                d_alpha["lat"] = MAVLINK_TELEMETRY["lat"]
                d_alpha["lon"] = MAVLINK_TELEMETRY["lon"]
                d_alpha["alt"] = MAVLINK_TELEMETRY["alt"]
                d_alpha["yaw"] = MAVLINK_TELEMETRY["yaw"]
                d_alpha["pitch"] = MAVLINK_TELEMETRY["pitch"]
                d_alpha["roll"] = MAVLINK_TELEMETRY["roll"]
                d_alpha["battery"] = MAVLINK_TELEMETRY["battery"]
                d_alpha["speed"] = MAVLINK_TELEMETRY["speed"]
                d_alpha["is_spraying"] = MAVLINK_TELEMETRY["is_spraying"]

            if "drone_beta" in MULTIPLAYER_DRONES and sim_beta:
                d_beta = MULTIPLAYER_DRONES["drone_beta"]
                d_beta["lat"] = home_lat + (sim_beta.pos[1] * lat_deg_per_meter)
                d_beta["lon"] = home_lon + (sim_beta.pos[0] * lon_deg_per_meter)
                d_beta["alt"] = sim_beta.pos[2]
                d_beta["yaw"] = sim_beta.attitude[2]
                d_beta["pitch"] = sim_beta.attitude[1]
                d_beta["roll"] = sim_beta.attitude[0]
                d_beta["battery"] = sim_beta.battery
                d_beta["speed"] = np.linalg.norm(sim_beta.vel)
                d_beta["is_spraying"] = sim_beta.is_spraying

        # Log current flight telemetry frame to history buffer
        append_to_buffer(MAVLINK_TELEMETRY)
        time.sleep(0.05) # 20Hz

import streamlit as st

@st.cache_resource
def init_mavlink_telemetry_service():
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()

    listener_thread = threading.Thread(target=start_mavlink_listener, daemon=True)
    listener_thread.start()

    # Wait for the ports to bind and update the shared state
    import time
    start_time = time.time()
    while (shared_state["TELEMETRY_PORT"] is None or shared_state["WEBSOCKET_PORT"] is None) and (time.time() - start_time < 2.0):
        time.sleep(0.05)

    print(f"MAVLink & WebSocket Telemetry Services Started: HTTP={shared_state['TELEMETRY_PORT']}, WS={shared_state['WEBSOCKET_PORT']}")
    return True

# Telemetry is started lazily by app.py on operations screens.
# ──────────────────────────────────────────────────────────────


# ── Local modules ─────────────────────────────────────────────
from src.indices.indices import compute_all_indices
from src.core.multispectral_loader import MultispectralImage, load_multispectral_tiff
from src.segmentation.stress_segmentation import (
    rule_based_stress_segmentation, mask_to_overlay,
    CLASS_LABELS, CLASS_COLORS, compute_iou,
)
from src.gis.field_zoning import (
    delineate_management_zones, extract_stress_regions,
    render_zone_map, compute_grid_statistics, plot_grid_heatmap,
)
from src.temporal.temporal_analytics import (
    build_temporal_report, plot_ndvi_progression, plot_stress_progression,
    plot_multi_index_radar, plot_canopy_stress_area, plot_index_heatmap,
    STAGES,
)
from src.weather.weather_stress_inference import fetch_weather, assess_weather_stress
import torch
import cv2
from src.segmentation.deeplabv3_model import build_model
from src.segmentation.gradcam_segmentation import class_activation_map, overlay_segmentation_cam

@st.cache_resource
def load_cached_segmentation_model():
    model = build_model()
    pth_path = Path("models/segmentation/deeplabv3_multispectral.pth")
    if pth_path.exists() and pth_path.stat().st_size > 0:
        try:
            model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        except Exception:
            pass
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────


plt.style.use("dark_background")
plt.rcParams.update({"axes.facecolor": "#11141e", "figure.facecolor": "#11141e", "text.color": "#ededf1", "axes.labelcolor": "#9ba2b3", "xtick.color": "#9ba2b3", "ytick.color": "#9ba2b3", "grid.color": "#242c42"})
from ui import metric_card, status_card, section_header, render_sidebar, render_header_status

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def colormap_array(arr: np.ndarray, cmap: str = "RdYlGn",
                   vmin: float = -1, vmax: float = 1) -> np.ndarray:
    """Convert float32 array to uint8 RGB using a matplotlib colormap."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(arr, bytes=True)
    return rgba[:, :, :3]   # drop alpha → (H,W,3) uint8


def plot_urgency_velocity(urgency: np.ndarray, velocity_x: np.ndarray, velocity_y: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(urgency, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intervention Urgency")

    step = max(1, min(urgency.shape[0], urgency.shape[1]) // 15)
    Y, X = np.mgrid[0:urgency.shape[0]:step, 0:urgency.shape[1]:step]
    U = velocity_x[0:urgency.shape[0]:step, 0:urgency.shape[1]:step]
    V = velocity_y[0:urgency.shape[0]:step, 0:urgency.shape[1]:step]

    if np.max(np.sqrt(U**2 + V**2)) > 1e-5:
        ax.quiver(X, Y, U, -V, color="cyan", alpha=0.8, scale=10.0, width=0.008)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_boundaries_contours(pathogen: np.ndarray, boundaries: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pathogen, cmap="Purples", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pathogen Density")

    levels = [0.1, 0.5, 0.8]
    if np.max(boundaries) > 0.05:
        try:
            from scipy.ndimage import gaussian_filter
            smooth_b = gaussian_filter(boundaries, sigma=1.0)
            cs = ax.contour(smooth_b, levels=levels, colors=["yellow", "orange", "red"], linewidths=[1.0, 1.5, 2.0], alpha=0.9)
            labels = {levels[0]: "50% Boundary", levels[1]: "75% Boundary", levels[2]: "90% Boundary"}
            ax.clabel(cs, fmt=labels, inline=True, fontsize=8, colors="white")
        except Exception:
            ax.imshow(boundaries, cmap="Oranges", alpha=0.4)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
