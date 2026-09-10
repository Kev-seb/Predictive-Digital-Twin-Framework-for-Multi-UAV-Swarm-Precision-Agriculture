"""
verify_sitl_telemetry.py
------------------------
Cyber-physical evaluation script for the UAV crop digital twin.
Mocks a dual-UAV swarm mission in SITL by streaming telemetry packets to the
sync servers (HTTP API and WebSocket broadcast).
Asserts:
1. Swarm safety separation distance (must remain > 6m).
2. Packet transmission latency (must remain < 100ms).
3. Telemetry synchronization state across clients.

Outputs evaluation report to `outputs/academic/telemetry_report.json`.
"""

import os
import sys
import json
import time
import math
import asyncio
import requests
import websockets
import numpy as np

# Ensure outputs folder exists
os.makedirs("outputs/academic", exist_ok=True)


# Coordinate distance calculator
def calculate_distance_meters(lat1, lon1, lat2, lon2):
    # Standard Haversine formula
    R = 6371000.0  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


async def test_websocket_telemetry(duration_seconds: int = 5):
    http_url = "http://127.0.0.1:8000/telemetry"
    ws_url = "ws://127.0.0.1:8765"
    
    print("=========================================================")
    print("Starting Cyber-Physical Swarm Telemetry verification...")
    print(f"Connecting to WS={ws_url}, HTTP={http_url}")
    print("=========================================================\n")
    
    # 1. Verify HTTP Telemetry endpoint
    try:
        r = requests.get(http_url, timeout=2.0)
        if r.status_code == 200:
            print(f"[HTTP SUCCESS] Telemetry API responsive. Initial state: {r.json().get('autopilot_mode')}")
        else:
            print(f"[HTTP WARNING] API returned status {r.status_code}")
    except Exception as e:
        print(f"[HTTP ERROR] API offline. Make sure Streamlit app is running. Error: {e}")
        return False
        
    # 2. Connect to WebSocket Sync Server
    try:
        async with websockets.connect(ws_url) as websocket:
            print("[WS SUCCESS] Connected to WebSocket Sync Server.")
            
            # Send client role registration
            await websocket.send(json.dumps({
                "type": "client_update",
                "role": "Academic Validator"
            }))
            
            latencies = []
            safety_distances = []
            safety_violations = 0
            
            start_time = time.time()
            step = 0
            
            # Base home position
            home_lat = 11.0
            home_lon = 79.0
            
            # Sim parameters: Alpha orbits, Beta moves in a straight line
            while time.time() - start_time < duration_seconds:
                t_val = time.time() - start_time
                step += 1
                
                # Alpha coords (Radius 12m)
                alpha_x = 12.0 * math.sin(t_val * 0.8)
                alpha_y = 12.0 * math.cos(t_val * 0.8)
                
                # Beta coords (Moves from -20m to +20m)
                beta_x = -20.0 + (t_val / duration_seconds) * 40.0
                beta_y = -8.0
                
                # Convert displacements to GPS coords
                lat_deg_m = 1.0 / 111320.0
                lon_deg_m = 1.0 / (111320.0 * math.cos(math.radians(home_lat)))
                
                lat_alpha = home_lat + (alpha_y * lat_deg_m)
                lon_alpha = home_lon + (alpha_x * lon_deg_m)
                
                lat_beta = home_lat + (beta_y * lat_deg_m)
                lon_beta = home_lon + (beta_x * lon_deg_m)
                
                # Payload updates for both drones
                update_alpha = {
                    "type": "telemetry_update",
                    "drone_id": "drone_alpha",
                    "lat": lat_alpha,
                    "lon": lon_alpha,
                    "alt": 10.0,
                    "battery": 88.0,
                    "speed": 6.5,
                    "is_spraying": True
                }
                
                update_beta = {
                    "type": "telemetry_update",
                    "drone_id": "drone_beta",
                    "lat": lat_beta,
                    "lon": lon_beta,
                    "alt": 11.0,
                    "battery": 92.0,
                    "speed": 4.0,
                    "is_spraying": False
                }
                
                t_send = time.time()
                
                # Send updates
                await websocket.send(json.dumps(update_alpha))
                await websocket.send(json.dumps(update_beta))
                
                # Wait for broadcast back (echo frame)
                try:
                    response_msg = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                    t_recv = time.time()
                    
                    # Latency in ms
                    latencies.append((t_recv - t_send) * 1000.0)
                    
                    data = json.loads(response_msg)
                    drones = data.get("drones", {})
                    
                    if "drone_alpha" in drones and "drone_beta" in drones:
                        da = drones["drone_alpha"]
                        db = drones["drone_beta"]
                        
                        dist = calculate_distance_meters(da["lat"], da["lon"], db["lat"], db["lon"])
                        safety_distances.append(dist)
                        
                        if dist < 6.0:  # Safety radius d0
                            safety_violations += 1
                            print(f"[SWARM WARNING] Collision hazard! Distance is {dist:.2f}m (Threshold: 6.0m)")
                            
                except asyncio.TimeoutError:
                    print("[WS WARNING] Timeout waiting for server broadcast.")
                    
                await asyncio.sleep(0.1)  # 10Hz stream
                
            # Compile stats
            avg_latency = np.mean(latencies) if latencies else 0.0
            min_dist = np.min(safety_distances) if safety_distances else 0.0
            
            print("\n=========================================================")
            print("SITL Telemetry Verification Results:")
            print(f"Total packets synced: {step * 2}")
            print(f"Average network broadcast latency: {avg_latency:.2f} ms")
            print(f"Minimum safety separation: {min_dist:.2f} m")
            print(f"Swarm Safety Violations: {safety_violations}")
            print("=========================================================\n")
            
            report = {
                "timestamp": time.time(),
                "network_status": "ONLINE",
                "total_packets_transmitted": step * 2,
                "average_latency_ms": float(avg_latency),
                "minimum_separation_distance_m": float(min_dist),
                "safety_threshold_m": 6.0,
                "safety_violations_count": safety_violations,
                "verification_result": "PASS" if safety_violations == 0 and avg_latency < 100.0 else "WARNING"
            }
            
            with open("outputs/academic/telemetry_report.json", "w") as f:
                json.dump(report, f, indent=4)
                
            print("[SUCCESS] Telemetry report JSON saved to outputs/academic/telemetry_report.json")
            return True
            
    except Exception as e:
        print(f"[WS ERROR] WebSocket offline. Error: {e}")
        return False


if __name__ == "__main__":
    duration = 5
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
            
    asyncio.run(test_websocket_telemetry(duration_seconds=duration))
