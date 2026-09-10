#!/usr/bin/env python3
"""
simulate_unity_client.py
------------------------
Headless mock Unity client to verify the spatial reconstruction, 
telemetry ingestion, and closed-loop potential-field controls.
"""

import asyncio
import json
import time
import cv2
import numpy as np
import requests
import websockets

API_URL = "http://127.0.0.1:8080"
WS_URL = "ws://127.0.0.1:8080/ws/telemetry"

def generate_synthetic_stereo_pair(disparity_val: int = 15):
    """
    Generate synthetic left/right stereo images (640x480) with a shifted circle.
    The SGBM algorithm in the backend will resolve this shift to a disparity/height.
    """
    # Background: dark gray field
    left_img = np.full((480, 640, 3), 40, dtype=np.uint8)
    right_img = np.full((480, 640, 3), 40, dtype=np.uint8)
    
    # Draw green leaf canopy pattern
    center_y = 240
    center_x_left = 320
    center_x_right = center_x_left - disparity_val
    radius = 60
    
    # Left image: Green circle at center
    cv2.circle(left_img, (center_x_left, center_y), radius, (30, 200, 30), -1)
    cv2.circle(left_img, (center_x_left, center_y), radius - 10, (50, 240, 50), -1)
    
    # Right image: Shifted green circle (creating disparity)
    cv2.circle(right_img, (center_x_right, center_y), radius, (30, 200, 30), -1)
    cv2.circle(right_img, (center_x_right, center_y), radius - 10, (50, 240, 50), -1)
    
    # Encode to JPEG
    _, left_jpg = cv2.imencode(".jpg", left_img)
    _, right_jpg = cv2.imencode(".jpg", right_img)
    
    return left_jpg.tobytes(), right_jpg.tobytes()

async def post_stereo_frames(frame_id: int):
    """POST a stereo frame pair to the perception endpoint."""
    left_bytes, right_bytes = generate_synthetic_stereo_pair(disparity_val=18)
    
    metadata = {
        "drone_id": "drone_01",
        "frame_id": frame_id,
        "timestamp": time.time(),
        "focal_length": 35.0,
        "baseline": 0.5,
        "position": [12.4, 5.2, 8.7],
        "orientation": [0.0, 35.0, 0.0]
    }
    
    files = {
        "left_file": ("left.jpg", left_bytes, "image/jpeg"),
        "right_file": ("right.jpg", right_bytes, "image/jpeg"),
    }
    data = {
        "metadata": json.dumps(metadata)
    }
    
    try:
        # Run request in executor to avoid blocking the asyncio event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(f"{API_URL}/api/frame/stereo", files=files, data=data, timeout=5)
        )
        if response.status_code == 200:
            res_json = response.json()
            print(f"[HTTP Perception] Frame {frame_id} POST OK: {res_json}")
            return res_json
        else:
            print(f"[HTTP Perception] Frame {frame_id} failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"[HTTP Perception] Ingestion connection error: {e}")
    return None

async def telemetry_loop(drone_id: str, start_pos: list):
    """Manage 20 Hz WebSocket connection and closed-loop position updates."""
    # State tracking variables
    position = list(start_pos) # start position
    velocity = [0.0, 0.0, 0.0]
    telemetry_seq = 100
    
    print(f"[{drone_id}] Connecting to telemetry channel at {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as ws:
            print(f"[{drone_id}] Telemetry connection open (WebSocket)")
            
            # Send 15 steps of closed-loop telemetry updates
            for step in range(15):
                # 1. Update position based on returned command (Euler integration)
                dt = 0.05  # 20 Hz
                position[0] += velocity[0] * dt
                position[1] += velocity[1] * dt
                position[2] += velocity[2] * dt
                
                # 2. Package telemetry
                packet = {
                    "type": "telemetry",
                    "drone_id": drone_id,
                    "telemetry_seq": telemetry_seq,
                    "position": position,
                    "velocity": velocity,
                    "orientation": [0.0, 15.0, 0.0],
                    "battery_pct": 92.0,
                    "timestamp": time.time()
                }
                
                await ws.send(json.dumps(packet))
                
                # 3. Listen for command reply
                try:
                    raw_reply = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    reply = json.loads(raw_reply)
                    if reply.get("type") == "velocity_command":
                        cmd_vel = reply.get("velocity", [0.0, 0.0, 0.0])
                        print(f"[{drone_id}] Pos=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] | Cmd Velocity=[{cmd_vel[0]:.2f}, {cmd_vel[1]:.2f}, {cmd_vel[2]:.2f}]")
                        # Apply to next step iteration
                        velocity = cmd_vel
                except asyncio.TimeoutError:
                    print(f"[{drone_id}] Command timeout!")
                    velocity = [0.0, 0.0, 0.0] # safety hover
                
                telemetry_seq += 1
                await asyncio.sleep(0.05) # 20 Hz yield
                
    except Exception as e:
        print(f"[{drone_id}] Telemetry connection error: {e}")

async def main():
    print("=== Starting Mock Unity Client ===")
    
    # 1. Send initial perception frame
    await post_stereo_frames(101)
    await asyncio.sleep(0.5)
    
    # 2. Run telemetry loops concurrently
    print("\n--- Starting Closed-Loop Swarm Simulation ---")
    await asyncio.gather(
        telemetry_loop("drone_01", [2.0, 1.0, 10.0]),
        telemetry_loop("drone_02", [-2.0, -1.0, 10.0])
    )
    
    # 3. Send final perception frame
    print("\n--- Sending Final Perception Frame ---")
    await post_stereo_frames(102)
    
    # 4. Query and validate /api/field/state (Standard Twin State)
    print("\n--- Querying /api/field/state (Standard) ---")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{API_URL}/api/field/state?rows=20&columns=20", timeout=5)
        )
        if response.status_code == 200:
            res_json = response.json()
            rows = res_json.get("rows", 0)
            cols = res_json.get("columns", 0)
            ndvi = res_json.get("ndvi", [])
            stress = res_json.get("stress", [])
            fallback = res_json.get("is_simulation_fallback", True)
            print(f"[Standard Twin State] Loaded: {rows}x{cols} grid")
            print(f"[Standard Twin State] Fallback state active: {fallback}")
            print(f"[Standard Twin State] Array lengths: ndvi={len(ndvi)}, stress={len(stress)}")
            if len(ndvi) == rows * cols and len(stress) == rows * cols:
                print("[Standard Twin State] VALIDATION PASSED: Array dimensions match.")
            else:
                print("[Standard Twin State] VALIDATION FAILED: Dimensions mismatch!")
        else:
            print(f"[Standard Twin State] failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"[Standard Twin State] Query error: {e}")

    # 5. Query and validate /api/field/state (Deterministic Test Fixture)
    print("\n--- Querying /api/field/state (Deterministic Test Fixture) ---")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{API_URL}/api/field/state?rows=20&columns=20&test_fixture=deterministic", timeout=5)
        )
        if response.status_code == 200:
            res_json = response.json()
            rows = res_json.get("rows", 0)
            cols = res_json.get("columns", 0)
            ndvi = res_json.get("ndvi", [])
            stress = res_json.get("stress", [])
            is_fixture = res_json.get("is_test_fixture", False)
            
            print(f"[Deterministic Fixture] Loaded: {rows}x{cols} grid")
            print(f"[Deterministic Fixture] Is test fixture: {is_fixture}")
            
            # Map index = row * columns + col to test mapping correctness
            tl_idx = 0 * cols + 0 # Top-left (0,0) -> Healthy corner
            br_idx = (rows - 1) * cols + (cols - 1) # Bottom-right (19,19) -> Mild stress
            ctr_idx = (rows // 2) * cols + (cols // 2) # Center (10,10) -> Severe stress
            
            tl_ndvi, tl_stress = ndvi[tl_idx], stress[tl_idx]
            br_ndvi, br_stress = ndvi[br_idx], stress[br_idx]
            ctr_ndvi, ctr_stress = ndvi[ctr_idx], stress[ctr_idx]
            
            print(f"[Deterministic Fixture] Top-Left Corner (Healthy): NDVI={tl_ndvi:.2f}, Stress={tl_stress:.2f} (Expected: 0.85, 0.05)")
            print(f"[Deterministic Fixture] Bottom-Right Region (Mild Stress): NDVI={br_ndvi:.2f}, Stress={br_stress:.2f} (Expected: 0.50, 0.35)")
            print(f"[Deterministic Fixture] Center Outbreak Hotspot (Severe Stress): NDVI={ctr_ndvi:.2f}, Stress={ctr_stress:.2f} (Expected: 0.15, 0.85)")
            
            # Validate ranges
            if (abs(tl_ndvi - 0.85) < 0.01 and abs(br_ndvi - 0.50) < 0.01 and abs(ctr_ndvi - 0.15) < 0.01):
                print("[Deterministic Fixture] VALIDATION PASSED: Spatial grid mapping aligns perfectly.")
            else:
                print("[Deterministic Fixture] VALIDATION FAILED: Spatial grid indices out of sync!")
        else:
            print(f"[Deterministic Fixture] failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"[Deterministic Fixture] Query error: {e}")
        
    print("=== Simulation Completed ===")

if __name__ == "__main__":
    asyncio.run(main())
