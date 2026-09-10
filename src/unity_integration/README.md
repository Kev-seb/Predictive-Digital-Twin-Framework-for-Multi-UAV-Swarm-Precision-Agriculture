# Unity 3D Simulation Bridge: Integration Guide

This guide explains how to integrate the [UnityBridgeClient.cs](file:///d:/AI-powered-predictive-digital-twin-for-agriculture/uav-crop-stress-intelligence/src/unity_integration/UnityBridgeClient.cs) C# script into your Unity 3D project. This connection enables real-time 3D photogrammetric spatial canopy reconstruction and closed-loop potential-field drone swarm coordination.

---

## 1. Setup in Unity

### A. Add Script to Project
1. Copy the C# script `UnityBridgeClient.cs` and place it anywhere under your Unity project's `Assets/` directory (e.g., `Assets/Scripts/DigitalTwin/`).

### B. Configure Drone Rigidbody
1. Attach the script to your Drone GameObject.
2. Unity will automatically add a **Rigidbody** component (due to `[RequireComponent(typeof(Rigidbody))]`).
3. In the Rigidbody inspector:
   * **Uncheck** `Use Gravity` (forces are managed by the potential-field solver).
   * Set **Drag** to `0` and **Angular Drag** to `0.05`.
   * Under **Constraints**, check Freeze Rotation (`X`, `Y`, `Z`) if you wish to let potential fields handle coordinates strictly.

### C. Build the Stereo Camera Rig
To enable depth disparity canopy mapping, construct a stereo rig under your drone:
1. Create two Camera GameObjects as children of your Drone: `LeftCamera` and `RightCamera`.
2. Align them to point straight down (Rotation: `90, 0, 0`).
3. Set the horizontal separation (Baseline):
   * Move **LeftCamera** to Local Position: `x = -0.25, y = 0, z = 0`.
   * Move **RightCamera** to Local Position: `x = 0.25, y = 0, z = 0`.
   * *This forms a 0.5-meter baseline rig.*
4. Link these two Cameras to the `Left Camera` and `Right Camera` slots in the `UnityBridgeClient` inspector.
5. Set `Baseline` in the inspector to `0.5`.

---

## 2. Server Configuration

In the `UnityBridgeClient` inspector, set:
*   **Api Host:** The local IP address of your Python server (e.g. `127.0.0.1` or `192.168.x.x` if running on separate machines).
*   **Api Port:** `8080` (or `8081` depending on port availability).
*   **Drone Id:** A unique string identifying this drone (e.g. `drone_01`, `drone_02`).

---

## 3. Communication Loop Workflow

Once you click **Play** in Unity:

1.  **WebSocket Telemetry Loop (20 Hz):**
    *   The C# script packages the drone's position, velocity, and orientation and pushes them to `ws://host:port/ws/telemetry`.
    *   The Python backend receives it, updates the digital twin tracker, calculates potential-field forces (attractive to destination, repulsive to other drones), and replies with a `velocity_command` JSON.
    *   The C# script receives the velocity command and writes it directly to `Rigidbody.velocity`.
2.  **HTTP Perception Loop (Every 2 Seconds):**
    *   The script captures the viewport of the Left and Right cameras, compiles them into JPEG arrays, packages the focal length/baseline metadata, and sends an HTTP POST to `http://host:port/api/frame/stereo`.
    *   Python decodes the frames, executes Semi-Global Block Matching (SGBM) to generate a Digital Surface Model (DSM), extracts the Canopy Height Model (CHM), and updates the digital twin state.
3.  **Fail-safe Timeout:**
    *   If Python drops offline or connection latency spikes beyond **500 ms**, the drone automatically stops its velocity commands and hovers safely in place to prevent flyaways.

---

## 4. Troubleshooting

*   **Connection Refused:** Ensure the Python FastAPI server is running (`.venv\Scripts\python.exe src/dashboard/dashboard.py` or through your test server script).
*   **Failed to Capture RenderTextures:** Ensure your cameras have target textures assigned or that the resolution size is compatible.
*   **Flipped Disparity:** If canopy height is negative or inverted, verify that your Left and Right cameras are not swapped in the Unity hierarchy.
