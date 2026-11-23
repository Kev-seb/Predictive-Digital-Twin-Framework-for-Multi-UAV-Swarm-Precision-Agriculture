"""Renovated presentation of the original agriculture workflow."""

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from frontend_shared import (
    buffer_lock,
    MAVLINK_TELEMETRY, CURRENT_ENV, REPLAY_STATE, TELEMETRY_BUFFER,
    MULTIPLAYER_DRONES, COLLABORATIVE_ANNOTATIONS, math, json, time, os, io,
    render_sidebar, render_header_status,
    metric_card, status_card, section_header,
    fig_to_bytes, colormap_array, plot_urgency_velocity, plot_boundaries_contours,
    np, pd, plt, cm, Image, torch, cv2,
    compute_all_indices, MultispectralImage, load_multispectral_tiff,
    rule_based_stress_segmentation, mask_to_overlay, CLASS_LABELS, CLASS_COLORS, compute_iou,
    delineate_management_zones, extract_stress_regions, render_zone_map,
    compute_grid_statistics, plot_grid_heatmap,
    build_temporal_report, plot_ndvi_progression, plot_stress_progression,
    plot_multi_index_radar, plot_canopy_stress_area, plot_index_heatmap, STAGES,
    fetch_weather, assess_weather_stress,
    build_model, class_activation_map, overlay_segmentation_cam,
    load_cached_segmentation_model, get_shared_state,
    get_live_camera_frame, UAVFlightDynamicsSimulator,
)
# Page configuration and navigation are owned by frontend/app.py.

shared_state = get_shared_state()

crop_stage, stress_threshold, grid_size, lat, lon = render_sidebar()
render_header_status()

# ══════════════════════════════════════════════════════════════
#  Live UAV Control  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Live UAV Control", "Real-time MAVLink telemetry, 6-DOF flight simulation, and CV stress overlay", category="operations")

# Dynamic synchronization of Home Lat/Lon if crop field is loaded
if "lat" in locals() and "lon" in locals():
    shared_state["HOME_LAT"] = lat
    shared_state["HOME_LON"] = lon

col_ctrl, col_video = st.columns([1, 1.2])

with col_ctrl:
    st.markdown("### Connection & Protocol Settings")

    # Connection parameters
    conn_str = st.text_input("MAVLink Connection String", value=shared_state.get("MAVLINK_CONNECTION_STRING", "udp:127.0.0.1:14550"))
    protocol = st.selectbox("Autopilot Protocol / Flavor", ["Generic MAVLink", "ArduPilot", "PX4"], index=["Generic MAVLink", "ArduPilot", "PX4"].index(shared_state.get("MAVLINK_PROTOCOL", "Generic MAVLink")))
    cam_src = st.selectbox("Video Input Source", ["Simulated UAV Camera", "Local Webcam"], index=["Simulated UAV Camera", "Local Webcam"].index(shared_state.get("CAMERA_SOURCE", "Simulated UAV Camera")))

    # Save to shared state
    shared_state["MAVLINK_CONNECTION_STRING"] = conn_str
    shared_state["MAVLINK_PROTOCOL"] = protocol
    shared_state["CAMERA_SOURCE"] = cam_src

    # Connection status feedback
    is_connected = MAVLINK_TELEMETRY.get("connected", False)
    if is_connected:
        st.success("MAVLink Connection Established — Streaming Telemetry")
    else:
        st.warning("MAVLink Offline — Running High-Fidelity 6-DOF Autopilot Simulation")

    st.write("---")
    st.markdown("### Guidance & Autopilot Controls")

    autopilot_mode = st.selectbox(
        "Autopilot Mode",
        ["manual_orbit", "stabilized", "terrain_follow", "rtl", "landing"],
        index=["manual_orbit", "stabilized", "terrain_follow", "rtl", "landing"].index(shared_state.get("AUTOPILOT_MODE", "manual_orbit"))
    )
    shared_state["AUTOPILOT_MODE"] = autopilot_mode

    if autopilot_mode in ["stabilized", "terrain_follow"]:
        st.write("**Target Coordinates (Cartesian offset from field center)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            tgt_x = st.slider("Target East (m)", -50.0, 50.0, 0.0)
        with c2:
            tgt_y = st.slider("Target North (m)", -50.0, 50.0, 0.0)
        with c3:
            tgt_z = st.slider("Target Alt (m)", 2.0, 30.0, 10.0)
        shared_state["UI_TARGET_POS"] = [tgt_x, tgt_y, tgt_z]

        tgt_yaw = st.slider("Target Yaw (Degrees)", -180.0, 180.0, 0.0)
        shared_state["UI_TARGET_YAW"] = math.radians(tgt_yaw)

    is_spraying = st.checkbox("Force Actuator Spray Trigger (Manual Override)", value=shared_state.get("UI_IS_SPRAYING", False))
    shared_state["UI_IS_SPRAYING"] = is_spraying

    st.write("---")
    st.markdown("### Interactive Fault & Wind Perturbation Panel")

    col_f1, col_f2, col_f3 = st.columns(3)
    sim = shared_state.get("PHYSICS_SIMULATOR")
    if sim is None:
        home_lat = shared_state.get("HOME_LAT", 11.0)
        home_lon = shared_state.get("HOME_LON", 79.0)
        sim = UAVFlightDynamicsSimulator(shared_state, home_lat=home_lat, home_lon=home_lon)
        shared_state["PHYSICS_SIMULATOR"] = sim

    with col_f1:
        if st.button("Inject Wind Gust (15m/s)", help="Simulates sudden wind gust perturbing PID stabilization"):
            if sim:
                sim.inject_wind_gust()
                st.success("Wind Gust Injected!")
            else:
                st.error("Simulator not initialized yet.")
    with col_f2:
        if st.button("Fault: Rotor 3 Jam", help="Jams rotor 3, testing PID fault stabilization"):
            if sim:
                sim.fault_rotor_failure = True
                st.error("Rotor 3 Jammed!")
            else:
                st.error("Simulator not initialized yet.")
    with col_f3:
        if st.button("Force Low Battery", help="Drops battery capacity to 14.5% to trigger RTL"):
            if sim:
                sim.battery = 14.5
                st.warning("Low battery injected!")
            else:
                st.error("Simulator not initialized yet.")

    if st.button("Reset Simulator, Recharge & Clear Faults", type="primary", use_container_width=True):
        if sim:
            sim.fault_rotor_failure = False
            sim.battery = 100.0
            sim.payload_mass = 10.0
            sim.pos = np.array([0.0, 0.0, 10.0])
            sim.vel = np.array([0.0, 0.0, 0.0])
            sim.attitude = np.array([0.0, 0.0, 0.0])
            sim.omega = np.array([0.0, 0.0, 0.0])
            st.success("All systems green, battery charged, payload refilled!")
        else:
            st.error("Simulator not initialized yet.")

with col_video:
    st.markdown("### Real-time CV Crop Stress Intelligence Overlay")
    st.markdown("Dynamic down-looking camera view rendered with aviation-style HUD and real-time contour stress intelligence.")

    telemetry_port = shared_state.get("TELEMETRY_PORT", 8000)

    # Display the live stream using st.components.v1.html for continuous frame updates
    st.components.v1.html(
        f"""
        <div style="background-color: #0c0f1d; border-radius: 12px; padding: 10px; border: 2px solid #3b82f6; text-align: center;">
            <img src="http://127.0.0.1:{telemetry_port}/camera" width="100%" style="border-radius: 8px; max-width: 640px; aspect-ratio: 4/3; object-fit: cover;" onerror="this.src='https://placehold.co/640x480/0f172a/ffffff?text=Waiting+for+UAV+Telemetry+Camera+Feed...'"/>
        </div>
        """,
        height=430
    )

    # Display live telemetry readouts
    st.markdown("### Active Telemetry Dashboard")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.metric("Latitude", f"{MAVLINK_TELEMETRY['lat']:.6f}")
        st.metric("Pitch / Roll", f"{math.degrees(MAVLINK_TELEMETRY['pitch']):.1f}° / {math.degrees(MAVLINK_TELEMETRY['roll']):.1f}°")
    with t_col2:
        st.metric("Longitude", f"{MAVLINK_TELEMETRY['lon']:.6f}")
        st.metric("Yaw / Heading", f"{math.degrees(MAVLINK_TELEMETRY['yaw']):.1f}°")
    with t_col3:
        st.metric("Altitude (Relative)", f"{MAVLINK_TELEMETRY['alt']:.2f} m")
        st.metric("Battery Remaining", f"{MAVLINK_TELEMETRY['battery']:.1f}%")

    st.write(f"**GPS Fix status:** {'Online' if is_connected else 'Offline (6-DOF SITL Sim Mode)'} | "
             f"**Sprayer State:** {'ACTIVE' if MAVLINK_TELEMETRY['is_spraying'] else 'INACTIVE'} | "
             f"**Estimated Payload Mass:** {MAVLINK_TELEMETRY.get('payload_mass', 10.0):.2f} kg | "
             f"**Autopilot Mode:** {MAVLINK_TELEMETRY.get('autopilot_mode', 'stabilized').upper()}")

st.write("---")
st.markdown("### Buffering Telemetry Database Logs")
st.markdown("Rolling 20-frame log cache captured at 20Hz. Useful for post-flight analysis, mission replay, or CSV export.")

with buffer_lock:
    if len(TELEMETRY_BUFFER) > 0:
        df_log = pd.DataFrame(list(TELEMETRY_BUFFER)[-20:])
        # Filter and reorder columns
        cols = ['lat', 'lon', 'alt', 'pitch', 'roll', 'yaw', 'battery', 'speed', 'is_spraying']
        df_log_filtered = df_log[[c for c in cols if c in df_log.columns]]
        st.dataframe(df_log_filtered, use_container_width=True, hide_index=True)

        c_csv1, c_csv2 = st.columns(2)
        with c_csv1:
            st.download_button(
                label="Export Full Telemetry Log (CSV)",
                data=pd.DataFrame(list(TELEMETRY_BUFFER)).to_csv(index=False),
                file_name="telemetry_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_csv2:
            if st.button("Clear Log Buffer", use_container_width=True):
                TELEMETRY_BUFFER.clear()
                st.success("Log buffer cleared!")
    else:
        st.info("No telemetry logs buffered yet. Telemetry will begin buffering once MAVLink or SITL begins streaming.")
