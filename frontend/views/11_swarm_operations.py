"""Renovated presentation of the original agriculture workflow."""

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from frontend_shared import (
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
#  Swarm Operations  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Swarm Operations", "Multi-UAV coordinated missions with decentralized collision avoidance", category="operations")

# Fetch simulator instances
sim_alpha = shared_state["PHYSICS_SIMULATORS"].get("drone_alpha")
sim_beta = shared_state["PHYSICS_SIMULATORS"].get("drone_beta")

col_sw_ctrl, col_sw_map = st.columns([1, 1.3])

with col_sw_ctrl:
    st.markdown("### Fleet Mission Commander")

    swarm_mission = st.selectbox(
        "Select Swarm Strategy",
        [
            "coordinated_spraying",
            "synchronized_scouting",
            "orbit_avoidance_test"
        ],
        format_func=lambda x: {
            "coordinated_spraying": "Coordinated Spraying (West vs. East Sectors)",
            "synchronized_scouting": "Synchronized scouting scan patterns",
            "orbit_avoidance_test": "Orbit collision avoidance & proximity test"
        }[x],
        index=[
            "coordinated_spraying",
            "synchronized_scouting",
            "orbit_avoidance_test"
        ].index(shared_state.get("SWARM_MISSION_TYPE", "coordinated_spraying"))
    )
    shared_state["SWARM_MISSION_TYPE"] = swarm_mission

    # Interactive Mission Status details
    if swarm_mission == "coordinated_spraying":
        st.info("**Coordinated Spraying:** Drone Alpha covers Sector A (Western half of field) while Drone Beta covers Sector B (Eastern half). Spray nozzles activate when crossing target stress zones.")
    elif swarm_mission == "synchronized_scouting":
        st.info("**Synchronized Scouting:** Drones sweep and map the field in parallel lines. Actuators remain inactive to conserve payload, prioritizing high-resolution multispectral scouting.")
    elif swarm_mission == "orbit_avoidance_test":
        st.warning("**Orbit Avoidance Test:** Both drones fly overlapping circular orbits intersecting at the center of the field. Shows potential field collision avoidance pushing the drones apart as they cross.")

    st.write("---")
    st.markdown("### Swarm Collision Avoidance Log")

    warnings = shared_state.get("SWARM_WARNINGS", [])
    if len(warnings) == 0:
        st.success("Active Collision Avoidance System: Normal Operations. No proximity violations detected.")
    else:
        st.error(f"{len(warnings)} Proximity Warnings Logged:")
        for w in warnings[-5:]:
            st.write(f"- `{w}`")
        if st.button("Clear Collision Warning Logs", use_container_width=True):
            shared_state["SWARM_WARNINGS"] = []
            st.rerun()

    st.write("---")
    st.markdown("### Swarm Dispatch Commands")

    cmd_c1, cmd_c2 = st.columns(2)
    with cmd_c1:
        if st.button("Inject Gust Fleetwide", help="Simulate a wind turbulence gust hitting all drones simultaneously"):
            if sim_alpha and sim_beta:
                sim_alpha.inject_wind_gust()
                sim_beta.inject_wind_gust()
                st.success("Gust injected to all drones!")
            else:
                st.error("Simulators not ready.")
        if st.button("Trigger Swarm RTL", help="Command all drones to return to launch location"):
            if sim_alpha and sim_beta:
                sim_alpha.pos = np.array([0.0, 0.0, 10.0])
                sim_beta.pos = np.array([0.0, 0.0, 10.0])
                st.warning("Fleet commanded to return to home base.")
    with cmd_c2:
        if st.button("Rotor Failure (Beta)", help="Inject single rotor fail onto Drone Beta"):
            if sim_beta:
                sim_beta.fault_rotor_failure = True
                st.error("Rotor 3 Jammed on Drone Beta!")
            else:
                st.error("Drone Beta offline.")
        if st.button("Fleet Recharge & Refill", help="Refill payloads and recharge batteries for all drones"):
            if sim_alpha and sim_beta:
                sim_alpha.fault_rotor_failure = False
                sim_beta.fault_rotor_failure = False
                sim_alpha.battery = 100.0
                sim_beta.battery = 100.0
                sim_alpha.payload_mass = 10.0
                sim_beta.payload_mass = 10.0
                st.success("Fleet recharged and payloads refilled!")

with col_sw_map:
    st.markdown("### Swarm 3D Spatial Digital Twin")
    st.markdown("Real-time 3D flight paths, active target vectors, safety proximity margins, and spray particle drift.")

    # Plotly 3D Graph
    import plotly.graph_objects as go

    if "alpha_trail" not in st.session_state:
        st.session_state["alpha_trail"] = []
    if "beta_trail" not in st.session_state:
        st.session_state["beta_trail"] = []

    if sim_alpha and getattr(sim_alpha, 'pos', None) is not None:
        st.session_state["alpha_trail"].append(sim_alpha.pos.copy())
        if len(st.session_state["alpha_trail"]) > 100:
            st.session_state["alpha_trail"].pop(0)
    if sim_beta and getattr(sim_beta, 'pos', None) is not None:
        st.session_state["beta_trail"].append(sim_beta.pos.copy())
        if len(st.session_state["beta_trail"]) > 100:
            st.session_state["beta_trail"].pop(0)

    fig = go.Figure()

    # Plot trails
    if len(st.session_state["alpha_trail"]) > 0:
        trail_a = np.array(st.session_state["alpha_trail"])
        fig.add_trace(go.Scatter3d(
            x=trail_a[:, 0], y=trail_a[:, 1], z=trail_a[:, 2],
            mode='lines',
            line=dict(color='#00f2fe', width=4),
            name='Drone Alpha Path'
        ))

    if len(st.session_state["beta_trail"]) > 0:
        trail_b = np.array(st.session_state["beta_trail"])
        fig.add_trace(go.Scatter3d(
            x=trail_b[:, 0], y=trail_b[:, 1], z=trail_b[:, 2],
            mode='lines',
            line=dict(color='#ff4b4b', width=4),
            name='Drone Beta Path'
        ))

    # Draw current positions
    if sim_alpha and getattr(sim_alpha, 'pos', None) is not None:
        fig.add_trace(go.Scatter3d(
            x=[sim_alpha.pos[0]], y=[sim_alpha.pos[1]], z=[sim_alpha.pos[2]],
            mode='markers+text',
            marker=dict(size=12, color='#00f2fe', symbol='diamond', line=dict(width=1, color='white')),
            text=["Alpha"],
            textposition="top center",
            name='Drone Alpha'
        ))
        if getattr(sim_alpha, 'target_pos', None) is not None:
            fig.add_trace(go.Scatter3d(
                x=[sim_alpha.target_pos[0]], y=[sim_alpha.target_pos[1]], z=[sim_alpha.target_pos[2]],
                mode='markers',
                marker=dict(size=7, color='#00f2fe', symbol='cross'),
                name='Alpha Waypoint'
            ))

    if sim_beta and getattr(sim_beta, 'pos', None) is not None:
        fig.add_trace(go.Scatter3d(
            x=[sim_beta.pos[0]], y=[sim_beta.pos[1]], z=[sim_beta.pos[2]],
            mode='markers+text',
            marker=dict(size=12, color='#ff4b4b', symbol='diamond', line=dict(width=1, color='white')),
            text=["Beta"],
            textposition="top center",
            name='Drone Beta'
        ))
        if getattr(sim_beta, 'target_pos', None) is not None:
            fig.add_trace(go.Scatter3d(
                x=[sim_beta.target_pos[0]], y=[sim_beta.target_pos[1]], z=[sim_beta.target_pos[2]],
                mode='markers',
                marker=dict(size=7, color='#ff4b4b', symbol='cross'),
                name='Beta Waypoint'
            ))

    # Collision avoidance wireframe spheres (6m diameter -> 3.0m radius)
    def make_sphere(cx, cy, cz, r=3.0, n_points=8):
        phi = np.linspace(0, 2*np.pi, n_points)
        theta = np.linspace(0, np.pi, n_points)
        phi, theta = np.meshgrid(phi, theta)
        x = cx + r * np.sin(theta) * np.cos(phi)
        y = cy + r * np.sin(theta) * np.sin(phi)
        z = cz + r * np.cos(theta)
        return x, y, z

    if sim_alpha:
        sx, sy, sz = make_sphere(sim_alpha.pos[0], sim_alpha.pos[1], sim_alpha.pos[2])
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            opacity=0.15,
            colorscale=[[0, '#00f2fe'], [1, '#00f2fe']],
            showscale=False,
            hoverinfo='skip',
            name='Alpha Proximity Guard'
        ))

    if sim_beta:
        sx, sy, sz = make_sphere(sim_beta.pos[0], sim_beta.pos[1], sim_beta.pos[2])
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            opacity=0.15,
            colorscale=[[0, '#ff4b4b'], [1, '#ff4b4b']],
            showscale=False,
            hoverinfo='skip',
            name='Beta Proximity Guard'
        ))

    # Render Active Spray Particles
    particles = shared_state.get("ACTIVE_PARTICLES", np.zeros((0, 3)))
    if len(particles) > 0:
        home_lat = shared_state.get("HOME_LAT", 11.0)
        home_lon = shared_state.get("HOME_LON", 79.0)
        lat_deg_per_meter = 1.0 / 111320.0
        lon_deg_per_meter = 1.0 / (111320.0 * math.cos(math.radians(home_lat)))

        p_x = (particles[:, 0] - home_lon) / lon_deg_per_meter
        p_y = (particles[:, 1] - home_lat) / lat_deg_per_meter
        p_z = particles[:, 2]

        if len(p_x) > 800:
            indices = np.random.choice(len(p_x), 800, replace=False)
            p_x, p_y, p_z = p_x[indices], p_y[indices], p_z[indices]

        fig.add_trace(go.Scatter3d(
            x=p_x, y=p_y, z=p_z,
            mode='markers',
            marker=dict(size=2.5, color='#3b82f6', opacity=0.35),
            name='Spray Particles'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='East (m)', range=[-40, 40], backgroundcolor="#0c0f1d", gridcolor="#1e293b"),
            yaxis=dict(title='North (m)', range=[-40, 40], backgroundcolor="#0c0f1d", gridcolor="#1e293b"),
            zaxis=dict(title='Altitude (m)', range=[0, 20], backgroundcolor="#0c0f1d", gridcolor="#1e293b"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.35)
        ),
        margin=dict(r=0, l=0, b=0, t=10),
        paper_bgcolor="#0c0f1d",
        font_color="white",
        height=460
    )
    st.plotly_chart(fig, use_container_width=True)

# Telemetry grids for side-by-side display
st.write("---")
st.markdown("### Swarm Parallel Telemetry Deck")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("### Drone Alpha (Lead)")
    if sim_alpha:
        a_c1, a_c2 = st.columns(2)
        with a_c1:
            drone_a = MULTIPLAYER_DRONES.get('drone_alpha', {})
            st.metric("Latitude", f"{drone_a.get('lat', 0.0):.6f}")
            st.metric("Longitude", f"{drone_a.get('lon', 0.0):.6f}")
            alt = sim_alpha.pos[2] if getattr(sim_alpha, 'pos', None) is not None else 0.0
            st.metric("Altitude (Relative)", f"{alt:.2f} m")
        with a_c2:
            spd = np.linalg.norm(sim_alpha.vel) if getattr(sim_alpha, 'vel', None) is not None else 0.0
            st.metric("Ground Speed", f"{spd:.2f} m/s")
            yaw = math.degrees(sim_alpha.attitude[2]) if getattr(sim_alpha, 'attitude', None) is not None else 0.0
            st.metric("Yaw / Heading", f"{yaw:.1f}°")
            st.metric("Sprayer Output State", "ACTIVE" if getattr(sim_alpha, 'is_spraying', False) else "INACTIVE")

        bat = getattr(sim_alpha, 'battery', 0.0)
        pmass = getattr(sim_alpha, 'payload_mass', 0.0)
        st.write(f"**Battery Status:** {bat:.1f}%")
        st.progress(bat / 100.0)
        st.write(f"**Estimated Payload Mass:** {pmass:.2f} kg / 10.00 kg")
    else:
        st.info("Drone Alpha Offline.")

with col_t2:
    st.markdown("### Drone Beta (Scout / Support)")
    if sim_beta:
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            drone_b = MULTIPLAYER_DRONES.get('drone_beta', {})
            st.metric("Latitude", f"{drone_b.get('lat', 0.0):.6f}")
            st.metric("Longitude", f"{drone_b.get('lon', 0.0):.6f}")
            alt = sim_beta.pos[2] if getattr(sim_beta, 'pos', None) is not None else 0.0
            st.metric("Altitude (Relative)", f"{alt:.2f} m")
        with b_c2:
            spd = np.linalg.norm(sim_beta.vel) if getattr(sim_beta, 'vel', None) is not None else 0.0
            st.metric("Ground Speed", f"{spd:.2f} m/s")
            yaw = math.degrees(sim_beta.attitude[2]) if getattr(sim_beta, 'attitude', None) is not None else 0.0
            st.metric("Yaw / Heading", f"{yaw:.1f}°")
            st.metric("Sprayer Output State", "ACTIVE" if getattr(sim_beta, 'is_spraying', False) else "INACTIVE")

        bat = getattr(sim_beta, 'battery', 0.0)
        pmass = getattr(sim_beta, 'payload_mass', 0.0)
        st.write(f"**Battery Status:** {bat:.1f}%")
        st.progress(bat / 100.0)
        st.write(f"**Estimated Payload Mass:** {pmass:.2f} kg / 10.00 kg")
    else:
        st.info("Drone Beta Offline.")
