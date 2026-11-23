"""Renovated presentation of the original agriculture workflow."""

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from frontend_shared import (
    WS_CLIENTS, WS_CLIENT_ROLES, buffer_lock,
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
#  Spatial Reconstruction  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Spatial reconstruction", "Explore surface elevation, canopy height, and overlapping UAV captures.", category="spatial")

if "ms_image" not in st.session_state or "indices" not in st.session_state:
    st.warning("Please upload or enable demo data in Upload & Process first.")
else:
    from src.spatial.reconstruction import UAVSpatialReconstruction
    recon = UAVSpatialReconstruction(
        focal_length_px=1200.0,
        baseline_meters=0.5,
        uav_altitude_meters=30.0
    )

    ms_image = st.session_state["ms_image"]
    indices = st.session_state["indices"]

    recon_col1, recon_col2 = st.columns([1, 1])

    with recon_col1:
        st.subheader("UAV Orthomosaic Stitching Engine")
        st.markdown(
            "Stitches overlapping raw drone snapshots into a single georeferenced field orthomosaic "
            "using feature matching (ORB) and perspective homography transformation."
        )

        # Prepare overlapping simulated images
        rgb_preview = ms_image.rgb_preview()
        h_p, w_p = rgb_preview.shape[:2]

        overlap_w = int(w_p * 0.6)
        img_left = rgb_preview[:, :overlap_w, :]
        img_right_raw = rgb_preview[:, w_p - overlap_w:, :]

        # Add small rotation & shift to simulate different camera viewpoint
        M = cv2.getRotationMatrix2D((img_right_raw.shape[1]/2, img_right_raw.shape[0]/2), 2.5, 0.98)
        img_right = cv2.warpAffine(img_right_raw, M, (img_right_raw.shape[1], img_right_raw.shape[0]))

        st.markdown("**Simulated Overlapping Captures:**")
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.image(img_left, caption="UAV Capture 1 (Left)", use_container_width=True)
        sub_col2.image(img_right, caption="UAV Capture 2 (Right)", use_container_width=True)

        if st.button("Run Orthomosaic Stitcher", key="btn_stitch"):
            with st.spinner("Finding keypoint matches & calculating homography matrix..."):
                # Use OpenCV to compute keypoint matches for visualization
                detector = cv2.ORB_create(1000)
                kp1, des1 = detector.detectAndCompute(img_left, None)
                kp2, des2 = detector.detectAndCompute(img_right, None)

                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[:100]

                match_img = cv2.drawMatches(
                    img_left, kp1, img_right, kp2, matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # Run the actual stitcher code
                try:
                    stitched, _ = recon.stitch_images([img_left, img_right])

                    st.success(f"Stitching successful! Found {len(matches)} valid keypoint match vectors.")
                    st.image(match_img, caption="Keypoint Registration Vectors (ORB Matcher)", use_container_width=True)
                    st.image(stitched, caption="Unified Orthomosaic Stitched Output (Feather Blended)", use_container_width=True)
                except Exception as e:
                    st.error(f"Image stitching failed: {e}")

    with recon_col2:
        st.subheader("3D Elevation Modeling (DSM & CHM)")
        st.markdown(
            "Generate elevation profiles representing surface heights (Digital Surface Model) "
            "and extract crop heights (Canopy Height Model) by filtering out the terrain base."
        )

        dsm_mode = st.selectbox(
            "DSM Elevation Modeling Source",
            ["Spectral Shading & Biomass Model (Single Composite)", "Stereo Disparity Estimation (Overlap Disparity)"]
        )

        # Generate DSM
        cache_key = f"{dsm_mode}_{rgb_preview.shape}_{indices['ndvi'].mean():.4f}"
        if "cached_elevation" not in st.session_state or st.session_state.get("cached_elevation_key") != cache_key:
            with st.spinner("Extracting surface elevation map..."):
                try:
                    if dsm_mode == "Stereo Disparity Estimation (Overlap Disparity)":
                        # Simulate stereo pair from left & right slices
                        dsm = recon.generate_dsm(img_left, img_right)
                    else:
                        dsm = recon.generate_dsm_from_single(rgb_preview, indices["ndvi"])

                    # Compute CHM (Canopy Height Model) and DTM (Digital Terrain Model)
                    chm, dtm = recon.generate_chm(dsm)
                    st.session_state["cached_elevation"] = (dsm, chm, dtm)
                    st.session_state["cached_elevation_key"] = cache_key
                except Exception as e:
                    st.error(f"Elevation modeling failed: {e}")
        else:
            dsm, chm, dtm = st.session_state["cached_elevation"]

        shared_state["CHM"] = chm
        shared_state["NDVI_MAP"] = indices["ndvi"]

        # Key statistics
        max_height = float(chm.max())
        mean_height = float(chm.mean())
        biomass_vol = float(np.sum(chm) * (0.05 * 0.05)) # GSD = 0.05m per pixel
        slope_range = float(dtm.max() - dtm.min())

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Max Canopy Height", f"{max_height:.2f} m")
        m_col1.metric("Mean Canopy Height", f"{mean_height:.2f} m")
        m_col2.metric("Est. Canopy Biomass Vol.", f"{biomass_vol:.1f} m³")
        m_col2.metric("Terrain Base Slope", f"{slope_range:.2f} m")

        # Scientific Stage-Height Comparison
        st.subheader("Canopy Height vs. Scientific Stage Thresholds")

        stage_expected = {
            "Nursery":    (0.05, 0.25),
            "Vegetative": (0.25, 0.70),
            "Flowering":  (0.70, 1.10),
            "Mature":     (0.80, 1.20),
        }

        expected_range = stage_expected.get(crop_stage, (0.0, 1.5))
        lo, hi = expected_range

        height_status = ""
        if mean_height < lo:
            height_status = f"<div style='background-color:#5c3e16;border-left:5px solid #f39c12;padding:12px;border-radius:5px;margin-top:10px;margin-bottom:15px;color:#fff3cd'><b>STUNTED GROWTH WARNING:</b> Current mean canopy height of <b>{mean_height:.2f} m</b> is below the expected range of <b>{lo} - {hi} m</b> for the <b>{crop_stage}</b> stage. Under-fertilization or cold temperatures suspected.</div>"
        elif mean_height > hi:
            height_status = f"<div style='background-color:#5a1818;border-left:5px solid #e74c3c;padding:12px;border-radius:5px;margin-top:10px;margin-bottom:15px;color:#f8d7da'><b>OVERGROWTH / LODGING RISK:</b> Current mean canopy height of <b>{mean_height:.2f} m</b> exceeds the expected stage limit of <b>{hi} m</b>. Crops are taller than average, increasing physical lodging susceptibility under high wind conditions.</div>"
        else:
            height_status = f"<div style='background-color:#1e4620;border-left:5px solid #2ecc71;padding:12px;border-radius:5px;margin-top:10px;margin-bottom:15px;color:#d4edda'><b>NORMAL CANOPY HEIGHT:</b> Mean canopy height of <b>{mean_height:.2f} m</b> is within the optimal scientific range of <b>{lo} - {hi} m</b> for the <b>{crop_stage}</b> stage.</div>"

        st.markdown(height_status, unsafe_allow_html=True)

        # Show map plots
        st.markdown("**Elevation Profiles Map:**")
        fig_elev, (ax_dsm, ax_chm) = plt.subplots(1, 2, figsize=(10, 4.5))

        # Digital Surface Model Plot
        im_dsm = ax_dsm.imshow(dsm, cmap="terrain")
        ax_dsm.set_title("Digital Surface Model (DSM)", color="#f8fafc")
        ax_dsm.axis("off")
        fig_elev.colorbar(im_dsm, ax=ax_dsm, label="Elevation (m)", shrink=0.7)

        # Canopy Height Model Plot
        im_chm = ax_chm.imshow(chm, cmap="viridis")
        ax_chm.set_title("Canopy Height Model (CHM)", color="#f8fafc")
        ax_chm.axis("off")
        fig_elev.colorbar(im_chm, ax=ax_chm, label="Crop Height (m)", shrink=0.7)

        fig_elev.patch.set_facecolor('#0e1117')
        st.pyplot(fig_elev)

    # 3D Canopy Visualization
    st.markdown("---")
    st.subheader("WebGL 3D Interactive Digital Twin Viewer")

    # Interactive 3D Configuration Options
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        v_mode = st.radio("3D Viewer Mode", ["WebGL (Three.js Interactive)", "Matplotlib (Static Wireframe)"], index=0)
        bg_style = st.selectbox("Background Style", ["Dark Space", "Engineering Slate"], index=0)
    with opt_col2:
        h_source = st.selectbox("3D Heightmap Source", ["Canopy Height Model (CHM)", "Digital Surface Model (DSM)"], index=0)
        exaggeration = st.slider("3D Height Exaggeration Scale", 1.0, 15.0, 6.0, step=0.5)
    with opt_col3:
        tex_source = st.selectbox(
            "3D Surface Overlay Texture",
            ["True Color (RGB Preview)", "Crop Stress Index Map", "Vegetation Index (NDVI) Map", "Moisture Index (NDWI) Map"],
            index=1
        )

    if v_mode == "WebGL (Three.js Interactive)":
        # Select heightmap array
        height_array = chm if h_source == "Canopy Height Model (CHM)" else dsm

        # Select texture array
        if tex_source == "True Color (RGB Preview)":
            tex_array = rgb_preview
        elif tex_source == "Vegetation Index (NDVI) Map":
            tex_array = colormap_array(indices["ndvi"], "RdYlGn", 0, 1)
        elif tex_source == "Moisture Index (NDWI) Map":
            tex_array = colormap_array(indices["ndwi"], "GnBu", -1, 1)
        else:
            tex_array = colormap_array(indices["stress_score"], "RdYlGn_r", 0, 1)

        # Convert arrays to Base64
        with st.spinner("Generating WebGL textures..."):
            h_b64 = recon.export_to_base64(height_array, is_grayscale=True)
            t_b64 = recon.export_to_base64(tex_array, is_grayscale=False)

        # Map colors based on background style
        bg_color_hex = "#0e1117" if bg_style == "Dark Space" else "#1e293b"
        fog_color_hex = "0x0e1117" if bg_style == "Dark Space" else "0x1e293b"
        grid_color_1 = "0x334155" if bg_style == "Dark Space" else "0x475569"
        grid_color_2 = "0x1e293b" if bg_style == "Dark Space" else "0x334155"

        # Three.js Dynamic WebGL Viewport
        import time
        WEBSOCKET_PORT = shared_state.get("WEBSOCKET_PORT", 8765)
        TELEMETRY_PORT = shared_state.get("TELEMETRY_PORT", 8000)
        three_html = f"""\n<!-- Cache Buster: {time.time()} -->\n<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>UAV 3D Terrain Digital Twin</title>
<style>
    body {{
        margin: 0;
        overflow: hidden;
        background-color: {bg_color_hex};
        font-family: 'Inter', Arial, sans-serif;
        user-select: none;
    }}
    #app-container {{
        display: flex;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
    }}
    #canvas-container {{
        flex: 1;
        height: 100%;
        position: relative;
        z-index: 1;
    }}
    #sidebar-container {{
        width: 270px;
        height: 100%;
        background: #0f121a;
        border-left: 1px solid rgba(255,255,255,0.1);
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        color: #cbd5e1;
        box-sizing: border-box;
        z-index: 5;
    }}
    .sidebar-section {{
        padding: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .sidebar-section h3 {{
        margin: 0 0 12px 0;
        color: #f8fafc;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    #loading-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(14, 17, 23, 0.95);
        color: #f8fafc;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 10;
        transition: opacity 0.5s ease;
        pointer-events: none;
    }}
    .spinner {{
        border: 4px solid rgba(255,255,255,0.1);
        width: 50px;
        height: 50px;
        border-radius: 50%;
        border-left-color: #38bdf8;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
    }}
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}

    .control-group {{
        margin-bottom: 12px;
    }}
    .control-group label {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        color: #94a3b8;
        font-size: 11px;
    }}
    .gui-btn {{
        background: #0284c7;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 7px 12px;
        cursor: pointer;
        font-weight: 600;
        font-size: 11px;
        transition: background 0.2s;
        flex: 1;
        text-align: center;
    }}
    .gui-btn:hover {{
        background: #0369a1;
    }}
    .gui-btn.active {{
        background: #22c55e;
    }}
    .btn-row {{
        display: flex;
        gap: 8px;
    }}
    .gui-select, .gui-slider {{
        width: 100%;
        background: #1e293b;
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 6px;
        padding: 6px;
        font-size: 11px;
        box-sizing: border-box;
    }}
    .gui-slider {{
        height: 6px;
        border-radius: 3px;
        outline: none;
        -webkit-appearance: none;
    }}
    .gui-slider::-webkit-slider-thumb {{
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #38bdf8;
        cursor: pointer;
    }}

    /* Monospace HUD details */
    .hud-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-family: monospace;
        font-size: 11px;
    }}
    .hud-label {{
        color: #94a3b8;
    }}
    .hud-val {{
        color: #38bdf8;
        font-weight: bold;
        text-align: right;
    }}
    .payload-bar {{
        height: 5px;
        background: #334155;
        border-radius: 3px;
        margin-top: 6px;
        overflow: hidden;
    }}
    .payload-fill {{
        height: 100%;
        background: #0ea5e9;
        width: 100%;
        transition: width 0.1s;
    }}

    .sensor-display {{
        margin-top: 12px;
        background: rgba(2, 6, 23, 0.4);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-size: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    .sensor-status {{
        font-size: 12px;
        font-weight: bold;
        margin-top: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
        box-shadow: 0 0 8px rgba(255,255,255,0.5);
    }}

    /* Legend HUD overlay inside canvas */
    .canvas-overlay-legend {{
        position: absolute;
        bottom: 16px;
        right: 16px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 10px 14px;
        color: #94a3b8;
        font-size: 10px;
        pointer-events: none;
        backdrop-filter: blur(4px);
        z-index: 3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        margin-bottom: 4px;
        color: #cbd5e1;
    }}
    .legend-color {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
<div id="loading-overlay">
    <div class="spinner"></div>
    <div style="font-size: 16px; font-weight: 500;">Generating 3D Digital Twin Mesh...</div>
    <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">Triangulating terrain elevation & wrapping stress overlay texture</div>
</div>

<div id="app-container">
    <div id="canvas-container">
        <div class="canvas-overlay-legend">
            <div style="font-weight:bold; margin-bottom:6px; color:#f8fafc; font-size:11px;">Terrain Stress Indicators</div>
            <div class="legend-item"><div class="legend-color" style="background:#22c55e;"></div>Healthy / High NDVI</div>
            <div class="legend-item"><div class="legend-color" style="background:#eab308;"></div>Moisture/Nitrogen Deficit</div>
            <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>Severe Disease Pathogen</div>

            <div style="font-weight:bold; margin-top:10px; margin-bottom:6px; color:#f8fafc; font-size:11px;">Active Spray Agents</div>
            <div class="legend-item"><div class="legend-color" style="background:#38bdf8;"></div>Blanket Spray (Blue: Standard)</div>
            <div class="legend-item"><div class="legend-color" style="background:#ec4899;"></div>Fungicide (Pink: Pathogen)</div>
            <div class="legend-item"><div class="legend-color" style="background:#eab308;"></div>Nutrient (Yellow: Nitrogen)</div>
        </div>
    </div>

    <div id="sidebar-container">
        <!-- Telemetry HUD -->
        <div class="sidebar-section">
            <h3>UAV Telemetry <span id="hud-status" style="color:#22c55e; font-size:10px;">AUTO</span></h3>
            <div class="hud-row">
                <span class="hud-label">WebSocket Sync:</span>
                <span id="ws-status-indicator" class="hud-val" style="background:#ef4444; color:#ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 9px; letter-spacing: 0.5px;">WS OFFLINE</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">GPS Latitude:</span>
                <span id="hud-lat" class="hud-val">0.00 m</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">GPS Longitude:</span>
                <span id="hud-lon" class="hud-val">0.00 m</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Altitude (GPS):</span>
                <span id="hud-alt" class="hud-val">12.00 m</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Altitude (AGL):</span>
                <span id="hud-agl" class="hud-val">10.00 m</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Pitch / Roll:</span>
                <span id="hud-pitch" class="hud-val">0° / 0°</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Air Speed:</span>
                <span id="hud-speed" class="hud-val">0.0 m/s</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Battery Level:</span>
                <span id="hud-battery" class="hud-val" style="color:#22c55e;">100.0%</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Power Draw:</span>
                <span id="hud-power" class="hud-val">0 W</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Energy Efficiency:</span>
                <span id="hud-efficiency" class="hud-val">0.0 Wh/km</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Total Mass:</span>
                <span id="hud-mass" class="hud-val">12.00 kg</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Rotors RPM (F/B):</span>
                <span id="hud-rpm" class="hud-val">0 / 0 RPM</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Wind Vector:</span>
                <span id="hud-wind" class="hud-val">8 m/s @ 45°</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Active Treatment:</span>
                <span id="hud-treatment" class="hud-val" style="color:#38bdf8;">None</span>
            </div>
            <div class="hud-row" style="margin-top:10px;">
                <span class="hud-label">Spray Payload:</span>
                <span id="hud-payload" class="hud-val">100.0%</span>
            </div>
            <div class="payload-bar">
                <div id="payload-fill" class="payload-fill"></div>
            </div>

            <div class="sensor-display">
                <span>LIVE GYRO SENSOR UNDERNEATH</span>
                <div class="sensor-status">
                    <span id="sensor-dot" class="status-dot" style="background:#22c55e;"></span>
                    <span id="sensor-text">HEALTHY TARGET ZONE</span>
                </div>
            </div>
        </div>

        <!-- Multiplayer & Supervision -->
        <div class="sidebar-section">
            <h3>Multiplayer Operations</h3>
            <div class="hud-row">
                <span class="hud-label">Role Profile:</span>
                <span id="hud-my-role" class="hud-val" style="background:#475569; color:#ffffff; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 9px; letter-spacing: 0.5px;">OBSERVER</span>
            </div>
            <div class="hud-row">
                <span class="hud-label">Connected Sessions:</span>
                <span id="hud-total-clients" class="hud-val">1 active</span>
            </div>

            <div class="control-group">
                <label class="control-label">Select Operator Role</label>
                <select id="select-multiplayer-role" class="gui-select">
                    <option value="observer">Observer (View-Only)</option>
                    <option value="operator_alpha">Operator Alpha (Drone Alpha)</option>
                    <option value="operator_beta">Operator Beta (Drone Beta)</option>
                    <option value="supervisor">Supervisor (Remote Command)</option>
                </select>
            </div>

            <div class="control-group" id="group-camera-sync">
                <label class="checkbox-label" style="display: flex; align-items: center; color: #cbd5e1; font-size: 11px; cursor: pointer; user-select: none;">
                    <input id="check-camera-sync" type="checkbox" style="margin-right: 8px;">
                    Sync to Supervisor Camera
                </label>
            </div>

            <div class="control-group">
                <div class="btn-row">
                    <button id="btn-add-pin" class="gui-btn" style="width:100%; display:flex; align-items:center; justify-content:center; gap:6px;">
                        Place 3D Field Pin
                    </button>
                </div>
                <div id="pin-placement-indicator" style="display:none; text-align:center; font-size:10px; color:#f43f5e; margin-top:5px; font-weight:500;">
                    Click on 3D Terrain to Place Pin
                </div>
            </div>
        </div>

        <!-- Controls -->
        <div class="sidebar-section">
            <h3>Flight Controller</h3>
            <div class="control-group">
                <div class="btn-row">
                    <button id="btn-play" class="gui-btn active">Pause</button>
                    <button id="btn-reset" class="gui-btn">Reset</button>
                </div>
            </div>

            <div class="control-group">
                <label>Mission Path</label>
                <select id="select-path" class="gui-select">
                    <option value="grid">Coverage Mode (Grid Survey)</option>
                    <option value="spot_visit">Waypoint Mode (Stress Spots)</option>
                    <option value="orbit">Circular Orbit Pattern</option>
                    <option value="mavlink">MAVLink PX4/SITL Sync</option>
                </select>
            </div>

            <div class="control-group">
                <label>Camera Angle</label>
                <select id="select-cam" class="gui-select">
                    <option value="orbit">Orbit (Free View)</option>
                    <option value="follow">Follow (Rear View)</option>
                    <option value="fpv">FPV (Down Sensor)</option>
                </select>
            </div>

            <div class="control-group">
                <label>Pesticide Spray</label>
                <button id="btn-spray" class="gui-btn active" style="width:100%;">Spray: ON</button>
            </div>

            <div class="control-group">
                <label>Active Chemical Agent</label>
                <select id="select-chemical" class="gui-select">
                    <option value="dynamic">Dynamic Sensor Tracking (Smart)</option>
                    <option value="blanket">Broad-Spectrum Blanket (Blue)</option>
                    <option value="fungicide">Targeted Fungicide (Pink)</option>
                    <option value="nutrient">Liquid Nitrogen Nutrient (Yellow)</option>
                </select>
            </div>

            <div class="control-group">
                <label>Season & Growth Stage</label>
                <select id="select-season" class="gui-select">
                    <option value="summer">Summer (Peak Green)</option>
                    <option value="spring">Spring (Early Growth)</option>
                    <option value="autumn">Autumn (Harvest Gold)</option>
                    <option value="winter">Winter (Fallow / Brown)</option>
                </select>
            </div>

            <div class="control-group">
                <label>Crop Canopy Density</label>
                <select id="select-density" class="gui-select">
                    <option value="medium">Medium Canopy (Dense Grid)</option>
                    <option value="sparse">Sparse Canopy (Light Grid)</option>
                    <option value="dense">Ultra Dense (Performance Heavy)</option>
                </select>
            </div>

            <div class="control-group">
                <label>CFD Visualizer</label>
                <button id="btn-toggle-cfd" class="gui-btn active" style="width:100%;">Airflow Vectors: ON</button>
            </div>

            <div class="control-group">
                <label><span>Wind Speed</span><span id="val-wind-speed">8 m/s</span></label>
                <input id="slider-wind-speed" type="range" min="0" max="20" value="8" class="gui-slider">
            </div>

            <div class="control-group">
                <label><span>Wind Angle</span><span id="val-wind-dir">45°</span></label>
                <input id="slider-wind-dir" type="range" min="0" max="360" value="45" class="gui-slider">
            </div>
        </div>
    </div>
</div>

<script>
    const heightmapSrc = "{h_b64}";
    const textureSrc = "{t_b64}";
    const maxHeightScale = {exaggeration};

    const container = document.getElementById('canvas-container');
    const loader = document.getElementById('loading-overlay');

    // Parameters for wind and spray
    let windSpeed = 8.0;
    let windDir = 45.0; // degrees
    let isPlaying = true;
    let isSpraying = true;
    let showCFD = true;
    let activeChemical = 'dynamic'; // dynamic vs blanket vs fungicide vs nutrient
    let flightPathMode = 'grid'; // grid vs orbit vs spot_visit
    let cameraMode = 'orbit'; // orbit vs follow vs FPV
    let remainingPayload = 100.0;
    let isCurrentlyEmitting = false;

    // Seasonal & Procedural Canopy State Variables
    let activeSeason = 'summer'; // summer vs spring vs autumn vs winter
    let cropDensity = 'medium'; // sparse vs medium vs dense
    let instancedCrops = null; // reference to InstancedMesh

    // UAV Aerodynamic Physics Engine variables (6-DOF State)
    const droneVelocity = new THREE.Vector3(0, 0, 0);
    const droneRotation = new THREE.Vector3(0, 0, 0); // Pitch, Roll, Yaw
    const droneAngularVelocity = new THREE.Vector3(0, 0, 0);

    let energyConsumedWh = 0.0;
    const batteryCapacityWh = 320.0; // typical lithium flight battery size
    let totalDistanceTraveledM = 0.0;
    let pathDirection = 1; // 1 for forward, -1 for backward

    const mass_base = 12.0; // kg
    const mass_payload_max = 8.0; // kg

    // Setup event listeners for UI Panel controls
    document.getElementById('btn-play').addEventListener('click', (e) => {{
        isPlaying = !isPlaying;
        e.target.textContent = isPlaying ? "Pause" : "Resume";
        if (isPlaying) e.target.classList.add('active');
        else e.target.classList.remove('active');
    }});

    document.getElementById('btn-reset').addEventListener('click', () => {{
        resetSimulation();
    }});

    document.getElementById('btn-spray').addEventListener('click', (e) => {{
        isSpraying = !isSpraying;
        e.target.textContent = isSpraying ? "Spray: ON" : "Spray: OFF";
        if (isSpraying) e.target.classList.add('active');
        else e.target.classList.remove('active');
    }});

    document.getElementById('select-chemical').addEventListener('change', (e) => {{
        activeChemical = e.target.value;
        sendEnvironmentUpdate();
    }});

    document.getElementById('select-season').addEventListener('change', (e) => {{
        activeSeason = e.target.value;
        buildCrops();
        sendEnvironmentUpdate();
    }});

    document.getElementById('select-density').addEventListener('change', (e) => {{
        cropDensity = e.target.value;
        buildCrops();
    }});

    document.getElementById('btn-toggle-cfd').addEventListener('click', (e) => {{
        showCFD = !showCFD;
        e.target.textContent = showCFD ? "Airflow Vectors: ON" : "Airflow Vectors: OFF";
        if (showCFD) e.target.classList.add('active');
        else e.target.classList.remove('active');
        if (cfdFlowLines) cfdFlowLines.visible = showCFD;
    }});

    document.getElementById('select-path').addEventListener('change', (e) => {{
        flightPathMode = e.target.value;
        if (flightPathMode !== 'mavlink') {{
            document.getElementById('hud-status').textContent = "AUTO";
            document.getElementById('hud-status').style.color = "#22c55e";
        }}
        generatePath();
    }});

    document.getElementById('select-cam').addEventListener('change', (e) => {{
        cameraMode = e.target.value;
        if (cameraMode === 'orbit') {{
            controls.enabled = true;
        }} else {{
            controls.enabled = false;
        }}
    }});

    document.getElementById('slider-wind-speed').addEventListener('input', (e) => {{
        windSpeed = parseFloat(e.target.value);
        updateWindHUD();
        sendEnvironmentUpdate();
    }});

    document.getElementById('slider-wind-dir').addEventListener('input', (e) => {{
        windDir = parseFloat(e.target.value);
        updateWindHUD();
        sendEnvironmentUpdate();
    }});

    function updateWindHUD() {{
        document.getElementById('hud-wind').textContent = `${{windSpeed}} m/s @ ${{windDir}}°`;
        document.getElementById('val-wind-speed').textContent = `${{windSpeed}} m/s`;
        document.getElementById('val-wind-dir').textContent = `${{windDir}}°`;
    }}

    // Multiplayer Control Listeners
    document.getElementById('select-multiplayer-role').addEventListener('change', (e) => {{
        myRole = e.target.value;
        document.getElementById('hud-my-role').textContent = myRole.toUpperCase().replace('_', ' ');

        if (myRole === 'operator_alpha') {{
            myDroneId = 'drone_alpha';
        }} else if (myRole === 'operator_beta') {{
            myDroneId = 'drone_beta';
        }} else {{
            myDroneId = null;
        }}

        const camSyncGroup = document.getElementById('group-camera-sync');
        if (myRole === 'supervisor') {{
            camSyncGroup.style.display = 'none';
            followSupervisor = false;
            document.getElementById('check-camera-sync').checked = false;
        }} else {{
            camSyncGroup.style.display = 'block';
        }}

        if (wsConnected && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{
                type: "client_update",
                role: myRole
            }}));
        }}
    }});

    document.getElementById('check-camera-sync').addEventListener('change', (e) => {{
        followSupervisor = e.target.checked;
    }});

    document.getElementById('btn-add-pin').addEventListener('click', (e) => {{
        isPinPlacementMode = !isPinPlacementMode;
        const pinIndicator = document.getElementById('pin-placement-indicator');
        if (isPinPlacementMode) {{
            pinIndicator.style.display = 'block';
            e.target.classList.add('active');
        }} else {{
            pinIndicator.style.display = 'none';
            e.target.classList.remove('active');
        }}
    }});

    // WebSocket sync client variables
    let ws = null;
    let wsConnected = false;
    let latestTelemetryData = null;

    // Multiplayer Operations variables
    let myRole = 'observer';
    let myDroneId = null;
    let followSupervisor = false;
    let latestSupervisorCamera = null;
    let lastCameraUpdate = 0;
    let isPinPlacementMode = false;
    const activePins = {{}};
    let multiDrones = {{}};

    function connectWebSocket() {{
        try {{
            ws = new WebSocket("ws://127.0.0.1:{WEBSOCKET_PORT}");

            ws.onopen = () => {{
                console.log("WebSocket Sync Active on port {WEBSOCKET_PORT}");
                wsConnected = true;
                const wsStatus = document.getElementById('ws-status-indicator');
                if (wsStatus) {{
                    wsStatus.textContent = "WS ACTIVE";
                    wsStatus.style.backgroundColor = "#22c55e";
                }}
            }};

            ws.onmessage = (event) => {{
                try {{
                    const payload = JSON.parse(event.data);

                    // Sync environment updates from Python
                    if (payload.environment) {{
                        if (payload.environment.wind_speed !== undefined) {{
                            windSpeed = payload.environment.wind_speed;
                            document.getElementById('slider-wind-speed').value = windSpeed;
                        }}
                        if (payload.environment.wind_direction !== undefined) {{
                            windDir = payload.environment.wind_direction;
                            document.getElementById('slider-wind-dir').value = windDir;
                        }}
                        updateWindHUD();

                        if (payload.environment.active_agent) {{
                            let chemVal = 'dynamic';
                            if (payload.environment.active_agent === "Targeted Fungicide") chemVal = 'fungicide';
                            else if (payload.environment.active_agent === "Liquid Nitrogen Nutrient") chemVal = 'nutrient';
                            else if (payload.environment.active_agent === "Broad-Spectrum Blanket") chemVal = 'blanket';

                            if (chemVal !== activeChemical) {{
                                activeChemical = chemVal;
                                const chemSelect = document.getElementById('select-chemical');
                                if (chemSelect) chemSelect.value = activeChemical;
                            }}
                        }}
                        if (payload.environment.season) {{
                            let seasonVal = 'spring';
                            if (payload.environment.season === "Midsummer Lush") seasonVal = 'summer';
                            else if (payload.environment.season === "Autumn Harvest") seasonVal = 'autumn';
                            else if (payload.environment.season === "Drought Parched") seasonVal = 'winter';

                            if (seasonVal !== activeSeason) {{
                                activeSeason = seasonVal;
                                const seasonSelect = document.getElementById('select-season');
                                if (seasonSelect) {{
                                    seasonSelect.value = activeSeason;
                                    buildCrops(); // rebuild to match season colors
                                }}
                            }}
                        }}
                    }}

                    // Save telemetry data for animation loops
                    if (payload.telemetry) {{
                        latestTelemetryData = payload.telemetry;
                    }}

                    // Sync home GPS anchor if needed
                    if (homeLat === null || homeLon === null) {{
                        if (payload.telemetry && payload.telemetry.lat) {{
                            homeLat = payload.telemetry.lat;
                            homeLon = payload.telemetry.lon;
                        }} else if (payload.drones && payload.drones.drone_alpha) {{
                            homeLat = payload.drones.drone_alpha.lat;
                            homeLon = payload.drones.drone_alpha.lon;
                        }}
                    }}

                    // Sync dynamic supervisor camera view
                    if (payload.sync_camera && followSupervisor && myRole !== 'supervisor') {{
                        latestSupervisorCamera = payload.sync_camera;
                    }} else {{
                        latestSupervisorCamera = null;
                    }}

                    // Sync connected clients count
                    if (payload.clients) {{
                        const totalClients = payload.clients.total;
                        document.getElementById('hud-total-clients').textContent = `${{totalClients}} active`;
                    }}

                    // Sync annotations (pins)
                    if (payload.annotations) {{
                        const currentPinIds = new Set(payload.annotations.map(a => a.id));

                        // Remove deleted pins
                        for (let id in activePins) {{
                            if (!currentPinIds.has(id)) {{
                                scene.remove(activePins[id]);
                                delete activePins[id];
                            }}
                        }}

                        // Add/update pins
                        payload.annotations.forEach(ann => {{
                            if (!activePins[ann.id]) {{
                                // 3D Pin Cone
                                const pinGeom = new THREE.ConeGeometry(0.3, 1.2, 8);
                                pinGeom.rotateX(Math.PI / 2);
                                const pinMat = new THREE.MeshBasicMaterial({{ color: 0xef4444 }});
                                const pinMesh = new THREE.Mesh(pinGeom, pinMat);
                                pinMesh.position.set(ann.x, ann.y, ann.z + 0.6);

                                // Canvas text label billboard sprite
                                const canvas = document.createElement('canvas');
                                canvas.width = 256;
                                canvas.height = 64;
                                const ctx = canvas.getContext('2d');
                                ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
                                ctx.fillRect(0, 0, 256, 64);
                                ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                                ctx.lineWidth = 2;
                                ctx.strokeRect(0, 0, 256, 64);
                                ctx.fillStyle = '#ffffff';
                                ctx.font = 'bold 12px sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(ann.label, 128, 28);
                                ctx.font = 'bold 9px sans-serif';
                                ctx.fillStyle = '#f43f5e';
                                ctx.fillText(ann.creator_role, 128, 48);

                                const texture = new THREE.CanvasTexture(canvas);
                                const spriteMat = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
                                const sprite = new THREE.Sprite(spriteMat);
                                sprite.position.set(ann.x, ann.y, ann.z + 2.0);
                                sprite.scale.set(4, 1.0, 1);

                                const pinGroup = new THREE.Group();
                                pinGroup.add(pinMesh);
                                pinGroup.add(sprite);
                                scene.add(pinGroup);

                                activePins[ann.id] = pinGroup;
                            }}
                        }});
                    }}

                    // Sync multiplayer drones
                    if (payload.drones) {{
                        for (let droneId in payload.drones) {{
                            if (myDroneId === droneId) {{
                                if (multiDrones[droneId]) {{
                                    scene.remove(multiDrones[droneId].group);
                                    delete multiDrones[droneId];
                                }}
                                continue;
                            }}

                            const droneData = payload.drones[droneId];
                            if (homeLat !== null && homeLon !== null) {{
                                const EARTH_RADIUS = 6378137.0;
                                const latRad = droneData.lat * Math.PI / 180;
                                const lonRad = droneData.lon * Math.PI / 180;
                                const homeLatRad = homeLat * Math.PI / 180;
                                const homeLonRad = homeLon * Math.PI / 180;
                                const dy = (latRad - homeLatRad) * EARTH_RADIUS;
                                const dx = (lonRad - homeLonRad) * EARTH_RADIUS * Math.cos(homeLatRad);

                                const limitX = sizeX / 2.0 - 1.5;
                                const limitY = sizeY / 2.0 - 1.5;
                                const localX = Math.max(-limitX, Math.min(limitX, dx));
                                const localY = Math.max(-limitY, Math.min(limitY, dy));
                                const localZ = droneData.alt;

                                if (!multiDrones[droneId]) {{
                                    const colHex = droneId === 'drone_alpha' ? 0x38bdf8 : 0xec4899;
                                    const model = createDroneModel(colHex);
                                    scene.add(model.group);
                                    multiDrones[droneId] = {{
                                        group: model.group,
                                        rotors: model.rotors,
                                        sprayNozzles: model.sprayNozzles,
                                        targetPos: new THREE.Vector3(localX, localY, localZ),
                                        targetRot: new THREE.Vector3(droneData.roll, droneData.pitch, droneData.yaw),
                                        isSpraying: droneData.is_spraying,
                                        particles: []
                                    }};
                                }} else {{
                                    multiDrones[droneId].targetPos.set(localX, localY, localZ);
                                    multiDrones[droneId].targetRot.set(droneData.roll, droneData.pitch, droneData.yaw);
                                    multiDrones[droneId].isSpraying = droneData.is_spraying;
                                }}
                            }}
                        }}
                    }}
                }} catch (e) {{
                    console.log("WS message error:", e);
                }}
            }};

            ws.onclose = () => {{
                console.log("WebSocket disconnected. Retrying in 3 seconds...");
                wsConnected = false;
                const wsStatus = document.getElementById('ws-status-indicator');
                if (wsStatus) {{
                    wsStatus.textContent = "WS OFFLINE (POLLING)";
                    wsStatus.style.backgroundColor = "#eab308";
                }}
                setTimeout(connectWebSocket, 3000);
            }};

            ws.onerror = (err) => {{
                ws.close();
            }};
        }} catch (e) {{
            console.log("WebSocket initialization error:", e);
            setTimeout(connectWebSocket, 3000);
        }}
    }}

    function sendEnvironmentUpdate() {{
        if (wsConnected && ws.readyState === WebSocket.OPEN) {{
            let pythonAgent = "Dynamic Smart Tracking";
            if (activeChemical === 'fungicide') pythonAgent = "Targeted Fungicide";
            else if (activeChemical === 'nutrient') pythonAgent = "Liquid Nitrogen Nutrient";
            else if (activeChemical === 'blanket') pythonAgent = "Broad-Spectrum Blanket";

            let pythonSeason = "Spring Green";
            if (activeSeason === 'summer') pythonSeason = "Midsummer Lush";
            else if (activeSeason === 'autumn') pythonSeason = "Autumn Harvest";
            else if (activeSeason === 'winter') pythonSeason = "Drought Parched";

            ws.send(JSON.stringify({{
                type: "environment_update",
                wind_speed: windSpeed,
                wind_direction: windDir,
                season: pythonSeason,
                active_agent: pythonAgent
            }}));
        }}
    }}

    // Initialize connection
    connectWebSocket();

    // Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("{bg_color_hex}");
    scene.fog = new THREE.FogExp2({fog_color_hex}, 0.012);

    // Camera Setup
    const camera = new THREE.PerspectiveCamera(45, (window.innerWidth - 270) / window.innerHeight, 0.1, 1000);
    camera.position.set(0, -60, 45);
    camera.up.set(0, 0, 1); // Z is up

    // Renderer Setup
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth - 270, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    renderer.domElement.addEventListener('click', onCanvasClick);

    // Orbit Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.minDistance = 10;
    controls.maxDistance = 180;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.0); // Slightly brighter for PBR canopy detail
    sunLight.position.set(40, -40, 60);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048; // High-res shadows
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 200;
    sunLight.shadow.camera.left = -40;
    sunLight.shadow.camera.right = 40;
    sunLight.shadow.camera.top = 40;
    sunLight.shadow.camera.bottom = -40;
    scene.add(sunLight);

    const sunFillLight = new THREE.DirectionalLight(0x88ccff, 0.4);
    sunFillLight.position.set(-40, 40, 10);
    scene.add(sunFillLight);

    // Grid Helper
    const gridHelper = new THREE.GridHelper(120, 40, {grid_color_1}, {grid_color_2});
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.z = -0.5;
    scene.add(gridHelper);

    // Asset Loading
    let heightLoaded = false;
    let textureLoaded = false;
    let heightPix = null;
    let texturePix = null;
    let heightW = 0, heightH = 0;
    let textureW = 0, textureH = 0;

    const heightImg = new Image();
    const textureImg = new Image();

    heightImg.src = heightmapSrc;
    textureImg.src = textureSrc;

    function initTerrain() {{
        if (heightLoaded && textureLoaded) {{
            buildMesh();
            loader.style.opacity = 0;
            setTimeout(() => loader.style.display = 'none', 500);
        }}
    }}

    heightImg.onload = () => {{ heightLoaded = true; initTerrain(); }};
    textureImg.onload = () => {{ textureLoaded = true; initTerrain(); }};

    let meshMesh;
    let drawCanvas, drawCtx, canvasTexture;
    let sizeX = 55;
    let sizeY = 55;

    // CFD Flow Fields Global variables
    let cfdFlowLines = null;
    let cfdParticles = [];
    const cfdNumSegs = 220;

    // CFD Math vector field calculation (Navier-Stokes approximation)
    function getAirflowAt(pos) {{
        const flow = new THREE.Vector3(0, 0, 0);

        // 1. Base Wind with logarithmic altitude scaling
        const windAngleRad = (windDir * Math.PI) / 180;
        const baseWindX = Math.cos(windAngleRad) * windSpeed;
        const baseWindY = Math.sin(windAngleRad) * windSpeed;

        // Altitude scaling: wind is slower near canopy, faster higher up
        const hLocal = getTerrainHeightAt(pos.x, pos.y);
        const heightAboveCanopy = Math.max(0.1, pos.z - hLocal);
        const windAltScale = Math.min(1.5, Math.log(heightAboveCanopy + 1.0) / Math.log(6.0));

        flow.x = baseWindX * windAltScale;
        flow.y = baseWindY * windAltScale;

        // 2. Terrain Deflection (Slope Interaction)
        const sampleDist = 2.0;
        const windDirX = Math.cos(windAngleRad);
        const windDirY = Math.sin(windAngleRad);
        const hAhead = getTerrainHeightAt(pos.x + windDirX * sampleDist, pos.y + windDirY * sampleDist);
        const slope = (hAhead - hLocal) / sampleDist;

        // If wind is hitting an uphill, deflect it upwards
        const deflection = slope * windSpeed * 0.7 * Math.max(0, 1.0 - heightAboveCanopy / 8.0);
        flow.z += deflection;

        // 3. Rotor Downwash & Canopy Deflection (Only influences air below drone, not drone itself)
        if (droneGroup) {{
            const dPos = droneGroup.position;
            const dx = pos.x - dPos.x;
            const dy = pos.y - dPos.y;
            const r = Math.sqrt(dx*dx + dy*dy);
            const dz = dPos.z - pos.z;

            if (dz > 0 && dz < 25.0) {{
                const downwashRadius = 2.5 + dz * 0.12; // expanding cone
                if (r < downwashRadius) {{
                    // Gaussian-like horizontal falloff
                    const radialFactor = Math.exp(- (r * r) / (2.0 * 2.0));
                    const verticalDecay = Math.max(0.0, 1.0 - dz / 20.0);

                    // Downward downwash speed column
                    const downwashForce = -16.0 * radialFactor * verticalDecay;
                    flow.z += downwashForce;

                    // Canopy ground outflow: downwash hits the ground and spreads radially outward
                    const canopyDist = pos.z - hLocal;
                    if (canopyDist < 6.0 && r > 0.1) {{
                        const spreadScale = Math.max(0.0, 1.0 - canopyDist / 6.0) * verticalDecay;
                        const spreadSpeed = 10.0 * (r / downwashRadius) * radialFactor * spreadScale;
                        flow.x += (dx / r) * spreadSpeed;
                        flow.y += (dy / r) * spreadSpeed;
                    }}
                }}
            }}
        }}

        // 4. Perlin-like pseudo-random turbulence
        const t = Date.now() * 0.003;
        const turbScale = 0.08 * windSpeed;
        const turbX = Math.sin(pos.x * 0.15 + t) * Math.cos(pos.y * 0.1 + t * 0.7) * turbScale;
        const turbY = Math.cos(pos.x * 0.1 + t * 0.8) * Math.sin(pos.y * 0.15 + t) * turbScale;
        const turbZ = Math.sin(pos.z * 0.2 + t * 1.2) * Math.cos(pos.x * 0.1 + t) * turbScale * 0.5;

        flow.x += turbX;
        flow.y += turbY;
        flow.z += turbZ;

        return flow;
    }}

    function getRandomAirflowSpawnPos() {{
        const rx = (Math.random() - 0.5) * sizeX;
        const ry = (Math.random() - 0.5) * sizeY;
        const rz = getTerrainHeightAt(rx, ry) + 1.0 + Math.random() * 20.0;
        return new THREE.Vector3(rx, ry, rz);
    }}

    function buildMesh() {{
        heightW = heightImg.width;
        heightH = heightImg.height;
        textureW = textureImg.width;
        textureH = textureImg.height;

        const canvas = document.createElement('canvas');
        canvas.width = heightW;
        canvas.height = heightH;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(heightImg, 0, 0);
        heightPix = ctx.getImageData(0, 0, heightW, heightH).data;

        // Create Dynamic Painting Canvas
        drawCanvas = document.createElement('canvas');
        drawCanvas.width = textureW;
        drawCanvas.height = textureH;
        drawCtx = drawCanvas.getContext('2d');
        drawCtx.drawImage(textureImg, 0, 0);

        // Extract color data for sensor query
        texturePix = drawCtx.getImageData(0, 0, textureW, textureH).data;

        const aspect = heightW / heightH;
        sizeX = 55 * aspect;
        sizeY = 55;

        const geom = new THREE.PlaneGeometry(sizeX, sizeY, heightW - 1, heightH - 1);
        const pos = geom.attributes.position;

        for (let y = 0; y < heightH; y++) {{
            for (let x = 0; x < heightW; x++) {{
                const idx = (y * heightW + x) * 4;
                const rawVal = heightPix[idx] / 255;
                const zVal = rawVal * maxHeightScale;

                const vIdx = y * heightW + x;
                pos.setZ(vIdx, zVal);
            }}
        }}

        pos.needsUpdate = true;
        geom.computeVertexNormals();

        canvasTexture = new THREE.CanvasTexture(drawCanvas);
        canvasTexture.needsUpdate = true;

        // Upgrade to PBR MeshStandardMaterial
        const mat = new THREE.MeshStandardMaterial({{
            map: canvasTexture,
            roughness: 0.88,
            metalness: 0.08,
            flatShading: false,
            side: THREE.DoubleSide
        }});

        meshMesh = new THREE.Mesh(geom, mat);
        meshMesh.castShadow = true;
        meshMesh.receiveShadow = true;
        scene.add(meshMesh);

        // CFD visualizer line segments
        const cfdGeom = new THREE.BufferGeometry();
        const cfdPositions = new Float32Array(cfdNumSegs * 6); // 2 vertices per segment
        cfdGeom.setAttribute('position', new THREE.BufferAttribute(cfdPositions, 3));

        const cfdMat = new THREE.LineBasicMaterial({{
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.38,
            depthWrite: false
        }});
        cfdFlowLines = new THREE.LineSegments(cfdGeom, cfdMat);
        scene.add(cfdFlowLines);

        // Initialize CFD particles
        cfdParticles = [];
        for (let i = 0; i < cfdNumSegs; i++) {{
            cfdParticles.push({{
                pos: getRandomAirflowSpawnPos(),
                age: Math.random() * 4.0
            }});
        }}

        // Drone model creation
        buildDroneModel();
        generatePath();

        // Render vegetated crop canopy grid
        buildCrops();
    }}

    // Procedural Low-Poly Cross-Blade Crop geometry template
    function createCropGeometry() {{
        const geom = new THREE.BufferGeometry();

        // Intersecting double vertical planes (X shape billboard)
        // Z is pointing UP in world space. Root of crop is at Z=0. Height is 1.4m.
        const vertices = new Float32Array([
            // Blade 1 (X-Z plane)
            -0.35, 0, 0,
             0.35, 0, 0,
             0.35, 0, 1.4,
            -0.35, 0, 0,
             0.35, 0, 1.4,
            -0.35, 0, 1.4,

            // Blade 2 (Y-Z plane)
            0, -0.35, 0,
            0,  0.35, 0,
            0,  0.35, 1.4,
            0, -0.35, 0,
            0,  0.35, 1.4,
            0, -0.35, 1.4
        ]);

        const uvs = new Float32Array([
            // Blade 1
            0, 0,  1, 0,  1, 1,
            0, 0,  1, 1,  0, 1,

            // Blade 2
            0, 0,  1, 0,  1, 1,
            0, 0,  1, 1,  0, 1
        ]);

        const normals = new Float32Array([
            // Blade 1
            0, 1, 0,  0, 1, 0,  0, 1, 0,
            0, 1, 0,  0, 1, 0,  0, 1, 0,

            // Blade 2
            1, 0, 0,  1, 0, 0,  1, 0, 0,
            1, 0, 0,  1, 0, 0,  1, 0, 0
        ]);

        geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

        return geom;
    }}

    // Custom PBR Material containing GLSL Vertex Shader modification for wind sway
    function createCropMaterial() {{
        const mat = new THREE.MeshStandardMaterial({{
            roughness: 0.9,
            metalness: 0.05,
            side: THREE.DoubleSide,
            shadowSide: THREE.DoubleSide
        }});

        mat.onBeforeCompile = function (shader) {{
            shader.uniforms.uTime = {{ value: 0 }};
            shader.uniforms.uWindSpeed = {{ value: 8.0 }};
            shader.uniforms.uWindDir = {{ value: 45.0 }};

            mat.userData.shader = shader;

            shader.vertexShader = `
                uniform float uTime;
                uniform float uWindSpeed;
                uniform float uWindDir;
            ` + shader.vertexShader;

            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                // Sway displacement calculations
                // position.z is local plant height (0 at root, 1.4 at tip)
                float swayFactor = position.z * position.z * 0.15;
                float windAng = uWindDir * 3.14159 / 180.0;

                // Wave propagation formula based on world coords
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                float wave = sin(uTime * 3.6 + worldPos.x * 0.28 + worldPos.y * 0.28) * 0.07 * uWindSpeed;

                // Displace along the local coordinates aligned with wind direction
                transformed.x += cos(windAng) * wave * swayFactor;
                transformed.y += sin(windAng) * wave * swayFactor;
                `
            );
        }};

        return mat;
    }}

    // Query spatial indices for plant colors
    function getStressColorAt(x, y, healthyColor, pathogenColor, deficitColor) {{
        if (!texturePix) return healthyColor;

        const u = (x + sizeX / 2) / sizeX;
        const v = (y + sizeY / 2) / sizeY;

        if (u < 0 || u > 1 || v < 0 || v > 1) return healthyColor;

        const tx = Math.floor(Math.min(Math.max(u * textureW, 0), textureW - 1));
        const ty = Math.floor(Math.min(Math.max((1 - v) * textureH, 0), textureH - 1));

        const idx = (ty * textureW + tx) * 4;
        const r = texturePix[idx];
        const g = texturePix[idx+1];
        const b = texturePix[idx+2];

        if (r > 150 && g < 100) {{
            return pathogenColor; // severe pathogen
        }} else if (r > 150 && g > 150 && b < 100) {{
            return deficitColor; // moisture deficit
        }} else {{
            return healthyColor;
        }}
    }}

    // Query canopy density height from heightmap
    function getCanopyHeightValAt(x, y) {{
        if (!heightPix) return 1.0;
        const u = (x + sizeX / 2) / sizeX;
        const v = (y + sizeY / 2) / sizeY;
        if (u < 0 || u > 1 || v < 0 || v > 1) return 1.0;

        const imgX = Math.floor(Math.min(Math.max(u * heightW, 0), heightW - 1));
        const imgY = Math.floor(Math.min(Math.max((1 - v) * heightH, 0), heightH - 1));

        const idx = (imgY * heightW + imgX) * 4;
        return heightPix[idx] / 255;
    }}

    // Build procedural Instanced Canopy Mesh grid
    function buildCrops() {{
        if (instancedCrops) {{
            scene.remove(instancedCrops);
            instancedCrops.geometry.dispose();
            instancedCrops.material.dispose();
            instancedCrops = null;
        }}

        let spacing = 1.6; // medium density
        if (cropDensity === 'sparse') {{
            spacing = 2.6;
        }} else if (cropDensity === 'dense') {{
            spacing = 0.95;
        }}

        const xPoints = [];
        const yPoints = [];
        const m = 2.0; // border safety margin

        for (let x = -sizeX/2 + m; x <= sizeX/2 - m; x += spacing) {{
            xPoints.push(x);
        }}
        for (let y = -sizeY/2 + m; y <= sizeY/2 - m; y += spacing) {{
            yPoints.push(y);
        }}

        const count = xPoints.length * yPoints.length;
        if (count === 0) return;

        const geom = createCropGeometry();
        const mat = createCropMaterial();

        instancedCrops = new THREE.InstancedMesh(geom, mat, count);
        instancedCrops.castShadow = true;
        instancedCrops.receiveShadow = true;

        const dummy = new THREE.Object3D();
        let index = 0;

        const colorHealthy = new THREE.Color(0x22c55e);
        const colorPathogen = new THREE.Color(0xec4899); // pink fungicide target
        const colorDeficit = new THREE.Color(0xeab308); // yellow moisture deficit

        // Seasonal parameter adjustments
        let seasonHeightScale = 1.0;
        const seasonColorHealthy = colorHealthy.clone();

        if (activeSeason === 'spring') {{
            seasonHeightScale = 0.52; // Short early shoots
            seasonColorHealthy.setHex(0xa3e635); // Light vibrant lime-green
        }} else if (activeSeason === 'autumn') {{
            seasonHeightScale = 1.15; // Tall, mature grain heads
            seasonColorHealthy.setHex(0xfacc15); // Harvest Gold / Wheat
        }} else if (activeSeason === 'winter') {{
            seasonHeightScale = 0.22; // Low dry stubble
            seasonColorHealthy.setHex(0x78350f); // Dead brown stalks
        }}

        for (let i = 0; i < xPoints.length; i++) {{
            for (let j = 0; j < yPoints.length; j++) {{
                // Add slight random jittering to offset straight grid lines
                const px = xPoints[i] + (Math.random() - 0.5) * spacing * 0.45;
                const py = yPoints[j] + (Math.random() - 0.5) * spacing * 0.45;

                const terrainH = getTerrainHeightAt(px, py);
                const chmHeight = getCanopyHeightValAt(px, py);

                // Height and width scaling with random natural variation
                const heightScale = Math.max(0.15, chmHeight * 0.85) * seasonHeightScale;
                const widthScale = (0.8 + Math.random() * 0.4) * (activeSeason === 'winter' ? 0.35 : 1.0);

                const stressColor = getStressColorAt(px, py, seasonColorHealthy, colorPathogen, colorDeficit);

                dummy.position.set(px, py, terrainH);
                dummy.rotation.set(0, 0, Math.random() * Math.PI * 2); // Random rotation heading
                dummy.scale.set(widthScale, widthScale, heightScale);
                dummy.updateMatrix();

                instancedCrops.setMatrixAt(index, dummy.matrix);
                instancedCrops.setColorAt(index, stressColor);

                index++;
            }}
        }}

        instancedCrops.instanceMatrix.needsUpdate = true;
        if (instancedCrops.instanceColor) {{
            instancedCrops.instanceColor.needsUpdate = true;
        }}

        scene.add(instancedCrops);
    }}

    // Altimeter calculation: reads height under coordinates
    function getTerrainHeightAt(x, y) {{
        if (!heightPix) return 0.0;
        const u = (x + sizeX / 2) / sizeX;
        const v = (y + sizeY / 2) / sizeY;
        if (u < 0 || u > 1 || v < 0 || v > 1) return 0.0;

        const imgX = Math.floor(Math.min(Math.max(u * heightW, 0), heightW - 1));
        const imgY = Math.floor(Math.min(Math.max((1 - v) * heightH, 0), heightH - 1));

        const idx = (imgY * heightW + imgX) * 4;
        return (heightPix[idx] / 255) * maxHeightScale;
    }}

    // Quadcopter 3D Model construction
    let droneGroup;
    let rotors = [];
    let sprayNozzles = [];

    function buildDroneModel() {{
        droneGroup = new THREE.Group();

        // Sleek Blue Fuselage
        const bodyGeom = new THREE.BoxGeometry(1.6, 0.5, 0.8);
        const bodyMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7, metalness: 0.9, roughness: 0.1 }});
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        droneGroup.add(body);

        // Arms
        const armGeom = new THREE.CylinderGeometry(0.08, 0.08, 3.8);
        armGeom.rotateX(Math.PI / 2);
        const armMat = new THREE.MeshStandardMaterial({{ color: 0x334155, metalness: 0.85 }});
        const arm1 = new THREE.Mesh(armGeom, armMat);
        arm1.rotation.z = Math.PI / 4;
        const arm2 = new THREE.Mesh(armGeom, armMat);
        arm2.rotation.z = -Math.PI / 4;
        droneGroup.add(arm1);
        droneGroup.add(arm2);

        // Rotors
        const rotorGeom = new THREE.CylinderGeometry(0.8, 0.8, 0.03, 16);
        const rotorMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.3, side: THREE.DoubleSide }});

        const rotorOffsets = [
            [1.34, 1.34, 0.2],
            [-1.34, 1.34, 0.2],
            [1.34, -1.34, 0.2],
            [-1.34, -1.34, 0.2]
        ];

        rotorOffsets.forEach(([rx, ry, rz]) => {{
            const rMesh = new THREE.Mesh(rotorGeom, rotorMat);
            rMesh.position.set(rx, ry, rz);
            rMesh.rotation.x = Math.PI / 2;
            droneGroup.add(rMesh);
            rotors.push(rMesh);
        }});

        // Camera Gimbal
        const gimbalGeom = new THREE.SphereGeometry(0.35, 16, 16);
        const gimbalMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.4 }});
        const gimbal = new THREE.Mesh(gimbalGeom, gimbalMat);
        gimbal.position.set(0, 0, -0.45);
        droneGroup.add(gimbal);

        // Gimbal Sensor Laser Lens
        const lensGeom = new THREE.CylinderGeometry(0.12, 0.12, 0.15);
        lensGeom.rotateX(Math.PI / 2);
        const lensMat = new THREE.MeshBasicMaterial({{ color: 0x22c55e }});
        const lens = new THREE.Mesh(lensGeom, lensMat);
        lens.position.set(0, 0, -0.6);
        droneGroup.add(lens);

        // Gimbal projecting frustum
        const frustumGeom = new THREE.BufferGeometry();
        const frustumMat = new THREE.LineBasicMaterial({{ color: 0x22c55e, transparent: true, opacity: 0.35 }});
        const frustumVerts = new Float32Array([
            0, 0, -0.5,  -3.5, -2.5, -12,
            0, 0, -0.5,   3.5, -2.5, -12,
            0, 0, -0.5,   3.5,  2.5, -12,
            0, 0, -0.5,  -3.5,  2.5, -12,
            -3.5, -2.5, -12,  3.5, -2.5, -12,
             3.5, -2.5, -12,  3.5,  2.5, -12,
             3.5,  2.5, -12, -3.5,  2.5, -12,
            -3.5,  2.5, -12, -3.5, -2.5, -12
        ]);
        frustumGeom.setAttribute('position', new THREE.BufferAttribute(frustumVerts, 3));
        const frustumLines = new THREE.LineSegments(frustumGeom, frustumMat);
        droneGroup.add(frustumLines);

        // Spray bar underneath
        const sprayBarGeom = new THREE.CylinderGeometry(0.04, 0.04, 2.0);
        sprayBarGeom.rotateZ(Math.PI / 2);
        const sprayBarMat = new THREE.MeshStandardMaterial({{ color: 0x475569 }});
        const sprayBar = new THREE.Mesh(sprayBarGeom, sprayBarMat);
        sprayBar.position.set(0, 0, -0.4);
        droneGroup.add(sprayBar);

        // Two spray nozzles
        const nozzleGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.2);
        const nozzleMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8 }});

        const nozzle1 = new THREE.Mesh(nozzleGeom, nozzleMat);
        nozzle1.position.set(-0.8, 0, -0.5);
        droneGroup.add(nozzle1);
        sprayNozzles.push(nozzle1);

        const nozzle2 = new THREE.Mesh(nozzleGeom, nozzleMat);
        nozzle2.position.set(0.8, 0, -0.5);
        droneGroup.add(nozzle2);
        sprayNozzles.push(nozzle2);

        scene.add(droneGroup);

        // Set initial position
        droneGroup.position.set(0, 0, 15);
        droneVelocity.set(0, 0, 0);
        droneRotation.set(0, 0, 0);
        droneAngularVelocity.set(0, 0, 0);
    }}

    // Path Waypoint logic
    let waypoints = [];
    let currentWaypointIdx = 0;
    let targetRings = [];

    function generatePath() {{
        waypoints = [];
        currentWaypointIdx = 0;
        pathDirection = 1;

        // Safe boundary margin to keep target flight waypoints comfortably inside field terrain limits
        const b = 7.5;
        if (flightPathMode === 'grid') {{
            let rightSide = true;
            const spacing = 5.5;
            for (let y = -sizeY/2 + b; y <= sizeY/2 - b; y += spacing) {{
                if (rightSide) {{
                    waypoints.push(new THREE.Vector3(-sizeX/2 + b, y, 12));
                    waypoints.push(new THREE.Vector3(sizeX/2 - b, y, 12));
                }} else {{
                    waypoints.push(new THREE.Vector3(sizeX/2 - b, y, 12));
                    waypoints.push(new THREE.Vector3(-sizeX/2 + b, y, 12));
                }}
                rightSide = !rightSide;
            }}
            cleanupRings();
        }} else if (flightPathMode === 'orbit') {{
            const rad = Math.min(sizeX, sizeY) * 0.31;
            const pts = 36;
            for (let i = 0; i <= pts; i++) {{
                const ang = (i / pts) * Math.PI * 2;
                waypoints.push(new THREE.Vector3(Math.cos(ang) * rad, Math.sin(ang) * rad, 13));
            }}
            cleanupRings();
        }} else {{
            // Smart Stress Spot Visit locations (comfortably inside mesh boundaries)
            waypoints.push(new THREE.Vector3(-11, -8, 12));
            waypoints.push(new THREE.Vector3(8, 11, 12));
            waypoints.push(new THREE.Vector3(14, -14, 12));
            waypoints.push(new THREE.Vector3(-6, 14, 12));
            buildTargetRings();
        }}
        if (droneGroup && waypoints.length > 0) {{
            droneGroup.position.copy(waypoints[0]);
            droneVelocity.set(0, 0, 0);
            droneRotation.set(0, 0, Math.atan2(waypoints[0].y, waypoints[0].x));
            droneAngularVelocity.set(0, 0, 0);
        }}
    }}

    function buildTargetRings() {{
        cleanupRings();
        const ringGeom = new THREE.RingGeometry(1.2, 1.4, 32);
        ringGeom.rotateX(-Math.PI / 2);

        const spotCoords = [
            [-11, -8],
            [8, 11],
            [14, -14],
            [-6, 14]
        ];

        spotCoords.forEach(([rx, ry], idx) => {{
            const h = getTerrainHeightAt(rx, ry) + 0.15;
            const ringMat = new THREE.MeshBasicMaterial({{
                color: idx % 2 === 0 ? 0xef4444 : 0xeab308,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.8
            }});
            const rMesh = new THREE.Mesh(ringGeom, ringMat);
            rMesh.position.set(rx, ry, h);
            scene.add(rMesh);
            targetRings.push(rMesh);
        }});
    }}

    function cleanupRings() {{
        targetRings.forEach(r => scene.remove(r));
        targetRings = [];
    }}

    let homeLat = null;
    let homeLon = null;

    function createDroneModel(colorHex) {{
        const group = new THREE.Group();

        // Fuselage with custom colorHex
        const bodyGeom = new THREE.BoxGeometry(1.6, 0.5, 0.8);
        const bodyMat = new THREE.MeshStandardMaterial({{ color: colorHex, metalness: 0.9, roughness: 0.1 }});
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        group.add(body);

        // Arms
        const armGeom = new THREE.CylinderGeometry(0.08, 0.08, 3.8);
        armGeom.rotateX(Math.PI / 2);
        const armMat = new THREE.MeshStandardMaterial({{ color: 0x334155, metalness: 0.85 }});
        const arm1 = new THREE.Mesh(armGeom, armMat);
        arm1.rotation.z = Math.PI / 4;
        const arm2 = new THREE.Mesh(armGeom, armMat);
        arm2.rotation.z = -Math.PI / 4;
        group.add(arm1);
        group.add(arm2);

        // Rotors
        const rotorGeom = new THREE.CylinderGeometry(0.8, 0.8, 0.03, 16);
        const rotorMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.3, side: THREE.DoubleSide }});

        const rotorOffsets = [
            [1.34, 1.34, 0.2],
            [-1.34, 1.34, 0.2],
            [1.34, -1.34, 0.2],
            [-1.34, -1.34, 0.2]
        ];

        const droneRotors = [];
        rotorOffsets.forEach(([rx, ry, rz]) => {{
            const rMesh = new THREE.Mesh(rotorGeom, rotorMat);
            rMesh.position.set(rx, ry, rz);
            rMesh.rotation.x = Math.PI / 2;
            group.add(rMesh);
            droneRotors.push(rMesh);
        }});

        // Camera Gimbal
        const gimbalGeom = new THREE.SphereGeometry(0.35, 16, 16);
        const gimbalMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.4 }});
        const gimbal = new THREE.Mesh(gimbalGeom, gimbalMat);
        gimbal.position.set(0, 0, -0.45);
        group.add(gimbal);

        // Gimbal Sensor Laser Lens
        const lensGeom = new THREE.CylinderGeometry(0.12, 0.12, 0.15);
        lensGeom.rotateX(Math.PI / 2);
        const lensMat = new THREE.MeshBasicMaterial({{ color: 0x22c55e }});
        const lens = new THREE.Mesh(lensGeom, lensMat);
        lens.position.set(0, 0.2, -0.5);
        group.add(lens);

        // Left/Right Spray Nozzles
        const nozzleGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.6);
        const nozzleMat = new THREE.MeshStandardMaterial({{ color: 0x475569, metalness: 0.5 }});

        const nMesh1 = new THREE.Mesh(nozzleGeom, nozzleMat);
        nMesh1.position.set(-1.0, 0, -0.3);
        nMesh1.rotation.x = Math.PI / 2;
        group.add(nMesh1);

        const nMesh2 = new THREE.Mesh(nozzleGeom, nozzleMat);
        nMesh2.position.set(1.0, 0, -0.3);
        nMesh2.rotation.x = Math.PI / 2;
        group.add(nMesh2);

        return {{
            group: group,
            rotors: droneRotors,
            sprayNozzles: [nMesh1, nMesh2]
        }};
    }}

    function onCanvasClick(event) {{
        if (!isPinPlacementMode) return;

        const rect = renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);

        if (!meshMesh) return;
        const intersects = raycaster.intersectObject(meshMesh);

        if (intersects.length > 0) {{
            const p = intersects[0].point;
            const label = prompt("Enter Annotation Label for this 3D location:");
            if (label && label.trim() !== "") {{
                const annId = "pin_" + Date.now() + "_" + Math.floor(Math.random()*1000);
                ws.send(JSON.stringify({{
                    type: "add_annotation",
                    annotation: {{
                        id: annId,
                        x: p.x,
                        y: p.y,
                        z: p.z,
                        label: label.trim(),
                        creator_role: myRole.toUpperCase().replace('_', ' ')
                    }}
                }}));
            }}

            isPinPlacementMode = false;
            document.getElementById('pin-placement-indicator').style.display = 'none';
            document.getElementById('btn-add-pin').classList.remove('active');
        }}
    }}

    function updateDroneFromTelemetry(data, dt) {{
        if (!droneGroup) return;

        // 1. Connection status in HUD
        const statusLabel = document.getElementById('hud-status');
        if (data.connected) {{
            statusLabel.textContent = "PX4/MAVLINK";
            statusLabel.style.color = "#38bdf8";
        }} else {{
            statusLabel.textContent = "MOCK SITL";
            statusLabel.style.color = "#eab308";
        }}

        // 2. Lock Home Anchor GPS on first packet
        if (homeLat === null || homeLon === null) {{
            homeLat = data.lat;
            homeLon = data.lon;
        }}

        // 3. Flat-Earth GPS-to-Local projection
        const EARTH_RADIUS = 6378137.0;
        const latRad = data.lat * Math.PI / 180;
        const lonRad = data.lon * Math.PI / 180;
        const homeLatRad = homeLat * Math.PI / 180;
        const homeLonRad = homeLon * Math.PI / 180;

        const dy = (latRad - homeLatRad) * EARTH_RADIUS;
        const dx = (lonRad - homeLonRad) * EARTH_RADIUS * Math.cos(homeLatRad);

        // 4. Update Drone Position
        const limitX = sizeX / 2.0 - 1.5;
        const limitY = sizeY / 2.0 - 1.5;

        droneGroup.position.x = Math.max(-limitX, Math.min(limitX, dx));
        droneGroup.position.y = Math.max(-limitY, Math.min(limitY, dy));
        droneGroup.position.z = data.alt;

        // 5. Update Drone Rotations
        droneRotation.x = data.roll;
        droneRotation.y = data.pitch;
        droneRotation.z = data.yaw;

        droneGroup.rotation.set(droneRotation.x, droneRotation.y, droneRotation.z, 'ZYX');
        droneGroup.updateMatrix();

        // 6. Update HUD Stats
        document.getElementById('hud-lat').textContent = `${{data.lat.toFixed(6)}}°`;
        document.getElementById('hud-lon').textContent = `${{data.lon.toFixed(6)}}°`;
        document.getElementById('hud-alt').textContent = `${{data.alt.toFixed(2)}} m`;

        const terrainH = getTerrainHeightAt(droneGroup.position.x, droneGroup.position.y);
        const aglAlt = droneGroup.position.z - terrainH;
        document.getElementById('hud-agl').textContent = `${{aglAlt.toFixed(2)}} m`;

        const pDeg = Math.round(data.pitch * (180/Math.PI));
        const rDeg = Math.round(data.roll * (180/Math.PI));
        document.getElementById('hud-pitch').textContent = `${{pDeg}}° / ${{rDeg}}°`;
        document.getElementById('hud-speed').textContent = `${{data.speed.toFixed(1)}} m/s`;

        const battHUD = document.getElementById('hud-battery');
        battHUD.textContent = `${{data.battery.toFixed(1)}}%`;
        if (data.battery > 50) battHUD.style.color = "#22c55e";
        else if (data.battery > 20) battHUD.style.color = "#eab308";
        else battHUD.style.color = "#ef4444";

        // 7. Update RPM mesh animations
        rotors[0].rotation.y += 18.0 * dt;
        rotors[1].rotation.y -= 18.0 * dt;
        rotors[2].rotation.y -= 18.0 * dt;
        rotors[3].rotation.y += 18.0 * dt;
        document.getElementById('hud-rpm').textContent = "AUTO (MAVLink)";

        // 8. Spray trigger syncing
        isSpraying = data.is_spraying;
        updateSensorLookup(droneGroup.position.x, droneGroup.position.y);
        triggerSprayParticles();

        // Simulate payload draining when spraying
        if (isCurrentlyEmitting && remainingPayload > 0) {{
            remainingPayload = Math.max(0, remainingPayload - 2.5 * dt);
            document.getElementById('hud-payload').textContent = `${{remainingPayload.toFixed(1)}}%`;
            document.getElementById('payload-fill').style.width = `${{remainingPayload}}%`;
            if (remainingPayload === 0) {{
                document.getElementById('btn-spray').textContent = "Spray Payload Empty";
                document.getElementById('btn-spray').classList.remove('active');
            }}
        }}
    }}

    function resetSimulation() {{
        generatePath();
        homeLat = null;
        homeLon = null;
        remainingPayload = 100.0;
        energyConsumedWh = 0.0;
        totalDistanceTraveledM = 0.0;
        pathDirection = 1;
        droneVelocity.set(0, 0, 0);
        droneRotation.set(0, 0, 0);
        droneAngularVelocity.set(0, 0, 0);
        document.getElementById('hud-battery').textContent = "100.0%";
        document.getElementById('hud-battery').style.color = "#22c55e";
        // Reset terrain map back to original pixels
        drawCtx.drawImage(textureImg, 0, 0);
        canvasTexture.needsUpdate = true;
        buildCrops(); // Rebuild instances
    }}

    // Particle physics engine list
    let particles = [];
    const particleGeom = new THREE.SphereGeometry(0.15, 8, 8);

    function triggerSprayParticles() {{
        if (!droneGroup) return;

        const sensorText = document.getElementById('sensor-text').textContent;
        let particleColor = 0x38bdf8;
        let treatment = "Broad-Spectrum Blanket Spray";
        let shouldEmit = isSpraying; // default to current spray toggle status

        let selectedChemical = activeChemical;

        if (selectedChemical === 'dynamic') {{
            if (sensorText === "SEVERE PATHOGEN STRESS") {{
                selectedChemical = 'fungicide';
                shouldEmit = true; // force spot spray on stress even if base spray is toggled off
            }} else if (sensorText === "NITROGEN / WATER DEFICIT") {{
                selectedChemical = 'nutrient';
                shouldEmit = true; // force spot spray on stress even if base spray is toggled off
            }} else {{
                selectedChemical = 'blanket';
                shouldEmit = isSpraying; // only blanket spray if Spray toggle is ON
            }}
        }}

        if (selectedChemical === 'fungicide') {{
            particleColor = 0xec4899; // pink
            treatment = "Spot Fungicide (Pathogen)";
        }} else if (selectedChemical === 'nutrient') {{
            particleColor = 0xeab308; // yellow
            treatment = "Spot Liquid Nitrogen (Nutrient)";
        }} else {{
            particleColor = 0x38bdf8; // blue
            treatment = "Broad-Spectrum Blanket Spray";
        }}

        // If payload is empty, shut down spray
        if (remainingPayload <= 0) {{
            shouldEmit = false;
        }}

        // Update HUD labels
        const treatVal = document.getElementById('hud-treatment');
        isCurrentlyEmitting = shouldEmit;

        if (shouldEmit) {{
            treatVal.textContent = treatment;
            if (selectedChemical === 'fungicide') treatVal.style.color = "#ec4899";
            else if (selectedChemical === 'nutrient') treatVal.style.color = "#eab308";
            else treatVal.style.color = "#38bdf8";
        }} else {{
            treatVal.textContent = "None (Spray Off)";
            treatVal.style.color = "#94a3b8";
        }}

        if (!shouldEmit || remainingPayload <= 0) return;

        // Emit particles from both nozzles
        sprayNozzles.forEach((nozzle, nIdx) => {{
            const nozzleWorldPos = new THREE.Vector3();
            nozzle.getWorldPosition(nozzleWorldPos);

            // Initial velocity relative to drone, plus wingtip vortex interaction
            const initialVel = new THREE.Vector3(droneVelocity.x, droneVelocity.y, droneVelocity.z - 3.5);

            // Add tip vortex vector perpendicular to drone body longitudinal axis
            const swirlSpeed = 2.8;
            const swirlVec = new THREE.Vector3(0, swirlSpeed, 0);
            swirlVec.applyAxisAngle(new THREE.Vector3(0, 0, 1), droneRotation.z);

            if (nIdx === 0) {{
                initialVel.addScaledVector(swirlVec, 1);
            }} else {{
                initialVel.addScaledVector(swirlVec, -1);
            }}

            const pMat = new THREE.MeshBasicMaterial({{ color: particleColor, transparent: true, opacity: 0.8 }});
            const p = new THREE.Mesh(particleGeom, pMat);
            p.position.copy(nozzleWorldPos);
            scene.add(p);

            particles.push({{
                mesh: p,
                vel: initialVel,
                age: 0
            }});
        }});
    }}

    // Dynamic Sensor Lookup logic
    function updateSensorLookup(x, y) {{
        if (!texturePix) return;

        const u = (x + sizeX / 2) / sizeX;
        const v = (y + sizeY / 2) / sizeY;

        if (u < 0 || u > 1 || v < 0 || v > 1) return;

        const tx = Math.floor(Math.min(Math.max(u * textureW, 0), textureW - 1));
        const ty = Math.floor(Math.min(Math.max((1 - v) * textureH, 0), textureH - 1));

        const idx = (ty * textureW + tx) * 4;
        const r = texturePix[idx];
        const g = texturePix[idx+1];
        const b = texturePix[idx+2];

        const dot = document.getElementById('sensor-dot');
        const label = document.getElementById('sensor-text');

        if (r > 150 && g < 100) {{
            dot.style.background = "#ef4444";
            label.textContent = "SEVERE PATHOGEN STRESS";
            label.style.color = "#ef4444";
        }} else if (r > 150 && g > 150 && b < 100) {{
            dot.style.background = "#eab308";
            label.textContent = "NITROGEN / WATER DEFICIT";
            label.style.color = "#eab308";
        }} else {{
            dot.style.background = "#22c55e";
            label.textContent = "HEALTHY TARGET ZONE";
            label.style.color = "#22c55e";
        }}
    }}

    window.addEventListener('resize', () => {{
        const w = window.innerWidth - 270;
        const h = window.innerHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }});

    // Frame Delta timer
    const clock = new THREE.Clock();

    function tick() {{
        requestAnimationFrame(tick);

        const dt = Math.min(clock.getDelta(), 0.05); // cap frame step to prevent extreme delta jumps

        // Spin target rings if present
        targetRings.forEach(r => {{
            r.rotation.z += 0.02;
            const scale = 1.0 + Math.sin(Date.now() * 0.005) * 0.15;
            r.scale.set(scale, scale, 1);
        }});

        // Update real-time CFD Airflow vector streamlines in WebGL
        if (cfdFlowLines && cfdFlowLines.visible) {{
            const positions = cfdFlowLines.geometry.attributes.position.array;

            cfdParticles.forEach((p, idx) => {{
                const localWindVector = getAirflowAt(p.pos);

                // Point A (Line start)
                positions[idx * 6] = p.pos.x;
                positions[idx * 6 + 1] = p.pos.y;
                positions[idx * 6 + 2] = p.pos.z;

                // Point B (Line end)
                positions[idx * 6 + 3] = p.pos.x + localWindVector.x * 0.18;
                positions[idx * 6 + 4] = p.pos.y + localWindVector.y * 0.18;
                positions[idx * 6 + 5] = p.pos.z + localWindVector.z * 0.18;

                // Move indicator along local wind field
                p.pos.addScaledVector(localWindVector, dt);
                p.age += dt;

                const hLocal = getTerrainHeightAt(p.pos.x, p.pos.y);

                // Respawn airflow particle if dead, goes too low, or exits bounding box
                if (p.age > 4.5 || p.pos.z < hLocal || Math.abs(p.pos.x) > sizeX/2 || Math.abs(p.pos.y) > sizeY/2 || p.pos.z > 30.0) {{
                    p.pos.copy(getRandomAirflowSpawnPos());
                    p.age = 0;
                }}
            }});

            cfdFlowLines.geometry.attributes.position.needsUpdate = true;
        }}

        // Update GLSL Shader uniforms for GPU wind-blown crop sway
        if (instancedCrops && instancedCrops.material.userData.shader) {{
            const shader = instancedCrops.material.userData.shader;
            shader.uniforms.uTime.value = Date.now() * 0.0012;
            shader.uniforms.uWindSpeed.value = windSpeed;
            shader.uniforms.uWindDir.value = windDir;
        }}

        if (isPlaying && droneGroup && waypoints.length > 0) {{
            if (myDroneId === null) {{
                // View-Only: follow the active drone telemetry
                if (wsConnected && latestTelemetryData) {{
                    updateDroneFromTelemetry(latestTelemetryData, dt);
                }} else {{
                    fetch('http://127.0.0.1:{TELEMETRY_PORT}/telemetry')
                        .then(response => response.json())
                        .then(data => {{
                            updateDroneFromTelemetry(data, dt);
                        }})
                        .catch(err => console.log("Telemetry fetch error:", err));
                }}
            }} else if (flightPathMode === 'mavlink') {{
                // Operator in manual flight mode: follow SITL telemetry
                if (wsConnected && latestTelemetryData) {{
                    updateDroneFromTelemetry(latestTelemetryData, dt);
                }} else {{
                    fetch('http://127.0.0.1:{TELEMETRY_PORT}/telemetry')
                        .then(response => response.json())
                        .then(data => {{
                            updateDroneFromTelemetry(data, dt);
                        }})
                        .catch(err => console.log("Telemetry fetch error:", err));
                }}

                // Stream state to other clients as my drone
                if (wsConnected && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{
                        type: "telemetry_update",
                        drone_id: myDroneId,
                        lat: 37.7749 + (droneGroup.position.y / 111320.0),
                        lon: -122.4194 + (droneGroup.position.x / (111320.0 * Math.cos(37.7749 * Math.PI / 180.0))),
                        alt: droneGroup.position.z,
                        pitch: droneRotation.y,
                        roll: droneRotation.x,
                        yaw: droneRotation.z,
                        battery: batteryPercent,
                        speed: droneVelocity.length(),
                        is_spraying: (isCurrentlyEmitting && remainingPayload > 0)
                    }}));
                }}
            }} else {{
                // Operator in Autopilot (Waypoint / Coverage) mode: run physics engine locally
                // --- UAV FLIGHT AERODYNAMICS & PHYSICS INTEGRATOR ---
                const currentPos = droneGroup.position;
                const targetPos = waypoints[currentWaypointIdx];

            // 1. Mass Calculation (Scales dynamically with chemical payload mass)
            const payloadWeight = mass_payload_max * (remainingPayload / 100.0);
            const currentMass = mass_base + payloadWeight;

            // 2. Autopilot / Stabilization Logic (Altitude & Position PID Controls)

            // Heading Target Calculation (Yaw face target, safe threshold when close)
            const dirToTarget = new THREE.Vector3().subVectors(targetPos, currentPos);
            const dist2D = Math.sqrt(dirToTarget.x*dirToTarget.x + dirToTarget.y*dirToTarget.y);

            let targetYaw = droneRotation.z;
            if (dist2D > 0.6) {{
                targetYaw = Math.atan2(dirToTarget.y, dirToTarget.x);
            }}

            // Altitude Control PID loop
            const altError = targetPos.z - currentPos.z;
            const kp_alt = 2.5;
            const kd_alt = 1.8;
            const targetAccZ = kp_alt * altError - kd_alt * droneVelocity.z;

            // Collective Thrust Force (Clamped between safety factors of gravity compensation)
            let thrustMag = currentMass * (9.81 + targetAccZ);
            thrustMag = Math.max(0.2 * currentMass * 9.81, Math.min(2.2 * currentMass * 9.81, thrustMag));

            // Horizontal Navigation PID loop
            const maxCruiseSpeed = 8.5; // m/s
            const kp_pos = 1.2;

            // Target velocity is proportional to distance from waypoint
            const targetVelocity2D = new THREE.Vector3(dirToTarget.x, dirToTarget.y, 0).normalize().multiplyScalar(
                Math.min(maxCruiseSpeed, dist2D * kp_pos)
            );

            // PING-PONG ROUTE TRAVERSAL TO PREVENT LONG CROSSINGS
            if (dist2D < 0.6) {{
                currentWaypointIdx += pathDirection;
                if (currentWaypointIdx >= waypoints.length) {{
                    currentWaypointIdx = Math.max(0, waypoints.length - 2);
                    pathDirection = -1;
                }} else if (currentWaypointIdx < 0) {{
                    currentWaypointIdx = Math.min(waypoints.length - 1, 1);
                    pathDirection = 1;
                }}
            }}

            // Calculate required horizontal acceleration
            const velError = new THREE.Vector3(targetVelocity2D.x - droneVelocity.x, targetVelocity2D.y - droneVelocity.y, 0);
            const kp_vel = 1.4;
            const desiredAccX = velError.x * kp_vel;
            const desiredAccY = velError.y * kp_vel;

            // Ambient Wind sampling for feed-forward compensation
            const ambientWind = new THREE.Vector3(
                Math.cos((windDir*Math.PI)/180) * windSpeed,
                Math.sin((windDir*Math.PI)/180) * windSpeed,
                0
            );
            // Introduce wind gusts
            ambientWind.x += Math.sin(currentPos.x * 0.15 + Date.now()*0.003) * windSpeed * 0.1;
            ambientWind.y += Math.cos(currentPos.y * 0.15 + Date.now()*0.003) * windSpeed * 0.1;

            // AERODYNAMIC DRAG AND WIND DISTURBANCE FEED-FORWARD COMPENSATION:
            const droneDragCoeff = 0.38;
            const accDemandX = desiredAccX + (droneVelocity.x - ambientWind.x) * (droneDragCoeff / currentMass);
            const accDemandY = desiredAccY + (droneVelocity.y - ambientWind.y) * (droneDragCoeff / currentMass);

            // Convert world acceleration demands into local pitch/roll banking angles
            const cosYaw = Math.cos(droneRotation.z);
            const sinYaw = Math.sin(droneRotation.z);
            const accLocalX = accDemandX * cosYaw + accDemandY * sinYaw;
            const accLocalY = -accDemandX * sinYaw + accDemandY * cosYaw;

            // Limit maximum roll/pitch angle to 26 degrees (0.45 rad)
            const maxTilt = 0.45;

            // CONTROL SIGN MAPPING:
            // Positive local X acceleration needs positive pitch (forward tilt).
            // Positive local Y acceleration needs negative roll (right tilt).
            const targetPitch = Math.max(-maxTilt, Math.min(maxTilt, accLocalX / 9.81));
            const targetRoll = Math.max(-maxTilt, Math.min(maxTilt, -accLocalY / 9.81));

            // Attitude Stabilization PD loop
            const kp_att = 8.5;
            const kd_att = 4.2;

            const rollError = targetRoll - droneRotation.x;
            const pitchError = targetPitch - droneRotation.y;

            let yawError = targetYaw - droneRotation.z;
            while (yawError < -Math.PI) yawError += Math.PI * 2;
            while (yawError > Math.PI) yawError -= Math.PI * 2;

            // Angular Acceleration outputs
            const rollAcc = rollError * kp_att - droneAngularVelocity.x * kd_att;
            const pitchAcc = pitchError * kp_att - droneAngularVelocity.y * kd_att;
            const yawAcc = yawError * kp_att - droneAngularVelocity.z * kd_att;

            // Integrate angular state
            droneAngularVelocity.x += rollAcc * dt;
            droneAngularVelocity.y += pitchAcc * dt;
            droneAngularVelocity.z += yawAcc * dt;

            droneRotation.x += droneAngularVelocity.x * dt;
            droneRotation.y += droneAngularVelocity.y * dt;
            droneRotation.z += droneAngularVelocity.z * dt;

            // Update 3D drone body rotation (Pitch/Roll/Yaw)
            droneGroup.rotation.set(droneRotation.x, droneRotation.y, droneRotation.z, 'ZYX');
            droneGroup.updateMatrix();

            // 3. Force Integrations (Gravity, Drag, Wind Disturbance, Thrust)
            const F_gravity = new THREE.Vector3(0, 0, -currentMass * 9.81);

            // Thrust Vector direction points along local Z vector of drone body
            const thrustDir = new THREE.Vector3(0, 0, 1).applyQuaternion(droneGroup.quaternion);
            const F_thrust = new THREE.Vector3().copy(thrustDir).multiplyScalar(thrustMag);

            // Wind aerodynamic drag force on UAV structure
            const relWindVel = new THREE.Vector3().subVectors(droneVelocity, ambientWind);
            const F_drag = new THREE.Vector3().copy(relWindVel).multiplyScalar(-droneDragCoeff);

            // Accumulate net forces
            const F_net = new THREE.Vector3().addVectors(F_gravity, F_thrust).add(F_drag);

            // Integrate linear state (Acceleration -> Velocity -> Position)
            const droneAcc = new THREE.Vector3().copy(F_net).multiplyScalar(1.0 / currentMass);
            droneVelocity.addScaledVector(droneAcc, dt);
            currentPos.addScaledVector(droneVelocity, dt);

            // GEOFENCE HARD BOUNDARY CLAMP TO PREVENT EXITING THE FIELD
            const limitX = sizeX / 2.0 - 1.5;
            const limitY = sizeY / 2.0 - 1.5;
            if (currentPos.x < -limitX) {{ currentPos.x = -limitX; droneVelocity.x = 0.0; }}
            if (currentPos.x > limitX)  {{ currentPos.x = limitX;  droneVelocity.x = 0.0; }}
            if (currentPos.y < -limitY) {{ currentPos.y = -limitY; droneVelocity.y = 0.0; }}
            if (currentPos.y > limitY)  {{ currentPos.y = limitY;  droneVelocity.y = 0.0; }}

            // Update Distance Traveled
            totalDistanceTraveledM += droneVelocity.length() * dt;

            // 4. Rotor RPM Simulation
            // Base RPM maps to thrust demand, plus differential roll/pitch/yaw control torque outputs
            const baseRPM = 4400.0 * Math.sqrt(thrustMag / (currentMass * 9.81));
            const pitchDiff = pitchError * 850.0 - droneAngularVelocity.y * 180.0;
            const rollDiff = rollError * 850.0 - droneAngularVelocity.x * 180.0;
            const yawDiff = yawError * 600.0 - droneAngularVelocity.z * 120.0;

            const rpm0 = Math.max(800.0, Math.min(8500.0, baseRPM - pitchDiff + rollDiff - yawDiff));
            const rpm1 = Math.max(800.0, Math.min(8500.0, baseRPM - pitchDiff - rollDiff + yawDiff));
            const rpm2 = Math.max(800.0, Math.min(8500.0, baseRPM + pitchDiff + rollDiff + yawDiff));
            const rpm3 = Math.max(800.0, Math.min(8500.0, baseRPM + pitchDiff - rollDiff - yawDiff));

            // Animate rotor mesh spin rate proportional to individual simulated rotor RPMs
            rotors[0].rotation.y += rpm0 * 0.004 * dt;
            rotors[1].rotation.y += -rpm1 * 0.004 * dt;
            rotors[2].rotation.y += -rpm2 * 0.004 * dt;
            rotors[3].rotation.y += rpm3 * 0.004 * dt;

            // Show rotor RPMs in HUD (averaged front/rear for readability)
            const frontRPM = Math.round((rpm0 + rpm1) / 2);
            const rearRPM = Math.round((rpm2 + rpm3) / 2);
            document.getElementById('hud-rpm').textContent = `${{frontRPM}} / ${{rearRPM}} RPM`;

            // 5. Battery and Energy efficiency model
            // Power consumption: P_total = P_avionics + P_thrust + P_pump
            const P_avionics = 45.0; // W
            const P_spray = isCurrentlyEmitting ? 85.0 : 0.0; // W (liquid pump power)
            // Thrust power draws exponentially based on force magnitude required (heavier payload increases power draw!)
            const P_thrust = 2.35 * Math.pow(thrustMag, 1.06);

            const totalPower = P_avionics + P_spray + P_thrust;

            // Drain battery Wh
            energyConsumedWh += (totalPower * dt) / 3600.0;
            const batteryPercent = Math.max(0.0, 100.0 - (energyConsumedWh / batteryCapacityWh) * 100.0);

            // Calculate energy efficiency metric (Wh/km)
            const efficiencyWhKm = totalDistanceTraveledM > 10.0 ? (energyConsumedWh / (totalDistanceTraveledM / 1000.0)) : 0.0;

            // Update Battery HUD elements
            const battHUD = document.getElementById('hud-battery');
            battHUD.textContent = `${{batteryPercent.toFixed(1)}}% (${{Math.max(0, Math.round(batteryCapacityWh - energyConsumedWh))}} Wh)`;
            if (batteryPercent > 50) battHUD.style.color = "#22c55e";
            else if (batteryPercent > 20) battHUD.style.color = "#eab308";
            else battHUD.style.color = "#ef4444";

            document.getElementById('hud-power').textContent = `${{Math.round(totalPower)}} W`;
            document.getElementById('hud-efficiency').textContent = `${{efficiencyWhKm.toFixed(1)}} Wh/km`;
            document.getElementById('hud-mass').textContent = `${{currentMass.toFixed(2)}} kg`;
            document.getElementById('hud-speed').textContent = `${{droneVelocity.length().toFixed(1)}} m/s`;

            // Trigger sensor reading & chemical sprays
            updateSensorLookup(currentPos.x, currentPos.y);
            triggerSprayParticles();

            if (isCurrentlyEmitting && remainingPayload > 0) {{
                remainingPayload = Math.max(0, remainingPayload - 2.5 * dt);
                document.getElementById('hud-payload').textContent = `${{remainingPayload.toFixed(1)}}%`;
                document.getElementById('payload-fill').style.width = `${{remainingPayload}}%`;
                if (remainingPayload === 0) {{
                    document.getElementById('btn-spray').textContent = "Spray Payload Empty";
                    document.getElementById('btn-spray').classList.remove('active');
                }}
            }}

            // Update HUD Telemetry text
            document.getElementById('hud-lat').textContent = `${{currentPos.y.toFixed(2)}} m`;
            document.getElementById('hud-lon').textContent = `${{currentPos.x.toFixed(2)}} m`;
            document.getElementById('hud-alt').textContent = `${{currentPos.z.toFixed(2)}} m`;

            const terrainH = getTerrainHeightAt(currentPos.x, currentPos.y);
            const aglAlt = currentPos.z - terrainH;
            document.getElementById('hud-agl').textContent = `${{aglAlt.toFixed(2)}} m`;

            const pDeg = Math.round(droneRotation.y * (180/Math.PI));
            const rDeg = Math.round(droneRotation.x * (180/Math.PI));
            document.getElementById('hud-pitch').textContent = `${{pDeg}}° / ${{rDeg}}°`;

            // Stream local simulator state back to Python over WebSocket
            if (wsConnected && ws.readyState === WebSocket.OPEN) {{
                const yawVal = droneRotation.z;
                const pitchVal = droneRotation.y;
                const rollVal = droneRotation.x;
                const speedVal = droneVelocity.length();

                ws.send(JSON.stringify({{
                    type: "telemetry_update",
                    drone_id: myDroneId,
                    lat: 37.7749 + (currentPos.y / 111320.0),
                    lon: -122.4194 + (currentPos.x / (111320.0 * Math.cos(37.7749 * Math.PI / 180.0))),
                    alt: currentPos.z,
                    pitch: pitchVal,
                    roll: rollVal,
                    yaw: yawVal,
                    battery: batteryPercent,
                    speed: speedVal,
                    is_spraying: (isCurrentlyEmitting && remainingPayload > 0)
                }}));
            }}
            }}
        }} else {{
            document.getElementById('hud-treatment').textContent = "None (Paused)";
            document.getElementById('hud-treatment').style.color = "#94a3b8";
        }}

        // Update Spray Particle physics loop with CFD Airflow forces
        const gravity = -9.8;
        const airDragCoeff = 1.8; // drag coefficient representing air resistance force

        for (let i = particles.length - 1; i >= 0; i--) {{
            const p = particles[i];
            p.age += dt;

            // Query local Computational Fluid Dynamics (CFD) airflow vector
            const localFlow = getAirflowAt(p.mesh.position);

            // Apply aerodynamic forces (Inertia + Wind Drag + Gravity)
            p.vel.x += (localFlow.x - p.vel.x) * airDragCoeff * dt;
            p.vel.y += (localFlow.y - p.vel.y) * airDragCoeff * dt;
            p.vel.z += (localFlow.z - p.vel.z) * airDragCoeff * dt + gravity * dt;

            // Integrate position
            p.mesh.position.addScaledVector(p.vel, dt);

            const pH = getTerrainHeightAt(p.mesh.position.x, p.mesh.position.y);

            // Collision Check with the 3D surface mesh
            if (p.mesh.position.z <= pH || p.age > 4.5) {{
                // Paint wet footprint on the dynamic texture canvas
                if (p.mesh.position.z <= pH && drawCtx) {{
                    const u = (p.mesh.position.x + sizeX / 2) / sizeX;
                    const v = (p.mesh.position.y + sizeY / 2) / sizeY;

                    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {{
                        const canvasX = u * textureW;
                        const canvasY = (1 - v) * textureH;

                        // Dynamic paint color footprint on collision
                        let paintColor = "rgba(14, 165, 233, 0.28)";
                        const colHex = p.mesh.material.color.getHex();
                        if (colHex === 0xec4899) {{
                            paintColor = "rgba(236, 72, 153, 0.35)"; // pink fungicide stain
                        }} else if (colHex === 0xeab308) {{
                            paintColor = "rgba(234, 179, 8, 0.35)"; // yellow nutrient stain
                        }}

                        drawCtx.fillStyle = paintColor;
                        drawCtx.beginPath();
                        drawCtx.arc(canvasX, canvasY, textureW * 0.02, 0, Math.PI * 2);
                        drawCtx.fill();
                        canvasTexture.needsUpdate = true;
                    }}
                }}

                // Remove particle from world
                scene.remove(p.mesh);
                particles.splice(i, 1);
            }}
        }}

        // Camera Management Views
        if (droneGroup) {{
            if (cameraMode === 'fpv') {{
                // FPV camera follows drone body with full pitch/roll/yaw rotations
                camera.position.set(droneGroup.position.x, droneGroup.position.y, droneGroup.position.z - 0.6);
                camera.rotation.set(-Math.PI / 2, 0, droneRotation.z);
            }} else if (cameraMode === 'follow') {{
                const offset = new THREE.Vector3(0, -9, 4.5);
                offset.applyAxisAngle(new THREE.Vector3(0, 0, 1), droneRotation.z);
                camera.position.copy(droneGroup.position).add(offset);
                camera.lookAt(droneGroup.position.x, droneGroup.position.y, droneGroup.position.z + 0.5);
                camera.up.set(0, 0, 1);
            }} else {{
                controls.update();
            }}
        }}

        renderer.render(scene, camera);
    }}
    tick();
</script>
</body>
</html>
"""
        import streamlit.components.v1 as components
        components.html(three_html, height=800, scrolling=False)

        # --- Live Mission Control & Replay Dashboard HUD ---
        st.markdown("---")
        st.subheader("Live Mission Sync & Replay Controller")

        # Setup directories for mission files
        import os
        import glob
        import datetime
        os.makedirs("outputs/missions", exist_ok=True)

        @st.fragment(run_every=1.0)
        def render_telemetry_hud():
            # Sync Streamlit session state widgets with persistent CURRENT_ENV
            if "LAST_SEEN_ENV" not in st.session_state:
                st.session_state["LAST_SEEN_ENV"] = {
                    "wind_speed": CURRENT_ENV["wind_speed"],
                    "wind_direction": CURRENT_ENV["wind_direction"],
                    "season": CURRENT_ENV["season"],
                    "active_agent": CURRENT_ENV["active_agent"]
                }

            # Make sure the session state widget keys are initialized to avoid warnings/exceptions
            if "ws_env_wind_speed" not in st.session_state:
                st.session_state["ws_env_wind_speed"] = float(CURRENT_ENV["wind_speed"])
            if "ws_env_wind_dir" not in st.session_state:
                st.session_state["ws_env_wind_dir"] = int(CURRENT_ENV["wind_direction"])
            if "ws_env_season" not in st.session_state:
                st.session_state["ws_env_season"] = CURRENT_ENV["season"]
            if "ws_env_agent" not in st.session_state:
                st.session_state["ws_env_agent"] = CURRENT_ENV["active_agent"]

            for key in ["wind_speed", "wind_direction", "season", "active_agent"]:
                val = CURRENT_ENV[key]
                last_val = st.session_state["LAST_SEEN_ENV"].get(key)
                if last_val is not None and val != last_val:
                    widget_key = f"ws_env_{key}"
                    if key == "wind_direction":
                        widget_key = "ws_env_wind_dir"
                    elif key == "active_agent":
                        widget_key = "ws_env_agent"
                    st.session_state[widget_key] = val
                st.session_state["LAST_SEEN_ENV"][key] = val

            # Replay and Telemetry layout columns
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 2])

            with ctrl_col1:
                st.markdown("**Mission Playback Controls**")

                # Check status and buttons
                is_rep = REPLAY_STATE["is_replaying"]
                paused = REPLAY_STATE["paused"]
                speed = REPLAY_STATE["speed"]

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("▶ Start Replay" if not is_rep else "⏹ Stop Replay", key="ws_btn_start"):
                        REPLAY_STATE["is_replaying"] = not is_rep
                        REPLAY_STATE["current_index"] = 0
                        REPLAY_STATE["paused"] = False
                        st.rerun()

                with col_btn2:
                    if is_rep:
                        if st.button("⏸ Pause" if not paused else "▶ Play", key="ws_btn_pause"):
                            REPLAY_STATE["paused"] = not paused
                            st.rerun()
                    else:
                        st.button("⏸ Pause", disabled=True, key="ws_btn_pause_dis")

                with col_btn3:
                    if is_rep:
                        speed_opts = [1, 2, 5, 10]
                        try:
                            curr_idx = speed_opts.index(int(speed))
                        except ValueError:
                            curr_idx = 0
                        new_speed = st.selectbox("Speed", speed_opts, index=curr_idx, key="ws_select_speed")
                        if new_speed != REPLAY_STATE["speed"]:
                            REPLAY_STATE["speed"] = new_speed
                            st.rerun()
                    else:
                        st.selectbox("Speed", [1], disabled=True, key="ws_select_speed_dis")

                # Buffer Frame Slider / Seek Scrubber
                if len(TELEMETRY_BUFFER) > 0:
                    max_idx = len(TELEMETRY_BUFFER) - 1
                    if is_rep:
                        current_idx = min(max_idx, REPLAY_STATE["current_index"])
                        new_idx = st.slider("Playback Seek", 0, max_idx, int(current_idx), key="ws_seek_slider")
                        if new_idx != current_idx:
                            REPLAY_STATE["current_index"] = new_idx
                    else:
                        st.slider("Buffered Flight Frames", 0, max_idx, max_idx, disabled=True, key="ws_buffer_slider")
                else:
                    st.info("No flight data currently buffered. Fly the drone to accumulate telemetry!")

            with ctrl_col2:
                st.markdown("**Mission Memory (IO)**")
                # Save current buffer to a recorded mission file
                if st.button(" Save Flight Profile", key="ws_save_profile"):
                    if len(TELEMETRY_BUFFER) > 0:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"outputs/missions/flight_{timestamp}.json"
                        with open(filename, "w") as f:
                            json.dump(list(TELEMETRY_BUFFER), f, indent=4)
                        st.success(f"Saved {len(TELEMETRY_BUFFER)} frames to `{filename}`!")
                    else:
                        st.error("Telemetry buffer is empty!")

                # Load existing mission files
                mission_files = sorted(glob.glob("outputs/missions/*.json"), reverse=True)
                if mission_files:
                    basenames = [os.path.basename(f) for f in mission_files]
                    selected_file_name = st.selectbox("Load Saved Flight", ["Select File..."] + basenames, key="ws_load_select")
                    if selected_file_name != "Select File...":
                        full_path = os.path.join("outputs/missions", selected_file_name)
                        try:
                            with open(full_path, "r") as f:
                                loaded_data = json.load(f)
                                with buffer_lock:
                                    TELEMETRY_BUFFER.clear()
                                    TELEMETRY_BUFFER.extend(loaded_data)
                            st.success(f"Loaded {len(loaded_data)} telemetry frames!")
                            REPLAY_STATE["total_frames"] = len(TELEMETRY_BUFFER)
                            REPLAY_STATE["is_replaying"] = True
                            REPLAY_STATE["current_index"] = 0
                            st.rerun()
                        except Exception as err:
                            st.error(f"Error loading: {err}")
                else:
                    st.caption("No saved flights found.")

            with ctrl_col3:
                st.markdown("**Live Sync & System Telemetry**")
                t_col1, t_col2 = st.columns(2)
                t_col1.metric("Altitude (MSL)", f"{MAVLINK_TELEMETRY['alt']:.2f} m")
                t_col1.metric("Battery Level", f"{MAVLINK_TELEMETRY['battery']:.1f} %")
                t_col2.metric("Ground Speed", f"{MAVLINK_TELEMETRY['speed']:.1f} m/s")
                t_col2.metric("Spray Pump Status", "ACTIVE" if MAVLINK_TELEMETRY["is_spraying"] else "OFF")

            # --- Live Graphing/Plotting ---
            if len(TELEMETRY_BUFFER) > 1:
                st.markdown("**Real-Time Telemetry Analytics Charts**")
                chart_col1, chart_col2 = st.columns(2)

                # Extract buffer coordinates and status for plotting
                with buffer_lock:
                    altitudes = [f["alt"] for f in TELEMETRY_BUFFER]
                    speeds = [f["speed"] for f in TELEMETRY_BUFFER]
                    batteries = [f["battery"] for f in TELEMETRY_BUFFER]
                    spraying = [1.0 if f["is_spraying"] else 0.0 for f in TELEMETRY_BUFFER]
                    timestamps = list(range(len(TELEMETRY_BUFFER)))

                with chart_col1:
                    # Plot Altitude & Spray Graph
                    fig_alt, ax_alt = plt.subplots(figsize=(6, 2.5))
                    ax_alt.plot(timestamps, altitudes, color="#38bdf8", label="Altitude (m)", linewidth=1.5)
                    ax_alt2 = ax_alt.twinx()
                    ax_alt2.fill_between(timestamps, spraying, color="#ec4899", alpha=0.15, label="Spraying")
                    ax_alt.set_title("Altitude and Spraying State Timeline", color="#f8fafc", fontsize=10)
                    ax_alt.set_facecolor('#0e1117')
                    fig_alt.patch.set_facecolor('#0e1117')
                    ax_alt.tick_params(colors='#94a3b8', labelsize=8)
                    ax_alt2.tick_params(colors='#94a3b8', labelsize=8)
                    ax_alt.grid(True, color="#334155", linestyle=":", alpha=0.5)
                    st.pyplot(fig_alt)

                with chart_col2:
                    # Plot Battery Decay Graph
                    fig_bat, ax_bat = plt.subplots(figsize=(6, 2.5))
                    ax_bat.plot(timestamps, batteries, color="#22c55e", label="Battery (%)", linewidth=1.5)
                    ax_bat.set_title("Battery Consumption Trajectory", color="#f8fafc", fontsize=10)
                    ax_bat.set_facecolor('#0e1117')
                    fig_bat.patch.set_facecolor('#0e1117')
                    ax_bat.tick_params(colors='#94a3b8', labelsize=8)
                    ax_bat.grid(True, color="#334155", linestyle=":", alpha=0.5)
                    st.pyplot(fig_bat)

            # --- Wind Control and Season Control Synchronizer ---
            st.markdown("---")
            st.markdown("**Real-Time Twin Physical Synchronization Controls**")
            st.markdown(
                "These sliders update the ambient conditions of the interactive 3D digital twin "
                "in real-time across all connected WebSockets *without* reloading the iframe."
            )
            sync_col1, sync_col2, sync_col3, sync_col4 = st.columns(4)
            with sync_col1:
                wind_speed_val = st.slider("Live Wind Speed (m/s)", 0.0, 25.0, step=0.5, key="ws_env_wind_speed")
                if wind_speed_val != CURRENT_ENV["wind_speed"]:
                    CURRENT_ENV["wind_speed"] = wind_speed_val
                    if "LAST_SEEN_ENV" not in st.session_state:
                        st.session_state["LAST_SEEN_ENV"] = {}
                    st.session_state["LAST_SEEN_ENV"]["wind_speed"] = wind_speed_val
            with sync_col2:
                wind_dir_val = st.slider("Live Wind Direction (°)", 0, 360, step=5, key="ws_env_wind_dir")
                if wind_dir_val != CURRENT_ENV["wind_direction"]:
                    CURRENT_ENV["wind_direction"] = wind_dir_val
                    if "LAST_SEEN_ENV" not in st.session_state:
                        st.session_state["LAST_SEEN_ENV"] = {}
                    st.session_state["LAST_SEEN_ENV"]["wind_direction"] = wind_dir_val
            with sync_col3:
                season_options = ["Spring Green", "Midsummer Lush", "Autumn Harvest", "Drought Parched"]
                if CURRENT_ENV["season"] not in season_options:
                    CURRENT_ENV["season"] = "Spring Green"
                season_val = st.selectbox("Live Season Variant", season_options, key="ws_env_season")
                if season_val != CURRENT_ENV["season"]:
                    CURRENT_ENV["season"] = season_val
                    if "LAST_SEEN_ENV" not in st.session_state:
                        st.session_state["LAST_SEEN_ENV"] = {}
                    st.session_state["LAST_SEEN_ENV"]["season"] = season_val
            with sync_col4:
                agent_options = ["Dynamic Smart Tracking", "Broad-Spectrum Blanket", "Targeted Fungicide", "Liquid Nitrogen Nutrient"]
                if CURRENT_ENV["active_agent"] not in agent_options:
                    CURRENT_ENV["active_agent"] = "Dynamic Smart Tracking"
                active_agent_val = st.selectbox("Live Spray Agent Active", agent_options, key="ws_env_agent")
                if active_agent_val != CURRENT_ENV["active_agent"]:
                    CURRENT_ENV["active_agent"] = active_agent_val
                    if "LAST_SEEN_ENV" not in st.session_state:
                        st.session_state["LAST_SEEN_ENV"] = {}
                    st.session_state["LAST_SEEN_ENV"]["active_agent"] = active_agent_val

            # --- Real-Time Multiplayer & Supervision Controls ---
            st.markdown("---")
            st.markdown(" **Real-Time Multiplayer Fleet Operations & Remote Supervision**")
            st.markdown(
                "Monitor active operators, coordinate the multi-UAV spray fleet, "
                "and review collaborative 3D annotations in real-time."
            )

            m_col1, m_col2, m_col3 = st.columns([1, 1, 2])
            with m_col1:
                st.markdown("**Connected Operator Sessions**")
                st.metric("Total Connections", len(WS_CLIENTS))
                roles_list = list(WS_CLIENT_ROLES.values())
                if roles_list:
                    for role in set(roles_list):
                        st.caption(f"• `{role}` ({roles_list.count(role)} active)")
                else:
                    st.caption("No active operator roles selected.")

            with m_col2:
                st.markdown("**Collaborative 3D Pins**")
                if COLLABORATIVE_ANNOTATIONS:
                    for idx, ann in enumerate(COLLABORATIVE_ANNOTATIONS):
                        ann_row = st.container()
                        with ann_row:
                            st.markdown(f" **{ann.get('label', 'Pin')}**  \n*{ann.get('creator_role', 'Observer')}*")
                            if st.button("Delete", key=f"del_ann_{ann.get('id', idx)}", help="Delete Pin"):
                                if idx < len(COLLABORATIVE_ANNOTATIONS):
                                    COLLABORATIVE_ANNOTATIONS.pop(idx)
                                st.rerun()
                    if st.button("Clear All Pins", key="clear_all_pins_btn"):
                        COLLABORATIVE_ANNOTATIONS.clear()
                        st.rerun()
                else:
                    st.caption("No annotations placed on the field.")

            with m_col3:
                st.markdown("**Multi-UAV Fleet Status**")
                d_cols = st.columns(2)
                for idx, (d_id, drone) in enumerate(MULTIPLAYER_DRONES.items()):
                    with d_cols[idx % 2]:
                        st.markdown(f"**{drone['label']}**")
                        st.caption(f"Battery: **{drone['battery']:.1f}%**")
                        st.caption(f"Altitude: **{drone['alt']:.2f} m**")
                        st.caption(f"Speed: **{drone['speed']:.1f} m/s**")
                        st.caption(f"Spray: **{'ON' if drone['is_spraying'] else 'OFF'}**")

        render_telemetry_hud()
    else:
        fig_3d = plt.figure(figsize=(10, 6.5))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        step = 4
        H_c, W_c = chm.shape
        x = np.arange(0, W_c, step) * 0.05
        y = np.arange(0, H_c, step) * 0.05
        X, Y = np.meshgrid(x, y)
        Z = chm[::step, ::step]

        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax_3d.set_title("3D Crop Structure Profile", color="#f8fafc")
        ax_3d.set_xlabel("Field Width (m)", color="#cbd5e1")
        ax_3d.set_ylabel("Field Length (m)", color="#cbd5e1")
        ax_3d.set_zlabel("Height (m)", color="#cbd5e1")

        fig_3d.patch.set_facecolor('#0e1117')
        ax_3d.set_facecolor('#0e1117')
        try:
            ax_3d.xaxis.set_pane_color((0.05, 0.07, 0.1, 1.0))
            ax_3d.yaxis.set_pane_color((0.05, 0.07, 0.1, 1.0))
            ax_3d.zaxis.set_pane_color((0.05, 0.07, 0.1, 1.0))
        except Exception:
            pass

        fig_3d.colorbar(surf, ax=ax_3d, shrink=0.45, label="Height (m)")
        st.pyplot(fig_3d)

# ══════════════════════════════════════════════════════════════
# TAB 14 — AI Report
# ══════════════════════════════════════════════════════════════
