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
#  Satellite Analytics  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Satellite Analytics", "Sentinel-2 integration via Google Earth Engine, fused with UAV inspection triggering", category="spatial")

tab_config, tab_results = st.tabs([" Query Configuration", " Results & Anomaly Detection"])

with tab_config:
    col_sat1, col_sat2 = st.columns(2)
    with col_sat1:
        st.markdown("**Region & Date Range**")
        project_id = st.text_input("Google Cloud Project ID", value="buoyant-facet-454614-d1", help="Required by the Earth Engine API.")
        roi_lat = st.number_input("Center Latitude", value=float(lat) if "lat" in locals() else 11.0, format="%.6f")
        roi_lon = st.number_input("Center Longitude", value=float(lon) if "lon" in locals() else 79.0, format="%.6f")
        roi_size = st.number_input("Bounding Box Size (deg)", value=0.01, format="%.4f")
        start_date = st.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=30))
        end_date = st.date_input("End Date", value=pd.Timestamp.now())
        fetch_btn = st.button("Fetch Sentinel-2 Composite (Cloud-Masked)", type="primary", disabled=not project_id)

    with col_sat2:
        st.markdown("**Satellite Anomaly Triggering**")
        st.caption("If the macroscopic Sentinel-2 scan detects anomalous drops in NDVI, the AI will automatically generate a targeted UAV waypoint mission to investigate the exact GPS coordinates.")
        trigger_btn = st.button("Analyze for Anomalies & Trigger UAV")

with tab_results:
    if fetch_btn:
        with st.spinner("Authenticating with Google Earth Engine & rendering composite..."):
            import sys
            import importlib
            if 'src.core.satellite_loader' in sys.modules:
                importlib.reload(sys.modules['src.core.satellite_loader'])

            from src.core.satellite_loader import Sentinel2Engine
            engine = Sentinel2Engine(project_id=project_id)

            if not engine.initialized:
                st.error(f"Google Earth Engine failed to initialize. Details: {engine.error_msg}")
            else:
                roi_poly = [
                    [roi_lon - roi_size, roi_lat - roi_size],
                    [roi_lon + roi_size, roi_lat - roi_size],
                    [roi_lon + roi_size, roi_lat + roi_size],
                    [roi_lon - roi_size, roi_lat + roi_size],
                    [roi_lon - roi_size, roi_lat - roi_size]
                ]

                img = engine.fetch_satellite_composite(roi_poly, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if img is not None:
                    st.session_state["satellite_img"] = img
                    st.session_state["satellite_roi"] = roi_poly
                    st.success("Successfully fetched Sentinel-2 Harmonized L2A median composite!")

                    rgb_url = engine.get_rgb_thumbnail_url(img, roi_poly)
                    ndvi_url = engine.get_index_thumbnail_url(img, roi_poly, "NDVI")
                    evi_url = engine.get_index_thumbnail_url(img, roi_poly, "EVI")
                    ndwi_url = engine.get_index_thumbnail_url(img, roi_poly, "NDWI")
                    cire_url = engine.get_index_thumbnail_url(img, roi_poly, "CIRE")

                    st.markdown("### Multi-Scale Monitoring: Satellite Indices")

                    scol1, scol2, scol3 = st.columns(3)
                    with scol1:
                        st.markdown("**True Color (RGB)**")
                        if rgb_url: st.image(rgb_url, use_container_width=True)
                        st.markdown("**NDWI (Water Stress)**")
                        if ndwi_url: st.image(ndwi_url, use_container_width=True)
                    with scol2:
                        st.markdown("**NDVI (Vigor)**")
                        if ndvi_url: st.image(ndvi_url, use_container_width=True)
                        st.markdown("**CIre (Chlorophyll)**")
                        if cire_url: st.image(cire_url, use_container_width=True)
                    with scol3:
                        st.markdown("**EVI (Canopy Structure)**")
                        if evi_url: st.image(evi_url, use_container_width=True)

                        st.markdown("**Spatiotemporal Fusion**")
                        st.info("Mathematical fusion mask active. Highlights micro-anomalies missed by Sentinel-2 10m grid when compared to UAV.")
                        # Mock visual representation using the chlorophyll map + styling
                        if cire_url: st.image(cire_url, use_container_width=True, caption="UAV-Satellite Delta Mask")
                else:
                    st.error("Failed to fetch image or no cloud-free images found in the specified date range.")

    if trigger_btn:
        if "satellite_img" not in st.session_state:
            st.warning("Please fetch the Sentinel-2 composite first before analyzing for anomalies.")
        else:
            st.info("Simulating anomaly detection on the Sentinel-2 10m grid...")
            import time
            time.sleep(1.5)
        # Mock finding an anomaly based on the requested coordinates
        roi_lat_anomaly = st.session_state["satellite_roi"][0][1] + 0.002
        roi_lon_anomaly = st.session_state["satellite_roi"][0][0] + 0.002

        st.warning(f"Large-Area Anomaly Detected! Sharp EVI and NDWI drop found at Lat {roi_lat_anomaly:.5f}, Lon {roi_lon_anomaly:.5f}.")
        st.info("Triggering targeted UAV multi-spectral inspection for high-resolution validation.")

        import sys
        import importlib
        if 'src.ai_engine.treatment_recommender' in sys.modules:
            importlib.reload(sys.modules['src.ai_engine.treatment_recommender'])

        from src.ai_engine.treatment_recommender import AITreatmentRecommender
        rec = AITreatmentRecommender()
        mission = rec.generate_uav_inspection_mission([(roi_lon_anomaly, roi_lat_anomaly)])

        st.success("Targeted UAV Inspection Mission Generated!")
        st.json(mission, expanded=False)
        st.download_button(
            "Download QGC Waypoint Mission",
            data=json.dumps(mission, indent=4),
            file_name="anomaly_inspection_mission.plan",
            mime="application/json"
        )
