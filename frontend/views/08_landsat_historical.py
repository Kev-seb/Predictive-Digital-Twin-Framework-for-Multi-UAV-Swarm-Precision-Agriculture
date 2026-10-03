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
#  Landsat Historical  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Landsat Historical", "10+ years of Landsat 8/9 archives for drought progression, productivity trends, and climate resilience", category="spatial")

tab_config, tab_results = st.tabs([" Engine Controls", " Historical Analytics"])

with tab_config:
    landsat_project_id = st.text_input("GCP Project ID (Landsat)", value="buoyant-facet-454614-d1", help="Required for GEE authentication.")

    # Pull lat/lon from general session state if available
    l_lat = st.number_input("Landsat Lat", value=lat if "lat" in locals() else 11.0, format="%.6f")
    l_lon = st.number_input("Landsat Lon", value=lon if "lon" in locals() else 79.0, format="%.6f")
    l_size = st.number_input("Landsat ROI Size (deg)", value=0.01, format="%.4f", help="Width of the square area in degrees.")

    s_year = st.slider("Start Year", 2013, 2026, 2015)
    e_year = st.slider("End Year", 2013, 2026, 2026)

    fetch_l_btn = st.button("Fetch Landsat Timeseries", type="primary", disabled=not landsat_project_id)

with tab_results:
    if fetch_l_btn:
        import sys
        import importlib
        if 'src.core.landsat_loader' in sys.modules:
            importlib.reload(sys.modules['src.core.landsat_loader'])
        from src.core.landsat_loader import LandsatEngine
        l_engine = LandsatEngine(project_id=landsat_project_id)

        if not l_engine.initialized:
            st.error(f"Failed to initialize Earth Engine: {l_engine.error_msg}")
        else:
            l_roi_poly = [
                [l_lon - l_size, l_lat - l_size],
                [l_lon + l_size, l_lat - l_size],
                [l_lon + l_size, l_lat + l_size],
                [l_lon - l_size, l_lat + l_size],
                [l_lon - l_size, l_lat - l_size]
            ]

            with st.spinner("Accessing GEE Landsat 8/9 Archive..."):
                df_raw = l_engine.fetch_historical_timeseries(l_roi_poly, s_year, e_year)

            if df_raw is not None and not df_raw.empty:
                st.success(f"Successfully loaded {len(df_raw)} historical observations from Landsat 8 & 9!")

                # 1. Multi-Season Evolution Plot
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_raw['date'], y=df_raw['NDVI'], name='NDVI (Vigor)', line=dict(color='#00ff88', width=2)))
                fig.add_trace(go.Scatter(x=df_raw['date'], y=df_raw['NDWI'], name='NDWI (Moisture)', line=dict(color='#00d2ff', width=2)))
                fig.update_layout(
                    title="Multi-Season Vegetation & Moisture Evolution",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Index Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # 2. Drought Progression
                df_drought = l_engine.analyze_drought_progression(df_raw)
                st.markdown("### Long-Term Drought Progression (NDWI-based)")

                # Color bar based on drought class
                colors = df_drought['drought_severity'].apply(
                    lambda x: '#ff4b4b' if x < -2.0 else ('#ffa500' if x < -1.0 else ('#ffff00' if x < -0.5 else '#00ff88'))
                )

                fig_drought = go.Figure(go.Bar(
                    x=df_drought['date'],
                    y=df_drought['drought_severity'],
                    marker_color=colors,
                    name="Drought Severity"
                ))
                fig_drought.update_layout(
                    title="Drought Index Deviation (Negative = Dry / Stress)",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Standard Deviation Anomaly"
                )
                st.plotly_chart(fig_drought, use_container_width=True)

                # 3. Climate Resilience Metrics
                metrics = l_engine.calculate_resilience_metrics(df_raw)
                st.markdown("### Climate Resilience & Long-Term Trend")

                m1, m2, m3 = st.columns(3)
                m1.metric("Resilience Score", f"{metrics['resilience_score']}/100",
                          delta="High Resilience" if metrics['resilience_score'] > 70 else "Vulnerable",
                          delta_color="normal" if metrics['resilience_score'] > 70 else "inverse")
                m2.metric("Avg Drought Recovery", f"{metrics['recovery_average_days']} Days")
                m3.metric("Annual NDVI Slope", f"{metrics['trend_slope'] * 365:.4f}/yr",
                          delta="Productivity Stable" if metrics['trend_slope'] >= 0 else "Degrading Trend",
                          delta_color="normal" if metrics['trend_slope'] >= 0 else "inverse")

                # 4. Long-Term Productivity Simulation
                st.markdown("### Long-Term Yield & Productivity Simulation")
                years_proj = st.slider("Simulation Horizon (Years)", 1, 10, 5)
                simulated_slope = metrics['trend_slope'] * 365.25

                # Base yield proxy
                current_yield = 8.5 # tons per hectare
                projected_yield = max(1.0, current_yield + (simulated_slope * current_yield * years_proj * 5))

                st.info(f"Assuming an annual vegetation drift of **{simulated_slope*100:.2f}%**, projected crop productivity in **{years_proj} years** is simulated at **{projected_yield:.2f} t/ha** (baseline: {current_yield} t/ha).")

            else:
                st.warning("No data retrieved for the specified dates and coordinates. Try increasing the date window or adjusting the ROI center.")
