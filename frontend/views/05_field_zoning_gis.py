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
#  Field Zoning (GIS)
# ══════════════════════════════════════════════════════════════

section_header("Field Zoning (GIS)", "Management zones, stress regions, grid statistics and interactive maps", category="spatial")

if "ms_image" not in st.session_state:
    st.warning(" Please upload or enable demo data on the **Upload & Process** page first.")
else:
    ms_image = st.session_state["ms_image"]
    idx = st.session_state["indices"]
    ndvi        = idx["ndvi"]
    stress_score= idx["stress_score"]

    zones = delineate_management_zones(ndvi, stress_score)
    zone_map_rgb = render_zone_map(zones, ndvi.shape)
    regions = extract_stress_regions(stress_score, threshold=stress_threshold)

    tab_zones, tab_rx, tab_grid, tab_regions, tab_maps = st.tabs(
        [" Zone Map", " Prescriptions", " Grid Analysis", " Stress Regions", " Interactive Maps"]
    )

    with tab_zones:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("**Management Zone Map**")
            st.image(zone_map_rgb, use_container_width=True)
        with col2:
            st.markdown("**Zone Statistics**")
            zone_df = pd.DataFrame([{
                "Zone": z.zone_name,
                "Area %": f"{z.area_pct:.1f}%",
                "NDVI Mean": f"{z.ndvi_mean:.3f}",
                "Stress Mean": f"{z.stress_mean:.3f}",
            } for z in zones])
            st.dataframe(zone_df, use_container_width=True, hide_index=True)

    with tab_rx:
        for z in zones:
            if z.ndvi_mean >= 0.60:
                z_status, kind = "Healthy Canopy (Optimal ≥ 0.60)", "good"
            elif z.ndvi_mean >= 0.30:
                z_status, kind = "Mildly Degraded (Warning 0.30–0.60)", "warn"
            else:
                z_status, kind = "Critical Vegetation Loss (< 0.30)", "crit"

            with st.expander(f"{z.zone_name} — {z.area_pct:.1f}% of field"):
                metric_card(z.zone_name, "", sub=z.prescription, badge=z_status, badge_kind=kind)
                st.caption(f"Mean NDVI: {z.ndvi_mean:.3f}  |  Stress Score: {z.stress_mean:.3f}")

    with tab_grid:
        st.markdown(f"**Field Grid Analysis ({grid_size}×{grid_size})**")
        grid = compute_grid_statistics(ndvi, stress_score, grid_size, grid_size)
        fig = plot_grid_heatmap(grid, f"Mean NDVI per Grid Cell ({grid_size}×{grid_size})")
        st.image(fig_to_bytes(fig), use_container_width=True)

    with tab_regions:
        st.markdown("**Spatial Stress Region Detection**")
        if regions:
            region_df = pd.DataFrame([{
                "Region ID":  r.region_id,
                "Severity":   r.severity,
                "Area %":     f"{r.area_pct:.2f}%",
                "Stress Score": f"{r.stress_mean:.3f}",
                "Centroid (y,x)": f"({r.centroid_yx[0]:.0f}, {r.centroid_yx[1]:.0f})",
            } for r in regions])
            st.dataframe(region_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant stress regions detected at current threshold.")

    with tab_maps:
        st.caption(
            "Visualise spatial patterns interactively. Use layers to inspect management zones, "
            "stress regions, and high-intensity thermal/index anomalies."
        )

        def get_lat_lon_grids(image, center_lat, center_lon, gsd=0.05):
            lat_deg_per_meter = 1.0 / 111120.0
            lon_deg_per_meter = 1.0 / (111120.0 * np.cos(np.radians(center_lat)))
            grid_h, grid_w = 50, 50
            rows = np.linspace(-grid_h/2, grid_h/2, grid_h) * gsd * lat_deg_per_meter
            cols = np.linspace(-grid_w/2, grid_w/2, grid_w) * gsd * lon_deg_per_meter
            lons_grid, lats_grid = np.meshgrid(cols + center_lon, rows + center_lat)
            return lats_grid.flatten(), lons_grid.flatten()

        map_col1, map_col2 = st.columns(2)

        with map_col1:
            st.markdown("**Management Zone & Stress Region Map**")
            from src.gis.field_zoning import create_folium_stress_map
            folium_map = create_folium_stress_map(zones, regions, (lat, lon))
            if folium_map is not None:
                from streamlit_folium import st_folium
                st_folium(folium_map, width=None, height=400, returned_objects=[], key="gis_folium_zones")
            else:
                st.warning("GIS map components unavailable.")

        with map_col2:
            st.markdown("**Continuous Stress Heatmap**")
            H_arr, W_arr = stress_score.shape
            y_indices = np.linspace(0, H_arr - 1, 50, dtype=int)
            x_indices = np.linspace(0, W_arr - 1, 50, dtype=int)

            sampled_stress = []
            sampled_lats = []
            sampled_lons = []

            lats_grid, lons_grid = get_lat_lon_grids(ms_image, lat, lon)
            lats_grid_2d = lats_grid.reshape(50, 50)
            lons_grid_2d = lons_grid.reshape(50, 50)

            for i, y in enumerate(y_indices):
                for j, x in enumerate(x_indices):
                    val = stress_score[y, x]
                    if val > stress_threshold:
                        sampled_stress.append(float(val))
                        sampled_lats.append(float(lats_grid_2d[i, j]))
                        sampled_lons.append(float(lons_grid_2d[i, j]))

            if sampled_stress:
                from src.gis.folium_maps import stress_heatmap
                heatmap_map = stress_heatmap(
                    np.array(sampled_lats),
                    np.array(sampled_lons),
                    np.array(sampled_stress),
                    center=(lat, lon)
                )
                from streamlit_folium import st_folium
                st_folium(heatmap_map, width=None, height=400, returned_objects=[], key="gis_folium_heatmap")
            else:
                st.info("No significant stress pixels detected to build spatial heatmap overlay.")
