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
#  Weather & Risk  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Weather & Risk", "Weather-aware crop stress risk assessment fused with vegetation indices", category="operations")

fetch_btn = st.button(" Fetch Weather Data", type="primary")

if fetch_btn or "weather_assessment" in st.session_state:
    if fetch_btn:
        with st.spinner(f"Fetching weather for ({lat:.3f}, {lon:.3f})..."):
            weather = fetch_weather(lat, lon)
            if weather is None:
                st.error("Weather API unavailable. Check internet connection or install `requests`.")
                st.stop()

            idx_now = st.session_state.get("indices", {})
            ndvi_m  = float(idx_now.get("ndvi", np.array([0.5])).mean()) if idx_now else 0.5
            ndwi_m  = float(idx_now.get("ndwi", np.array([0.0])).mean()) if idx_now else 0.0

            assessment = assess_weather_stress(weather, crop_stage, ndvi_m, ndwi_m)
            st.session_state["weather_assessment"] = assessment

    assessment = st.session_state.get("weather_assessment")
    if assessment:
        weather = assessment.weather
        risk_kind = {
            "Low": "good", "Medium": "warn",
            "High": "warn", "Critical": "crit"
        }.get(assessment.overall_risk, "info")

        # Overall risk banner
        metric_card("Overall Risk Level", assessment.overall_risk,
                    sub=f"Composite risk score: {assessment.composite_risk_score:.2f}",
                    badge=assessment.overall_risk.upper(), badge_kind=risk_kind)
        st.markdown("")

        # Current weather metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Current Temp", f"{weather.current_temp:.1f}°C", badge_kind="info")
        with c2:
            metric_card("Humidity", f"{weather.current_humidity:.0f}%", badge_kind="info")
        with c3:
            metric_card("Today's Rain", f"{weather.current_precip:.1f} mm", badge_kind="info")
        with c4:
            metric_card("Crop Stage", crop_stage, badge_kind="info")

        # Scientific Weather Threshold Comparison
        st.markdown("### Weather Parameters vs. Optimal Paddy Rice Limits")

        temp_status = "NORMAL (20°C - 35°C)"
        temp_color = "#2ecc71"
        if weather.current_temp > 35.0:
            temp_status = "HEAT RISK (>35°C)"
            temp_color = "#e74c3c"
        elif weather.current_temp < 15.0:
            temp_status = "COLD RISK (<15°C)"
            temp_color = "#3498db"

        hum_status = "OPTIMAL (70% - 90%)"
        hum_color = "#2ecc71"
        if weather.current_humidity > 95.0:
            hum_status = "BLAST/FUNGUS RISK (>95%)"
            hum_color = "#e67e22"
        elif weather.current_humidity < 50.0:
            hum_status = "ARIDITY RISK (<50%)"
            hum_color = "#f39c12"

        rain_status = "NO RAIN"
        rain_color = "#2ecc71"
        if weather.current_precip > 10.0:
            rain_status = "HEAVY RAIN (>10mm)"
            rain_color = "#e74c3c"
        elif weather.current_precip > 0.1:
            rain_status = "LIGHT RAIN"
            rain_color = "#f39c12"

        st.markdown(f"""
        <table style='width:100%; border-collapse: collapse; font-size:0.9rem; margin-top: 10px; margin-bottom: 20px;'>
            <tr style='border-bottom: 1px solid #334155; color:#cbd5e1;'>
                <th style='text-align:left; padding:8px;'>Weather Metric</th>
                <th style='text-align:left; padding:8px;'>Observed</th>
                <th style='text-align:left; padding:8px;'>Paddy Rice Limits</th>
                <th style='text-align:left; padding:8px;'>Scientific Assessment</th>
            </tr>
            <tr>
                <td style='padding:8px;'>Temperature</td>
                <td style='padding:8px;'>{weather.current_temp:.1f}°C</td>
                <td style='padding:8px;'>15.0°C - 35.0°C</td>
                <td style='padding:8px; color:{temp_color};'><b>{temp_status}</b></td>
            </tr>
            <tr>
                <td style='padding:8px;'>Relative Humidity</td>
                <td style='padding:8px;'>{weather.current_humidity:.0f}%</td>
                <td style='padding:8px;'>70% - 90%</td>
                <td style='padding:8px; color:{hum_color};'><b>{hum_status}</b></td>
            </tr>
            <tr>
                <td style='padding:8px;'>Precipitation</td>
                <td style='padding:8px;'>{weather.current_precip:.1f} mm</td>
                <td style='padding:8px;'>&lt; 10.0 mm (during treatment)</td>
                <td style='padding:8px; color:{rain_color};'><b>{rain_status}</b></td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("### Active Risk Factors")
        if assessment.risk_factors:
            for rf in assessment.risk_factors:
                sev_icon = {"High": "", "Medium": "", "Low": ""}.get(rf.severity, "")
                with st.expander(f"{sev_icon} {rf.name} — {rf.severity}"):
                    st.markdown(f"**Observation:** {rf.description}")
                    st.markdown(f"**Recommendation:** {rf.recommendation}")
        else:
            st.success("No active weather stress factors detected.")

        # AI recommendation
        st.markdown("### AI Crop Management Recommendation")
        st.info(assessment.ai_recommendation)

        # Stage warnings
        if assessment.stage_specific_warnings:
            st.markdown("### Stage-Specific Alerts")
            for w in assessment.stage_specific_warnings:
                st.warning(w)

        # Weather chart
        if weather.temperature_max:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            days = list(range(1, len(weather.temperature_max) + 1))

            axes[0].plot(days, weather.temperature_max, "r-o", label="Tmax")
            axes[0].plot(days, weather.temperature_min, "b-o", label="Tmin")
            axes[0].axhline(35, ls="--", color="red", alpha=0.4, label="Heat threshold")
            axes[0].axhline(15, ls="--", color="blue", alpha=0.4, label="Cold threshold")
            axes[0].set_title("Temperature (°C)"); axes[0].legend(); axes[0].grid(alpha=0.3)

            axes[1].bar(days, weather.precipitation, color="#3498DB")
            axes[1].set_title("Daily Precipitation (mm)"); axes[1].grid(axis="y", alpha=0.3)

            fig.tight_layout()
            st.image(fig_to_bytes(fig), use_container_width=True)
