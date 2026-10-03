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
#  Vegetation Analytics  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Vegetation Analytics", "Explore vegetation indices computed from the multispectral field data")

if "indices" not in st.session_state:
    st.warning(" Please upload or enable demo data on the **Upload & Process** page first.")
else:
    idx = st.session_state["indices"]

    index_choice = st.selectbox(
        "Select Index to Visualise",
        ["NDVI", "NDRE", "NDWI", "GNDVI", "EVI", "SAVI", "Stress Score"],
    )

    interpretations = {
        "NDVI": "NDVI > 0.6 = healthy dense canopy | 0.3–0.6 = sparse/stressed | < 0.3 = bare/stressed.",
        "NDRE": "NDRE > 0.4 = good chlorophyll | Declining NDRE = early N-deficiency or stress.",
        "NDWI": "NDWI > 0.3 = water/flooding | < -0.1 = canopy water stress / drought.",
        "GNDVI": "GNDVI highly correlated with chlorophyll concentration (Gitelson 1996).",
        "EVI":   "EVI reduces atmospheric noise vs NDVI. Best for dense-canopy paddy.",
        "SAVI":  "SAVI corrects for soil background — useful at early Nursery stage.",
        "Stress Score": "Composite stress index (NDVI + NDRE + NDWI weighted). >0.5 = intervention recommended.",
    }

    index_map = {
        "NDVI":        (idx["ndvi"],         "RdYlGn", -1, 1),
        "NDRE":        (idx["ndre"],         "RdYlGn", -1, 1),
        "NDWI":        (idx["ndwi"],         "RdBu",   -1, 1),
        "GNDVI":       (idx["gndvi"],        "RdYlGn", -1, 1),
        "EVI":         (idx["evi"],          "RdYlGn", -1, 1),
        "SAVI":        (idx["savi"],         "RdYlGn", -1, 1),
        "Stress Score":(idx["stress_score"],"RdYlGn_r", 0, 1),
    }
    arr, cmap, vmin, vmax = index_map[index_choice]

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = plot_index_heatmap(arr, f"{index_choice} Heatmap", cmap, vmin, vmax)
        st.image(fig_to_bytes(fig), use_container_width=True)

    with col2:
        st.markdown(f"**{index_choice} Statistics**")
        m1, m2 = st.columns(2)
        with m1:
            metric_card("Mean", f"{arr.mean():.4f}", badge_kind="info")
            metric_card("Min",  f"{arr.min():.4f}", badge_kind="info")
            metric_card("P10",  f"{np.percentile(arr, 10):.4f}", badge_kind="info")
        with m2:
            metric_card("Std",  f"{arr.std():.4f}", badge_kind="info")
            metric_card("Max",  f"{arr.max():.4f}", badge_kind="info")
            metric_card("P90",  f"{np.percentile(arr, 90):.4f}", badge_kind="info")

        # Histogram
        fig_hist, ax = plt.subplots(figsize=(4, 3))
        ax.hist(arr.ravel(), bins=60, color="#2ECC71", edgecolor="white", alpha=0.8)
        ax.set_xlabel(index_choice)
        ax.set_ylabel("Pixel Count")
        ax.set_title(f"{index_choice} Distribution")
        fig_hist.tight_layout()
        st.image(fig_to_bytes(fig_hist), use_container_width=True)

    # Scientific Comparison and Interpretation
    st.markdown("---")
    section_header("Scientific Threshold Comparison & Interpretation")
    mean_val = float(arr.mean())

    label, detail, kind = "Index Mean", f"Average {index_choice} value is {mean_val:.3f}.", "info"
    if index_choice == "NDVI":
        if mean_val >= 0.60:
            label, detail, kind = "Healthy Canopy", f"Mean NDVI ({mean_val:.3f}) is above the scientific threshold of 0.60. Photosynthetic activity and canopy vigor are normal.", "good"
        elif mean_val >= 0.30:
            label, detail, kind = "Mild Stress / Sparse Vegetation", f"Mean NDVI ({mean_val:.3f}) lies in the sub-optimal range (0.30–0.60). Early nutrient deficiency or water stress suspected.", "warn"
        else:
            label, detail, kind = "Critical Degradation", f"Mean NDVI ({mean_val:.3f}) is below the bare soil/high-stress threshold of 0.30. Urgent treatment or irrigation required.", "crit"
    elif index_choice == "NDRE":
        if mean_val >= 0.40:
            label, detail, kind = "Good Chlorophyll", f"Mean NDRE ({mean_val:.3f}) is above the target threshold of 0.40, indicating robust leaf chlorophyll concentration and nitrogen sufficiency.", "good"
        else:
            label, detail, kind = "Nitrogen Deficiency / Stress", f"Mean NDRE ({mean_val:.3f}) is below 0.40. Early chlorophyll degradation detected. Nitrogen top-dressing recommended.", "crit"
    elif index_choice == "NDWI":
        if mean_val >= 0.30:
            label, detail, kind = "Waterlogging / Flooding", f"Mean NDWI ({mean_val:.3f}) is above 0.30. Open water or extreme soil saturation detected. Drainage checks recommended.", "info"
        elif mean_val >= -0.10:
            label, detail, kind = "Optimal Water Content", f"Mean NDWI ({mean_val:.3f}) is in the normal range (-0.10 to 0.30), indicating adequate canopy hydration.", "good"
        else:
            label, detail, kind = "Water Stress / Drought", f"Mean NDWI ({mean_val:.3f}) is below -0.10. Crop is experiencing hydration deficits. Irrigation is recommended.", "crit"
    elif index_choice == "Stress Score":
        if mean_val < 0.30:
            label, detail, kind = "Stress-Free", f"Composite stress score ({mean_val:.3f}) is below the threshold of 0.30. Swarm intervention is not required.", "good"
        elif mean_val < 0.50:
            label, detail, kind = "Mild Field-Level Stress", f"Composite stress score ({mean_val:.3f}) is moderate (0.30–0.50). Monitor weather and soil trends.", "warn"
        else:
            label, detail, kind = "Severe Systemic Stress", f"Composite stress score ({mean_val:.3f}) exceeds the intervention threshold of 0.50. Spray prescription recommended.", "crit"

    metric_card(label, "", sub=detail, badge=kind.upper(), badge_kind=kind)
    st.info(interpretations.get(index_choice, ""))
