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
#  Temporal Analytics  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Temporal Analytics", "Track vegetation and stress progression across crop growth stages")
st.markdown("Simulated temporal analytics across Nursery → Vegetative → Flowering → Mature stages.")

# Synthetic temporal dataset (replace with real dataset loader)
@st.cache_data
def make_synthetic_temporal():
    """Synthetic index progressions modelled on paddy phenology literature."""
    np.random.seed(7)
    data = {
        "Nursery":    {"ndvi_mean": 0.22, "ndre_mean": 0.18, "ndwi_mean": 0.10,
                       "gndvi_mean": 0.25, "stress_mean": 0.58, "n": 8},
        "Vegetative": {"ndvi_mean": 0.62, "ndre_mean": 0.41, "ndwi_mean":-0.05,
                       "gndvi_mean": 0.55, "stress_mean": 0.32, "n": 12},
        "Flowering":  {"ndvi_mean": 0.71, "ndre_mean": 0.48, "ndwi_mean":-0.12,
                       "gndvi_mean": 0.63, "stress_mean": 0.25, "n": 10},
        "Mature":     {"ndvi_mean": 0.45, "ndre_mean": 0.30, "ndwi_mean":-0.08,
                       "gndvi_mean": 0.40, "stress_mean": 0.42, "n": 9},
    }
    return data

temporal = make_synthetic_temporal()
stages = list(temporal.keys())

ndvi_means  = [temporal[s]["ndvi_mean"]   for s in stages]
stress_means= [temporal[s]["stress_mean"] for s in stages]
ndre_means  = [temporal[s]["ndre_mean"]   for s in stages]
ndwi_means  = [temporal[s]["ndwi_mean"]   for s in stages]
gndvi_means = [temporal[s]["gndvi_mean"]  for s in stages]

# Build DataFrame for display
df = pd.DataFrame({
    "Stage":       stages,
    "NDVI Mean":   ndvi_means,
    "NDRE Mean":   ndre_means,
    "NDWI Mean":   ndwi_means,
    "GNDVI Mean":  gndvi_means,
    "Stress Mean": stress_means,
    "N Images":    [temporal[s]["n"] for s in stages],
})
st.dataframe(df.set_index("Stage"), use_container_width=True)

# Plots
c1, c2 = st.columns(2)

with c1:
    # NDVI progression
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stages))
    ax.plot(x, ndvi_means, "o-", color="#2ECC71", lw=2.5, ms=8, label="NDVI")
    ax.plot(x, ndre_means, "s--",color="#3498DB", lw=2, ms=7, label="NDRE")
    ax.plot(x, gndvi_means,"^-.", color="#9B59B6", lw=2, ms=7, label="GNDVI")
    ax.axhline(0.6, ls="--", color="grey", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel("Index Value"); ax.set_title("Vegetation Index Progression")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.image(fig_to_bytes(fig), use_container_width=True)
    plt.close(fig)

with c2:
    # Stress progression
    colors = ["#90EE90", "#32CD32", "#FFD700", "#FF8C00"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stages, stress_means, color=colors, edgecolor="white")
    for bar, val in zip(bars, stress_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Composite Stress Score")
    ax.set_title("Stress Progression Across Growth Stages")
    ax.set_ylim(0, 0.8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.image(fig_to_bytes(fig), use_container_width=True)
    plt.close(fig)

# Change detection
st.write("---")
st.markdown("---")
section_header("Interactive Two-Date Survey Comparison", "Change detection between two growth stages")
st.markdown(
    "Upload a second historical GeoTIFF image to compare with the active image, "
    "or use a simulated historical scan (active image shifted and stressed) to demo the change detection pipeline."
)

if "ms_image" not in st.session_state:
    st.warning("Please upload or enable demo data in Upload & Process first.")
else:
    ms_image = st.session_state["ms_image"]
    idx = st.session_state["indices"]

    t1_file = st.file_uploader("Upload Historical UAV GeoTIFF (Time 1)", type=["tif", "tiff"], key="t1_file_uploader")

    # Prepare ndvi_t2
    ndvi_t2 = idx["ndvi"]

    if t1_file is not None:
        try:
            with st.spinner("Loading Time 1 image..."):
                # Save temp file
                temp_dir = Path("outputs/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / t1_file.name
                with open(temp_path, "wb") as f:
                    f.write(t1_file.getbuffer())

                t1_image = load_multispectral_tiff(temp_path)

                # Compute NDVI for T1
                from src.indices.indices import compute_ndvi
                ndvi_t1 = compute_ndvi(t1_image.nir, t1_image.red)

                # Resize if shape mismatch
                if ndvi_t1.shape != ndvi_t2.shape:
                    ndvi_t1 = cv2.resize(ndvi_t1, (ndvi_t2.shape[1], ndvi_t2.shape[0]))

                st.success(f"Successfully loaded {t1_file.name} for comparison.")
        except Exception as e:
            st.error(f"Error loading Time 1 image: {e}")
            t1_file = None

    if t1_file is None:
        # Generate simulated Time 1: slightly less healthy (lower NDVI)
        st.info("No Time 1 image uploaded. Using simulated historical survey (15% lower NDVI baseline).")
        ndvi_t1 = np.clip(ndvi_t2 - 0.15 + np.random.normal(0, 0.03, ndvi_t2.shape), -1.0, 1.0).astype(np.float32)

    # Run NDVI change detection
    from src.temporal.change_detection import ndvi_difference, plot_change_maps

    change_thresh = st.slider("ΔNDVI Change Threshold", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

    with st.spinner("Performing change detection..."):
        result = ndvi_difference(ndvi_t1, ndvi_t2, threshold=change_thresh)
        fig_change = plot_change_maps(result, t1_label="Historical", t2_label="Active")
        st.pyplot(fig_change)
        plt.close(fig_change)

    # Display change stats
    c_stats1, c_stats2, c_stats3 = st.columns(3)
    with c_stats1:
        metric_card("Total Changed Area", f"{result.pct_changed:.1f}%", badge_kind="info")
    with c_stats2:
        metric_card("Mean ΔNDVI Difference", f"{result.mean_change:+.3f}", badge_kind="info")
    with c_stats3:
        metric_card("Max Change Magnitude", f"{result.max_magnitude:.3f}", badge_kind="info")
