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
#  Upload & Process  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Upload & Process", "Load a multispectral orthomosaic or generate a synthetic paddy field to begin analysis", category="analysis")

_data_ready = bool(st.session_state.get("indices"))

# ── Data source: tabbed instead of side-by-side ─────────────
src_tab1, src_tab2 = st.tabs(["  Upload Real Data", "  Demo Mode"])

with src_tab1:
    uploaded = st.file_uploader(
        "Upload 4-band GeoTIFF (Green / Red / RedEdge / NIR)",
        type=["tif", "tiff"],
    )
    with st.expander(" Expected band layout"):
        st.markdown("""
        | Band | Wavelength | Purpose |
        |------|------------|---------|
        | 1    | Green ~550 nm | Chlorophyll, NDWI |
        | 2    | Red ~670 nm | NDVI denominator |
        | 3    | Red Edge ~720 nm | Early stress, NDRE |
        | 4    | NIR ~840 nm | Biomass, vigour |
        """)

with src_tab2:
    use_demo = st.toggle("Generate synthetic paddy field data", value=False)
    if use_demo:
        st.caption(" 256×256 synthetic field with randomly placed stressed zones.")

# ── Load or generate image (unchanged logic) ────────────────

ms_image = None

if uploaded is not None:
    with st.spinner("Loading multispectral TIFF..."):
        import tempfile
        tmp_path = Path(tempfile.gettempdir()) / "uploaded.tif"
        tmp_path.write_bytes(uploaded.read())
        try:
            ms_image = load_multispectral_tiff(tmp_path)
            st.success(f"Loaded: {ms_image.height}×{ms_image.width} px")
        except Exception as e:
            st.error(f"Failed to load TIFF: {e}")

elif use_demo:
    np.random.seed(42)
    H, W = 256, 256

    base_nir = np.random.normal(0.7, 0.1, (H, W)).clip(0, 1).astype(np.float32)
    base_red = np.random.normal(0.2, 0.05, (H, W)).clip(0, 1).astype(np.float32)
    base_re  = np.random.normal(0.5, 0.08, (H, W)).clip(0, 1).astype(np.float32)
    base_grn = np.random.normal(0.35, 0.06, (H, W)).clip(0, 1).astype(np.float32)

    for _ in range(6):
        cy, cx = np.random.randint(30, H-30), np.random.randint(30, W-30)
        r = np.random.randint(15, 35)
        yy, xx = np.ogrid[:H, :W]
        circle = (yy - cy)**2 + (xx - cx)**2 < r**2
        sev = np.random.uniform(0.3, 0.7)
        base_nir[circle] *= (1 - sev)
        base_red[circle] *= (1 + sev * 0.5)

    ms_image = MultispectralImage(bands={
        "green":    base_grn,
        "red":      base_red,
        "red_edge": base_re,
        "nir":      base_nir,
    })

if ms_image is not None:
    st.session_state["survey_source"] = "Uploaded survey" if uploaded is not None else "Demo survey"
    st.session_state["ms_image"]   = ms_image
    # crop_stage already lives in session_state via the sidebar widget's
    # own key (see shared.render_sidebar) — do not reassign it here.

    idx = compute_all_indices(ms_image.bands)
    st.session_state["indices"] = idx

    st.divider()

    result_tab1, result_tab2 = st.tabs(["  Composites", "  Index Summary"])

    with result_tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(ms_image.false_color_cir(), use_container_width=True,
                      caption="False Color CIR (NIR→R, Red→G, Green→B)")
        with c2:
            st.image(ms_image.false_color_vegetation(), use_container_width=True,
                      caption="False Color — Vegetation Stress")
        with c3:
            st.image(ms_image.false_color_redge_emphasis(), use_container_width=True,
                      caption="False Color — Red Edge Emphasis")

    with result_tab2:
        q1, q2, q3, q4, q5 = st.columns(5)
        with q1:
            metric_card("NDVI", f"{idx['ndvi'].mean():.3f}", badge_kind="info")
        with q2:
            metric_card("NDRE", f"{idx['ndre'].mean():.3f}", badge_kind="info")
        with q3:
            metric_card("NDWI", f"{idx['ndwi'].mean():.3f}", badge_kind="info")
        with q4:
            metric_card("GNDVI", f"{idx['gndvi'].mean():.3f}", badge_kind="info")
        with q5:
            stress_val = idx['stress_score'].mean()
            kind = "crit" if stress_val > 0.5 else ("warn" if stress_val > 0.3 else "good")
            metric_card("Stress Score", f"{stress_val:.3f}", badge_kind=kind)

    st.success(" Data ready — head to **Vegetation Analytics** or any other module in the sidebar to continue.")
else:
    st.info(" Upload a file or turn on Demo Mode above to get started.")
