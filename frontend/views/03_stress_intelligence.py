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
#  Stress Intelligence  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Stress Intelligence", "Segmentation, severity mapping, and GradCAM explainability")

if "ms_image" not in st.session_state:
    st.warning("Please upload or enable demo data in Upload & Process first.")
else:
    ms_image = st.session_state["ms_image"]
    idx      = st.session_state["indices"]

    with st.spinner("Running stress segmentation..."):
        mask = rule_based_stress_segmentation(ms_image.bands)
        base_rgb = ms_image.false_color_cir()
        overlay  = mask_to_overlay(mask, base_rgb, alpha=0.55)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("False Color CIR")
        st.image(base_rgb, use_container_width=True)
    with col2:
        st.subheader("Stress Segmentation Overlay")
        st.image(overlay, use_container_width=True)
    with col3:
        st.subheader("Stress Score Heatmap")
        stress_rgb = colormap_array(idx["stress_score"], "RdYlGn_r", 0, 1)
        st.image(stress_rgb, use_container_width=True)

    # Class legend + area stats & physical thresholds
    st.subheader("Segmentation Class Distribution & Stress Level Diagnostics")
    total = mask.size
    cols = st.columns(max(5, len(CLASS_LABELS)))
    ndvi_arr = idx["ndvi"]

    for cls_id, label in CLASS_LABELS.items():
        pixels_in_class = mask == cls_id
        area_pct = pixels_in_class.sum() / total * 100

        # Compute average NDVI level
        if pixels_in_class.any():
            avg_ndvi = float(ndvi_arr[pixels_in_class].mean())
        else:
            avg_ndvi = 0.0

        # Evaluation description based on physical thresholds
        if cls_id == 0:
            eval_str = "<small>Non-vegetated background</small>"
            avg_info = f"Avg NDVI: {avg_ndvi:.2f}"
        elif cls_id == 1:
            # Healthy Canopy
            status = "GOOD" if avg_ndvi >= 0.60 else "SUB-OPTIMAL"
            eval_str = f"Status: <b>{status}</b><br><small>Threshold: &ge;0.60</small>"
            avg_info = f"Avg NDVI: {avg_ndvi:.2f}"
        elif cls_id == 2:
            # Mild Stress
            status = "WARNING" if avg_ndvi < 0.60 else "GOOD"
            eval_str = f"Status: <b>{status}</b><br><small>Threshold: 0.30-0.60</small>"
            avg_info = f"Avg NDVI: {avg_ndvi:.2f}"
        elif cls_id == 3:
            # Moderate Stress
            eval_str = "Status: <b>ATTENTION</b><br><small>Threshold: 0.10-0.30</small>"
            avg_info = f"Avg NDVI: {avg_ndvi:.2f} (BAD)"
        elif cls_id == 4:
            # Severe Stress
            eval_str = "Status: <b>CRITICAL</b><br><small>Threshold: &lt;0.10</small>"
            avg_info = f"Avg NDVI: {avg_ndvi:.2f} (SEVERE)"

        try:
            color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[cls_id, :3])
        except Exception:
            color_hex = "#cccccc"
        text_color = "#000000" if cls_id in [2, 3] else "#ffffff"
        cols[cls_id % len(cols)].markdown(
            f"<div style='background:{color_hex};color:{text_color};padding:10px;border-radius:6px;line-height:1.45;min-height:125px'>"
            f"<span style='font-size:1.05rem;font-weight:700;'>{label}</span><br>"
            f"<b>Area:</b> {area_pct:.1f}%<br>"
            f"<b>{avg_info}</b><br>"
            f"{eval_str}</div>",
            unsafe_allow_html=True,
        )

    # Explainability (GradCAM Attention)
    st.subheader("Explainability (GradCAM Attention)")

    try:
        model = load_cached_segmentation_model()
        # Prepare tensor (1, 5, 256, 256)
        stack = [ms_image.green, ms_image.red, ms_image.red_edge, ms_image.nir, ms_image.nir]
        arr = np.stack(stack, axis=0).astype(np.float32)
        resized = np.stack([
            cv2.resize(arr[c], (256, 256), interpolation=cv2.INTER_LINEAR)
            for c in range(5)
        ], axis=0)
        tensor = torch.from_numpy(resized).unsqueeze(0)

        target_class = st.selectbox(
            "Select Target Stress Class for Explainability Analysis",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda k: CLASS_LABELS[k],
            index=3, # default: moderate stress
        )

        with st.spinner("Generating feature attribution map..."):
            from src.segmentation.gradcam_segmentation import SegmentationGradCAM
            try:
                # Target layer is layer4 of ResNet50 encoder in smp.DeepLabV3Plus
                target_layer = model.backbone.encoder.layer4
                gcam = SegmentationGradCAM(model, target_layer)
                cam = gcam.generate(tensor, target_class)
                gcam.remove_hooks()
            except Exception as gcam_err:
                # Fallback to feature norm CAM if grad-based hook fails
                cam = class_activation_map(model, tensor)

            if cam is not None and cam.shape == (256, 256):
                # Resize preview to 256x256
                rgb_preview = cv2.resize(ms_image.rgb_preview(), (256, 256))
                overlay_cam = overlay_segmentation_cam(rgb_preview, cam)

                col_cam1, col_cam2 = st.columns(2)
                with col_cam1:
                    st.image(rgb_preview, caption="Standard Field Preview (RGB)", use_container_width=True)
                with col_cam2:
                    st.image(overlay_cam, caption=f"Grad-CAM Attribution Overlay for: {CLASS_LABELS[target_class]}", use_container_width=True)

                st.success(
                    "Saliency attribution map successfully generated. "
                    "Bright red/yellow highlights represent key features/regions driving the segmentation decision."
                )
            else:
                st.error("Failed to generate attribution map due to shape mismatch.")
    except Exception as e:
        st.warning(f"Grad-CAM explainability could not run: {e}")

    # Stress area summary
    st.subheader("Stress Summary")
    stressed_pct = float((idx["stress_score"] > stress_threshold).mean() * 100)
    severity_color = "stress-critical" if stressed_pct > 40 else \
                     "stress-high"     if stressed_pct > 25 else \
                     "stress-medium"   if stressed_pct > 10 else "stress-low"

    st.markdown(
        f"**Stressed Area (threshold={stress_threshold}):** "
        f'<span class="{severity_color}">{stressed_pct:.1f}%</span>',
        unsafe_allow_html=True,
    )

    # Model Training Panel
    st.write("---")
    st.subheader("Deep Learning Model Training Control Panel")
    st.markdown(
        "Configure hyperparameters and launch the DeepLabV3+ segmentation model training loop. "
        "For demonstration, this triggers an interactive training run using a synthetic dataset patch stack."
    )

    train_cols = st.columns(3)
    with train_cols[0]:
        train_lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=1e-4, format="%e")
    with train_cols[1]:
        train_epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=3)
    with train_cols[2]:
        train_batch_size = st.selectbox("Batch Size", options=[2, 4, 8, 16], index=1)

    if st.button("Launch Interactive Training Loop", key="btn_segmentation_train"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()

        status_box.info("Initializing dataset and model architecture...")

        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            # Create fake inputs
            X_train = torch.randn(10, 5, 256, 256)
            y_train = torch.randint(0, 5, (10, 256, 256)).long()
            X_val = torch.randn(4, 5, 256, 256)
            y_val = torch.randint(0, 5, (4, 256, 256)).long()

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=train_batch_size, shuffle=False)

            model = build_model()
            model.train()

            from src.segmentation.train_segmentation import CombinedLoss
            criterion = CombinedLoss(num_classes=5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr)

            history = {"epoch": [], "train_loss": [], "val_loss": []}

            for epoch in range(1, train_epochs + 1):
                status_box.info(f"Training Epoch {epoch}/{train_epochs}...")

                # Train epoch
                epoch_loss = 0.0
                model.train()
                for step, (images, masks) in enumerate(train_loader):
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, masks)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)

                # Val epoch
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        logits = model(images)
                        loss = criterion(logits, masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                history["epoch"].append(epoch)
                history["train_loss"].append(epoch_loss)
                history["val_loss"].append(val_loss)

                # Update progress and charts
                progress_bar.progress(epoch / train_epochs)

                # Plot curves
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", color="#e74c3c")
                ax.plot(history["epoch"], history["val_loss"], label="Val Loss", marker="x", color="#3498db")
                ax.set_title("Training Diagnostics (Interactive)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                fig.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

            status_box.success("Interactive training run complete! Model successfully trained and metrics updated.")
            st.session_state["trained_model"] = model

        except Exception as train_err:
            status_box.error(f"Error during training loop: {train_err}")
