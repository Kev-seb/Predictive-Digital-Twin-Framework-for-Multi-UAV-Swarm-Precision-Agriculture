"""
live_video_panel.py
--------------------
Live video display panel for the Ground Station tab.

Renders:
    - Live annotated camera feed (from controller)
    - Frame info (FPS, resolution, quality score)
    - Heatmap toggle
    - Frame capture button (save current frame)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st


def render_live_video_panel(controller) -> None:
    """Render the live annotated video feed."""
    st.markdown("#### 📹 Live Camera Feed")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Connection status
        if controller.phone_connected:
            st.success("📱 Phone Connected — Live Stream Active", icon="🟢")
        elif controller.is_active:
            st.warning("⏳ Waiting for phone stream...", icon="📡")
        else:
            st.info("▶ Start a mission to activate live stream", icon="📷")

        # Video frame
        frame_placeholder = st.empty()
        annotated = controller.latest_annotated_frame

        if annotated is not None:
            frame_placeholder.image(
                annotated,
                channels="RGB",
                use_container_width=True,
                caption=f"Live Field AI  ·  {annotated.shape[1]}×{annotated.shape[0]}",
            )
        else:
            # Show placeholder
            placeholder_img = _make_placeholder_frame()
            frame_placeholder.image(
                placeholder_img,
                channels="RGB",
                use_container_width=True,
                caption="No video signal",
            )

    with col2:
        # QoS panel
        qos = controller.qos_metrics
        st.markdown("**Stream QoS**")
        st.metric("FPS", f"{qos.get('fps', 0):.1f}")
        st.metric("Avg AI ms", f"{qos.get('avg_inference_ms', 0):.0f}")
        st.metric("Frames OK", f"{qos.get('frames_accepted', 0)}")
        st.metric("Elapsed", _fmt_elapsed(qos.get('elapsed_s', 0)))

        # Latest result info
        result = controller.latest_result
        if result:
            st.markdown("---")
            st.markdown("**Latest AI**")
            stress_color = _stress_color(result.stress_label)
            st.markdown(
                f"<div style='color:{stress_color}; font-size:1.0rem; font-weight:700'>"
                f"{result.stress_label}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(result.crop_stress_score))
            st.caption(f"Conf: {result.meta.confidence:.0%}")
            if result.meta.low_confidence_reason:
                st.warning(result.meta.low_confidence_reason, icon="⚠️")


def _make_placeholder_frame(w: int = 640, h: int = 360) -> np.ndarray:
    """Generate a dark 'waiting for signal' placeholder image."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(img, "No Signal", (w//2 - 70, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
    cv2.putText(img, "Connect phone to start streaming",
                (w//2 - 160, h//2 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    # Scanline effect
    for y in range(0, h, 4):
        img[y, :] = np.clip(img[y, :].astype(int) - 8, 0, 255)
    return img


def _stress_color(label: str) -> str:
    colors = {
        "Healthy":        "#22c55e",
        "Mild Stress":    "#eab308",
        "Moderate Stress":"#f97316",
        "Severe Stress":  "#ef4444",
        "Critical Stress":"#7f1d1d",
    }
    return colors.get(label, "#94a3b8")


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s//60:02d}:{s%60:02d}"
