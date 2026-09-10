"""
live_ai_panel.py
-----------------
AI inference results panel for the Ground Station tab.

Shows:
    - Stress score + label
    - Disease detections
    - Weed coverage
    - Crop stage
    - RGB vegetation indices (GRVI, VARI, ExG)
    - Model confidence + inference timing
"""

from __future__ import annotations

import streamlit as st


def render_live_ai_panel(controller) -> None:
    """Render live AI inference results."""
    result = controller.latest_result

    st.markdown("#### 🤖 RGB AI Results")
    st.caption("⚠️ RGB Estimates Only — Not equivalent to multispectral NDVI/NDRE")

    if result is None:
        st.info("Waiting for first inference result...", icon="⏳")
        return

    # ── Stress score ──────────────────────────────────────────────
    score = result.crop_stress_score
    label = result.stress_label
    colors = {
        "Healthy":        "#22c55e",
        "Mild Stress":    "#eab308",
        "Moderate Stress":"#f97316",
        "Severe Stress":  "#ef4444",
        "Critical Stress":"#7f1d1d",
    }
    c = colors.get(label, "#94a3b8")

    st.markdown(
        f"""
        <div style='background:rgba(255,255,255,0.04); border-radius:10px; padding:14px;
                    border-left: 4px solid {c}; margin-bottom:12px;'>
          <div style='font-size:0.75rem; color:#94a3b8; text-transform:uppercase;
                      letter-spacing:1px; margin-bottom:6px;'>Crop Stress (RGB AI)</div>
          <div style='font-size:1.6rem; font-weight:800; color:{c};'>{label}</div>
          <div style='font-size:0.8rem; color:#94a3b8; margin-top:4px;'>
            Score: {score:.3f} &nbsp;|&nbsp; Confidence: {result.meta.confidence:.0%}
            &nbsp;|&nbsp; AI: {result.meta.inference_time_ms:.0f}ms
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score)

    # ── Metrics grid ──────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.metric("GRVI", f"{result.grvi:+.3f}",
                  help="Green-Red Vegetation Index (RGB-only)")
    with cols[1]:
        st.metric("VARI", f"{result.vari:+.3f}",
                  help="Visible Atmospherically Resistant Index")
    with cols[2]:
        st.metric("ExG", f"{result.exg:+.3f}",
                  help="Excess Green Index")
    with cols[3]:
        st.metric("Crop Stage", result.crop_stage)

    # ── Weed coverage ─────────────────────────────────────────────
    weed_pct = result.weed_coverage_pct
    cols2 = st.columns(2)
    with cols2[0]:
        st.metric("Weed Coverage", f"{weed_pct:.1f}%",
                  delta="⚠ High" if weed_pct > 30 else "Normal",
                  delta_color="inverse" if weed_pct > 30 else "off")
    with cols2[1]:
        st.metric("Disease Detections", len(result.disease_detections))

    # ── Disease list ──────────────────────────────────────────────
    if result.disease_detections:
        st.markdown("**Disease Detections**")
        for det in result.disease_detections[:5]:
            st.markdown(
                f"🔴 **{det.label}** — confidence: {det.confidence:.0%}",
            )

    # ── Weed detections ───────────────────────────────────────────
    if result.weed_detections:
        st.markdown("**Weed Detections**")
        for det in result.weed_detections[:3]:
            st.markdown(f"🌿 Weed patch — confidence: {det.confidence:.0%}")

    # ── Low confidence warning ────────────────────────────────────
    if result.meta.low_confidence_reason:
        st.warning(
            f"📷 Low Confidence: {result.meta.low_confidence_reason}",
            icon="⚠️",
        )
