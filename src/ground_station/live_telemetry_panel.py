"""
live_telemetry_panel.py
------------------------
Real-time telemetry display panel for the Ground Station tab.

Displays GPS, attitude, battery, signal, and speed data from the
fused telemetry coming from the phone.
"""

from __future__ import annotations

import streamlit as st


def render_live_telemetry_panel(controller) -> None:
    """Render telemetry gauges and data."""
    tele = controller.latest_telemetry

    st.markdown("#### 📡 Live Telemetry")

    # GPS row
    cols = st.columns(3)
    with cols[0]:
        lat_val = f"{tele.lat:.6f}" if tele.lat else "—"
        st.metric("Latitude", lat_val)
    with cols[1]:
        lon_val = f"{tele.lon:.6f}" if tele.lon else "—"
        st.metric("Longitude", lon_val)
    with cols[2]:
        st.metric("GPS Accuracy", f"{tele.accuracy_m:.1f} m",
                  delta="Good" if tele.accuracy_m < 5 else "Poor",
                  delta_color="normal" if tele.accuracy_m < 5 else "inverse")

    # Attitude row
    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("Heading", f"{tele.heading:.1f}°")
    with cols2[1]:
        st.metric("Pitch", f"{tele.pitch:.1f}°")
    with cols2[2]:
        st.metric("Roll", f"{tele.roll:.1f}°")
    with cols2[3]:
        st.metric("Speed", f"{tele.speed:.2f} m/s")

    # Battery + signal
    cols3 = st.columns(2)
    with cols3[0]:
        bat = tele.battery
        bat_color = "🟢" if bat > 50 else "🟡" if bat > 20 else "🔴"
        st.metric("Battery", f"{bat:.0f}%",
                  delta=bat_color + " OK" if bat > 20 else "⚠ Low")
        st.progress(max(0.0, min(1.0, bat / 100.0)))
    with cols3[1]:
        sig = tele.signal_strength
        bars = "▮" * sig + "▯" * (4 - sig)
        st.metric("Signal", bars, delta=f"{sig}/4 bars")

    # Digital Twin status row
    st.markdown("---")
    cols4 = st.columns(2)
    canopy_height = "—"
    drone_id = "—"
    if hasattr(controller, "digital_twin_state") and controller.digital_twin_state:
        height_val = controller.digital_twin_state.get("canopy_max_height")
        if height_val is not None:
            canopy_height = f"{height_val:.2f} m"
        drone_id = controller.digital_twin_state.get("last_active_drone", "—")

    with cols4[0]:
        st.metric("Twin Max Canopy Height", canopy_height)
    with cols4[1]:
        st.metric("Twin Active Drone ID", drone_id)

    # Phone connection staleness
    if controller.is_active and not controller.phone_connected:
        st.warning("📵 Phone disconnected — last telemetry may be stale", icon="⚠️")
