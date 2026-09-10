"""
live_field_tab.py
------------------
Main assembly for the "🌿 Live Field Mode" Streamlit tab.

Layout:
    ┌────────────────────────────────────────────────────────────┐
    │ MODE SELECTOR: Research Mode | Live Field Mode             │
    │ CONNECTION STATUS + QR CODE                                 │
    ├──────────────────────────┬─────────────────────────────────┤
    │  Mission Control Panel   │  Live Camera Feed               │
    │  - Start / Stop mission  │  (annotated)                    │
    │  - Field settings        │                                 │
    ├──────────────────────────┴─────────────────────────────────┤
    │ TABS:                                                       │
    │  [AI Results] [Telemetry] [Map] [QoS] [Replay] [Report]   │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import streamlit as st


# ── Cached controller (singleton per session) ──────────────────────────

@st.cache_resource
def _get_controller():
    """Returns a singleton LiveFieldController for this Streamlit session."""
    from src.live_mode.live_field_controller import LiveFieldController
    controller = LiveFieldController()
    return controller


@st.cache_resource
def _get_api_port():
    """Start the FastAPI server once and return the port it's listening on."""
    controller = _get_controller()
    try:
        from src.mobile_api.mobile_api_server import start_mobile_api_server
        port = start_mobile_api_server(controller, port=8080)
        return port
    except Exception as e:
        st.warning(f"Mobile API server failed to start: {e}")
        return None


def render_live_field_tab() -> None:
    """
    Main entry point for the Live Field Mode tab.
    Call this from dashboard.py inside the appropriate tab.
    """
    # Ensure controller and server are running
    controller = _get_controller()
    api_port = _get_api_port()

    # ── Header + connection status ─────────────────────────────────
    col_title, col_status = st.columns([2, 1])
    with col_title:
        st.markdown(
            """
            <h2 style='margin:0; font-size:1.4rem; font-weight:800;'>
              🌿 Live Field Mode
            </h2>
            <p style='margin:0; color:#94a3b8; font-size:0.8rem;'>
              Smartphone-powered ground station &nbsp;·&nbsp; AI crop scouting
            </p>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        if controller.is_active:
            if controller.phone_connected:
                st.success("🟢 Mission Active — Phone Online", icon="📱")
            else:
                st.warning("🟡 Mission Active — Waiting for phone", icon="📡")
        elif controller.is_replay_active:
            st.info("▶ Replaying mission...", icon="⏩")
        else:
            st.info("⚡ Ready — Start a mission below", icon="🌿")

    st.divider()

    # ── QR Code + Connection guide ─────────────────────────────────
    with st.expander("📱 Connect Android Phone — Scan QR Code", expanded=not controller.is_active):
        _render_qr_section(api_port)

    # ── Mission control + Video (side by side) ────────────────────
    ctrl_col, video_col = st.columns([1, 2])

    with ctrl_col:
        _render_mission_control(controller)

    with video_col:
        from src.ground_station.live_video_panel import render_live_video_panel
        render_live_video_panel(controller)

    st.divider()

    # ── Alert ticker ──────────────────────────────────────────────
    _render_alert_ticker(controller)

    # ── Sub-tabs ──────────────────────────────────────────────────
    subtab_labels = [
        "🤖 AI Results",
        "📡 Telemetry",
        "🗺 Map",
        "📊 QoS",
        "▶ Replay",
        "📋 Report",
    ]
    subtabs = st.tabs(subtab_labels)

    with subtabs[0]:
        from src.ground_station.live_ai_panel import render_live_ai_panel
        render_live_ai_panel(controller)

    with subtabs[1]:
        from src.ground_station.live_telemetry_panel import render_live_telemetry_panel
        render_live_telemetry_panel(controller)

    with subtabs[2]:
        from src.ground_station.mission_map_panel import render_mission_map_panel
        render_mission_map_panel(controller)

    with subtabs[3]:
        from src.ground_station.live_qos_panel import render_live_qos_panel
        render_live_qos_panel(controller)

    with subtabs[4]:
        from src.ground_station.mission_replay_panel import render_mission_replay_panel
        render_mission_replay_panel(controller)

    with subtabs[5]:
        from src.ground_station.live_report_generator import render_report_generator
        render_report_generator(controller)

    # ── Auto-refresh during active mission ────────────────────────
    if controller.is_active or controller.is_replay_active:
        time.sleep(0.05)  # Brief yield
        st.rerun()


# ── Mission control panel ──────────────────────────────────────────────

def _render_mission_control(controller) -> None:
    """Mission start/stop controls with field settings."""
    st.markdown("#### ⚙ Mission Control")

    with st.form("mission_form", clear_on_submit=False):
        mission_name = st.text_input(
            "Mission Name",
            value=f"Mission {time.strftime('%b %d %H:%M')}",
            placeholder="e.g. North Field Scout",
        )
        field_name = st.text_input("Field Name", value="Field Alpha")

        col1, col2 = st.columns(2)
        with col1:
            field_lat = st.number_input("Approx. Lat", value=11.0, format="%.5f")
        with col2:
            field_lon = st.number_input("Approx. Lon", value=79.0, format="%.5f")

        camera_profile = st.selectbox(
            "Camera Profile",
            ["default", "samsung_a52", "pixel_6", "custom"],
            help="Select your phone model or use calibration to create a custom profile",
        )

        col_start, col_stop = st.columns(2)
        with col_start:
            start_pressed = st.form_submit_button(
                "▶ Start Mission",
                disabled=controller.is_active,
                type="primary",
            )
        with col_stop:
            stop_pressed = st.form_submit_button(
                "■ Stop Mission",
                disabled=not controller.is_active,
            )

    if start_pressed and not controller.is_active:
        mission_id = controller.start_mission(
            name=mission_name,
            field_lat=field_lat,
            field_lon=field_lon,
            field_name=field_name,
            camera_profile=camera_profile,
        )
        st.success(f"Mission started! ID: {mission_id[:8]}...", icon="✅")
        st.rerun()

    if stop_pressed and controller.is_active:
        mid = controller.stop_mission()
        st.info(f"Mission {(mid or '')[:8]} completed and saved.", icon="💾")
        st.rerun()

    # Pause controls
    if controller.is_active:
        p_col, r_col = st.columns(2)
        with p_col:
            if st.button("⏸ Pause", disabled=controller.is_paused):
                controller.pause_mission()
                st.rerun()
        with r_col:
            if st.button("▶ Resume", disabled=not controller.is_paused):
                controller.resume_mission()
                st.rerun()

        # Current mission stats
        qos = controller.qos_metrics
        st.metric("Elapsed", _fmt_elapsed(qos.get("elapsed_s", 0)))
        st.metric("Frames OK", qos.get("frames_accepted", 0))


# ── QR Code section ────────────────────────────────────────────────────

def _render_qr_section(api_port: Optional[int]) -> None:
    """Show connection QR code and manual URL."""
    if api_port is None:
        st.error("Mobile API server failed to start. Check logs.")
        return

    controller = _get_controller()

    # Automatically set default if not manually configured
    if "tunnel_url_override" not in st.session_state:
        st.session_state["tunnel_url_override"] = ""

    auto_url = getattr(controller, "public_tunnel_url", None)
    if not st.session_state["tunnel_url_override"] and auto_url:
        st.session_state["tunnel_url_override"] = auto_url

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Allow setting a public tunnel URL override (e.g. localtunnel, ngrok)
    tunnel_url = st.text_input(
        "🔗 Public Tunnel URL (e.g. https://afraid-bees-hope.loca.lt)",
        help="If your phone is on a different network, use a tunnel service (like localtunnel or ngrok) and enter the public URL here to update the QR code.",
        key="tunnel_url_override"
    )

    if tunnel_url.strip():
        pwa_url = f"{tunnel_url.strip().rstrip('/')}/mobile"
    else:
        pwa_url = f"http://{local_ip}:{api_port}/mobile"

    col1, col2 = st.columns([1, 2])
    with col1:
        # Try to render QR code
        try:
            import qrcode
            import io
            from PIL import Image
            qr = qrcode.QRCode(box_size=5, border=2)
            qr.add_data(pwa_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, width=180, caption="Scan to open companion app")
        except ImportError:
            st.info("Install `qrcode[pil]` to generate QR code")

    with col2:
        st.markdown(f"**Ground Station URL:**")
        st.code(pwa_url, language=None)
        
        if tunnel_url.strip():
            st.info("💡 **Tunnel Active:** Ensure you have bypassed the localtunnel/ngrok landing page on your phone's browser before tapping Connect.")

        st.markdown("""
        **Setup instructions:**
        1. Connect phone to the **same WiFi network** (or use the Public Tunnel URL if on different networks/cellular)
        2. Scan the QR code or open the URL in Chrome/Safari
        3. Add to Home Screen for best experience (PWA)
        4. Tap **Connect** → **Start Mission**
        
        **Supported transports:**
        - ⚡ WebRTC (best): real-time low-latency video
        - 📷 JPEG fallback: ~5 FPS, works on all browsers
        """)


# ── Alert ticker ───────────────────────────────────────────────────────

def _render_alert_ticker(controller) -> None:
    """Show the latest alerts in a compact banner."""
    alerts = controller.latest_alerts
    if not alerts:
        return

    recent = alerts[-3:]  # Show last 3
    for alert in reversed(recent):
        if alert["severity"] == "critical":
            st.error(f"🔴 {alert['message']}", icon="⚠️")
        else:
            st.warning(f"🟡 {alert['message']}", icon="⚡")


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m:02d}m {sec:02d}s"
