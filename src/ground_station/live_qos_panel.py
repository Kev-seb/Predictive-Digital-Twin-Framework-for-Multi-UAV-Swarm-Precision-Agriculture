"""
live_qos_panel.py
------------------
Quality of Service monitoring panel for Live Field Mode.

Shows:
    - Streaming mode (WebRTC / JPEG)
    - FPS
    - Network latency estimate
    - CPU / memory usage
    - Inference timing histogram
    - Frame accept/reject rate
"""

from __future__ import annotations

import streamlit as st


def render_live_qos_panel(controller) -> None:
    """Render QoS monitoring metrics."""
    st.markdown("#### 📊 Quality of Service")

    qos = controller.qos_metrics

    # ── Key metrics ────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        fps = qos.get("fps", 0)
        fps_delta = "✓" if fps >= 3 else "⚠ Low"
        st.metric("Streaming FPS", f"{fps:.1f}",
                  delta=fps_delta,
                  delta_color="normal" if fps >= 3 else "inverse")
    with cols[1]:
        ai_ms = qos.get("avg_inference_ms", 0)
        st.metric("Avg AI Latency", f"{ai_ms:.0f} ms",
                  delta="Fast" if ai_ms < 200 else "Slow",
                  delta_color="normal" if ai_ms < 200 else "inverse")
    with cols[2]:
        total = max(qos.get("frames_total", 1), 1)
        accepted = qos.get("frames_accepted", 0)
        accept_rate = accepted / total * 100
        st.metric("Accept Rate", f"{accept_rate:.0f}%",
                  delta="Good" if accept_rate > 70 else "Check camera",
                  delta_color="normal" if accept_rate > 70 else "inverse")
    with cols[3]:
        phone = qos.get("phone_connected", False)
        st.metric("Phone Link",
                  "🟢 Live" if phone else "🔴 Offline")

    # ── System resource snapshot ───────────────────────────────────
    try:
        import psutil
        cpu_pct  = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        ram_pct  = ram_info.percent

        cols2 = st.columns(2)
        with cols2[0]:
            st.metric("CPU Usage", f"{cpu_pct:.0f}%",
                      delta_color="inverse" if cpu_pct > 80 else "off")
            st.progress(cpu_pct / 100.0)
        with cols2[1]:
            st.metric("RAM Usage", f"{ram_pct:.0f}%",
                      delta_color="inverse" if ram_pct > 85 else "off")
            st.progress(ram_pct / 100.0)
    except ImportError:
        pass

    # ── Frame counters ─────────────────────────────────────────────
    st.caption(
        f"Frames processed: {qos.get('frames_total', 0)} total  |  "
        f"{qos.get('frames_accepted', 0)} accepted  |  "
        f"Elapsed: {_fmt_elapsed(qos.get('elapsed_s', 0))}"
    )


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
