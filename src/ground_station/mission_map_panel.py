"""
mission_map_panel.py
---------------------
GIS map panel for the Ground Station tab.

Shows:
    - Live GPS track on a Folium map
    - Stress-coloured observation markers
    - Alert pins
    - Field coverage polygon
    - Mission area estimation
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


def render_mission_map_panel(controller) -> None:
    """Render the live GPS track map."""
    st.markdown("#### 🗺 Live Mission Map")

    if not FOLIUM_AVAILABLE:
        st.warning("Install folium and streamlit-folium for the map panel.", icon="⚠️")
        return

    tele = controller.latest_telemetry
    alerts = controller.latest_alerts

    # Default location
    center_lat = tele.lat if tele.lat else 11.0
    center_lon = tele.lon if tele.lon else 79.0
    zoom = 18 if tele.lat else 12

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="Esri.WorldImagery",
    )

    # ── Current position marker ────────────────────────────────────
    if tele.lat and tele.lon:
        folium.Marker(
            [tele.lat, tele.lon],
            icon=folium.Icon(color="green", icon="mobile", prefix="fa"),
            popup=folium.Popup(
                f"<b>Current Position</b><br>"
                f"Lat: {tele.lat:.6f}<br>"
                f"Lon: {tele.lon:.6f}<br>"
                f"Heading: {tele.heading:.0f}°<br>"
                f"Battery: {tele.battery:.0f}%",
                max_width=200,
            ),
            tooltip="📱 Phone Position",
        ).add_to(m)

    # ── GPS track from database ────────────────────────────────────
    if controller.mission_id:
        try:
            from src.mission import mission_database as db
            track = db.get_gps_track(controller.mission_id)
            if len(track) >= 2:
                points = [[p["lat"], p["lon"]] for p in track]
                folium.PolyLine(
                    points,
                    color="#22c55e",
                    weight=3,
                    opacity=0.8,
                    tooltip="GPS Track",
                ).add_to(m)

            # Stress-coloured observation markers (subsample for performance)
            obs = db.get_observations(controller.mission_id, limit=200)
            step = max(1, len(obs) // 50)
            for o in obs[::step]:
                stress = o.get("crop_stress_score", 0)
                color = _stress_hex(stress)
                lat, lon = o.get("lat", 0), o.get("lon", 0)
                if lat and lon:
                    folium.CircleMarker(
                        [lat, lon],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.8,
                        popup=f"Stress: {stress:.2f} | Stage: {o.get('crop_stage', '?')}",
                    ).add_to(m)
        except Exception:
            pass

    # ── Alert markers ─────────────────────────────────────────────
    for alert in alerts[-20:]:
        lat = alert.get("lat", 0)
        lon = alert.get("lon", 0)
        if lat and lon:
            icon_color = "red" if alert["severity"] == "critical" else "orange"
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color=icon_color, icon="warning", prefix="fa"),
                tooltip=alert["message"][:60],
            ).add_to(m)

    # ── Render ────────────────────────────────────────────────────
    st_folium(m, height=400, use_container_width=True)

    # Stats below map
    if controller.is_active:
        qos = controller.qos_metrics
        st.caption(
            f"📍 Track: {_count_gps_points(controller.mission_id)} points  |  "
            f"⏱ {int(qos.get('elapsed_s', 0) // 60)}m {int(qos.get('elapsed_s', 0) % 60)}s  |  "
            f"⚠ {len(alerts)} alerts"
        )


def _stress_hex(stress: float) -> str:
    if stress < 0.2:   return "#22c55e"
    if stress < 0.4:   return "#eab308"
    if stress < 0.6:   return "#f97316"
    if stress < 0.8:   return "#ef4444"
    return "#7f1d1d"


def _count_gps_points(mission_id: str) -> int:
    try:
        from src.mission import mission_database as db
        track = db.get_gps_track(mission_id)
        return len(track)
    except Exception:
        return 0
