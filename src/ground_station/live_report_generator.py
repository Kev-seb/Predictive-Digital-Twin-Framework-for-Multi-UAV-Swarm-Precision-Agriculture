"""
live_report_generator.py
-------------------------
Generates a PDF field report for a completed Live Field mission.

Report sections:
    1. Mission summary (date, GPS, sensor source)
    2. Statistics table (frames, stress, distance, area, duration)
    3. GPS track map (embedded)
    4. Stress timeline chart
    5. Disease detection summary
    6. Alert log
    7. Treatment recommendations (based on stress levels)
    8. Weather log summary
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


def render_report_generator(controller) -> None:
    """Render the report generation UI."""
    st.markdown("#### 📋 Field Report Generator")

    try:
        from src.mission import mission_database as db
        missions = db.list_missions(limit=30)
    except Exception:
        st.warning("Mission database not available.")
        return

    completed = [m for m in missions if m.get("status") == "completed"]
    if not completed:
        if controller.is_active:
            st.info("Complete a mission first to generate a report.", icon="ℹ️")
        else:
            st.info("No completed missions available.", icon="ℹ️")
        return

    options = {
        f"{m.get('name', 'Mission')} — {m.get('started_at', '')[:16]}": m["id"]
        for m in completed
    }
    selected_label = st.selectbox("Select Mission for Report", list(options.keys()))
    selected_id = options.get(selected_label)

    if not selected_id:
        return

    mission = next((m for m in completed if m["id"] == selected_id), None)

    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating field report..."):
            pdf_bytes = _generate_pdf(mission, selected_id)
            if pdf_bytes:
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"field_report_{selected_id[:8]}.pdf",
                    mime="application/pdf",
                )
                st.success("Report generated!", icon="✅")
            else:
                st.error("Report generation failed. Check fpdf2 is installed.")


def _generate_pdf(mission: Optional[Dict], mission_id: str) -> Optional[bytes]:
    """Generate a PDF report using fpdf2."""
    try:
        from fpdf import FPDF
        from src.mission import mission_database as db
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()

    # ── Title ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 14, "Live Field Mission Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, f"UAV Crop Stress Intelligence Platform — Live Field Mode", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(34, 197, 94)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Mission summary ─────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Mission Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)

    if mission:
        rows = [
            ("Mission Name",   mission.get("name", "—")),
            ("Started",        mission.get("started_at", "—")[:19] if mission.get("started_at") else "—"),
            ("Ended",          mission.get("ended_at", "—")[:19] if mission.get("ended_at") else "—"),
            ("Field Location", f"{mission.get('field_lat', 0):.5f}, {mission.get('field_lon', 0):.5f}"),
            ("Sensor Source",  mission.get("sensor_source", "phone_rgb").replace("_", " ").title()),
            ("Camera Profile", mission.get("camera_profile", "default")),
            ("Status",         mission.get("status", "—").title()),
        ]
        for label, val in rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(55, 8, label + ":", ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, str(val), ln=True)

    pdf.ln(4)

    # ── Statistics ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Field Statistics (RGB AI Estimates)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "Note: GRVI/VARI/ExG are RGB approximations — not multispectral NDVI.", ln=True)
    pdf.ln(2)

    if mission:
        stats_rows = [
            ("Total Frames",        str(mission.get("total_frames", 0))),
            ("Frames Accepted",     str(mission.get("frames_accepted", 0))),
            ("Frames Rejected",     str(mission.get("frames_rejected", 0))),
            ("Distance Covered",    f"{mission.get('total_distance_m', 0):.1f} m"),
            ("Duration",            f"{int(mission.get('duration_seconds', 0) // 60)}m "
                                    f"{int(mission.get('duration_seconds', 0) % 60)}s"),
            ("Mean Stress Score",   f"{mission.get('mean_stress_score', 0):.4f}"),
            ("Max Stress Score",    f"{mission.get('max_stress_score', 0):.4f}"),
            ("Disease Detections",  str(mission.get("disease_count", 0))),
            ("Critical Alerts",     str(mission.get("alerts_critical", 0))),
            ("Warning Alerts",      str(mission.get("alerts_warning", 0))),
        ]
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 65, 85)
        for label, val in stats_rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(65, 8, label + ":", ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, val, ln=True)

    pdf.ln(4)

    # ── Alerts ──────────────────────────────────────────────────
    try:
        alerts = db.get_alerts(mission_id)
        if alerts:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 10, f"Alert Log ({len(alerts)} alerts)", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(51, 65, 85)
            for a in alerts[:20]:
                sev = a.get("severity", "warning").upper()
                msg = a.get("message", "")[:80]
                ts_str = time.strftime('%H:%M:%S', time.localtime(a.get("timestamp", 0)))
                pdf.multi_cell(0, 7, f"[{ts_str}] [{sev}] {msg}")
    except Exception:
        pass

    pdf.ln(4)

    # ── Treatment recommendation ─────────────────────────────────
    if mission:
        mean_s = mission.get("mean_stress_score", 0)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Treatment Recommendation", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 65, 85)

        if mean_s < 0.2:
            rec = "Crop appears healthy. Continue monitoring. No immediate intervention required."
        elif mean_s < 0.4:
            rec = ("Mild stress detected. Check irrigation schedule and soil moisture. "
                   "Consider foliar nutrition if persists.")
        elif mean_s < 0.6:
            rec = ("Moderate stress detected. Immediate irrigation recommended. "
                   "Inspect for pest or disease. Apply appropriate treatment.")
        elif mean_s < 0.8:
            rec = ("Severe stress — urgent field inspection required. "
                   "Likely combined water + nutrient deficit or active disease outbreak. "
                   "Prioritize affected zones for treatment.")
        else:
            rec = ("CRITICAL STRESS — Immediate intervention. "
                   "Coordinate with agronomist. Consider emergency irrigation and "
                   "targeted pesticide/fungicide application.")

        pdf.multi_cell(0, 8, rec)

    pdf.ln(4)

    # ── Footer ───────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8,
             f"Generated by UAV Crop Stress Intelligence Platform — Live Field Mode — {generated}",
             align="C")

    return pdf.output()
