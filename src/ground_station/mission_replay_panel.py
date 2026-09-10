"""
mission_replay_panel.py
------------------------
Mission replay panel for the Ground Station tab.

Allows loading and replaying any completed mission:
    - Select mission from list
    - Scrub through observations
    - Play at adjustable speed
    - Rewatch annotated frames
    - View per-frame AI results
"""

from __future__ import annotations

import streamlit as st


def render_mission_replay_panel(controller) -> None:
    """Render the mission replay controls."""
    st.markdown("#### ▶ Mission Replay")

    # ── Load missions ─────────────────────────────────────────────
    try:
        from src.mission import mission_database as db
        missions = db.list_missions(limit=30)
    except Exception:
        st.warning("Mission database not available.", icon="⚠️")
        return

    if not missions:
        st.info("No completed missions yet. Run a Live Field session to record data.")
        return

    completed = [m for m in missions if m.get("status") == "completed"]
    if not completed:
        st.info("No completed missions found.", icon="ℹ️")
        return

    # ── Mission selector ──────────────────────────────────────────
    mission_options = {
        f"{m.get('name', 'Mission')} — {m.get('started_at', '')[:16]}": m["id"]
        for m in completed
    }
    selected_label = st.selectbox("Select Mission", list(mission_options.keys()))
    selected_id = mission_options.get(selected_label)

    if not selected_id:
        return

    # Show mission summary
    mission = next((m for m in completed if m["id"] == selected_id), None)
    if mission:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Frames", mission.get("total_frames", 0))
        with cols[1]:
            st.metric("Mean Stress", f"{mission.get('mean_stress_score', 0):.3f}")
        with cols[2]:
            dist_km = mission.get("total_distance_m", 0) / 1000
            st.metric("Distance", f"{dist_km:.2f} km")
        with cols[3]:
            dur_s = int(mission.get("duration_seconds", 0))
            st.metric("Duration", f"{dur_s // 60}m {dur_s % 60}s")

    st.divider()

    # ── Replay controls ───────────────────────────────────────────
    col1, col2 = st.columns([3, 1])

    with col2:
        speed = st.slider("Replay Speed", 0.5, 10.0, 2.0, 0.5,
                          format="%.1fx")

    with col1:
        if controller.is_replay_active:
            progress = controller.replay_progress
            st.progress(progress, text=f"Replaying... {progress:.0%}")
            if st.button("⏹ Stop Replay", type="secondary"):
                controller.stop_replay()
        else:
            if st.button("▶ Start Replay", type="primary"):
                ok = controller.start_replay(selected_id, speed=speed)
                if not ok:
                    st.error("Failed to load mission observations.")
                else:
                    st.success(f"Replaying mission at {speed:.1f}x speed...")

    # ── View observations table ───────────────────────────────────
    with st.expander("🔍 Browse Observations"):
        try:
            from src.mission import mission_database as db
            obs = db.get_observations(selected_id, limit=100)
            if obs:
                import pandas as pd
                df = pd.DataFrame([{
                    "Seq":      o.get("sequence_num"),
                    "Lat":      round(o.get("lat", 0), 5),
                    "Lon":      round(o.get("lon", 0), 5),
                    "Stress":   round(o.get("crop_stress_score", 0) or 0, 3),
                    "Stage":    o.get("crop_stage", "—"),
                    "GRVI":     round(o.get("grvi", 0) or 0, 3),
                    "VARI":     round(o.get("vari", 0) or 0, 3),
                    "Quality":  round(o.get("quality_score", 0) or 0, 2),
                } for o in obs])
                st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Could not load observations: {e}")
