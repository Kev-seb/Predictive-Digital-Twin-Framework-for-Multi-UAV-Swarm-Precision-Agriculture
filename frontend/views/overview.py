"""Data-aware landing page for the renovated agriculture workspace."""
from html import escape
import numpy as np
import streamlit as st
from ui import section_header, metric_card, page_link

section_header("Your field. In perspective.", "One workspace for crop health, predictive intelligence, and coordinated field operations.")
idx = st.session_state.get("indices", {})
ready = bool(idx)
source = st.session_state.get("survey_source", "No survey loaded")
st.html('<div class="hero"><div class="hero-art"></div><div class="hero-label">PRECISION STARTS WITH PERSPECTIVE</div><h2>From a field observation<br>to an informed decision.</h2><p>Bring your surveys, crop insights, and UAV operations together. Start with a multispectral survey, then explore what your field is telling you.</p></div>')
actions = st.columns([1, 1, .4])
with actions[0]: page_link("upload", "Upload a survey")
with actions[1]: page_link("twin", "Explore digital twin")

st.html('<div class="section-label"><h2>Field at a glance</h2><span>Current survey · values appear after processing</span></div>')
with st.container(key="overview_metrics"):
    cols = st.columns(4)
values = [
    ("Mean vegetation index", f"{np.nanmean(idx['ndvi']):.3f}" if ready else "—", "NDVI · canopy vigor", "SURVEY" if ready else "AWAITING DATA", "info"),
    ("Area above stress threshold", f"{np.mean(idx['stress_score'] > st.session_state.get('stress_threshold', .55)) * 100:.1f}%" if ready else "—", "Share of survey pixels", "OBSERVED" if ready else "AWAITING DATA", "warn"),
    ("Crop growth stage", st.session_state.get("crop_stage", "Vegetative"), "Selected in field settings", "FIELD CONTEXT", "good"),
    ("Survey source", "Demo field" if source == "Demo survey" else "Uploaded" if ready else "No survey", "4-band multispectral imagery", "SYNTHETIC" if source == "Demo survey" else "READY" if ready else "GET STARTED", "info"),
]
for col, values_ in zip(cols, values):
    with col: metric_card(*values_)

st.html('<div class="section-label"><h2>A clearer path through your field</h2><span>From observation to action</span></div>')
for col, (n, title, desc, slug, action) in zip(st.columns(3), [
    ("01", "Observe & understand", "Explore vegetation indices, stress patterns, and changes across growth stages.", "vegetation", "Open field analysis"),
    ("02", "Model what comes next", "Explore field scenarios, disease spread, and potential treatment outcomes.", "twin", "Open digital twin"),
    ("03", "Plan your intervention", "Compare treatment options and coordinate your next field operation.", "optimizer", "Open input optimizer"),
]):
    with col:
        st.html(f'<div class="workflow-card"><div class="workflow-number">{n}</div><h3>{title}</h3><p>{desc}</p></div>')
        page_link(slug, action)

st.html('<div class="section-label"><h2>Workspace briefing</h2><span>Your current field context</span></div>')
left, right = st.columns([1.5, 1])
with left:
    if ready:
        with st.container(border=True):
            st.markdown("**Current vegetation survey**")
            from frontend_shared import colormap_array
            st.image(colormap_array(idx["ndvi"], "RdYlGn", -1, 1), caption=f"{source} · NDVI: red (low) to green (high)", use_container_width=True)
            page_link("zoning", "Explore management zones")
    else:
        st.html('<div class="insight-panel"><h3>Your survey will appear here</h3><div class="empty-note">Upload a four-band GeoTIFF to begin.<br>Or choose a clearly labeled demo survey to explore the workspace.</div></div>')
        page_link("upload", "Load your first survey")
with right:
    name = escape(st.session_state.get("field_name", "Field Alpha"))
    stage = escape(st.session_state.get("crop_stage", "Vegetative"))
    lat, lon = st.session_state.get("field_lat", 11.), st.session_state.get("field_lon", 79.)
    st.html(f'<div class="insight-panel"><h3>Field context</h3><div class="detail-row"><span>Active field</span><b>{name}</b></div><div class="detail-row"><span>Growth stage</span><b>{stage}</b></div><div class="detail-row"><span>Coordinates</span><b>{lat:.4f}, {lon:.4f}</b></div><div class="detail-row"><span>Data status</span><b>{escape(source)}</b></div></div>')
    page_link("weather", "Check weather & risk")
    page_link("reports", "Prepare a field report")
st.html('<div class="footer-note"><span>GARUDA · Precision agriculture workspace</span><span>Field analysis / Mapping / Operations / Intelligence</span></div>')
