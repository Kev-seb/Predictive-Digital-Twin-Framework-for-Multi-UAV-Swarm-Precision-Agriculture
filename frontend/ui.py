"""Presentation components shared by every renovated page."""
from datetime import datetime
from html import escape
from pathlib import Path
import streamlit as st
from navigation import GROUPS, ITEMS

HERE = Path(__file__).resolve().parent
STAGES = ["Nursery", "Vegetative", "Flowering", "Mature"]

def apply_theme():
    st.html("<style>" + (HERE / "assets/theme.css").read_text(encoding="utf-8") + "</style>")

def page_link(slug, label=None):
    _, title, icon, file = ITEMS[slug]
    st.page_link("views/" + file, label=label or title, icon=f":material/{icon}:", use_container_width=True)

def render_shell(active):
    with st.sidebar:
        st.html('<div class="brand"><div class="brand-mark">G<span></span></div><div><strong>GARUDA<span class="brand-period">.</span></strong><small>FIELD INTELLIGENCE</small></div></div>')
        with st.container(key="upload_action"):
            page_link("upload", "Upload survey")
        for group, items in GROUPS.items():
            st.html(f'<div class="nav-label">{escape(group)}</div>')
            for slug, title, icon, file in items:
                with st.container(key="nav_" + slug + ("_active" if active == slug else "")):
                    page_link(slug)
        st.divider()
        with st.expander("Field settings", icon=":material/settings:"):
            st.text_input("Field name", value="Field Alpha", key="field_name")
            st.selectbox("Crop growth stage", STAGES, index=1, key="crop_stage")
            st.slider("Stress threshold", 0.30, 0.80, 0.55, 0.05, key="stress_threshold")
            st.slider("Field grid size", 3, 10, 5, key="grid_size")
            st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=11.0, format="%.4f", key="field_lat")
            st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=79.0, format="%.4f", key="field_lon")
        st.html('<div class="sidebar-note"><span class="status-dot"></span> Precision agriculture workspace<small>Observe. Understand. Act.</small></div>')
    group, title, _, _ = ITEMS[active]
    ready = bool(st.session_state.get("indices"))
    source = st.session_state.get("survey_source", "Survey loaded" if ready else "No survey loaded")
    with st.container(key="workspace_topbar"):
        st.html(f'<div class="workspace-bar"><div><span class="workspace-label">WORKSPACE</span><b>{escape(st.session_state.get("field_name", "Field Alpha"))}</b><span class="bar-separator">/</span><span>{escape(group)}</span></div><div><span class="source-chip {"ready" if ready else ""}"><i></i>{escape(source)}</span><span class="bar-date">{datetime.now():%d %b %Y}</span></div></div>')

def render_sidebar():
    # Widgets belong to the entrypoint, preserving their values across routes.
    from frontend_shared import get_shared_state
    state = get_shared_state()
    lat, lon = st.session_state.get("field_lat", 11.0), st.session_state.get("field_lon", 79.0)
    state["HOME_LAT"], state["HOME_LON"] = lat, lon
    return (st.session_state.get("crop_stage", "Vegetative"), st.session_state.get("stress_threshold", .55), st.session_state.get("grid_size", 5), lat, lon)

def render_header_status():
    """The entrypoint renders the shared workspace bar once."""

def section_header(title, subtitle="", category=""):
    first = not st.session_state.get("_page_heading_rendered", False)
    st.session_state["_page_heading_rendered"] = True
    group = ITEMS.get(st.session_state.get("_active_page", "overview"), ("Workspace",))[0]
    tag = "h1" if first else "h2"
    eyebrow = f'<div class="eyebrow">{escape(group)}</div>' if first else ""
    st.html(f'<div class="page-heading {"primary-heading" if first else "subsection-heading"}">{eyebrow}<{tag}>{escape(title)}</{tag}><p>{escape(subtitle)}</p></div>')

def metric_card(label, value, sub="", badge="", badge_kind="info"):
    kind = badge_kind if badge_kind in {"good", "warn", "crit", "info"} else "info"
    badge_html = f'<span class="gd-card-badge gd-badge-{kind}">{escape(str(badge))}</span>' if badge else ""
    st.html(f'<div class="gd-card metric-tile"><div class="gd-card-label">{escape(str(label))}</div><div class="gd-card-value">{escape(str(value))}</div><div class="gd-card-sub">{escape(str(sub))}</div>{badge_html}</div>')

def status_card(title, value, sub="", kind="info"):
    metric_card(title, value, sub, kind.upper(), kind)
