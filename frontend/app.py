"""Separate entry point. Run from the repository root with frontend/run.ps1."""
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND))

import streamlit as st

st.set_page_config(page_title="Garuda | Field Intelligence", page_icon=":material/eco:", layout="wide", initial_sidebar_state="auto")

from navigation import GROUPS
from ui import apply_theme, render_shell

routes = {}
for group, items in GROUPS.items():
    routes[group] = [st.Page("views/" + file, title=title, icon=f":material/{icon}:", url_path=slug, default=slug == "overview") for slug, title, icon, file in items]
page = st.navigation(routes, position="hidden")
active = page.url_path or "overview"
st.session_state["_active_page"] = active
st.session_state["_page_heading_rendered"] = False
apply_theme()
render_shell(active)

# Start telemetry only on screens that use it; the overview needs no hardware.
if active in {"spatial", "uav", "swarm"}:
    from frontend_shared import init_mavlink_telemetry_service
    init_mavlink_telemetry_service()

page.run()
