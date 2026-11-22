"""
overview.py - the dashboard's home page: live mission control for the predictive twin.

Reads the same /ue5/frame the UE5 PaddyTwin level uses, so the numbers match the Unreal panel:
KPIs (harvest now / in 7 days, worst disease, fungicide saved, twin error, missions),
TWIN / REALITY / FORECAST +7 DAYS field maps, the narration, and the actions.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
import streamlit as st

from src.dashboard.ui_shell import C

API = f"http://{os.getenv('UAV_TWIN_API_HOST', '127.0.0.1')}:{os.getenv('UAV_TWIN_API_PORT', '8008')}"
GRID = 20

RISK_SCALE = [[0.0, "#1F7A3A"], [0.3, "#8DBE3A"], [0.55, "#E0B83C"], [0.78, "#A2592A"], [1.0, "#5E1A12"]]
LAYERS = {
    "Crop health": ("health", RISK_SCALE, 0.0, 1.0),
    "NDVI": ("ndvi", "Greens", 0.2, 0.9),
    "Stress": ("stress", "YlOrBr", 0.0, 0.4),
    "Fungus": ("fungus", "OrRd", 0.0, 0.6),
    "Uncertainty": ("std", "Greys", 0.0, 0.12),
}


# ----------------------------------------------------------------- API
def _get(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: dict, timeout: float = 8.0):
    req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _act(label: str, path: str, body: dict) -> None:
    try:
        _post(path, body)
        st.toast(f"{label}: started", icon=":material/check_circle:")
    except urllib.error.HTTPError as e:
        st.toast("Busy: wait for the current step, or press Stop" if e.code == 409 else f"{label}: HTTP {e.code}",
                 icon=":material/hourglass_top:")
    except Exception:
        st.toast(f"Cannot reach the twin API at {API}", icon=":material/cloud_off:")


# ----------------------------------------------------------------- helpers
def _grid(field: dict, prefix: str, kind: str) -> np.ndarray | None:
    def arr(name):
        v = field.get(name)
        return None if not v else np.asarray(v, dtype=float).reshape(GRID, GRID)
    s, f = arr(f"{prefix}stress"), arr(f"{prefix}fungus")
    if kind == "health":
        if s is None or f is None:
            return None
        return np.clip(np.maximum(np.clip(s * 2.5, 0, 1) * 0.6, np.clip(f * 1.6, 0, 1)), 0, 1)
    if kind == "std":
        return arr("forecast_std" if prefix == "forecast_" else ("stress_std" if prefix == "" else "__none__"))
    return arr(f"{prefix}{kind}")


def _heatmap(z: np.ndarray, scale, zmin: float, zmax: float, title_color: str):
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(z=z, colorscale=scale, zmin=zmin, zmax=zmax, showscale=False,
                               xgap=1, ygap=1, hovertemplate="row %{y}, col %{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False, constrain="domain"),
                      yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"))
    return fig


def _kpi(label: str, value: str, unit: str = "", foot: str = "") -> str:
    u = f"<small>{unit}</small>" if unit else ""
    return (f'<div class="pt-kpi"><div class="pt-kpi-label">{label}</div>'
            f'<div class="pt-kpi-value">{value}{u}</div><div class="pt-kpi-foot">{foot}</div></div>')


def _offline() -> None:
    with st.container(border=True):
        st.markdown(f"#### :material/cloud_off: The twin API is not running")
        st.markdown(f"The live overview reads the twin at `{API}`. Start it, then this page connects by itself.")
        st.code("start_demo.bat          # Windows, from the repository folder\n"
                "python digital_twin_api.py   # or directly", language="bash")
        st.caption("Every other page in the sidebar works without the API.")


def _open_unreal() -> None:
    try:
        from src.ue5_integration.ue_launcher import open_paddy_twin
        with st.spinner("Contacting Unreal Engine..."):
            st.session_state["ue_open_msg"] = open_paddy_twin()
    except Exception as e:
        st.session_state["ue_open_msg"] = (False, f"Could not open Unreal Engine: {e}")


# ----------------------------------------------------------------- page
def render() -> None:
    st.markdown('<div class="pt-crumb">Live operations &nbsp;›&nbsp; <b>Live twin &amp; drones</b></div>', unsafe_allow_html=True)
    head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
    with head_l:
        st.markdown('<div class="pt-title">Field Alpha &nbsp;·&nbsp; Paddy rice</div>'
                    '<div class="pt-sub">The twin estimates the field from drone scouting, forecasts the next 7 days, '
                    'and sends the drones to spray only where it matters.</div>', unsafe_allow_html=True)
    with head_r:
        t_col, b_col = st.columns([1, 1.4], vertical_alignment="center")
        live = t_col.toggle("Live updates", value=st.query_params.get("live", "1") != "0",
                            help="Refresh every 2 seconds (open with ?live=0 to start paused)")
        b_col.button("Open in Unreal Engine", icon=":material/sports_esports:", type="primary",
                     use_container_width=True, on_click=_open_unreal,
                     help="Opens the 3D PaddyTwin level in Unreal Engine on this PC (or brings it to the front) "
                          "and starts the paddy twin there. Actions on this page then move the drones in Unreal too.")
    msg = st.session_state.pop("ue_open_msg", None)
    if msg:
        (st.success if msg[0] else st.error)(msg[1], icon=":material/sports_esports:")
    _live_panel(live)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown("##### How the twin works")
    steps = [("01", "Sense", "Drones fly multispectral scouting passes over the paddy."),
             ("02", "Assimilate", "Every frame updates the twin's estimate of stress, disease and moisture."),
             ("03", "Predict", "The twin simulates the next 7 days with and without spraying."),
             ("04", "Act", "Only zones above the action threshold are sprayed, by the swarm."),
             ("05", "Verify", "Later passes confirm recovery; twin-vs-reality error keeps falling.")]
    st.markdown('<div class="pt-flow">' + "".join(
        f'<div class="pt-step"><div class="pt-step-n">{n}</div><div class="pt-step-t">{t}</div>'
        f'<div class="pt-step-d">{d}</div></div>' for n, t, d in steps) + "</div>", unsafe_allow_html=True)


def _live_panel(live: bool) -> None:
    @st.fragment(run_every=2.0 if live else None)
    def panel():
        try:
            frame = _get("/ue5/frame?since=-1")
        except Exception:
            _offline()
            return
        field, m, status = frame.get("field") or {}, frame.get("metrics") or {}, frame.get("status") or {}
        _status_row(frame, field, m, status)
        _kpis(m, field)
        left, right = st.columns([2.35, 1], gap="medium")
        with left:
            _maps(field)
        with right:
            _controls(frame)
            _error_chart(m)
    panel()


def _status_row(frame, field, m, status) -> None:
    running = frame.get("demo_running")
    drones = (frame.get("drones") or {}).get("drones") or {}
    flying = sum(1 for p in drones.values() if p and p[2] > 1.5)
    spraying = sum(1 for p in drones.values() if p and p[3])
    rain = (frame.get("weather") or {}).get("precipitation_mm", 0.0)
    pills = [
        (C["accent"], "Twin API connected"),
        (C["amber"] if running else C["muted"], "Demo running" if running else "Demo idle"),
        (C["blue"], f"Day {m.get('sim_day', field.get('sim_day', '-'))} · {m.get('growth_stage', field.get('growth_stage', ''))}"),
        (C["violet"] if flying else C["muted"], f"{flying}/{len(drones) or 3} drones airborne" + (f" · {spraying} spraying" if spraying else "")),
        (C["blue"] if rain > 1 else C["muted"], f"Rain {rain:.1f} mm"),
    ]
    st.markdown("".join(f'<span class="pt-pill"><span class="pt-dot" style="background:{c}"></span>{t}</span>'
                        for c, t in pills), unsafe_allow_html=True)
    text = status.get("text") or "Waiting for the first step. Press Start demo or Next day."
    st.markdown(f'<div class="pt-status" style="margin:.6rem 0 .9rem 0">{text}</div>', unsafe_allow_html=True)


def _kpis(m, field) -> None:
    if not m:
        return
    h, f, ms, e = m["harvest_t_ha"], m["fungicide"], m["missions"], m["twin_error"]
    wi = field.get("what_if") or {}
    d_no = h["in_7d_no_spray"] - h["now"]
    d_sp = h["in_7d_if_sprayed"] - h["in_7d_no_spray"]
    cards = [
        _kpi("Expected harvest", f"{h['now']:.2f}", "t/ha",
             f"7 days, no spray <span class='{'pt-down' if d_no < 0 else 'pt-up'}'>{d_no:+.2f}</span>"),
        _kpi("If sprayed (7 days)", f"{h['in_7d_if_sprayed']:.2f}", "t/ha",
             f"<span class='pt-up'>{d_sp:+.2f}</span> vs no spray" + (
                 f" · zones {', '.join(str(z) for z in h.get('spray_zones') or [])}" if h.get("spray_zones") else "")),
        _kpi("Peak disease, 7 days",
             f"{wi['no_spray']['fungus_peak'] * 100:.0f}%" if wi else "–", "",
             f"<span class='pt-up'>{wi['if_sprayed']['fungus_peak'] * 100:.0f}%</span> if sprayed" if wi else ""),
        _kpi("Fungicide saved", f"{f['saved_pct']:.0f}%" if f.get("blanket_kg") else "–", "",
             f"{f['used_kg']:.1f} kg vs {f['blanket_kg']:.0f} kg whole-field" if f.get("blanket_kg") else "none used yet"),
        _kpi("Twin vs reality error", f"{e['now']:.3f}" if e.get("now") is not None else "–", "",
             f"from {e['first']:.3f} at start" if e.get("first") is not None else ""),
        _kpi("Missions", f"{ms['scout'] + ms['spray']}", "",
             f"{ms['scout']} scout · {ms['spray']} spray · {ms['flight_min']:.0f} min"),
    ]
    cols = st.columns(len(cards), gap="small")
    for c, html in zip(cols, cards):
        c.markdown(html, unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)


def _maps(field) -> None:
    with st.container(border=True):
        top_l, top_r = st.columns([1, 2.2], vertical_alignment="center")
        top_l.markdown("**Field maps** &nbsp;<span style='color:#8D9CB6;font-size:.85rem'>20 × 20 cells of 5 m</span>",
                       unsafe_allow_html=True)
        layer = top_r.segmented_control("Layer", list(LAYERS), default="Crop health", key="ov_layer",
                                        label_visibility="collapsed") or "Crop health"
        kind, scale, zmin, zmax = LAYERS[layer]
        cols = st.columns(3, gap="small")
        views = [("", "TWIN", "what the twin believes now", C["blue"]),
                 ("truth_", "REALITY", "hidden ground truth", C["amber"]),
                 ("forecast_", "FORECAST +7 DAYS", "if nothing is done", C["violet"])]
        for col, (prefix, name, sub, colr) in zip(cols, views):
            with col:
                st.markdown(f'<div class="pt-fieldlabel" style="color:{colr}">{name}</div>'
                            f'<div class="pt-fieldsub">{sub}</div>', unsafe_allow_html=True)
                z = _grid(field, prefix, kind)
                if z is None:
                    st.caption("Not available for this field" if kind == "std" and prefix == "truth_" else "No data yet")
                else:
                    st.plotly_chart(_heatmap(z, scale, zmin, zmax, colr), use_container_width=True,
                                    config={"displayModeBar": False}, key=f"ov_map_{prefix}{kind}")
        if layer == "Crop health":
            st.markdown('<div class="pt-legend">' + "".join(
                f'<span><i style="background:{c}"></i>{t}</span>' for c, t in
                [("#1F7A3A", "Healthy"), ("#8DBE3A", "Mild stress"), ("#E0B83C", "Stressed"),
                 ("#A2592A", "Severe"), ("#5E1A12", "Fungal disease")]) + "</div>", unsafe_allow_html=True)


def _controls(frame) -> None:
    running = frame.get("demo_running")
    with st.container(border=True):
        st.markdown("**Mission control**")
        if running:
            st.button("Stop demo", icon=":material/stop_circle:", type="primary", use_container_width=True,
                      on_click=_act, args=("Stop", "/demo/stop", {}))
        else:
            st.button("Start demo", icon=":material/play_circle:", type="primary", use_container_width=True,
                      on_click=_act, args=("Demo", "/demo/start", {}))
        a, b = st.columns(2)
        a.button("Next day", icon=":material/today:", use_container_width=True, disabled=running,
                 on_click=_act, args=("Next day", "/live/action", {"action": "next_day"}))
        b.button("Scout", icon=":material/travel_explore:", use_container_width=True, disabled=running,
                 on_click=_act, args=("Scout", "/live/action", {"action": "scout"}))
        a.button("Spray", icon=":material/water_drop:", use_container_width=True, disabled=running,
                 on_click=_act, args=("Spray", "/live/action", {"action": "spray"}))
        b.button("Reset", icon=":material/restart_alt:", use_container_width=True, disabled=running,
                 on_click=_act, args=("Reset", "/live/action", {"action": "reset"}))
        st.button("Scout + spray", icon=":material/bolt:", use_container_width=True, disabled=running,
                  on_click=_act, args=("Scout + spray", "/live/action", {"action": "auto"}))
        with st.popover("What-if: inject outbreak", icon=":material/coronavirus:", use_container_width=True,
                        disabled=running):
            st.caption("Plants a disease patch in the hidden REALITY field. The twin isn't told; scouting finds it.")
            r = st.slider("Row (north → south)", 0, 19, 14, key="ov_ob_row")
            c = st.slider("Column (west → east)", 0, 19, 5, key="ov_ob_col")
            st.button("Inject", type="primary", use_container_width=True, on_click=_act,
                      args=("Outbreak", "/live/action", {"action": "outbreak", "row": r + 0.5, "col": c + 0.5}))



def _error_chart(m) -> None:
    hist = ((m or {}).get("twin_error") or {}).get("history") or []
    if len(hist) < 2:
        return
    import plotly.graph_objects as go
    days = [h[0] for h in hist]
    errs = [h[1] for h in hist]
    fig = go.Figure(go.Scatter(x=list(range(len(errs))), y=errs, mode="lines", line=dict(color=C["accent"], width=2.5),
                               fill="tozeroy", fillcolor="rgba(43,214,123,0.10)",
                               customdata=days, hovertemplate="day %{customdata}: %{y:.3f}<extra></extra>"))
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=6, b=0), xaxis=dict(visible=False),
                      yaxis=dict(title=None, tickformat=".2f"))
    with st.container(border=True):
        st.markdown("**Twin vs reality error** <span style='color:#8D9CB6;font-size:.8rem'>(lower is better)</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="ov_err")
