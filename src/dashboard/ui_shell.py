"""
ui_shell.py - application shell for the dashboard: theme, grouped sidebar navigation, page chrome.

dashboard.py keeps every feature page's code unchanged; each former tab block is now
`if show_page(N):` and only the selected page runs (faster than rendering all 15 tabs each rerun).
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

# ----------------------------------------------------------------- design tokens
C = {
    "bg": "#0A1020",
    "surface": "#101A2E",
    "surface2": "#15223B",
    "border": "#22314D",
    "text": "#E7EDF7",
    "muted": "#8D9CB6",
    "accent": "#2BD67B",       # healthy / primary
    "accent_dim": "#1E9E5B",
    "blue": "#4EA8FF",
    "amber": "#F5B841",
    "red": "#F0525A",
    "violet": "#B69CFF",
}

APP_NAME = "PaddyTwin"
APP_TAGLINE = "Predictive digital twin for UAV precision agriculture"


@dataclass(frozen=True)
class Page:
    key: str
    label: str
    icon: str
    tab: int | None        # index of the former st.tabs block in dashboard.py (None = new page)
    blurb: str


# The main workflow: upload one field image, then every step builds on it, ending with the report.
WORKFLOW: list[Page] = [
    Page("upload", "Upload field image", ":material/upload_file:", 0, "Multispectral GeoTIFF of the field"),
    Page("vegetation", "Vegetation analytics", ":material/eco:", 1, "NDVI, NDRE, NDWI and index maps"),
    Page("stress", "Stress intelligence", ":material/coronavirus:", 2, "Where the crop is stressed, and why"),
    Page("zoning", "Field zones", ":material/grid_on:", 4, "Management zones for targeted treatment"),
    Page("weather", "Weather & risk", ":material/thunderstorm:", 5, "Forecast-driven disease and water risk"),
    Page("twin", "Digital twin forecast", ":material/hub:", 6, "Simulate the next days and the yield"),
    Page("optimizer", "Treatment plan", ":material/tune:", 7, "Variable-rate inputs for each zone"),
    Page("report", "AI report", ":material/description:", 13, "Agronomy report for the field"),
]
# What finishes a step: its result in the session (action steps) or having opened it (viewing steps).
STEP_RESULT = {"upload": "ms_image", "weather": "weather_assessment", "twin": "simulation_result",
               "optimizer": "opt_report"}
STEP_ACTION = {"weather": "Press **Fetch Weather Data** above: the twin forecast and the treatment plan use it.",
               "twin": "Press **Run Scenario Simulation Model** above to forecast the field.",
               "optimizer": "Press **Run AI Optimization Solver** above: the report includes its treatment plan."}

NAV: list[tuple[str, list[Page]]] = [
    ("Workflow", WORKFLOW),
    ("Live operations", [
        Page("overview", "Live twin & drones", ":material/space_dashboard:", None, "Live mission control for the field twin"),
        Page("leafcheck", "Leaf disease check", ":material/local_florist:", None, "Photo of a leaf: likely disease"),
        Page("swarm", "Swarm operations", ":material/scatter_plot:", 12, "Fleet planning and missions"),
        Page("uav", "UAV control center", ":material/flight:", 11, "Live single-drone control"),
        Page("livefield", "Live field mode", ":material/sensors:", 14, "Real-time field feed"),
    ]),
    ("More analysis", [
        Page("temporal", "Temporal analytics", ":material/timeline:", 3, "Change over the season"),
        Page("reconstruction", "3D reconstruction", ":material/view_in_ar:", 8, "UAV photogrammetry and 3D field"),
        Page("satellite", "Satellite analytics", ":material/satellite_alt:", 9, "Sentinel-2 field monitoring"),
        Page("landsat", "Landsat history", ":material/history:", 10, "Decades of field history"),
    ]),
]
WORKFLOW_KEYS = [p.key for p in WORKFLOW]
NEEDS_FIELD = set(WORKFLOW_KEYS[1:]) | {"temporal", "reconstruction"}

PAGES = {p.key: p for _, pages in NAV for p in pages}
GROUP_OF = {p.key: g for g, pages in NAV for p in pages}
_BY_TAB = {p.tab: p for p in PAGES.values() if p.tab is not None}


# ----------------------------------------------------------------- theme
def apply_theme() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{ font-family: 'Inter', sans-serif !important; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.stApp {{ background: radial-gradient(1200px 600px at 75% -10%, rgba(43,214,123,0.06), transparent 60%), {C['bg']}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebarContent"] {{ padding-top: 0.6rem; }}
.pt-brand {{ display:flex; align-items:center; gap:.65rem; padding:.35rem .2rem 1rem .2rem; }}
.pt-logo {{ width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
           background: linear-gradient(135deg, {C['accent']} 0%, {C['blue']} 100%); color:#05121F; font-weight:800; font-size:.95rem; }}
.pt-brand-name {{ font-weight:800; font-size:1.05rem; color:{C['text']}; letter-spacing:-.2px; line-height:1.1; }}
.pt-brand-sub {{ font-size:.72rem; color:{C['muted']}; }}
.pt-navgroup {{ font-size:.68rem; font-weight:700; letter-spacing:1.1px; text-transform:uppercase;
               color:{C['muted']}; margin:.9rem 0 .25rem .45rem; }}
section[data-testid="stSidebar"] .stButton button {{
    width:100%; justify-content:flex-start; border-radius:8px; padding:.38rem .6rem; min-height:0;
    font-size:.88rem; font-weight:500; border:1px solid transparent; background:transparent; color:{C['text']};
}}
section[data-testid="stSidebar"] .stButton button:hover {{ background:{C['surface2']}; border-color:{C['border']}; color:{C['text']}; }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: rgba(43,214,123,0.12); border:1px solid rgba(43,214,123,0.35); color:{C['accent']}; font-weight:600;
}}
section[data-testid="stSidebar"] .stButton {{ margin-bottom:-.55rem; }}
section[data-testid="stSidebar"] .stButton button > div {{ justify-content:flex-start; width:100%; }}
section[data-testid="stSidebar"] .stButton button p {{ text-align:left; }}
section[data-testid="stSidebar"] .stButton button[disabled] {{ opacity:.45; background:transparent; border-color:transparent; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {{ display:none !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{ border:1px solid {C['border']}; border-radius:10px; background:{C['bg']}; }}

/* ---------- page chrome ---------- */
.pt-crumb {{ font-size:.78rem; color:{C['muted']}; margin-bottom:.15rem; letter-spacing:.2px; }}
.pt-crumb b {{ color:{C['text']}; font-weight:600; }}
.pt-title, .pt-title p {{ font-size:1.9rem !important; font-weight:800; color:{C['text']}; letter-spacing:-.6px; margin:0; line-height:1.2; }}
.pt-sub, .pt-sub p {{ font-size:.95rem !important; color:{C['muted']}; margin:.25rem 0 1.1rem 0; }}
h1, h2 {{ letter-spacing:-.5px; }}
.stMainBlockContainer h2, [data-testid="stHeadingWithActionElements"] h2 {{ font-weight:800; }}

/* ---------- cards, metrics ---------- */
div[data-testid="stMetric"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:14px 18px; }}
div[data-testid="stMetricValue"] {{ font-size:1.8rem; font-weight:700; letter-spacing:-.5px; }}
div[data-testid="stMetricLabel"] {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; color:{C['muted']}; font-weight:600; }}
div[data-testid="stFileUploader"] {{ background:{C['surface']}; border:1px dashed {C['border']}; border-radius:12px; padding:12px 16px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{C['border']} !important; border-radius:14px !important; background:{C['surface']}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {C['border']}; }}
.stTabs [data-baseweb="tab"] {{ padding:8px 14px; border-radius:8px 8px 0 0; font-weight:600; color:{C['muted']}; }}
.stTabs [aria-selected="true"] {{ color:{C['accent']} !important; }}
.stButton > button {{ border-radius:9px; font-weight:600; }}
.stAlert {{ border-radius:10px; }}
code, .pt-mono {{ font-family:'JetBrains Mono', monospace !important; }}

/* ---------- overview widgets ---------- */
.pt-kpi {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:14px; padding:14px 16px 12px; height:100%; }}
.pt-kpi-label {{ font-size:.72rem; font-weight:700; letter-spacing:.9px; text-transform:uppercase; color:{C['muted']}; }}
.pt-kpi-value {{ font-size:1.75rem; font-weight:800; color:{C['text']}; letter-spacing:-.6px; margin-top:.2rem; }}
.pt-kpi-value small {{ font-size:.9rem; font-weight:600; color:{C['muted']}; margin-left:.25rem; }}
.pt-kpi-foot {{ font-size:.8rem; color:{C['muted']}; margin-top:.15rem; }}
.pt-up {{ color:{C['accent']}; font-weight:700; }} .pt-down {{ color:{C['red']}; font-weight:700; }} .pt-warn {{ color:{C['amber']}; font-weight:700; }}
.pt-pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.22rem .65rem; border-radius:999px; font-size:.78rem; font-weight:600;
           border:1px solid {C['border']}; background:{C['surface']}; color:{C['text']}; margin-right:.4rem; }}
.pt-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.pt-status {{ background:{C['surface']}; border:1px solid {C['border']}; border-left:3px solid {C['accent']}; border-radius:10px;
             padding:.7rem .9rem; color:{C['text']}; font-size:.92rem; }}
.pt-legend {{ display:flex; flex-wrap:wrap; gap:.9rem; font-size:.8rem; color:{C['muted']}; margin-top:.2rem; }}
.pt-legend span i {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:.35rem; vertical-align:-1px; }}
.pt-flow {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:.6rem; }}
.pt-step {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:.8rem .9rem; }}
.pt-step-n {{ font-size:.7rem; font-weight:700; color:{C['accent']}; letter-spacing:1px; }}
.pt-step-t {{ font-weight:700; color:{C['text']}; margin:.15rem 0 .2rem; }}
.pt-step-d {{ font-size:.8rem; color:{C['muted']}; line-height:1.35; }}
.pt-fieldlabel {{ font-weight:700; font-size:.86rem; letter-spacing:.6px; }}
.pt-fieldsub {{ font-size:.76rem; color:{C['muted']}; }}

/* legacy classes used by feature pages */
.main-header {{ font-size:1.75rem; font-weight:800; color:{C['text']}; }}
.sub-header {{ color:{C['muted']}; }}
.stress-critical {{ color:{C['red']}; font-weight:800; }}
.stress-high {{ color:#F97316; font-weight:800; }}
.stress-medium {{ color:{C['amber']}; font-weight:800; }}
.stress-low {{ color:{C['accent']}; font-weight:800; }}
.pt-fieldcard {{ display:flex; align-items:center; gap:.5rem; font-size:.82rem; color:{C['text']};
                 background:{C['bg']}; border:1px solid {C['border']}; border-radius:9px; padding:.45rem .65rem; margin:-.3rem 0 .2rem; }}
.pt-navcount {{ color:{C['accent']}; letter-spacing:0; font-weight:700; }}
.pt-stepper {{ display:flex; align-items:center; gap:0; margin:.1rem 0 1.0rem; overflow-x:auto; padding-bottom:.2rem; }}
.pt-chipline {{ flex:1 1 18px; min-width:14px; height:2px; background:{C['border']}; }}
.pt-chip {{ display:flex; align-items:center; gap:.45rem; white-space:nowrap; font-size:.78rem; font-weight:600;
           padding:.3rem .65rem .3rem .3rem; border-radius:999px; border:1px solid {C['border']}; color:{C['muted']};
           background:{C['surface']}; }}
.pt-chip span {{ display:grid; place-items:center; width:20px; height:20px; border-radius:50%; font-size:.72rem;
                background:{C['surface2']}; color:{C['muted']}; }}
.pt-chip-done {{ color:{C['text']}; }}
.pt-chip-done span {{ background:rgba(43,214,123,.18); color:{C['accent']}; }}
.pt-chip-cur {{ color:{C['accent']}; border-color:rgba(43,214,123,.55); background:rgba(43,214,123,.10); }}
.pt-chip-cur span {{ background:{C['accent']}; color:#05121F; }}
@media (max-width: 900px) {{ .pt-flow {{ grid-template-columns: 1fr 1fr; }} }}
</style>
""", unsafe_allow_html=True)
    _plotly_template()


def _plotly_template() -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return
    if "paddytwin" in pio.templates:
        return
    t = go.layout.Template(pio.templates["plotly_dark"])
    t.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        colorway=[C["accent"], C["blue"], C["amber"], C["red"], C["violet"], "#38D9C5"],
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    pio.templates["paddytwin"] = t
    pio.templates.default = "paddytwin"


def matplotlib_theme() -> None:
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": C["surface"], "figure.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "text.color": C["text"], "axes.labelcolor": C["muted"], "axes.edgecolor": C["border"],
        "xtick.color": C["muted"], "ytick.color": C["muted"], "grid.color": C["border"],
        "font.family": "sans-serif",
    })


# ----------------------------------------------------------------- navigation + workflow
def _set_page(key: str) -> None:
    st.session_state["pt_page"] = key
    st.query_params["page"] = key


def current_page() -> str:
    if "pt_page" not in st.session_state:
        q = st.query_params.get("page", "upload")
        st.session_state["pt_page"] = q if q in PAGES else "upload"
    page = st.session_state["pt_page"]
    if page in NEEDS_FIELD and not has_field():
        page = st.session_state["pt_page"] = "upload"      # every step starts from the uploaded field
    return page


def has_field() -> bool:
    return "ms_image" in st.session_state


def mark_done(key: str) -> None:
    st.session_state.setdefault("pt_done", set()).add(key)


def reset_progress() -> None:
    st.session_state["pt_done"] = set()


def step_done(key: str) -> bool:
    if key in STEP_RESULT:
        return STEP_RESULT[key] in st.session_state
    return key in st.session_state.get("pt_done", set())


def sidebar_nav() -> str:
    """Brand + workflow steps + other tools. Returns the selected page key."""
    page = current_page()
    field = has_field()
    if field and page in WORKFLOW_KEYS and page not in STEP_RESULT:
        mark_done(page)                     # count the step being opened in the progress shown below
    st.sidebar.markdown(
        f'<div class="pt-brand"><div class="pt-logo">PT</div><div>'
        f'<div class="pt-brand-name">{APP_NAME}</div><div class="pt-brand-sub">{APP_TAGLINE}</div></div></div>',
        unsafe_allow_html=True)
    if field:
        name = st.session_state.get("field_name", "Field")
        st.sidebar.markdown(f'<div class="pt-fieldcard"><span class="pt-dot" style="background:{C["accent"]}"></span>'
                            f'<b>{name}</b></div>', unsafe_allow_html=True)
    for group, pages in NAV:
        done_n = sum(step_done(p.key) for p in pages) if group == "Workflow" else None
        label = f"{group} &nbsp;<span class='pt-navcount'>{done_n}/{len(pages)}</span>" if done_n is not None else group
        st.sidebar.markdown(f'<div class="pt-navgroup">{label}</div>', unsafe_allow_html=True)
        for i, p in enumerate(pages):
            locked = p.key in NEEDS_FIELD and not field
            if group == "Workflow":
                mark = "✓" if step_done(p.key) else str(i + 1)
                text = f"{mark}   {p.label}"
            else:
                text = p.label
            st.sidebar.button(text, key=f"nav_{p.key}", icon=None if group == "Workflow" else p.icon,
                              on_click=_set_page, args=(p.key,), disabled=locked,
                              help="Upload a field image first" if locked else p.blurb,
                              type="primary" if p.key == page else "secondary", use_container_width=True)
    st.sidebar.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    return page


def stepper(page_key: str) -> None:
    """Workflow progress across the top of a workflow page."""
    chips = []
    for i, p in enumerate(WORKFLOW):
        state = "cur" if p.key == page_key else ("done" if step_done(p.key) else "todo")
        mark = "✓" if state == "done" else str(i + 1)
        chips.append(f'<div class="pt-chip pt-chip-{state}"><span>{mark}</span>{p.label}</div>')
    st.markdown('<div class="pt-stepper">' + '<div class="pt-chipline"></div>'.join(chips) + "</div>",
                unsafe_allow_html=True)


def page_header(page_key: str) -> None:
    p = PAGES[page_key]
    if page_key in WORKFLOW_KEYS:
        stepper(page_key)
        n = WORKFLOW_KEYS.index(page_key) + 1
        st.markdown(f'<div class="pt-crumb">Workflow &nbsp;›&nbsp; Step {n} of {len(WORKFLOW)} &nbsp;›&nbsp; '
                    f'<b>{p.label}</b></div>', unsafe_allow_html=True)
    else:
        group = GROUP_OF.get(page_key) or "Home"
        st.markdown(f'<div class="pt-crumb">{group} &nbsp;›&nbsp; <b>{p.label}</b></div>', unsafe_allow_html=True)


def show_page(tab_index: int) -> bool:
    """True when the feature page that used to be tabs[tab_index] is selected (and draws its header)."""
    p = _BY_TAB.get(tab_index)
    if p is None or current_page() != p.key:
        return False
    page_header(p.key)
    if p.key in WORKFLOW_KEYS and p.key not in STEP_RESULT:
        mark_done(p.key)                    # viewing steps are done once opened with a field loaded
    return True


def workflow_footer() -> None:
    """Bottom of a workflow page: what to do here, and the button to the next step."""
    page = current_page()
    if page not in WORKFLOW_KEYS or not has_field():
        return
    i = WORKFLOW_KEYS.index(page)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        left, mid, right = st.columns([1.2, 3, 1.6], vertical_alignment="center")
        if i > 0:
            prev = WORKFLOW[i - 1]
            left.button("Back", key="wf_back", help=f"Back to {prev.label}", icon=":material/arrow_back:",
                        on_click=_set_page, args=(prev.key,), use_container_width=True)
        if i + 1 < len(WORKFLOW):
            nxt = WORKFLOW[i + 1]
            hint = STEP_ACTION.get(page) if not step_done(page) else None
            mid.markdown(f"**Step {i + 1} of {len(WORKFLOW)}** · next: {nxt.label}  \n"
                         + (f":orange[{hint}]" if hint else f"<span style='color:{C['muted']}'>{nxt.blurb}</span>"),
                         unsafe_allow_html=True)
            right.button(f"Continue: {nxt.label}", key="wf_next", type="primary", icon=":material/arrow_forward:",
                         on_click=_set_page, args=(nxt.key,), use_container_width=True)
        else:
            mid.markdown("**Workflow complete.** The report above covers this field.  \n"
                         f"<span style='color:{C['muted']}'>Next: watch the twin and the drones act on a field "
                         "in real time.</span>", unsafe_allow_html=True)
            right.button("Open live twin & drones", key="wf_live", type="primary", icon=":material/flight_takeoff:",
                         on_click=_set_page, args=("overview",), use_container_width=True)


def goto_button(label: str, key: str, page_key: str) -> None:
    st.button(label, key=key, on_click=_set_page, args=(page_key,))
