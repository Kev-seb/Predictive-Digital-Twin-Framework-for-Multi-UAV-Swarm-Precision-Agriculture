"""
farm_home.py - the farmer's home page: the field in plain words, what to do, and alerts.
Phone-first: one column on small screens, large type, few choices. English or Tamil (i18n.py).
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from src.dashboard import ui_shell
from src.dashboard.i18n import NEEDS_REVIEW, lang, t
from src.dashboard.ui_shell import C


def _frame():
    from src.dashboard import twin_link
    try:
        return twin_link._request("GET", "/ue5/frame?since=-1", timeout=15.0)
    except Exception:
        return None


def _card(label: str, value: str, colour: str = None, foot: str = "") -> str:
    c = colour or C["text"]
    return (f"<div class='pt-kpi' style='padding:16px 18px'><div class='pt-kpi-label'>{label}</div>"
            f"<div class='pt-kpi-value' style='font-size:1.45rem;color:{c}'>{value}</div>"
            f"<div class='pt-kpi-foot'>{foot}</div></div>")


def render() -> None:
    st.markdown(f'<div class="pt-title">{t("farm_title")}</div><div class="pt-sub">{t("farm_sub")}</div>',
                unsafe_allow_html=True)
    if NEEDS_REVIEW.get(lang()):
        st.caption(t("review_note"))
    f = _frame()
    if not f:
        st.warning(t("no_twin"), icon=":material/cloud_off:")
        _buttons()
        return
    field, m = f.get("field") or {}, f.get("metrics") or {}
    if (f.get("field_source") or {}).get("source") == "upload":
        st.markdown(f"**{f['field_source']['name']}**")
    stress = np.asarray(field.get("stress") or [0.0], float)
    fungus = np.asarray(field.get("fungus") or [0.0], float)
    healthy_pct = float(np.mean((stress < 0.15) & (fungus < 0.10)) * 100)
    wi = field.get("what_if") or {}
    peak = (wi.get("no_spray") or {}).get("fungus_peak", 0.0)
    risk, rc = (("risk_low", C["accent"]) if peak < 0.15 else
                ("risk_medium", C["amber"]) if peak < 0.30 else ("risk_high", C["red"]))
    h = m.get("harvest_t_ha") or {}
    rain = (f.get("weather") or {}).get("precipitation_mm", 0.0) or 0.0

    cols = st.columns(4, gap="small")
    cols[0].markdown(_card(t("health"), t("health_val", pct=healthy_pct),
                           C["accent"] if healthy_pct >= 80 else C["amber"]), unsafe_allow_html=True)
    cols[1].markdown(_card(t("disease"), t(risk), rc), unsafe_allow_html=True)
    cols[2].markdown(_card(t("harvest"), t("harvest_val", t=h.get("now", 0.0))), unsafe_allow_html=True)
    cols[3].markdown(_card(t("weather"), t("rain_yes", mm=rain) if rain > 1 else t("rain_no"),
                           C["blue"] if rain > 1 else C["text"],
                           f"{(f.get('weather') or {}).get('temperature_c', 0):.0f} °C"), unsafe_allow_html=True)

    # one clear recommendation
    zones = h.get("spray_zones") or []
    std = float(np.mean(field.get("stress_std") or [0.0]))
    if zones:
        todo = t("todo_rain") if rain > 1 else t("todo_spray", zones=", ".join(map(str, zones)),
                                                 gain=max(0.0, h.get("in_7d_if_sprayed", 0) - h.get("in_7d_no_spray", 0)))
        tone = C["amber"]
    elif std > 0.06:
        todo, tone = t("todo_scout"), C["blue"]
    else:
        todo, tone = t("todo_nospray"), C["accent"]
    st.markdown(f"<div class='pt-status' style='border-left-color:{tone};font-size:1.05rem;margin:.9rem 0'>"
                f"<b>{t('todo')}:</b> {todo}</div>", unsafe_allow_html=True)
    fz = m.get("fungicide") or {}
    if fz.get("blanket_kg") and fz.get("used_kg"):
        st.caption(t("saved", pct=fz.get("saved_pct", 0)))

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        _map(field)
    with right:
        _buttons()
    _alerts(f)


def _map(field) -> None:
    if not field.get("stress"):
        return
    import plotly.graph_objects as go
    s = np.asarray(field["stress"], float).reshape(20, 20)
    fu = np.asarray(field["fungus"], float).reshape(20, 20)
    z = np.clip(np.maximum(np.clip(s * 2.5, 0, 1) * 0.6, np.clip(fu * 1.6, 0, 1)), 0, 1)
    fig = go.Figure(go.Heatmap(z=z, zmin=0, zmax=1, showscale=False, xgap=1, ygap=1,
                               colorscale=[[0, "#1F7A3A"], [0.3, "#8DBE3A"], [0.55, "#E0B83C"], [0.78, "#A2592A"],
                                           [1, "#5E1A12"]], hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False),
                      yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"))
    st.markdown(f"**{t('map_title')}**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="farm_map")


def _buttons() -> None:
    for label, page, icon in (("btn_leaf", "leafcheck", ":material/local_florist:"),
                              ("btn_upload", "upload", ":material/upload_file:"),
                              ("btn_tasks", "optimizer", ":material/checklist:"),
                              ("btn_live", "overview", ":material/flight:")):
        st.button(t(label), icon=icon, use_container_width=True, key=f"farm_{page}",
                  on_click=ui_shell._set_page, args=(page,),
                  disabled=page in ui_shell.NEEDS_FIELD and not ui_shell.has_field())


def _alerts(frame) -> None:
    from src.alerts import alert_engine as ae
    st.markdown(f"#### {t('alerts')}")
    alerts = ae.evaluate(frame)
    if not alerts:
        st.success(t("alert_none"), icon=":material/notifications:")
    for a in alerts:
        (st.error if a["level"] == "high" else st.warning)(ae.message(a, lang()), icon=":material/notifications_active:")
    with st.expander("Send alerts by SMS / WhatsApp / email", icon=":material/send:"):
        ok = ae.channels_configured()
        st.caption("Channels: " + " · ".join(f"{k} {'✅' if v else '— not set up'}" for k, v in ok.items())
                   + ". SMS and WhatsApp need a Twilio account; email needs an SMTP server "
                     "(settings: src/alerts/alert_engine.py). Until then alerts go to the log.")
        import pandas as pd
        recips = ae.load_recipients() or [{"name": "Farmer", "language": "ta", "channel": "log", "address": ""}]
        ed = st.data_editor(pd.DataFrame(recips), num_rows="dynamic", hide_index=True, use_container_width=True,
                            key="farm_recips",
                            column_config={"language": st.column_config.SelectboxColumn("language", options=["en", "ta"]),
                                           "channel": st.column_config.SelectboxColumn(
                                               "channel", options=["log", "sms", "whatsapp", "email"]),
                                           "address": st.column_config.TextColumn("phone (+91…) or email")})
        a, b = st.columns(2)
        if a.button("Save recipients", key="farm_save_recips", use_container_width=True):
            ae.save_recipients([r for r in ed.fillna("").to_dict("records") if r.get("name")])
            st.toast("Saved", icon=":material/check:")
        if b.button("Send current alerts now", key="farm_send", type="primary", use_container_width=True,
                    disabled=not alerts):
            name = (frame.get("field_source") or {}).get("name", "your field")
            res = ae.send(alerts, [r for r in ed.fillna("").to_dict("records") if r.get("name")], name, force=True)
            st.dataframe(pd.DataFrame(res), hide_index=True, use_container_width=True)
        if alerts:
            st.caption("Preview: " + " | ".join(f"[{l}] {ae.message(alerts[0], l)}" for l in ("en", "ta")))
