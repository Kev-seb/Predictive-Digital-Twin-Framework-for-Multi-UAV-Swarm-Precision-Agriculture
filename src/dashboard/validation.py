"""
validation.py - "how accurate is the twin?" from replays of the real drone flights.

scripts/replay_flights.py runs the twin through each field's real flights, oldest first: before every
flight it predicts the flight's NDVI, then assimilates what the drone measured. This panel reads the
reports (outputs/replay/<field>/report.json) and shows prediction error, twin vs the crop model alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.dashboard.ui_shell import C

ROOT = Path(__file__).resolve().parents[2]
REPLAY_DIR = ROOT / "outputs" / "replay"


def load_reports() -> list[dict]:
    reports = []
    for p in sorted(REPLAY_DIR.glob("*/report.json")):
        try:
            reports.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return reports


def summary(rep: dict) -> dict | None:
    scored = [r for r in rep.get("flights", [])[1:] if "twin_rmse" in r]   # flight 1 has nothing to learn from
    if not scored:
        return None
    tw = sum(r["twin_rmse"] for r in scored) / len(scored)
    mo = sum(r["model_only_rmse"] for r in scored) / len(scored)
    return {"flights": len(rep.get("flights", [])), "scored": len(scored), "twin_rmse": tw, "model_rmse": mo,
            "better_pct": (1 - tw / mo) * 100 if mo > 0 else 0.0}


def render(expanded: bool = False) -> None:
    reports = load_reports()
    with st.expander("Validated on real drone flights", icon=":material/verified:", expanded=expanded):
        if not reports:
            st.caption("No validation yet. Run `python scripts/calibrate_paddy.py` once, then "
                       "`python scripts/replay_flights.py --field F1` (and F2) where the raw flight dataset is.")
            return
        st.caption("Before each real flight the twin predicted the field's NDVI, then learned from what the drone "
                   "measured. Lower error is better; the crop model alone shows what the twin adds.")
        cols = st.columns(len(reports))
        for col, rep in zip(cols, reports):
            s = summary(rep)
            with col:
                st.markdown(f"**Field {rep.get('field')}** · {len(rep.get('flights', []))} flights · "
                            f"<span style='color:{C['muted']}'>{rep.get('weather', '')}</span>",
                            unsafe_allow_html=True)
                if s is None:
                    st.caption("Not enough usable flights to score.")
                    continue
                a, b = st.columns(2)
                a.metric("Twin error (NDVI RMSE)", f"{s['twin_rmse']:.3f}")
                b.metric("Model alone", f"{s['model_rmse']:.3f}",
                         delta=f"{s['better_pct']:.0f}% better with the twin" if s["better_pct"] > 0 else
                         f"{-s['better_pct']:.0f}% worse with the twin", delta_color="normal" if s["better_pct"] > 0
                         else "inverse")
                _chart(rep)


def _chart(rep: dict) -> None:
    import plotly.graph_objects as go
    rows = [r for r in rep.get("flights", []) if "observed_ndvi" in r]
    if len(rows) < 2:
        return
    x = [r["date"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["observed_ndvi"] for r in rows], name="Drone measured",
                             mode="lines+markers", line=dict(color=C["text"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=[r["twin_predicted_ndvi"] for r in rows], name="Twin predicted",
                             mode="lines+markers", line=dict(color=C["accent"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=[r["model_only_ndvi"] for r in rows], name="Model alone",
                             mode="lines", line=dict(color=C["amber"], width=1.5, dash="dot")))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Mean NDVI",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key=f"val_{rep.get('field')}")
