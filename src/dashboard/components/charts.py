"""
charts.py
---------
Reusable Streamlit charts and plots (Plotly/Matplotlib wrappers).
"""

from __future__ import annotations

import streamlit as st
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def render_metric_bar_chart(title: str, labels: list[str], values: list[float], colors: list[str] | None = None) -> None:
    """Render a horizontal bar chart of metrics using Plotly."""
    if not HAS_PLOTLY:
        st.bar_chart(data=dict(zip(labels, values)))
        return
        
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors if colors else '#3498db'
    ))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_timeseries_chart(title: str, dates: list, values: list[float], ylabel: str = "Value") -> None:
    """Render a time-series line chart using Plotly."""
    if not HAS_PLOTLY:
        st.line_chart(data=values)
        return
        
    fig = px.line(x=dates, y=values, title=title, markers=True)
    fig.update_layout(yaxis_title=ylabel, xaxis_title="Date", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
