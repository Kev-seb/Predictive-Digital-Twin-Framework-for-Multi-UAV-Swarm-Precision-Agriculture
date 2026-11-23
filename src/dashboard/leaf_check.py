"""
leaf_check.py - "Leaf disease check" page: a close-up leaf photo -> likely disease, and (optionally) the
finding sent to the live twin as a geotagged observation, like a drone frame.
"""
from __future__ import annotations

import math

import streamlit as st

from src.dashboard.ui_shell import C


def _send_to_twin(row: int, col: int, fungal: float, label: str) -> tuple[bool, str]:
    """Assimilate the photo's fungal probability at a cell of the 20 x 20 field map."""
    from src.dashboard import twin_link
    try:
        g = twin_link._request("GET", "/twin/state", timeout=5.0)["grid"]
    except Exception:
        return False, "The live twin is not running (start_demo.bat), so the finding was not sent."
    cell = g["cell_size_m"] * g["rows"] / 20.0                       # the map shows 20 x 20 cells
    south, east = (row + 0.5) * cell, (col + 0.5) * cell
    lat = g["origin_lat"] - south / 110_540.0
    lon = g["origin_lon"] + east / (111_320.0 * math.cos(math.radians(g["origin_lat"])))
    try:
        twin_link._request("POST", "/twin/observe/point", {
            "lat": lat, "lon": lon, "values": {"fungus": round(fungal, 4)}, "source": "leaf_photo",
            "footprint_m": max(1.0, cell / 2), "influence_radius_m": max(4.0, cell * 1.5)})
    except Exception as e:
        return False, f"The twin did not accept the observation: {e}"
    return True, (f"Sent to the live twin: {label} at row {row}, column {col} (fungal probability "
                  f"{fungal * 100:.0f}%). The twin's fungus map is updated around that spot.")


def render() -> None:
    from src.ai_engine import leaf_disease as ld
    st.markdown('<div class="pt-crumb">Live operations &nbsp;›&nbsp; <b>Leaf disease check</b></div>',
                unsafe_allow_html=True)
    st.header("Leaf disease check")
    st.markdown("Take or upload a **close-up photo of one rice leaf**. The AI suggests the most likely disease, "
                "and you can add the finding to the live twin at the spot where the leaf was taken.")
    if not ld.available():
        st.warning("The disease model has not been trained yet: run `python research/10_train_disease_model.py`.")
        return
    card = ld.card() or {}
    ev = card.get("evaluation", {})

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        src = st.segmented_control("Photo", ["Upload", "Camera"], default="Upload", key="leaf_src",
                                   label_visibility="collapsed") or "Upload"
        photo = (st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"], key="leaf_upload")
                 if src == "Upload" else st.camera_input("Leaf photo", key="leaf_cam"))
        if photo is not None:
            show_why = st.toggle("Show where the AI looked", value=False, key="leaf_why",
                                 help="Grad-CAM heatmap: red areas drove the answer most")
            if show_why:
                from PIL import Image
                with st.spinner("Computing the heatmap..."):
                    st.image(ld.explain(Image.open(photo)), use_container_width=True,
                             caption="Red = the parts of the leaf that drove the answer")
            else:
                st.image(photo, use_container_width=True)
    with right:
        if photo is None:
            st.info("Tips: fill the frame with one leaf, in daylight, in focus, lesions visible.",
                    icon=":material/photo_camera:")
            _credits(card)
            return
        from PIL import Image
        with st.spinner("Analysing the leaf..."):
            results = ld.predict(Image.open(photo), top=len(card.get("classes", [])) or 7)
        top = results[0]
        conf = top["probability"]
        colour = C["accent"] if top["disease"] == "Healthy" else (C["red"] if conf >= 0.6 else C["amber"])
        st.markdown(f"<div class='pt-kpi'><div class='pt-kpi-label'>Most likely</div>"
                    f"<div class='pt-kpi-value' style='color:{colour}'>{top['disease']}</div>"
                    f"<div class='pt-kpi-foot'>{conf * 100:.0f}% confidence · {top.get('kind', '')}</div></div>",
                    unsafe_allow_html=True)
        if conf < 0.5:
            st.warning("The model is unsure about this photo. Take another, closer photo in daylight, or ask an "
                       "agronomist.", icon=":material/help:")
        st.markdown(f"**What it means:** {top.get('note', '')}")
        st.markdown("**Other possibilities**")
        for r in results[1:3]:
            st.progress(min(1.0, r["probability"]), text=f"{r['disease']} · {r['probability'] * 100:.0f}%")
        fungal = ld.fungal_probability(results)

        with st.container(border=True):
            st.markdown("**Add this finding to the live twin**")
            st.caption("Where in the field was the leaf? Row 0 is the north edge, column 0 the west edge "
                       "(the same 20 × 20 map as on Live twin & drones).")
            a, b = st.columns(2)
            row = a.number_input("Row", 0, 19, 10, key="leaf_row")
            col = b.number_input("Column", 0, 19, 10, key="leaf_col")
            if st.button("Send to twin", icon=":material/hub:", type="primary", key="leaf_send"):
                ok, msg = _send_to_twin(int(row), int(col), fungal, top["disease"])
                (st.success if ok else st.warning)(msg)
        st.caption(f"Model tested on leaves it never saw: {ev.get('test_accuracy', 0) * 100:.0f}% overall, "
                   f"{(ev.get('field_photos_accuracy') or 0) * 100:.0f}% on real field photos. A photo is not a "
                   "diagnosis: confirm with an agronomist before treating.")
        _credits(card)


def _credits(card: dict) -> None:
    if card.get("credits"):
        st.caption(f"Training data: {card['credits']}")
