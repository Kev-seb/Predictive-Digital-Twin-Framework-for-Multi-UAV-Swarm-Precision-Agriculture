"""
field_report.py - the field report as a PDF: summary, maps, plan and costs, sustainability,
crop-damage evidence (for insurance), and data credits.

    pdf_bytes = build_report(ctx)

ctx keys (all optional except field_name): field_name, location (lat, lon), footprint_m (w, h), crop_stage,
indices {ndvi, stress_score, ...}, zone_labels, plan {zones, summary, economics}, flights [rows], twin (frame),
weather (assessment.weather dict), stage_gdd, models (cards).
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict

import numpy as np

GREEN, AMBER, RED, INK, MUTED = (31, 122, 58), (196, 140, 20), (190, 50, 50), (20, 28, 40), (110, 120, 135)


def _png(arr, cmap, vmin, vmax, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=130)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
    buf.seek(0)
    return buf


def _ascii(s: str) -> str:
    rep = {"₹": "Rs ", "–": "-", "—": "-", "×": "x", "°": " deg", "→": "->", "·": "-", "’": "'", "“": '"', "”": '"',
           "≥": ">=", "≤": "<=", "✓": "v"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def build_report(ctx: Dict[str, Any]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_title(_ascii(f"Field report - {ctx.get('field_name', 'field')}"))

    def h1(text):
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*INK)
        pdf.cell(0, 10, _ascii(text), new_x="LMARGIN", new_y="NEXT")

    def h2(text):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, _ascii(text), new_x="LMARGIN", new_y="NEXT")

    def para(text, size=10, colour=INK):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*colour)
        pdf.multi_cell(0, 5.2, _ascii(text), new_x="LMARGIN", new_y="NEXT")

    def table(rows, widths, header=True):
        pdf.set_font("Helvetica", "B" if header else "", 9)
        for i, row in enumerate(rows):
            pdf.set_font("Helvetica", "B" if header and i == 0 else "", 9)
            for w, cell in zip(widths, row):
                pdf.cell(w, 6.2, _ascii(str(cell))[:60], border=1)
            pdf.ln()

    name = ctx.get("field_name", "Field")
    lat, lon = ctx.get("location") or (None, None)
    fp = ctx.get("footprint_m")
    idx = ctx.get("indices") or {}
    twin = ctx.get("twin") or {}
    m = twin.get("metrics") or {}
    plan = ctx.get("plan") or {}

    # ---------------------------------------------------------------- 1. summary
    pdf.add_page()
    h1(f"Field report: {name}")
    para(f"Generated {time.strftime('%d %B %Y, %H:%M')} by PaddyTwin.", 9, MUTED)
    rows = [["Item", "Value"]]
    if lat is not None:
        rows.append(["Location (field centre)", f"{lat:.5f} N, {lon:.5f} E"])
    if fp:
        rows.append(["Image footprint", f"{fp[0]:.0f} x {fp[1]:.0f} m ({fp[0] * fp[1] / 10000:.2f} ha)"])
    if ctx.get("crop_stage"):
        rows.append(["Growth stage (analysis)", ctx["crop_stage"]])
    sg = ctx.get("stage_gdd")
    if sg:
        rows.append(["Stage from transplanting", f"{sg['stage']} (day {sg['days_after_transplanting']}, "
                                                 f"{sg['gdd']:.0f} degree-days)"])
    if "ndvi" in idx:
        nd = np.asarray(idx["ndvi"])
        st_ = np.asarray(idx.get("stress_score", np.zeros_like(nd)))
        rows += [["Mean NDVI", f"{np.nanmean(nd):.3f}"],
                 ["Area with high stress (score > 0.5)", f"{np.nanmean(st_ > 0.5) * 100:.1f} %"]]
    h = m.get("harvest_t_ha") or {}
    if h:
        rows += [["Expected harvest now", f"{h['now']:.2f} t/ha"],
                 ["In 7 days, no spray / if sprayed", f"{h['in_7d_no_spray']:.2f} / {h['in_7d_if_sprayed']:.2f} t/ha"]]
    ec = plan.get("economics") or {}
    if ec:
        rows += [["Cost of the treatment plan", f"Rs {ec['plan_total_rs']:,.0f}"],
                 ["Saved vs spraying the whole field", f"Rs {ec['spray_saving_rs']:,.0f} "
                                                       f"({ec['fungicide_saved_pct']:.0f}% less fungicide)"]]
    h2("Summary")
    table(rows, [75, 105])
    w = ctx.get("weather") or {}
    if w:
        def first7(key):
            v = w.get(key)
            return [float(x) for x in v[:7]] if isinstance(v, (list, tuple)) and v else None
        tmax, tmin, rain, hum, wind = (first7(k) for k in ("temperature_max", "temperature_min", "precipitation",
                                                            "humidity", "wind_speed"))
        bits = []
        if tmin and tmax:
            bits.append(f"temperature {min(tmin):.0f}-{max(tmax):.0f} deg C")
        if rain:
            bits.append(f"rain {sum(rain):.0f} mm in total ({sum(1 for r in rain if r > 1)} wet days)")
        if hum:
            bits.append(f"humidity {np.mean(hum):.0f}% on average")
        if wind:
            bits.append(f"wind up to {max(wind):.0f} km/h")
        if bits:
            h2("Weather, next 7 days")
            para("; ".join(bits) + ".")

    # ---------------------------------------------------------------- 2. maps
    if "ndvi" in idx:
        pdf.add_page()
        h1("Field maps")
        x0, y0 = pdf.get_x(), pdf.get_y()
        pdf.image(_png(np.asarray(idx["ndvi"]), "RdYlGn", -0.2, 0.9, "NDVI (vegetation vigour)"), x=x0, y=y0, w=92)
        if "stress_score" in idx:
            pdf.image(_png(np.asarray(idx["stress_score"]), "YlOrRd", 0, 1, "Stress score"), x=x0 + 96, y=y0, w=92)
        pdf.set_y(y0 + 82)
        zl = ctx.get("zone_labels")
        if zl is not None:
            pdf.image(_png(np.asarray(zl), "tab10", 0, 9, "Management zones"), x=x0, y=pdf.get_y(), w=92)
            pdf.set_y(pdf.get_y() + 82)
        para("Row 0 / top of each map is the north edge of the image.", 8, MUTED)

    # ---------------------------------------------------------------- 3. plan and costs
    zones = plan.get("zones") or []
    if zones:
        pdf.add_page()
        h1("Treatment plan and prescription")
        rows = [["Zone", "N kg/ha", "P kg/ha", "K kg/ha", "Irrig. mm", "Fung. L/ha", "Risk"]]
        rows += [[f"{z['ZONE']} {z['NAME']}"[:28], z["N_KG_HA"], z["P_KG_HA"], z["K_KG_HA"], z["IRR_MM"],
                  z["FUNG_L_HA"], z["RISK"]] for z in zones]
        table(rows, [60, 20, 20, 20, 20, 22, 18])
        s = plan.get("summary") or {}
        if s:
            para(f"Over {s['area_ha']:.2f} ha: {s['N_KG_total']:.1f} kg N, {s['P_KG_total']:.1f} kg P2O5, "
                 f"{s['K_KG_total']:.1f} kg K2O and {s['FUNG_L_total']:.2f} L fungicide on {s['FUNG_area_ha']:.2f} ha.")
        if ec:
            h2("Costs and returns")
            rows = [["Item", "Rupees"], ["Fertiliser", f"{ec['fertiliser_rs']:,.0f}"],
                    ["Spraying (product + drone hire)", f"{ec['spray_rs']:,.0f}"],
                    ["Total cost of this plan", f"{ec['plan_total_rs']:,.0f}"],
                    ["Spraying the whole field instead", f"{ec['blanket_spray_rs']:,.0f}"],
                    ["Saved by targeting", f"{ec['spray_saving_rs']:,.0f}"]]
            if "yield_value_rs" in ec:
                rows += [["Yield value protected (7-day twin forecast)", f"{ec['yield_value_rs']:,.0f}"],
                         ["Net benefit", f"{ec['net_benefit_rs']:,.0f}"]]
            table(rows, [110, 70])

        # ------------------------------------------------------------ 4. sustainability
        h2("Sustainability")
        lines = []
        if ec.get("blanket_fungicide_l"):
            lines.append(f"Fungicide: {ec['fungicide_l']:.2f} L instead of {ec['blanket_fungicide_l']:.2f} L "
                         f"({ec['fungicide_saved_pct']:.0f}% less chemical reaching soil and water).")
        if zones and s:
            n_uniform = max(z["N_KG_HA"] for z in zones) * s["area_ha"]
            if n_uniform > 0:
                lines.append(f"Nitrogen: {s['N_KG_total']:.1f} kg placed by zone instead of {n_uniform:.1f} kg at the "
                             f"highest zone rate everywhere ({(1 - s['N_KG_total'] / n_uniform) * 100:.0f}% less).")
        fz = m.get("fungicide") or {}
        if fz.get("blanket_kg"):
            lines.append(f"Season so far (live twin): {fz['used_kg']:.1f} kg fungicide used vs {fz['blanket_kg']:.0f} kg "
                         f"for whole-field spraying ({fz['saved_pct']:.0f}% saved).")
        para(" ".join(lines) or "Run the treatment plan to compute savings.")

    # ---------------------------------------------------------------- 5. damage evidence
    pdf.add_page()
    h1("Crop condition evidence")
    para("Supporting evidence for a crop-loss or insurance claim (for example under PMFBY). It documents what the "
         "drone images and the field twin recorded; it is not an official damage assessment.", 9, MUTED)
    if "stress_score" in idx:
        st_ = np.asarray(idx["stress_score"])
        rows = [["Condition", "Share of field"], ["Healthy (stress < 0.25)", f"{np.nanmean(st_ < 0.25) * 100:.1f} %"],
                ["Moderate (0.25 - 0.5)", f"{np.nanmean((st_ >= 0.25) & (st_ < 0.5)) * 100:.1f} %"],
                ["Severe (> 0.5)", f"{np.nanmean(st_ >= 0.5) * 100:.1f} %"]]
        if fp:
            area = fp[0] * fp[1] / 10000
            rows.append(["Severe area", f"{np.nanmean(st_ >= 0.5) * area:.3f} ha of {area:.2f} ha imaged"])
        table(rows, [90, 90])
    fl = ctx.get("flights") or []
    if fl:
        h2("Flights (images) of this field")
        rows = [["#", "Image", "Loaded", "NDVI mean", "Stress"]]
        rows += [[f["Flight"], f["Image"][:38], f["Loaded"], f["NDVI mean"], f["Stress score"]] for f in fl]
        table(rows, [10, 75, 40, 28, 27])
    if lat is not None:
        para(f"Georeference: centre {lat:.6f} N, {lon:.6f} E (WGS84).", 9)

    # ---------------------------------------------------------------- 6. credits and accuracy
    h2("Models, accuracy and data credits")
    for c in ctx.get("models") or []:
        ev = c.get("evaluation", {})
        acc = ev.get("test_accuracy", ev.get("test_acc"))
        para(f"- {c.get('task', c.get('model'))}: tested accuracy {acc * 100:.0f}% on held-out data. "
             f"{c.get('credits', '')}", 9)
    para("Twin validation: on 20 real flights over two fields, the twin predicted each next flight's NDVI 20-27% "
         "better than the crop model alone (scripts/replay_flights.py).", 9)
    return bytes(pdf.output())
