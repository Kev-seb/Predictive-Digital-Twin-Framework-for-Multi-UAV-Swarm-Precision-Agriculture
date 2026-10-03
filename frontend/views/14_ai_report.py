"""Renovated presentation of the original agriculture workflow."""

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from frontend_shared import (
    MAVLINK_TELEMETRY, CURRENT_ENV, REPLAY_STATE, TELEMETRY_BUFFER,
    MULTIPLAYER_DRONES, COLLABORATIVE_ANNOTATIONS, math, json, time, os, io,
    render_sidebar, render_header_status,
    metric_card, status_card, section_header,
    fig_to_bytes, colormap_array, plot_urgency_velocity, plot_boundaries_contours,
    np, pd, plt, cm, Image, torch, cv2,
    compute_all_indices, MultispectralImage, load_multispectral_tiff,
    rule_based_stress_segmentation, mask_to_overlay, CLASS_LABELS, CLASS_COLORS, compute_iou,
    delineate_management_zones, extract_stress_regions, render_zone_map,
    compute_grid_statistics, plot_grid_heatmap,
    build_temporal_report, plot_ndvi_progression, plot_stress_progression,
    plot_multi_index_radar, plot_canopy_stress_area, plot_index_heatmap, STAGES,
    fetch_weather, assess_weather_stress,
    build_model, class_activation_map, overlay_segmentation_cam,
    load_cached_segmentation_model, get_shared_state,
    get_live_camera_frame, UAVFlightDynamicsSimulator,
)
# Page configuration and navigation are owned by frontend/app.py.

shared_state = get_shared_state()

crop_stage, stress_threshold, grid_size, lat, lon = render_sidebar()
render_header_status()

# ══════════════════════════════════════════════════════════════
#  AI Report  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

# Premium card aesthetics now sourced from the same design tokens
# defined in the global stylesheet (var(--gd-*)) instead of a separate
# one-off palette, so this tab matches the rest of the app.
st.markdown("""
<style>
.premium-card {
    background: linear-gradient(145deg, var(--gd-surface), var(--gd-surface-2));
    border-radius: var(--gd-radius);
    padding: 20px;
    box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.35);
    border: 1px solid var(--gd-border);
    margin-bottom: 20px;
}
.premium-title {
    color: var(--gd-cyan);
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    margin-top: 0;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--gd-text);
}
.metric-label {
    font-size: 1rem;
    color: var(--gd-text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.slide-bar-container {
    padding: 10px 0 30px 0;
}
</style>
""", unsafe_allow_html=True)

section_header("Automated Precision Agriculture Report", "AI-generated summary, risk assessment, and recommendations")

idx = st.session_state.get("indices", {})
if not idx:
    st.warning("Process an image first (Upload & Process).")
else:
    ndvi_mean  = float(idx["ndvi"].mean())
    ndre_mean  = float(idx["ndre"].mean())
    ndwi_mean  = float(idx["ndwi"].mean())
    stress_mean= float(idx["stress_score"].mean())
    stressed_pct = float((idx["stress_score"] > stress_threshold).mean() * 100)

    risk_level = ("Critical" if stressed_pct > 40 else
                  "High"     if stressed_pct > 25 else
                  "Medium"   if stressed_pct > 10 else "Low")

    # Build Recommendations Dynamically to avoid blank lines and bad numbering
    recs = []
    if ndwi_mean < -0.15: recs.append("**Immediate irrigation** — NDWI indicates water stress.")
    if ndwi_mean > 0.30: recs.append("**Drainage check** — NDWI indicates waterlogging risk.")
    if ndre_mean < 0.30: recs.append("**Nitrogen application** — NDRE suggests chlorophyll decline.")
    if stressed_pct > 25: recs.append("**Ground truthing** — >25% stressed area warrants field inspection.")
    if stressed_pct <= 10 and len(recs) == 0: recs.append("**Standard monitoring** — Continue weekly UAV surveys.")
    if len(recs) == 0: recs.append("**Moderate monitoring** — Conditions are fair, but keep an eye on stress progression.")

    recs_md = "\n".join([f"{i+1}. {r}" for i, r in enumerate(recs)])

    # Include EVI and CIre if they exist
    evi_mean = float(idx.get("evi", np.zeros_like(idx["ndvi"])).mean())
    cire_mean = float(idx.get("cire", np.zeros_like(idx["ndvi"])).mean())

    # Yield and Harvest Forecasting details for report
    from src.ai_engine.yield_predictor import CropYieldPredictor
    report_predictor = CropYieldPredictor(crop_type="Paddy Rice")

    # Build weather
    report_weather = {"temperature": 25.0, "humidity": 75.0, "precipitation": 0.0, "wind_speed": 10.0}
    if "weather_assessment" in st.session_state:
        w_obj = st.session_state["weather_assessment"].weather
        report_weather = {
            "temperature": getattr(w_obj, "current_temp", 25.0),
            "humidity": getattr(w_obj, "current_humidity", 75.0),
            "precipitation": getattr(w_obj, "current_precip", 0.0),
            "wind_speed": getattr(w_obj, "wind_speed", [10.0])[0] if isinstance(getattr(w_obj, "wind_speed", 10.0), list) else getattr(w_obj, "wind_speed", 10.0)
        }

    report_forecast_weather = []
    if "weather_assessment" in st.session_state:
        w_obj = st.session_state["weather_assessment"].weather
        for d in range(7):
            temp_max_val = w_obj.temperature_max[d] if len(w_obj.temperature_max) > d else 25.0
            temp_min_val = w_obj.temperature_min[d] if len(w_obj.temperature_min) > d else 18.0
            precip_val = w_obj.precipitation[d] if len(w_obj.precipitation) > d else 0.0
            humidity_val = w_obj.humidity[d] if len(w_obj.humidity) > d else 75.0
            wind_val = w_obj.wind_speed[d] if len(w_obj.wind_speed) > d else 10.0
            precip_prob = 10.0 if precip_val == 0.0 else 80.0

            report_forecast_weather.append({
                "temperature": (temp_max_val + temp_min_val) / 2.0,
                "humidity": humidity_val,
                "precipitation": precip_val,
                "wind_speed": wind_val,
                "precipitation_probability": precip_prob
            })
    else:
        report_forecast_weather = [
            {"temperature": 25.0, "humidity": 75.0, "precipitation": 0.0, "wind_speed": 10.0, "precipitation_probability": 15.0}
            for _ in range(7)
        ]

    # Estimate Above-Ground Biomass (AGB)
    report_biomass = report_predictor.estimate_biomass(
        ndvi=idx["ndvi"],
        ndre=idx["ndre"],
        growth_stage=crop_stage
    )

    # Predict grain yield
    report_yield = report_predictor.predict_yield(
        biomass_map=report_biomass,
        stress_score=idx["stress_score"],
        weather=report_weather,
        growth_stage=crop_stage
    )

    # Generate default GDD
    report_t_base = 10.0
    report_daily_gdd = max(0.0, report_weather["temperature"] - report_t_base)
    report_dat = 45 if crop_stage == "Vegetative" else (70 if crop_stage == "Flowering" else 95)
    report_gdd_accum = float(report_dat * report_daily_gdd)

    # Generate harvest forecast
    report_field_area_ha = 1.5
    report_forecast = report_predictor.generate_harvest_forecast(
        yield_map=report_yield,
        biomass_map=report_biomass,
        current_gdd_accumulated=report_gdd_accum,
        weather_forecast=report_forecast_weather,
        growth_stage=crop_stage,
        days_after_transplanting=report_dat,
        field_area_ha=report_field_area_ha
    )

    report_limiting_factors_md = "\n".join([f"- {factor}" for factor in report_forecast.limiting_factors])
    report_recommendations_md = "\n".join([f"- {rec}" for rec in report_forecast.harvest_recommendations])

    # --- Weather variables for report ---
    _rtemp     = report_weather.get("temperature", 25.0)
    _rhumidity = report_weather.get("humidity", 75.0)
    _rprecip   = report_weather.get("precipitation", 0.0)
    _rwind     = report_weather.get("wind_speed", 10.0)
    _spray_ok  = _rwind <= 15.0 and _rhumidity < 90.0
    _gdd_today = max(0.0, _rtemp - 10.0)

    # --- Pull AI Optimization report if available ---
    _opt = st.session_state.get("opt_report", None)
    if _opt is not None:
        _opt_cost_md       = f"${_opt.total_estimated_cost:.2f}"
        _opt_benefit_md    = f"{_opt.total_projected_benefit:.1f} pts"
        _opt_roi_md        = f"{_opt.average_roi_ratio:.3f}"
        _opt_spray_md      = "Open" if _opt.spraying_feasible else "Blocked"
        _opt_access_md     = "Accessible" if _opt.ground_machinery_accessible else "Restricted"
        _opt_var_md        = f"{_opt.var_95:.1f}%"
        _opt_cvar_md       = f"{_opt.cvar_95:.1f}%"
        _opt_expected_md   = f"{_opt.expected_yield:.1f}%"
        _opt_water_eff_md  = f"{_opt.water_efficiency_score:.1f}%"
        _opt_fert_eff_md   = f"{_opt.fertilizer_efficiency_score:.1f}%"
        _opt_chem_eff_md   = f"{_opt.chemical_efficiency_score:.1f}%"
        _opt_uav_cost_md   = f"${_opt.uav_mission_cost:.2f}/ha"
        _zone_rows = []
        for _act in (_opt.actions[:12] if _opt.actions else []):
            _zone_rows.append(
                f"| {_act.zone_name} | {_act.action_type} | {_act.action_dosage} "
                f"| Day {_act.suggested_day} | ${_act.estimated_cost_usd_ha:.2f}/ha "
                f"| {_act.health_benefit_score:.1f} pts | {_act.net_roi_index:.3f} | {_act.feasibility} |"
            )
        _opt_actions_md = "\n".join(_zone_rows) if _zone_rows else \
            "| — | No actions generated yet | — | — | — | — | — | — |"
        _opt_schedule_md = "\n".join([
            f"- **{day}:** " + "; ".join(tasks)
            for day, tasks in _opt.schedule.items()
        ]) if _opt.schedule else "- No treatment schedule generated yet."
    else:
        _opt_cost_md = _opt_benefit_md = _opt_roi_md = "N/A (Run AI Optimizer tab)"
        _opt_spray_md = _opt_access_md = "Unknown"
        _opt_var_md = _opt_cvar_md = _opt_expected_md = "N/A"
        _opt_water_eff_md = _opt_fert_eff_md = _opt_chem_eff_md = "N/A"
        _opt_uav_cost_md = "N/A"
        _opt_actions_md = "| — | Run AI Optimizer tab to populate actions | — | — | — | — | — | — |"
        _opt_schedule_md = "- Open the **AI Treatment Optimizer** tab and run optimization to populate this section."

    report_md = f"""\
# UAV Crop Stress Intelligence Report

| Field | Value |
|---|---|
| **Platform** | AI-Powered UAV Crop Stress Intelligence Platform v2.0 |
| **Report Generated** | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC |
| **Crop Type** | Paddy Rice |
| **Growth Stage** | {crop_stage} |
| **Field GPS Center** | {lat:.5f}°N, {lon:.5f}°E |
| **Field Area Modeled** | {report_field_area_ha:.1f} hectares |
| **Stress Detection Threshold** | {stress_threshold} |
| **Overall Risk Classification** | **{risk_level}** |

---

## 1. Executive Summary

Full multispectral analysis at **{crop_stage}** growth stage. Composite stress index = **{stress_mean:.4f}**, with **{stressed_pct:.1f}%** of the field classified as stressed (threshold = {stress_threshold}).

{"**CRITICAL ALERT:** Immediate multi-input intervention required. Yield losses without treatment could exceed 30%." if risk_level == "Critical" else "**HIGH ALERT:** Significant crop stress. Zone-level treatment within 5 days strongly recommended." if risk_level == "High" else "**MODERATE ALERT:** Localized stress hotspots detected. Precision intervention in affected zones advised." if risk_level == "Medium" else "**LOW:** Crop health is within acceptable limits. Standard monitoring is sufficient."}

Biomass model: **{report_forecast.estimated_biomass_t_ha:.2f} t/ha** dry matter. Yield model: **{report_forecast.average_yield_t_ha:.2f} t/ha** grain, total **{report_forecast.total_production_t:.2f} tonnes** over {report_field_area_ha:.1f} ha. Harvest readiness: **{report_forecast.harvest_readiness_pct:.1f}%**. Projected harvest: **{report_forecast.predicted_harvest_date.strftime("%B %d, %Y")}** ({report_forecast.days_to_harvest} days from today).

---

## 2. Crop Production & Yield Forecast

### 2.1 Biomass & Yield Model Output

| Parameter | Estimate | Methodology |
|---|---|---|
| Above-Ground Dry Biomass | **{report_forecast.estimated_biomass_t_ha:.2f} t/ha** | LUE: sqrt(NDVI × NDRE) × Stage multiplier × Peak AGB (14.5 t/ha) |
| Predicted Grain Yield | **{report_forecast.average_yield_t_ha:.2f} t/ha** | Harvest Index (0.48) × Stress penalty map |
| Total Field Production | **{report_forecast.total_production_t:.2f} tonnes** | Avg yield × {report_field_area_ha:.1f} ha |
| Harvest Readiness | **{report_forecast.harvest_readiness_pct:.1f}%** | Days After Transplanting / 91-day cycle |
| Daily GDD (today) | **{_gdd_today:.1f} GDD** | max(T_avg − T_base 10°C, 0) |

### 2.2 Harvest Forecast Calendar

| Event | Date | Days from Today |
|---|---|---|
| **Predicted Harvest Date** | **{report_forecast.predicted_harvest_date.strftime("%B %d, %Y")}** | {report_forecast.days_to_harvest} days |
| Optimal Window Opens | {report_forecast.optimal_window_start.strftime("%B %d, %Y")} | {report_forecast.days_to_harvest - 2} days |
| Optimal Window Closes | {report_forecast.optimal_window_end.strftime("%B %d, %Y")} | {report_forecast.days_to_harvest + 2} days |

### 2.3 Primary Yield Limiting Factors

{report_limiting_factors_md}

### 2.4 Agronomic Harvest Recommendations

{report_recommendations_md}

---

## 3. Vegetation Index Deep Analysis

### 3.1 Index Summary Table

| Index | Mean | Threshold | Pass/Fail | Severity |
|---|---|---|---|---|
| NDVI | {ndvi_mean:.4f} | >0.50 healthy | {"PASS" if ndvi_mean > 0.5 else "FAIL"} | {"Excellent" if ndvi_mean > 0.65 else "Good" if ndvi_mean > 0.5 else "Mild" if ndvi_mean > 0.35 else "Severe"} |
| NDRE | {ndre_mean:.4f} | >0.35 adequate | {"PASS" if ndre_mean > 0.35 else "FAIL"} | {"Excellent" if ndre_mean > 0.50 else "Good" if ndre_mean > 0.35 else "Mild" if ndre_mean > 0.25 else "Severe"} |
| NDWI | {ndwi_mean:.4f} | −0.10 to 0.25 | {"PASS" if -0.1 <= ndwi_mean <= 0.25 else "FAIL"} | {"Optimal" if -0.05 <= ndwi_mean <= 0.20 else "Mild stress" if -0.15 <= ndwi_mean <= 0.30 else "Severe"} |
| EVI | {evi_mean:.4f} | >0.40 robust | {"PASS" if evi_mean > 0.4 else "FAIL"} | {"Excellent" if evi_mean > 0.55 else "Good" if evi_mean > 0.40 else "Thin" if evi_mean > 0.25 else "Very thin"} |
| CIre | {cire_mean:.4f} | >1.00 high N | {"PASS" if cire_mean > 1.0 else "FAIL"} | {"Excellent" if cire_mean > 1.8 else "Good" if cire_mean > 1.0 else "Mild deficit" if cire_mean > 0.6 else "Severe deficit"} |
| Stress Score | {stress_mean:.4f} | <0.35 low | {"PASS" if stress_mean < 0.35 else "FAIL"} | {"None" if stress_mean < 0.20 else "Moderate" if stress_mean < 0.50 else "High"} |
| Stressed Area | {stressed_pct:.1f}% | <10% low risk | {"PASS" if stressed_pct < 10 else "FAIL"} | {"None" if stressed_pct < 5 else "Low" if stressed_pct < 10 else "Medium" if stressed_pct < 25 else "High" if stressed_pct < 40 else "Critical"} |

### 3.2 NDVI — Canopy Density & Photosynthesis

**Formula:** NDVI = (NIR − Red) / (NIR + Red)

NDVI = **{ndvi_mean:.4f}** at **{crop_stage}** stage. {"Canopy density is strong. Radiation interception and photosynthetic efficiency are high." if ndvi_mean > 0.60 else "Canopy density is moderate. Adequate but sub-optimal photosynthetic capacity." if ndvi_mean > 0.45 else "Canopy is sparse or stressed. Radiation interception is significantly reduced — yield potential is at risk."}

{"**Note (Flowering/Mature stage):** NDVI should exceed 0.55 for maximum grain-filling efficiency. Values below 0.45 at this stage indicate significant grain-fill failure risk." if crop_stage in ["Flowering", "Mature"] else "**Note (Vegetative stage):** NDVI > 0.50 confirms healthy tillering and canopy establishment. Values below 0.40 indicate poor stand or early biotic/abiotic stress."}

### 3.3 NDRE — Chlorophyll & Leaf Nitrogen

**Formula:** NDRE = (NIR − RedEdge) / (NIR + RedEdge)

NDRE = **{ndre_mean:.4f}**. {"Strong leaf chlorophyll. Nitrogen nutrition is adequate for the current growth stage." if ndre_mean > 0.40 else "Mild chlorophyll decline. Early nitrogen deficiency possible." if ndre_mean > 0.28 else "Significant chlorophyll reduction. Nitrogen deficiency is highly likely — immediate N application is recommended."}

CIre = **{cire_mean:.4f}**, indicating **{"efficient nitrogen uptake and healthy mesophyll photosynthetic activity." if cire_mean > 1.2 else "moderate canopy N — split urea application (40–80 kg N/ha) advisable within 5–7 days." if cire_mean > 0.7 else "severe nitrogen limitation. Priority nutrient top-dress required."}**

### 3.4 NDWI — Canopy Water Content

**Formula:** NDWI = (Green − NIR) / (Green + NIR)

NDWI = **{ndwi_mean:.4f}**. {"Significant water deficit. Crop is under water stress — irrigation should be prioritized immediately." if ndwi_mean < -0.10 else "Elevated canopy moisture. Risk of waterlogging and root anoxia — clear drainage channels." if ndwi_mean > 0.25 else "Canopy water content is within the agronomically optimal range. No irrigation action required at this time."}

### 3.5 EVI — Enhanced Vegetation Index

EVI = **{evi_mean:.4f}**. {"Robust, structurally sound crop canopy. Soil background contamination in the spectral signal is minimal." if evi_mean > 0.40 else "Thin or structurally compromised canopy. Soil background pixels are influencing spectral measurement — ground truth verification is recommended."}

---

## 4. Weather Intelligence

| Parameter | Value | Agronomic Assessment |
|---|---|---|
| Temperature | {_rtemp:.1f}°C | {"Within optimal paddy rice range (22–32°C)." if 22 <= _rtemp <= 32 else "Below optimal — GDD accumulation and crop growth rate will be reduced." if _rtemp < 22 else "Above optimal — heat-induced spikelet sterility risk is elevated."} |
| Relative Humidity | {_rhumidity:.1f}% | {"Ideal canopy humidity." if 50 <= _rhumidity <= 80 else "Low humidity — increased transpiration demand and water stress." if _rhumidity < 50 else "High humidity — elevated blast and sheath blight fungal pathogen risk."} |
| Precipitation | {_rprecip:.1f} mm | {"No rainfall — check irrigation schedule." if _rprecip < 1 else "Moderate rainfall — monitor drainage." if _rprecip < 10 else "Heavy rainfall — field saturation risk. Priority drainage required."} |
| Wind Speed | {_rwind:.1f} km/h | {"Favorable for UAV spray operations (< 15 km/h)." if _rwind <= 15 else "Elevated — UAV spray operations should be deferred to avoid drift."} |
| UAV Spray Window | {"OPEN" if _spray_ok else "BLOCKED"} | {"All weather parameters are within safe UAV spray thresholds." if _spray_ok else "Wind or humidity exceed safe spray thresholds. Reschedule UAV spray."} |
| Daily GDD | {_gdd_today:.1f} GDD | {"Normal thermal accumulation for tropical paddy rice." if _gdd_today >= 10 else "Low GDD — crop phenological development will be delayed."} |

---

## 5. AI Treatment Optimization Report

### 5.1 Optimization Performance Metrics

| Metric | Value |
|---|---|
| Total Optimized Treatment Cost | {_opt_cost_md}/ha |
| Total Projected Health Benefit | {_opt_benefit_md} |
| Benefit / Cost ROI Ratio | {_opt_roi_md} |
| UAV Mission Cost | {_opt_uav_cost_md} |
| UAV Spray Window | {_opt_spray_md} |
| Ground Machinery Access | {_opt_access_md} |
| Stochastic Expected Yield Score | {_opt_expected_md} |
| Value at Risk (VaR 95th pct) | {_opt_var_md} |
| Conditional VaR (CVaR 95th pct) | {_opt_cvar_md} |
| Water Use Efficiency | {_opt_water_eff_md} |
| Fertilizer Efficiency | {_opt_fert_eff_md} |
| Chemical Safety Score | {_opt_chem_eff_md} |

### 5.2 Zone-Level Precision Treatment Actions

| Zone | Action | Dosage | Day | Cost/ha | Benefit | ROI | Feasibility |
|---|---|---|---|---|---|---|---|
{_opt_actions_md}

### 5.3 7-Day Treatment Schedule

{_opt_schedule_md}

---

## 6. Agronomic Recommendations

{recs_md}

---

## 7. Methodology & Technical Details

### 7.1 Sensor & UAV Platform
- **Multispectral Sensor:** 4-band — Green (560 nm), Red (660 nm), Red Edge (730 nm), NIR (840 nm)
- **RGB Camera:** 20 MP for visual ground-truth and texture analysis
- **Dataset:** UAV Multispectral & RGB Dataset — Multi-Stage Paddy Crop Monitoring

### 7.2 Image Processing Pipeline
- Orthorectification via SfM photogrammetric 3D reconstruction
- Radiometric calibration from calibrated reflectance panels
- Atmospheric correction using MODTRAN-based window normalization
- Segmentation via NDVI + NDRE composite rule-based thresholding

### 7.3 AI & Machine Learning Models
- **Classification:** EfficientNet-B0 fine-tuned on 4-channel multispectral input
- **Stress Score:** NDVI (50%) + NDRE (30%) + NDWI (20%) weighted composite
- **Biomass:** LUE model — sqrt(NDVI × NDRE) × Stage multiplier × Peak AGB (14.5 t/ha)
- **Yield:** Harvest Index model — HI baseline 0.48; heat sterility & stress penalties applied
- **Harvest Forecast:** Cumulative GDD thermal tracking; T_base = 10°C
- **Treatment Optimizer:** Q-learning MDP + Monte Carlo Rollout (multi-objective, 7-day horizon)
- **Risk Quantification:** 100–500 Monte Carlo paths; VaR/CVaR at 95th percentile
- **Spatial Zoning:** Connected-component analysis (8-connectivity kernel)

### 7.4 Model Assumptions & Data Quality
- All indices computed from calibrated top-of-canopy reflectance
- Field area set to {report_field_area_ha:.1f} ha — update in sidebar for accurate production totals
- Harvest Index baseline = 0.48 per IRRI paddy rice standard
- GDD T_base = 10°C; total required GDD to maturity = 1,350

---

## 8. Disclaimer

This report is generated by an AI decision support system. All values are model-based estimates derived from UAV multispectral imagery and meteorological data. Actual field conditions may vary. This report should supplement — not replace — agronomist consultation, field scouting, and laboratory analysis before making treatment or harvesting decisions.

---

*Generated by AI-Powered UAV Crop Stress Intelligence Platform v2.0*
*Based on: UAV Multispectral & RGB Dataset for Multi-Stage Paddy Crop Monitoring*
"""




    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Executive Summary</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-label">Overall Risk Level</div>', unsafe_allow_html=True)
        color = "#ff4b4b" if risk_level in ["Critical", "High"] else "#00d2ff"
        st.markdown(f'<div class="metric-value" style="color:{color};">{risk_level}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-label">Stressed Area</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stressed_pct:.1f}%</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-label">Growth Stage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{crop_stage}</div>', unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.info(f"Analysis generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} for coordinates {lat:.4f}°N, {lon:.4f}°E.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Vegetation Index Summary</h2>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI (Vigor)", f"{ndvi_mean:.3f}", delta="Healthy" if ndvi_mean > 0.5 else "Stressed", delta_color="normal" if ndvi_mean > 0.5 else "inverse")
    c2.metric("NDRE (Nitrogen)", f"{ndre_mean:.3f}", delta="Good" if ndre_mean > 0.35 else "Low N", delta_color="normal" if ndre_mean > 0.35 else "inverse")
    c3.metric("NDWI (Water)", f"{ndwi_mean:.3f}", delta="Normal" if -0.1 <= ndwi_mean <= 0.3 else "Stress/Waterlog", delta_color="normal" if -0.1 <= ndwi_mean <= 0.3 else "inverse")
    c4.metric("Stress Score", f"{stress_mean:.3f}", delta=f"{risk_level} Risk", delta_color="inverse" if stress_mean > 0.35 else "normal")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("EVI (Canopy)", f"{evi_mean:.3f}", delta="Robust" if evi_mean > 0.4 else "Thin", delta_color="normal" if evi_mean > 0.4 else "inverse")
    c6.metric("CIre (Chlorophyll)", f"{cire_mean:.3f}", delta="High" if cire_mean > 1.0 else "Low", delta_color="normal" if cire_mean > 1.0 else "inverse")


    st.markdown("#### Stress Progression")
    st.progress(min(stress_mean, 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Stress Assessment & Disease Risk</h2>', unsafe_allow_html=True)
    st.warning(f"**Stressed Field Area:** {stressed_pct:.1f}% (threshold = {stress_threshold})")
    st.markdown(f"### NDVI Interpretation\nNDVI of {ndvi_mean:.3f} at {crop_stage} stage is **{'within expected range' if ndvi_mean > 0.4 else 'below expected — intervention recommended'}**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    c_w, c_n = st.columns(2)
    with c_w:
        st.markdown('<h2 class="premium-title">Irrigation & Water Stress (NDWI)</h2>', unsafe_allow_html=True)
        if ndwi_mean < -0.15:
            st.error(f"NDWI of {ndwi_mean:.3f} indicates severe water stress. Immediate irrigation recommended.")
        elif ndwi_mean > 0.3:
            st.warning(f"NDWI of {ndwi_mean:.3f} indicates potential waterlogging risk.")
        else:
            st.success(f"NDWI of {ndwi_mean:.3f} indicates normal canopy moisture.")
    with c_n:
        st.markdown('<h2 class="premium-title">Chlorophyll Status (NDRE)</h2>', unsafe_allow_html=True)
        if ndre_mean > 0.35:
            st.success(f"NDRE of {ndre_mean:.3f} indicates adequate chlorophyll content.")
        else:
            st.error(f"NDRE of {ndre_mean:.3f} indicates early chlorophyll decline — possible nitrogen deficiency.")
        st.metric("Mean NDRE", f"{ndre_mean:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Recommended Actions</h2>', unsafe_allow_html=True)
    for r in recs:
        st.markdown(r)
    if "Standard monitoring" in "\n".join(recs):
        st.success("No immediate critical actions required. Crop is in good health.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Crop Production & Harvest Forecasting</h2>', unsafe_allow_html=True)
    cy1, cy2, cy3, cy4 = st.columns(4)
    cy1.metric("Predicted Avg Yield", f"{report_forecast.average_yield_t_ha:.2f} t/ha")
    cy2.metric("Total Production", f"{report_forecast.total_production_t:.2f} tonnes")
    cy3.metric("Estimated Biomass", f"{report_forecast.estimated_biomass_t_ha:.2f} t/ha")
    cy4.metric("Harvest Readiness", f"{report_forecast.harvest_readiness_pct:.1f}%")

    st.info(
        f"**Projected Harvest Date:** {report_forecast.predicted_harvest_date.strftime('%B %d, %Y')} "
        f"({report_forecast.days_to_harvest} days remaining)\n\n"
        f"**Optimal Harvest Window:** {report_forecast.optimal_window_start.strftime('%b %d')} to {report_forecast.optimal_window_end.strftime('%b %d, %Y')}"
    )

    c_lf, c_rg = st.columns(2)
    with c_lf:
        st.markdown("#### Primary Yield Limiting Factors")
        for factor in report_forecast.limiting_factors:
            if "Risk" in factor or "Penalty" in factor or "Deficit" in factor or "Retardation" in factor:
                st.error(factor)
            else:
                st.success(factor)
    with c_rg:
        st.markdown("#### Agronomic Harvesting Recommendations")
        for rec in report_forecast.harvest_recommendations:
            st.warning(rec)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="premium-title">Methodology & Export</h2>', unsafe_allow_html=True)
    st.markdown("- **Sensor:** 4-band multispectral UAV camera\n- **Segmentation:** Rule-based index thresholding\n- **Classification:** EfficientNet-B0 (4-channel input)\n- **Stress Score:** Weighted composite index\n- **GIS Zoning:** Connected-component spatial analysis")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Export Full Report")

    # PDF Generation with Markdown parsing
    import tempfile
    from fpdf import FPDF
    import markdown

    class PDF(FPDF):
        def header(self):
            import os
            logo_path = os.path.join(os.path.dirname(__file__), "assets", "garuda_logo.jpg")
            if os.path.exists(logo_path):
                self.image(logo_path, x=10, y=8, w=35)
            self.set_font('Helvetica', 'B', 15)
            self.cell(0, 10, '    UAV Crop Stress Intelligence Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Comprehensive Unicode → ASCII sanitization for Helvetica PDF font compatibility
    _unicode_map = {
        "\u2212": "-",      # minus sign −
        "\u2014": "--",     # em dash —
        "\u2013": "-",      # en dash –
        "\u00d7": "x",      # multiplication sign ×
        "\u00b0": " deg",   # degree sign °
        "\u2265": ">=",     # greater than or equal ≥
        "\u2264": "<=",     # less than or equal ≤
        "\u00b2": "^2",     # superscript 2 ²
        "\u00b3": "^3",     # superscript 3 ³
        "\u03bc": "u",      # mu µ
        "\u2019": "'",      # right single quote '
        "\u2018": "'",      # left single quote '
        "\u201c": '"',      # left double quote "
        "\u201d": '"',      # right double quote "
        "\u2026": "...",    # ellipsis …
        "\u00e9": "e",      # é
        "\u00e8": "e",      # è
        "\u00ea": "e",      # ê
        "\u00e0": "a",      # à
        "\u00e2": "a",      # â
        "\u00f4": "o",      # ô
        "\u2022": "-",      # bullet •
        "\u25cf": "-",      # black circle ●
        "\u2713": "OK",     # check mark
        "\u2717": "X",      # cross mark
        "\u00a0": " ",      # non-breaking space
        "\u00ad": "-",      # soft hyphen
        "\u00d7": "x",      # × multiplication
        "\u2248": "~",      # approximately ≈
        "\u00b1": "+/-",    # plus-minus ±
        "\u2192": "->",     # right arrow →
        "\u2190": "<-",     # left arrow ←
        "\u00b5": "u",      # micro µ
        "\u03b1": "alpha",  # α
        "\u03b2": "beta",   # β
        "\u03b3": "gamma",  # γ
        "\u00f7": "/",      # division ÷
    }
    safe_report_md = report_md
    for uni_char, ascii_sub in _unicode_map.items():
        safe_report_md = safe_report_md.replace(uni_char, ascii_sub)
    # Final fallback: encode to latin-1, replacing any remaining unmapped chars
    safe_report_md = safe_report_md.encode("latin-1", errors="replace").decode("latin-1")
    html_content = markdown.markdown(safe_report_md, extensions=['tables'])

    pdf = PDF()
    pdf.add_page()

    try:
        # fpdf2 allows HTML rendering directly which preserves formatting
        pdf.write_html(html_content)
        pdf_bytes = bytes(pdf.output())
    except Exception as e:
        st.error(f"PDF Generation Error: {e}")
        pdf_bytes = b""

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download Full Report (Markdown)",
            data=report_md,
            file_name=f"crop_stress_report_{crop_stage}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"crop_stress_report_{crop_stage}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 10 — Satellite Analytics
# ══════════════════════════════════════════════════════════════
