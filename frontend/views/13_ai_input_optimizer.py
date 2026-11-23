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
#  AI Input Optimizer  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Input & treatment optimizer", "Compare treatment schedules, resource allocation, and projected outcomes.", category="intelligence")

if "ms_image" not in st.session_state or "indices" not in st.session_state:
    st.warning("Please upload or enable demo data in Upload & Process first.")
else:
    import importlib
    import src.ai_engine.treatment_recommender
    import src.ai_engine.treatment_optimizer

    importlib.reload(src.ai_engine.treatment_recommender)
    importlib.reload(src.ai_engine.treatment_optimizer)

    from src.ai_engine.treatment_recommender import AITreatmentRecommender
    from src.ai_engine.treatment_optimizer import AITreatmentOptimizer

    # Prepare weather
    weather_data = {"temperature": 25.0, "humidity": 75.0, "precipitation_probability": 20.0, "wind_speed": 10.0}
    if "weather_assessment" in st.session_state:
        w = st.session_state["weather_assessment"].weather

        # Determine wind speed float
        wind_val = 10.0
        if hasattr(w, "wind_speed"):
            if isinstance(w.wind_speed, list) and len(w.wind_speed) > 0:
                wind_val = float(w.wind_speed[0])
            elif isinstance(w.wind_speed, (int, float)):
                wind_val = float(w.wind_speed)

        # Determine precipitation probability (defaulting to 80% if there is current rain)
        precip_prob = 10.0
        if hasattr(w, "current_precip") and w.current_precip > 0.5:
            precip_prob = 80.0
        elif hasattr(w, "precipitation") and isinstance(w.precipitation, list) and len(w.precipitation) > 0:
            if w.precipitation[0] > 0.5:
                precip_prob = 80.0

        weather_data = {
            "temperature": float(w.current_temp),
            "humidity": float(w.current_humidity),
            "precipitation_probability": float(precip_prob),
            "wind_speed": float(wind_val)
        }

    recommender = AITreatmentRecommender()
    optimizer = AITreatmentOptimizer()

    # Generate clustered prescriptions
    prescriptions, zone_labels = recommender.generate_zone_prescriptions(
        indices=st.session_state["indices"],
        weather=weather_data,
        crop_stage=crop_stage,
        n_zones=grid_size
    )

    # Show configuration column layout
    opt_col1, opt_col2 = st.columns([1, 1])
    with opt_col1:
        st.subheader("Optimization Settings")
        optimization_model = st.selectbox(
            "AI Optimization Engine",
            ["Heuristic Knapsack", "Reinforcement Learning (MDP)", "Monte Carlo Rollout"],
            help="Heuristic: Greedy ROI. RL: Markov Decision Process with state-dependent Q-learning. Monte Carlo: stochastic rollout planner."
        )
        budget_limit = st.slider("Total Intervention Budget ($/ha)", 100, 1000, 450, 50)

        risk_profile = st.select_slider(
            "Risk Aversion Tolerance",
            options=["Risk-Averse", "Risk-Neutral", "Risk-Seeking"],
            value="Risk-Neutral",
            help="Risk-Averse: Optimizes CVaR to protect against extreme weather/disease events. Risk-Seeking: Maximizes peak potential returns."
        )

        mc_runs = st.slider("Monte Carlo Simulation Runs", 10, 300, 100, 10, help="Number of weather-perturbed rollout evaluations.")

        with st.expander("Multi-Objective Utility Weights", expanded=False):
            w_yield = st.slider("Yield Maximization Weight", 0.0, 2.0, 1.0, 0.1)
            w_cost = st.slider("Cost Minimization Weight", 0.0, 2.0, 1.0, 0.1)
            w_water = st.slider("Water Conservation Weight", 0.0, 2.0, 1.0, 0.1)
            w_chem = st.slider("Chemical Safety Weight", 0.0, 2.0, 1.0, 0.1)
            w_uav = st.slider("UAV Flight Efficiency Weight", 0.0, 2.0, 1.0, 0.1)

        run_opt = st.button("Run AI Optimization Solver", type="primary")

    with opt_col2:
        st.subheader("Environmental Threshold Checks")
        feasible_spray, spray_reason = optimizer.evaluate_weather_sprayability(weather_data)
        field_avg_ndwi = float(sum(p.ndwi_mean for p in prescriptions) / len(prescriptions))
        accessible, access_reason = optimizer.evaluate_field_accessibility(field_avg_ndwi)

        # Diagnostic status comparisons
        wind_ok = weather_data['wind_speed'] < 15.0
        rain_ok = weather_data['precipitation_probability'] < 50.0
        ndwi_ok = field_avg_ndwi < 0.30

        st.markdown(f"""
        <table style='width:100%; border-collapse: collapse; font-size:0.9rem; margin-bottom:15px;'>
            <tr style='border-bottom: 1px solid #334155; color:#cbd5e1;'>
                <th style='text-align:left; padding:8px;'>Parameter</th>
                <th style='text-align:left; padding:8px;'>Value</th>
                <th style='text-align:left; padding:8px;'>Scientific Limit</th>
                <th style='text-align:left; padding:8px;'>Status</th>
            </tr>
            <tr>
                <td style='padding:8px;'>Wind Speed</td>
                <td style='padding:8px;'>{weather_data['wind_speed']:.1f} km/h</td>
                <td style='padding:8px;'>&lt; 15.0 km/h</td>
                <td style='padding:8px; color:{"#4ade80" if wind_ok else "#f87171"};'><b>{"SAFE" if wind_ok else "HIGH DRIFT RISK"}</b></td>
            </tr>
            <tr>
                <td style='padding:8px;'>Precipitation Prob.</td>
                <td style='padding:8px;'>{weather_data['precipitation_probability']:.1f}%</td>
                <td style='padding:8px;'>&lt; 50.0%</td>
                <td style='padding:8px; color:{"#4ade80" if rain_ok else "#f87171"};'><b>{"SAFE" if rain_ok else "WASHOUT RISK"}</b></td>
            </tr>
            <tr>
                <td style='padding:8px;'>Field Satiation (NDWI)</td>
                <td style='padding:8px;'>{field_avg_ndwi:.3f}</td>
                <td style='padding:8px;'>&lt; 0.300</td>
                <td style='padding:8px; color:{"#4ade80" if ndwi_ok else "#f87171"};'><b>{"ACCESSIBLE" if ndwi_ok else "WATERLOGGED"}</b></td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        if feasible_spray:
            st.success("Spread window open: Favorable wind/rain window.")
        else:
            st.error(f"Spray window blocked: {spray_reason}")

        if accessible:
            st.success("Field accessible: Heavy machinery can enter.")
        else:
            st.warning(f"Field saturated: {access_reason}")

    if run_opt or "opt_report" in st.session_state:
        if run_opt:
            report = optimizer.optimize_treatment_plan(
                prescriptions=prescriptions,
                weather=weather_data,
                budget_limit=budget_limit,
                optimization_model=optimization_model,
                objective_weights={
                    "yield": w_yield,
                    "cost": w_cost,
                    "water": w_water,
                    "chem": w_chem,
                    "uav": w_uav
                },
                risk_profile=risk_profile,
                mc_runs=mc_runs
            )
            st.session_state["opt_report"] = report

        report = st.session_state["opt_report"]

        st.subheader("Autonomous Optimization & ROI Summary")
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        stat_c1.metric("Projected Total Cost", f"${report.total_estimated_cost:.2f}/ha")
        stat_c2.metric("Projected Crop Recovery Benefit", f"{report.total_projected_benefit:.1f} pts")
        stat_c3.metric("Benefit/Cost ROI Ratio", f"{report.average_roi_ratio:.3f}")

        # Draw extended risk-aware stats scorecard if AI models are used
        if optimization_model != "Heuristic Knapsack":
            st.markdown("##### Extended Risk & Resource Efficiency Analytics")
            es1, es2, es3, es4 = st.columns(4)
            es1.metric("Value at Risk (VaR 95%)", f"{report.var_95:.1f}%")
            es2.metric("Worst-case Yield (CVaR 95%)", f"{report.cvar_95:.1f}%")
            es3.metric("UAV Flight Cost Allocation", f"${report.uav_mission_cost:.2f}/ha")
            es4.metric("Water Savings Index", f"{report.water_efficiency_score:.1f}%")

            # Plot charts side-by-side
            ch_col1, ch_col2 = st.columns(2)

            # Chart 1: Yield distribution
            with ch_col1:
                fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                ax1.hist(report.yield_samples, bins=15, density=True, color="#818cf8", alpha=0.65, edgecolor="#4f46e5", label="Simulated Paths")
                ax1.axvline(report.expected_yield, color="#10b981", linestyle="--", linewidth=2, label=f"Expected Yield: {report.expected_yield:.1f}%")
                ax1.axvline(report.var_95, color="#f59e0b", linestyle="-.", linewidth=2, label=f"VaR (95%): {report.var_95:.1f}%")
                ax1.axvline(report.cvar_95, color="#ef4444", linestyle=":", linewidth=2, label=f"CVaR (95%): {report.cvar_95:.1f}%")
                ax1.set_title("Stochastic Yield Probability Curve", fontsize=10, color="white", weight="bold")
                ax1.set_xlabel("Projected Crop Yield (%)", fontsize=8, color="white")
                ax1.set_ylabel("Probability Density", fontsize=8, color="white")
                ax1.legend(fontsize=7, facecolor="#1e1b4b", edgecolor="none", labelcolor="white")

                fig1.patch.set_facecolor("#0f121a")
                ax1.set_facecolor("#1e293b")
                ax1.spines['bottom'].set_color('#475569')
                ax1.spines['left'].set_color('#475569')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.tick_params(colors='white', labelsize=7)
                ax1.grid(color="#334155", linestyle=":", alpha=0.5)
                st.pyplot(fig1)

            # Chart 2: Pareto scores
            with ch_col2:
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                labels = list(report.pareto_scores.keys())
                values = list(report.pareto_scores.values())
                colors = ["#10b981", "#3b82f6", "#06b6d4", "#ec4899", "#8b5cf6"]
                bars = ax2.barh(labels, values, color=colors, height=0.55, edgecolor="none")
                ax2.set_xlim(0, 100)
                ax2.set_title("AI Multi-Objective Pareto Performance", fontsize=10, color="white", weight="bold")
                ax2.set_xlabel("Performance Score (0-100)", fontsize=8, color="white")

                fig2.patch.set_facecolor("#0f121a")
                ax2.set_facecolor("#1e293b")
                ax2.spines['bottom'].set_color('#475569')
                ax2.spines['left'].set_color('#475569')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(colors='white', labelsize=7)
                ax2.grid(color="#334155", linestyle=":", alpha=0.5)
                for bar in bars:
                    width = bar.get_width()
                    ax2.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}",
                             va='center', ha='left', color='white', fontsize=7, weight='bold')
                st.pyplot(fig2)

        # Action item table
        st.subheader("Recommended Variable-Rate Actions")
        action_rows = []
        for act in report.actions:
            action_rows.append({
                "Zone": act.zone_name,
                "Action Type": act.action_type,
                "Target Dosage": act.action_dosage,
                "Est Cost ($/ha)": f"${act.estimated_cost_usd_ha:.2f}",
                "Benefit Score": f"{act.health_benefit_score:.1f}",
                "Priority": act.priority,
                "Feasibility": act.feasibility
            })
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

        # 7-day schedule calendar
        st.subheader("Optimal 7-Day Intervention Schedule")
        for day, day_actions in report.schedule.items():
            with st.expander(f"{day} Calendar"):
                for a in day_actions:
                    st.markdown(f"- {a}")

        # Export GIS VRA map
        st.subheader("Export Precision Agriculture GIS Layer")
        st.markdown(
            "Export variable-rate application maps in standard GIS GeoJSON format. "
            "This format is directly compatible with modern onboard tractor computer control units."
        )

        # Generate export path
        export_path = "outputs/prescriptions/field_vra_prescription.geojson"
        recommender.export_gis_geojson(
            prescriptions=prescriptions,
            zone_labels=zone_labels,
            center_lat=lat,
            center_lon=lon,
            output_file=export_path
        )

        try:
            with open(export_path, "r") as f:
                geojson_data = f.read()

            st.download_button(
                label="Download VRA GIS Layer (GeoJSON)",
                data=geojson_data,
                file_name="field_vra_prescription.geojson",
                mime="application/geo+json"
            )
        except FileNotFoundError:
            st.error("GeoJSON export failed. File not found.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — Spatial Reconstruction (UAV Photogrammetry)
# ══════════════════════════════════════════════════════════════
