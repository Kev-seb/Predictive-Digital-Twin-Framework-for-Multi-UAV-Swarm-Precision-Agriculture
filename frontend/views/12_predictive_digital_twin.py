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
#  Predictive Digital Twin  (original Tab content, unchanged, dedented)
# ══════════════════════════════════════════════════════════════

section_header("Predictive digital twin", "Explore field scenarios, crop trajectories, and treatment outcomes.", category="intelligence")

if "ms_image" not in st.session_state or "indices" not in st.session_state:
    st.warning("Please upload or enable demo data in Upload & Process first.")
else:
    import importlib
    import src.ai_engine.epidemiology
    import src.digital_twin.simulator
    import src.digital_twin.twin

    importlib.reload(src.ai_engine.epidemiology)
    importlib.reload(src.digital_twin.simulator)
    importlib.reload(src.digital_twin.twin)

    from src.digital_twin.twin import FieldDigitalTwin
    from src.ai_engine.treatment_recommender import AITreatmentRecommender
    from src.temporal.temporal_analytics import plot_index_heatmap

    # Instantiate Digital Twin
    twin = FieldDigitalTwin(data_dir=os.environ.get("GARUDA_TWIN_MEMORY_DIR", str(DASHBOARD_DIR / "runtime/digital_twin_memory")))

    # Sync current state
    weather_data = {"temperature": 25.0, "humidity": 75.0, "precipitation": 0.0, "wind_speed": 10.0}
    if "weather_assessment" in st.session_state:
        w = st.session_state["weather_assessment"].weather
        weather_data = {
            "temperature": getattr(w, "current_temp", 25.0),
            "humidity": getattr(w, "current_humidity", 75.0),
            "precipitation": getattr(w, "current_precip", 0.0),
            "wind_speed": getattr(w, "wind_speed", [10.0])[0] if isinstance(getattr(w, "wind_speed", 10.0), list) else getattr(w, "wind_speed", 10.0)
        }

    ndvi_map = st.session_state["indices"]["ndvi"]
    stress_map = st.session_state["indices"]["stress_score"]

    twin.synchronize_twin_state(
        date_str=pd.Timestamp.now().strftime("%Y-%m-%d"),
        ndvi=ndvi_map,
        stress_score=stress_map,
        weather=weather_data,
        active_stage=crop_stage
    )

    # Display Twin Dashboard Metrics
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Cumulative Stress Index", f"{twin.state['cumulative_stress_index']:.3f}")
    dm2.metric("Health Trajectory", twin.state['health_trajectory'])
    dm3.metric("Surveys Logged", len(twin.state['surveys_logged']))
    dm4.metric("Avg Climate Temp", f"{twin.state['historical_weather_summary']['mean_temperature_c']:.1f}°C")

    # Create sub-tabs for the Digital Twin tab
    twin_subtabs = st.tabs(["Ecosystem Simulation & Playback", "Yield & Harvest Forecasting"])

    with twin_subtabs[0]:
        # ----------------- SCENARIO PLAYBACK ENGINE -----------------
        st.subheader("Predictive Scenario Simulation Playback")
        st.markdown(
            "Configure and compare agronomic simulation scenarios (Do Nothing, Custom Treatments, or AI Autonomous Plan) "
            "over a 7-day future window. Visualizes crop recovery, lateral soil nutrient diffusion, moisture evapotranspiration, and fungal spread."
        )

        # Generate prescription zones on the fly
        recommender = AITreatmentRecommender()
        prescriptions, zone_labels = recommender.generate_zone_prescriptions(
            indices=st.session_state["indices"],
            weather={**weather_data, "precipitation_probability": 20.0},
            crop_stage=crop_stage,
            n_zones=grid_size
        )
        zone_names = {p.zone_id: p.zone_name for p in prescriptions}

        # Select Scenario
        scenario_mode = st.selectbox(
            "Select Simulation Scenario",
            ["Do Nothing", "Custom Interventions", "AI Autonomous Plan"]
        )

        scenario_key = {
            "Do Nothing": "do_nothing",
            "Custom Interventions": "custom",
            "AI Autonomous Plan": "ai_planned"
        }[scenario_mode]

        # Init session states for custom interventions
        if "custom_interventions" not in st.session_state:
            st.session_state["custom_interventions"] = []

        ai_budget = 500.0

        if scenario_key == "custom":
            st.write("**Custom Intervention Scheduler**")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                c_day = st.slider("Schedule Day", 1, 7, 1)
            with col_c2:
                c_zone = st.selectbox("Target Zone", list(zone_names.values()))
                c_zone_id = next(k for k, v in zone_names.items() if v == c_zone)
            with col_c3:
                c_type = st.selectbox("Action Type", ["Precision Irrigation", "Nutrient Top-Dress", "Fungicide Spray"])

            c_cost = {"Precision Irrigation": 25.0, "Nutrient Top-Dress": 179.0, "Fungicide Spray": 49.4}[c_type]

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                if st.button("Add Action to Custom Scenario"):
                    st.session_state["custom_interventions"].append({
                        "day": c_day,
                        "zone_id": c_zone_id,
                        "zone_name": c_zone,
                        "type": c_type,
                        "cost": c_cost
                    })
                    st.success(f"Added {c_type} on Day {c_day} targeting {c_zone}.")
            with c_col2:
                if st.button("Reset Custom Actions"):
                    st.session_state["custom_interventions"] = []
                    st.warning("Cleared all custom scenario actions.")

            if st.session_state["custom_interventions"]:
                st.dataframe(pd.DataFrame(st.session_state["custom_interventions"]), hide_index=True)

        elif scenario_key == "ai_planned":
            st.write("**AI Autonomous Intervention Planner**")
            ai_budget = st.slider("AI Budget Limit ($/ha)", 100, 1000, 500, 50)
        else:
            ai_budget = 500.0

        st.write("---")
        st.write("**Epidemiological Contagion & Microclimate Controls**")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            prop_model = st.selectbox(
                "Contagion Propagation Model",
                ["Fisher-Kolmogorov PDE (Anisotropic)", "Directed Graph Neural Network (GNN)", "Hybrid PDE-GNN Spore Model", "Baseline Diffusion"],
                index=0
            )
            prop_key = {
                "Fisher-Kolmogorov PDE (Anisotropic)": "pde",
                "Directed Graph Neural Network (GNN)": "gnn",
                "Hybrid PDE-GNN Spore Model": "hybrid",
                "Baseline Diffusion": "baseline"
            }[prop_model]
        with ec2:
            wind_dir = st.slider("Predominant Wind Direction (Degrees)", 0, 360, 45, 15, help="0° = North, 90° = East, 180° = South, 270° = West")
        with ec3:
            stage_select = st.selectbox(
                "Canopy Susceptibility Crop Stage",
                ["Emergence", "Vegetative", "Flowering", "Senescence"],
                index=["Emergence", "Vegetative", "Flowering", "Senescence"].index(crop_stage) if crop_stage in ["Emergence", "Vegetative", "Flowering", "Senescence"] else 1
            )

        # Construct weather forecast
        weather_forecast = []
        if "weather_assessment" in st.session_state:
            w = st.session_state["weather_assessment"].weather
            for d in range(7):
                temp_max_val = w.temperature_max[d] if len(w.temperature_max) > d else 25.0
                temp_min_val = w.temperature_min[d] if len(w.temperature_min) > d else 18.0
                precip_val = w.precipitation[d] if len(w.precipitation) > d else 0.0
                humidity_val = w.humidity[d] if len(w.humidity) > d else 75.0
                wind_val = w.wind_speed[d] if len(w.wind_speed) > d else 10.0
                precip_prob = 10.0 if precip_val == 0.0 else 80.0

                weather_forecast.append({
                    "temperature": (temp_max_val + temp_min_val) / 2.0,
                    "humidity": humidity_val,
                    "precipitation": precip_val,
                    "wind_speed": wind_val,
                    "wind_direction": wind_dir,
                    "precipitation_probability": precip_prob
                })
        else:
            weather_forecast = [
                {"temperature": 25.0, "humidity": 75.0, "precipitation": 0.0, "wind_speed": 10.0, "wind_direction": wind_dir, "precipitation_probability": 15.0}
                for _ in range(7)
            ]

        # Trigger Simulation
        if st.button("Run Scenario Simulation Model", type="primary"):
            with st.spinner("Executing dynamic agronomic simulation loops..."):
                try:
                    sim_res = twin.run_scenario_simulation(
                        scenario_type=scenario_key,
                        forecast_days=7,
                        weather_forecast=weather_forecast,
                        custom_interventions=st.session_state["custom_interventions"],
                        budget_limit=ai_budget,
                        indices=st.session_state["indices"],
                        zone_labels=zone_labels,
                        zone_names=zone_names,
                        center_lat=lat if "lat" in locals() else 11.0,
                        center_lon=lon if "lon" in locals() else 79.0,
                        propagation_model=prop_key,
                        growth_stage=stage_select
                    )
                    st.session_state["simulation_result"] = sim_res
                    st.success("Simulation complete! Use the playback slider below to explore forecast timelines.")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

        # Playback section
        if "simulation_result" in st.session_state:
            sim_res = st.session_state["simulation_result"]

            st.write("---")
            st.subheader("High-Performance GPU Rendering Pipeline (Live Drone State)")
            st.markdown("Powered by PyTorch (CUDA) physics compute & WebGL/Deck.gl rendering.")

            import pydeck as pdk
            # Render Live Drone & GPU Particles
            live_monitor = st.checkbox("Enable Live GPU Particle & Telemetry Feed (Runs for 15s)", value=False)


            if live_monitor:
                placeholder = st.empty()
                status_placeholder = st.empty()

                # Get or create a local GPU engine for this rendering loop
                engine = shared_state.get("GPU_ENGINE")
                if engine is None:
                    from src.digital_twin.gpu_physics import GPUPhysicsEngine
                    engine = GPUPhysicsEngine()
                    shared_state["GPU_ENGINE"] = engine

                # Use drone telemetry position as center
                telemetry = shared_state["MAVLINK_TELEMETRY"]
                center_lat = telemetry.get("lat", 11.0)
                center_lon = telemetry.get("lon", 79.0)

                # Force valid position if at origin
                if abs(center_lat) < 0.01 and abs(center_lon) < 0.01:
                    center_lat, center_lon = 11.0, 79.0
                    telemetry["lat"] = center_lat
                    telemetry["lon"] = center_lon

                # Generate terrain heatmap once
                terrain_data = engine.generate_terrain_heatmap(
                    width=64, height=64,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    extent_deg=0.005
                )
                df_terrain = pd.DataFrame(terrain_data) if terrain_data else pd.DataFrame(columns=['lon', 'lat', 'weight'])
                if not df_terrain.empty and 'color' not in df_terrain.columns:
                    df_terrain['color'] = df_terrain.apply(
                        lambda row: [255, int(255 - row.get('weight', 0) * 255), 0, 120], axis=1
                    )

                import time as _time

                for frame_i in range(150):
                    # Force spraying on so particles emit
                    telemetry["is_spraying"] = True
                    telemetry["connected"] = True
                    if telemetry["alt"] < 2.0:
                        telemetry["alt"] = 10.0

                    drone_lon = telemetry["lon"]
                    drone_lat = telemetry["lat"]
                    drone_alt = telemetry["alt"]

                    # === DIRECTLY DRIVE THE PHYSICS ENGINE ===
                    # Emit new spray particles from drone position
                    engine.emit_particles(
                        count=80,
                        source_pos=(drone_lon, drone_lat, drone_alt - 1.0),
                        initial_velocity=(0.0, 0.0, -3.0),
                        spread=1.5
                    )

                    # Step physics with wind
                    wind_speed = CURRENT_ENV.get("wind_speed", 5.0)
                    wind_dir = math.radians(CURRENT_ENV.get("wind_direction", 45.0))
                    wx = math.cos(wind_dir) * wind_speed * 9e-6
                    wy = math.sin(wind_dir) * wind_speed * 9e-6
                    engine.update_particles(dt=0.05, wind_vector=(wx, wy, 0.0))

                    # Get particle positions
                    particle_arr = engine.get_active_particles_numpy()

                    df_drone = pd.DataFrame([{
                        "lon": drone_lon, "lat": drone_lat, "alt": drone_alt
                    }])

                    if len(particle_arr) > 0:
                        df_particles = pd.DataFrame(particle_arr, columns=["lon", "lat", "alt"])
                    else:
                        df_particles = pd.DataFrame(columns=["lon", "lat", "alt"])

                    # Build Deck.gl Layers
                    layers = []

                    if not df_terrain.empty:
                        layers.append(pdk.Layer(
                            "GridCellLayer",
                            data=df_terrain,
                            get_position='[lon, lat]',
                            get_elevation='weight * 10',
                            get_fill_color='color',
                            elevation_scale=1,
                            cell_size=10,
                            extruded=True,
                        ))

                    if not df_particles.empty:
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=df_particles,
                            get_position='[lon, lat, alt]',
                            get_fill_color=[0, 150, 255, 200],
                            get_radius=0.5,
                            radius_min_pixels=2,
                            radius_max_pixels=10
                        ))

                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=df_drone,
                        get_position='[lon, lat, alt]',
                        get_fill_color=[255, 0, 0, 255],
                        get_radius=2.0,
                        radius_min_pixels=5,
                        radius_max_pixels=15
                    ))

                    view_state = pdk.ViewState(
                        longitude=drone_lon,
                        latitude=drone_lat,
                        zoom=18,
                        pitch=45,
                        bearing=telemetry.get("yaw", 0.0) * (180/math.pi)
                    )

                    deck = pdk.Deck(
                        layers=layers,
                        initial_view_state=view_state,
                        map_provider="carto",
                        map_style="dark"
                    )

                    with placeholder:
                        st.pydeck_chart(deck)

                    with status_placeholder:
                        st.caption(f"Frame {frame_i+1}/150 | Particles: {len(particle_arr)} | Drone: ({drone_lat:.4f}, {drone_lon:.4f}) @ {drone_alt:.1f}m")

                    _time.sleep(0.1)

                st.info("Live feed paused. Uncheck and recheck to resume.")

            # ----------------- STATIC SIMULATION RESULTS -----------------
            st.write("---")
            st.subheader("Simulation Forecast Results")
            max_day = max(0, len(sim_res["maps_history"]) - 1)
            playback_day = st.slider("Playback Timeline (Day)", 0, max_day, 0)
            day_maps = sim_res["maps_history"][playback_day]

            # Static grid maps
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                fig_ndvi = plot_index_heatmap(day_maps["ndvi"], f"Predicted NDVI (Day {playback_day})", "RdYlGn", -1.0, 1.0)
                st.image(fig_to_bytes(fig_ndvi), use_container_width=True)
                fig_soil = plot_index_heatmap(day_maps["moisture"], f"Soil Moisture Grid (Day {playback_day})", "Blues", 0.0, 1.0)
            st.image(fig_to_bytes(fig_soil), use_container_width=True)
            with m_col2:
                fig_n = plot_index_heatmap(day_maps["nitrogen"], f"Soil Nitrogen Grid (Day {playback_day})", "YlOrBr", 0.0, 1.0)
                st.image(fig_to_bytes(fig_n), use_container_width=True)
                fig_fung = plot_index_heatmap(day_maps["fungus"], f"Fungal Load Grid (Day {playback_day})", "Purples", 0.0, 1.0)
                st.image(fig_to_bytes(fig_fung), use_container_width=True)

            # Fungal propagation vectors & boundaries
            if "fungus_urgency" in day_maps:
                st.write("---")
                st.subheader("Spatiotemporal Pathogen Contagion Analytics")
                st.markdown("Dynamic epidemiological projections using reaction-diffusion & wind-dispersal advective modeling.")

                ep_col1, ep_col2 = st.columns(2)
                with ep_col1:
                    fig_urg = plot_urgency_velocity(
                        urgency=day_maps["fungus_urgency"],
                        velocity_x=day_maps["fungus_direction"][0] * day_maps["fungus_velocity"] if "fungus_direction" in day_maps else np.zeros_like(day_maps["fungus_urgency"]),
                        velocity_y=day_maps["fungus_direction"][1] * day_maps["fungus_velocity"] if "fungus_direction" in day_maps else np.zeros_like(day_maps["fungus_urgency"]),
                        title=f"Treatment Urgency & Outbreak Expansion Vectors (Day {playback_day})"
                    )
                    st.image(fig_to_bytes(fig_urg), use_container_width=True)
                with ep_col2:
                    fig_bound = plot_boundaries_contours(
                        pathogen=day_maps["fungus"],
                        boundaries=day_maps["fungus_boundaries"] if "fungus_boundaries" in day_maps else np.zeros_like(day_maps["fungus"]),
                        title=f"Contagion Progression & Probabilistic Boundaries (Day {playback_day})"
                    )
                    st.image(fig_to_bytes(fig_bound), use_container_width=True)

                # Epidemiological Scorecard Metrics
                st.write("#### Epidemiological Forecast Scorecard")
                es1, es2, es3, es4 = st.columns(4)
                with es1:
                    inf_area = np.mean(day_maps["fungus"] > 0.10) * 100
                    st.metric("Contagion Area", f"{inf_area:.1f}%", help="Percentage of field with pathogen pressure > 10%")
                with es2:
                    inf_velocity = np.mean(day_maps["fungus_velocity"]) * 100 if "fungus_velocity" in day_maps else 0.0
                    st.metric("Spore Spread Velocity", f"{inf_velocity:.2f} %/day", help="Contagion wavefront expansion speed")
                with es3:
                    peak_pressure = np.max(day_maps["fungus"]) * 100
                    st.metric("Outbreak Intensity", f"{peak_pressure:.1f}%", help="Maximum pathogen density in the field")
                with es4:
                    max_urgency = np.max(day_maps["fungus_urgency"]) * 100
                    st.metric("Max Urgency", f"{max_urgency:.1f}%", help="Peak intervention priority rating")

            # Display Timeline Logs for that day
            st.subheader(f"Timeline Engine Event Log (Up to Day {playback_day})")

            for line in sim_res["timeline"]:
                # Render lines belonging to days <= playback_day
                for d in range(playback_day + 1):
                    if f"Day {d}:" in line or f"Day {d} " in line or f"--- Day {d} ---" in line:
                        st.info(line)
                        break

            # AI-Assisted Mission Generation Download Block
            if sim_res.get("qgc_mission") is not None:
                st.write("---")
                st.subheader("AI-Assisted Mission Flight Plan")
                st.markdown(
                    "The system has compiled the scheduled spatial treatments into a standard MAVLink "
                    "waypoint flight plan. You can download the QGroundControl Plan directly."
                )

                import json
                qgc_json = json.dumps(sim_res["qgc_mission"], indent=2)
                st.download_button(
                    label="Download QGroundControl Flight Plan (.mission)",
                    data=qgc_json,
                    file_name=f"{scenario_key}_precision_mission.mission",
                    mime="application/json"
                )

    with twin_subtabs[1]:
        st.subheader("Predictive Crop Yield, Biomass & Harvest Forecasting Dashboard")
        st.markdown(
            "A predictive agronomic model simulating pixel-level crop yield and above-ground biomass accumulated, "
            "integrated with thermal Growing Degree Days (GDD) tracking and climatic forecast risks."
        )

        # Interactive Control Panel
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            crop_choice = st.selectbox("Predictive Crop Parameter Model", ["Paddy Rice", "Corn", "Wheat"], index=0)
            field_area = st.number_input("Field Area (Hectares)", min_value=0.1, max_value=100.0, value=1.5, step=0.1)

        with col_p2:
            dat_slider = st.slider(
                "Days After Planting / Transplanting (DAT)",
                1, 120,
                45 if crop_stage == "Vegetative" else (70 if crop_stage == "Flowering" else 95)
            )

            # Default GDD based on standard daily thermal accumulation
            t_base_val = 10.0 if crop_choice in ["Paddy Rice", "Corn"] else 4.0
            daily_gdd_est = max(0.0, weather_data["temperature"] - t_base_val)
            default_gdd = float(dat_slider * daily_gdd_est)

            gdd_accumulated = st.number_input(
                "Accumulated Growing Degree Days (GDD) to Date (°C-days)",
                min_value=0.0, max_value=2000.0,
                value=default_gdd, step=10.0
            )

        # Run calculations
        from src.ai_engine.yield_predictor import CropYieldPredictor
        yield_pred = CropYieldPredictor(crop_type=crop_choice)

        # Estimate AGB (Biomass)
        biomass_map = yield_pred.estimate_biomass(
            ndvi=ndvi_map,
            ndre=st.session_state["indices"]["ndre"],
            growth_stage=crop_stage
        )

        # Predict Yield Map
        yield_map = yield_pred.predict_yield(
            biomass_map=biomass_map,
            stress_score=stress_map,
            weather=weather_data,
            growth_stage=crop_stage
        )

        # Generate harvest forecast
        h_forecast = yield_pred.generate_harvest_forecast(
            yield_map=yield_map,
            biomass_map=biomass_map,
            current_gdd_accumulated=gdd_accumulated,
            weather_forecast=weather_forecast,
            growth_stage=crop_stage,
            days_after_transplanting=dat_slider,
            field_area_ha=field_area
        )

        # Display Forecast Summary Cards
        st.markdown("### Harvest Forecasting Dashboard & Scorecard")
        card1, card2, card3, card4 = st.columns(4)
        with card1:
            st.metric("Avg Predicted Yield", f"{h_forecast.average_yield_t_ha:.2f} t/ha")
        with card2:
            st.metric("Total Expected Production", f"{h_forecast.total_production_t:.2f} tonnes")
        with card3:
            st.metric("Estimated Biomass", f"{h_forecast.estimated_biomass_t_ha:.2f} t/ha")
        with card4:
            st.metric("Harvest Readiness Index", f"{h_forecast.harvest_readiness_pct:.1f}%")

        # Progress bar
        st.progress(h_forecast.harvest_readiness_pct / 100.0)

        # Sub-panel for Dates & Windows
        st.info(
            f"**Projected Harvest Date:** {h_forecast.predicted_harvest_date.strftime('%B %d, %Y')} "
            f"({h_forecast.days_to_harvest} days remaining)\n\n"
            f"**Optimal Harvest Window:** {h_forecast.optimal_window_start.strftime('%b %d')} to {h_forecast.optimal_window_end.strftime('%b %d, %Y')}"
        )

        # Map Rendering
        st.markdown("### Spatiotemporal Yield & Biomass Maps")
        map_col1, map_col2 = st.columns(2)
        with map_col1:
            fig_y = plot_index_heatmap(
                yield_map,
                "Predicted Local Crop Yield Map (t/ha)",
                "YlGn",
                0.0,
                max(1.0, float(yield_map.max()))
            )
            st.image(fig_to_bytes(fig_y), use_container_width=True)
            st.caption("Grain yield map modeling nitrogen/chlorophyll efficiency & stress penalty factor.")
        with map_col2:
            fig_b = plot_index_heatmap(
                biomass_map,
                "Estimated Above-Ground Biomass Map (t/ha)",
                "Greens",
                0.0,
                max(1.0, float(biomass_map.max()))
            )
            st.image(fig_to_bytes(fig_b), use_container_width=True)
            st.caption("Total accumulated vegetative biomass (dry matter) before crop senescence.")

        # Limiting Factors & Recommendations
        lf_col1, lf_col2 = st.columns(2)
        with lf_col1:
            st.markdown("### Primary Yield Limiting Factors")
            for factor in h_forecast.limiting_factors:
                if "Risk" in factor or "Penalty" in factor or "Deficit" in factor or "Retardation" in factor:
                    st.error(factor)
                else:
                    st.success(factor)
        with lf_col2:
            st.markdown("### Agronomic Harvesting Recommendations")
            for rec in h_forecast.harvest_recommendations:
                st.warning(rec)


# ══════════════════════════════════════════════════════════════
# TAB 8 — AI Input Optimizer
# ══════════════════════════════════════════════════════════════
