"""
twin.py
-------
Core Field Digital Twin state tracking, memory synchronization, and simulation engine.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.digital_twin.simulator import AgronomicDigitalTwinSimulator, QGCMissionGenerator


class FieldDigitalTwin:
    def __init__(self, field_id: str = "field_alpha", data_dir: str = "outputs/digital_twin_memory"):
        self.field_id = field_id
        self.memory_dir = Path(data_dir) / field_id
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.memory_dir / "twin_state.json"
        self.history_dir = self.memory_dir / "historical_surveys"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        """
        Load the persistent digital twin state from disk.
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    if "event_timeline" not in state:
                        state["event_timeline"] = []
                    return state
            except Exception:
                pass
                
        # Default fresh digital twin state
        return {
            "field_id": self.field_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
            "active_growth_stage": "Vegetative",
            "cumulative_stress_index": 0.0,
            "health_trajectory": "Stable",  # Stable, Improving, Declining
            "surveys_logged": [],           # list of survey date strings
            "applied_treatments_log": [],  # list of dicts: date, type, cost
            "historical_weather_summary": {
                "cumulative_precipitation_mm": 0.0,
                "mean_temperature_c": 24.5
            },
            "event_timeline": []
        }

    def save_state(self) -> None:
        """
        Save the state to persistent JSON store.
        """
        self.state["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4)

    def log_intervention(self, action_type: str, cost: float, notes: str) -> None:
        """
        Record a crop treatment event in the digital twin's memory.
        """
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        event = {
            "timestamp": now_str,
            "action_type": action_type,
            "cost_usd": cost,
            "notes": notes
        }
        self.state["applied_treatments_log"].append(event)
        
        # Log to timeline
        self.state["event_timeline"].append({
            "timestamp": now_str,
            "type": "Treatment",
            "description": f"Applied {action_type} (Cost: ${cost:.2f}/ha): {notes}"
        })
        
        # Simulating treatment effects on trajectory
        if "Fungicide" in action_type or "Nutrient" in action_type or "Irrigation" in action_type:
            self.state["health_trajectory"] = "Improving"
            
        self.save_state()

    def synchronize_twin_state(
        self,
        date_str: str,
        ndvi: np.ndarray,
        stress_score: np.ndarray,
        weather: Dict[str, Any],
        active_stage: str
    ) -> None:
        """
        Synchronize the twin with new UAV flight imagery and environmental data.
        Saves the compressed arrays on disk for historical analysis.
        """
        # Save compressed arrays to history
        npz_path = self.history_dir / f"survey_{date_str}.npz"
        np.savez_compressed(npz_path, ndvi=ndvi, stress=stress_score)
        
        # Update list of surveys
        if date_str not in self.state["surveys_logged"]:
            self.state["surveys_logged"].append(date_str)
            self.state["surveys_logged"].sort()
            
        # Update growth stage
        self.state["active_growth_stage"] = active_stage
        
        # Calculate new cumulative stress index
        mean_stress = float(stress_score.mean())
        self.state["cumulative_stress_index"] = round(
            0.7 * self.state["cumulative_stress_index"] + 0.3 * mean_stress, 3
        )
        
        # Update weather history metrics
        rain = weather.get("precipitation", 0.0)
        temp = weather.get("temperature", 24.5)
        self.state["historical_weather_summary"]["cumulative_precipitation_mm"] += rain
        self.state["historical_weather_summary"]["mean_temperature_c"] = round(
            0.9 * self.state["historical_weather_summary"]["mean_temperature_c"] + 0.1 * temp, 2
        )
        
        # Log to timeline
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        self.state["event_timeline"].append({
            "timestamp": now_str,
            "type": "UAV Flight",
            "description": f"UAV Flight survey reconstructed on {date_str}. Mean stress score: {mean_stress:.3f}."
        })

        # Trajectory calculation if we have multiple flights
        if len(self.state["surveys_logged"]) > 1:
            prev_date = self.state["surveys_logged"][-2]
            try:
                prev_data = np.load(self.history_dir / f"survey_{prev_date}.npz")
                prev_stress = float(prev_data["stress"].mean())
                if mean_stress < prev_stress - 0.05:
                    self.state["health_trajectory"] = "Improving"
                elif mean_stress > prev_stress + 0.05:
                    self.state["health_trajectory"] = "Declining"
                else:
                    self.state["health_trajectory"] = "Stable"
            except Exception:
                pass
                
        self.save_state()

    def get_historical_sequence(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Fetch the loaded history of NDVI and stress maps in temporal order.
        """
        sequence = []
        for date_str in self.state["surveys_logged"]:
            npz_path = self.history_dir / f"survey_{date_str}.npz"
            if npz_path.exists():
                try:
                    data = np.load(npz_path)
                    sequence.append((data["ndvi"], data["stress"]))
                except Exception:
                    pass
        return sequence

    def run_predictive_simulation(
        self,
        forecast_days: int = 7
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs predictive ConvLSTM stress forecasting.
        """
        seq = self.get_historical_sequence()
        if not seq:
            return None
            
        from src.ai_engine.disease_evolution import CropStressEvolutionForecaster
        forecaster = CropStressEvolutionForecaster()
        
        if len(seq) < 3:
            current_ndvi, current_stress = seq[-1]
            seq = CropStressEvolutionForecaster.generate_synthetic_historical_sequence(
                current_ndvi, current_stress, n_steps=3
            )
            
        ndvi_pred, stress_pred, expansion = forecaster.predict_future_evolution(
            seq, forecast_days=forecast_days
        )
        return ndvi_pred, stress_pred, expansion

    def run_scenario_simulation(
        self,
        scenario_type: str,
        forecast_days: int = 7,
        weather_forecast: Optional[List[Dict[str, Any]]] = None,
        custom_interventions: Optional[List[Dict[str, Any]]] = None,
        budget_limit: float = 500.0,
        indices: Optional[Dict[str, np.ndarray]] = None,
        zone_labels: Optional[np.ndarray] = None,
        zone_names: Optional[Dict[int, str]] = None,
        center_lat: float = 37.7749,
        center_lon: float = -122.4194,
        propagation_model: str = "pde",
        growth_stage: str = "Vegetative"
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-day spatial simulation playback for different agronomic scenarios.
        Returns maps history, timeline logs, and generated MAVLink mission files.
        """
        if indices is None or zone_labels is None or zone_names is None:
            raise ValueError("indices, zone_labels, and zone_names are required to initialize simulation grids.")

        # Default standard 7-day weather forecast if none supplied
        if not weather_forecast:
            weather_forecast = [
                {"temperature": 25.0, "humidity": 75.0, "precipitation": 0.0, "wind_speed": 10.0, "precipitation_probability": 15.0}
                for _ in range(forecast_days)
            ]

        # Initialize simulator
        sim = AgronomicDigitalTwinSimulator(
            ndvi=indices["ndvi"],
            stress=indices["stress_score"],
            ndwi=indices["ndwi"],
            cire=indices["cire"],
            zone_labels=zone_labels,
            zone_names=zone_names
        )

        timeline = ["Day 0: Simulation initialization. Starting spatial maps loaded from active flight reconstruct."]
        maps_history = [sim.get_state_maps()]
        
        # 1. Scenario-based Intervention Setup
        planned_interventions = []
        
        if scenario_type == "ai_planned":
            # AI Planner evaluates initial layers
            for z_id in zone_names:
                mask = zone_labels == z_id
                if not np.any(mask):
                    continue
                mean_stress = float(indices["stress_score"][mask].mean())
                ndwi_mean = float(indices["ndwi"][mask].mean())
                cire_mean = float(indices["cire"][mask].mean())
                moist_val = float((ndwi_mean + 1.0) / 2.0)
                
                # Fungal Risk estimation
                fungal_risk = mean_stress * 0.45 + np.clip(ndwi_mean, 0, 1) * 0.3
                
                # Fungicide checks
                if fungal_risk > 0.35:
                    planned_interventions.append({
                        "day": 1,
                        "zone_id": z_id,
                        "type": "Fungicide Spray",
                        "cost": 49.40,
                        "priority": 5
                    })
                # Nitrogen checks
                if cire_mean < 1.4:
                    planned_interventions.append({
                        "day": 2,
                        "zone_id": z_id,
                        "type": "Nutrient Top-Dress",
                        "cost": 179.00,
                        "priority": 3
                    })
                # Water stress checks
                if moist_val < 0.38:
                    planned_interventions.append({
                        "day": 1,
                        "zone_id": z_id,
                        "type": "Precision Irrigation",
                        "cost": 25.00,
                        "priority": 4
                    })

            # Sort by priority
            planned_interventions.sort(key=lambda x: -x["priority"])
            
            # Apply weather feasibility checks & budget limits
            allocated = []
            running_budget = 0.0
            for intv in planned_interventions:
                day = intv["day"]
                # Weather check for Day 1
                if intv["type"] == "Fungicide Spray" and day == 1:
                    day_w = weather_forecast[0]
                    wind = float(day_w.get("wind_speed", 10.0))
                    rain_prob = float(day_w.get("precipitation_probability", 15.0))
                    if wind > 15.0 or rain_prob > 60.0:
                        intv["day"] = 2 # Defer to day 2 due to weather block
                        timeline.append(f"AI Plan: Deferred Fungicide on {zone_names[intv['zone_id']]} to Day 2 due to Day 1 wind/rain.")

                if running_budget + intv["cost"] <= budget_limit:
                    running_budget += intv["cost"]
                    allocated.append(intv)
                else:
                    timeline.append(f"AI Plan: Deferred treatment on {zone_names[intv['zone_id']]} (Cost: ${intv['cost']:.2f}) - budget limit reached.")
            
            planned_interventions = allocated

        elif scenario_type == "custom":
            planned_interventions = custom_interventions if custom_interventions else []

        # 2. Multi-day Playback Simulation Loop
        for day in range(1, forecast_days + 1):
            day_weather = weather_forecast[day - 1] if day <= len(weather_forecast) else weather_forecast[-1]
            day_interventions = [intv for intv in planned_interventions if intv.get("day", 1) == day]
            
            # Advance Simulator
            day_logs = sim.simulate_day(
                weather=day_weather,
                interventions=day_interventions,
                propagation_model=propagation_model,
                growth_stage=growth_stage
            )
            
            # Record events
            timeline.append(f"--- Day {day} ---")
            for log in day_logs:
                timeline.append(f"Day {day}: {log}")
            
            # Add to state maps
            maps_history.append(sim.get_state_maps())

        # 3. AI Flight Mission Compilation
        qgc_mission = None
        if scenario_type in ["ai_planned", "custom"] and len(planned_interventions) > 0:
            # We map the prescription targets to QGC waypoints
            from src.ai_engine.treatment_recommender import AITreatmentRecommender
            recommender = AITreatmentRecommender()
            prescriptions, _ = recommender.generate_zone_prescriptions(indices, weather_forecast[0], "Vegetative")
            
            # Generate the mission
            qgc_mission = QGCMissionGenerator.generate_precision_flight_plan(
                prescriptions=prescriptions,
                zone_labels=zone_labels,
                center_lat=center_lat,
                center_lon=center_lon
            )

        return {
            "scenario": scenario_type,
            "maps_history": maps_history,
            "timeline": timeline,
            "planned_interventions": planned_interventions,
            "qgc_mission": qgc_mission
        }

    # ── Live Field Mode integration ────────────────────────────────────

    def synchronize_live_observation(
        self,
        observation_dict: Dict[str, Any],
    ) -> None:
        """
        Ingest a real-time Live Field Mode observation into the Digital Twin state.

        This allows the twin to accumulate live scouting data alongside
        historical TIFF/multispectral surveys. The twin does NOT distinguish
        between sensor sources — it stores normalized metrics only.

        Parameters
        ----------
        observation_dict : dict
            Keys expected (all optional with safe defaults):
              - timestamp       : float (Unix epoch)
              - lat, lon        : float (GPS)
              - crop_stress_score : float 0-1
              - stress_label    : str
              - crop_stage      : str
              - grvi, vari, exg : float (RGB vegetation indices)
              - disease_detections : list
              - weed_coverage_pct : float
              - model_confidence : float
        """
        import datetime
        ts = observation_dict.get("timestamp", 0)
        ts_iso = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z" if ts else None

        # Update live_field_observations list (rolling window of 200)
        if "live_field_observations" not in self.state:
            self.state["live_field_observations"] = []

        self.state["live_field_observations"].append({
            "ts":                ts_iso,
            "lat":               observation_dict.get("lat"),
            "lon":               observation_dict.get("lon"),
            "crop_stress_score": observation_dict.get("crop_stress_score"),
            "stress_label":      observation_dict.get("stress_label"),
            "crop_stage":        observation_dict.get("crop_stage"),
            "grvi":              observation_dict.get("grvi"),
            "vari":              observation_dict.get("vari"),
            "exg":               observation_dict.get("exg"),
            "weed_coverage_pct": observation_dict.get("weed_coverage_pct"),
            "disease_count":     len(observation_dict.get("disease_detections", [])),
            "confidence":        observation_dict.get("model_confidence"),
            "source":            "phone_rgb",
        })

        # Keep rolling window
        if len(self.state["live_field_observations"]) > 200:
            self.state["live_field_observations"] = \
                self.state["live_field_observations"][-200:]

        # Update summary stats
        obs = self.state["live_field_observations"]
        stress_vals = [o["crop_stress_score"] for o in obs
                       if o.get("crop_stress_score") is not None]
        if stress_vals:
            self.state["live_mean_stress"]  = float(np.mean(stress_vals))
            self.state["live_max_stress"]   = float(np.max(stress_vals))
            self.state["live_obs_count"]    = len(obs)
            self.state["live_last_updated"] = ts_iso

        # Persist
        self.save_state()

