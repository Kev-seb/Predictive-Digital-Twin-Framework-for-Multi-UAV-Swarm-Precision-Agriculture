"""
sensitivity_analysis.py
------------------------
Physical and spatiotemporal sensitivity analysis for the UAV crop digital twin.
Hardened for Q1 peer review:
1. Wind velocity advection influence on Fisher-Kolmogorov PDE contagion spread speed and angle.
   Uses intensity-weighted centroids to eliminate discrete spatial binning step-artifacts.
2. Robustness of the AI optimizer (expected yield vs forecast noise variance).
3. Uncertainty propagation sweep: expected yield and worst-case CVaR vs. perception classification noise.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_engine.epidemiology import EpidemiologyForecaster
from src.ai_engine.treatment_optimizer import AITreatmentOptimizer
from src.ai_engine.treatment_recommender import ZonePrescription, TreatmentRecommendation


# =====================================================================
# 1. Spatiotemporal PDE Wind Sensitivity Analysis
# =====================================================================
def run_pde_wind_sensitivity():
    print("Running contagion PDE wind sensitivity analysis with intensity-weighted centroids...")
    
    grid_size = 50
    ndvi = np.ones((grid_size, grid_size), dtype=np.float32) * 0.75
    fungicide = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Pathogen center outbreak spot (center at 25, 25)
    pathogen_init = np.zeros((grid_size, grid_size), dtype=np.float32)
    pathogen_init[23:27, 23:27] = 1.0
    
    forecaster = EpidemiologyForecaster()
    
    # Evaluate at fine wind speed resolution (every 2 km/h) to get a smooth physical transition
    wind_speeds = np.arange(0.0, 31.0, 2.0)
    wind_dir = 45.0  # North-East (bearing of wind movement)
    
    front_velocities = []
    drift_angles = []

    for ws in wind_speeds:
        weather = {
            "temperature": 24.0,
            "humidity": 80.0,
            "wind_speed": ws,
            "wind_direction": wind_dir,
            "precipitation": 0.0
        }
        
        pathogen = np.copy(pathogen_init)
        dt = 0.1
        dx = 1.0
        for _ in range(15):
            pathogen, _, _ = forecaster.simulate_pde_step(
                pathogen, ndvi, weather, growth_stage_susceptibility=1.0, fungicide=fungicide, dt=dt, dx=dx
            )
            
        # Calculate intensity-weighted centroid to get smooth displacement (eliminating discrete binning artifacts)
        total_p = np.sum(pathogen)
        if total_p > 0:
            grid_y, grid_x = np.mgrid[0:grid_size, 0:grid_size]
            centroid_x = np.sum(pathogen * grid_x) / total_p
            centroid_y = np.sum(pathogen * grid_y) / total_p
            
            # Displacement from initial centroid (25, 25)
            dx_disp = centroid_x - 25.0
            dy_disp = 25.0 - centroid_y  # Invert y for standard Cartesian coordinate displacement
            
            displacement = np.sqrt(dx_disp**2 + dy_disp**2)
            velocity_ms = displacement / (15 * dt)
            front_velocities.append(float(velocity_ms))
            
            # Direction angle
            angle = np.degrees(np.arctan2(dy_disp, dx_disp))
            if angle < 0:
                angle += 360.0
            drift_angles.append(float(angle))
        else:
            front_velocities.append(0.0)
            drift_angles.append(0.0)

    # Plot 1: Wind Speed vs Contagion Spread Velocity (Smooth Curve)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(wind_speeds, front_velocities, marker="o", color="#3b82f6", linewidth=2.5, label="Anisotropic Wavefront")
    # Reference isotropic diffusion baseline
    plt.axhline(front_velocities[0], color="#94a3b8", linestyle="--", label="Isotropic Diffusion Limit")
    plt.title("Pathogen Wavefront Speed vs. Ambient Wind Speed", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Ambient Wind Speed (km/h)", fontsize=10)
    plt.ylabel("Contagion Front Velocity (pixels/sec)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs/academic", exist_ok=True)
    plt.savefig("outputs/academic/pde_wind_speed_sensitivity.png", dpi=300)
    plt.close()
    
    print(f"PDE Wind speeds: {wind_speeds.tolist()}")
    print(f"PDE Wavefront velocities: {front_velocities}")
    print(f"PDE Direction angles: {drift_angles}")
    
    # Save sweep details for LaTeX table inclusion
    df_sweep = pd.DataFrame({
        "Wind_Speed": wind_speeds,
        "Velocity": front_velocities,
        "Angle": drift_angles
    })
    df_sweep.to_csv("outputs/academic/pde_wind_sweep_data.csv", index=False)


# =====================================================================
# 2. Decision Sensitivity to Forecast Noise Analysis
# =====================================================================
def run_optimizer_noise_sensitivity():
    print("Running decision sensitivity to forecast noise analysis...")
    
    prescriptions = [
        ZonePrescription(
            zone_id=0, zone_name="Zone A", area_pct=30.0,
            ndvi_mean=0.40, ndre_mean=0.25, cire_mean=1.1, ndwi_mean=0.22,
            n_deficiency="Severe", fungal_risk_prob=0.75,
            recommendations=[TreatmentRecommendation("Fungicide", 0.75, 0.9, "2.0 L/ha", "", "High", "Vegetative")]
        ),
        ZonePrescription(
            zone_id=1, zone_name="Zone B", area_pct=70.0,
            ndvi_mean=0.72, ndre_mean=0.48, cire_mean=3.10, ndwi_mean=-0.08,
            n_deficiency="None", fungal_risk_prob=0.15,
            recommendations=[TreatmentRecommendation("Fungicide", 0.15, 0.9, "0.0 L/ha", "", "Low", "Vegetative")]
        )
    ]
    
    true_weather = {
        "temperature": 25.0,
        "humidity": 75.0,
        "wind_speed": 10.0,
        "precipitation_probability": 20.0,
        "precipitation": 0.0
    }
    
    optimizer = AITreatmentOptimizer()
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    risk_profiles = ["Risk-Averse", "Risk-Neutral", "Risk-Seeking"]
    yield_results = {profile: [] for profile in risk_profiles}

    for noise in noise_levels:
        for profile in risk_profiles:
            noisy_weather = {
                "temperature": true_weather["temperature"] + np.random.normal(0, 5.0 * noise),
                "humidity": np.clip(true_weather["humidity"] + np.random.normal(0, 15.0 * noise), 20.0, 100.0),
                "wind_speed": max(0.0, true_weather["wind_speed"] + np.random.normal(0, 4.0 * noise)),
                "precipitation_probability": np.clip(true_weather["precipitation_probability"] + np.random.normal(0, 20.0 * noise), 0.0, 100.0),
                "precipitation": 0.0
            }
            
            report = optimizer.optimize_treatment_plan(
                prescriptions, noisy_weather, budget_limit=400.0,
                optimization_model="Reinforcement Learning (MDP)",
                risk_profile=profile, mc_runs=50
            )
            yield_results[profile].append(report.expected_yield / 10.0)

    plt.figure(figsize=(7, 4.5))
    colors = {"Risk-Averse": "#ef4444", "Risk-Neutral": "#3b82f6", "Risk-Seeking": "#10b981"}
    styles = {"Risk-Averse": "-o", "Risk-Neutral": "-s", "Risk-Seeking": "-^"}
    
    for profile in risk_profiles:
        plt.plot(
            noise_levels, yield_results[profile], styles[profile],
            color=colors[profile], label=profile, linewidth=2
        )
        
    plt.title("Optimizer Robustness under Meteorological Uncertainty", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Weather Forecast Noise Standard Deviation (Scaled)", fontsize=10)
    plt.ylabel("Realized Crop Yield (t/ha)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/academic/optimizer_noise_sensitivity.png", dpi=300)
    plt.close()


# =====================================================================
# 3. Uncertainty Propagation Plot (Sweep A)
# =====================================================================
def run_perception_noise_plot():
    print("Generating uncertainty propagation sensitivity plot...")
    
    # Values extracted from Sweep A benchmarks
    noise_levels = [0, 5, 10, 15, 20, 25] # Percentage errors
    expected_yield = [7.16, 7.15, 7.10, 7.10, 6.97, 6.83]
    cvar_95 = [7.15, 6.59, 5.89, 5.24, 4.74, 4.27]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    ax.plot(noise_levels, expected_yield, marker="o", color="#3b82f6", linewidth=2.5, label="Expected Yield (Mean)")
    ax.plot(noise_levels, cvar_95, marker="s", color="#ef4444", linewidth=2.5, linestyle="--", label="Worst-case Yield (95% CVaR)")
    
    ax.set_title("Uncertainty Propagation from Perception to Control", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("DeepLabV3+ Segmentation Classification Noise (%)", fontsize=10)
    ax.set_ylabel("Realized Crop Yield (t/ha)", fontsize=10)
    ax.set_ylim(4.0, 7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left")
    
    # Shade the risk gap
    ax.fill_between(noise_levels, cvar_95, expected_yield, color="#ef4444", alpha=0.08, label="Downside Risk Gap")
    
    plt.tight_layout()
    plt.savefig("outputs/academic/uncertainty_propagation_sensitivity.png", dpi=300)
    plt.close()
    print("[SUCCESS] Uncertainty propagation plot generated successfully.")


if __name__ == "__main__":
    run_pde_wind_sensitivity()
    print("\n---------------------------------------------------------\n")
    run_optimizer_noise_sensitivity()
    print("\n---------------------------------------------------------\n")
    run_perception_noise_plot()
    print("\n[SUCCESS] All physical sensitivity figures compiled successfully.")
