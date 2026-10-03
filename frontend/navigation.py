"""Single source of truth for the renovated workspace navigation."""
GROUPS = {
    "Workspace": [("overview", "Overview", "dashboard", "overview.py")],
    "Field analysis": [
        ("upload", "Upload & process", "upload_file", "01_upload_and_process.py"),
        ("vegetation", "Vegetation analytics", "eco", "02_vegetation_analytics.py"),
        ("stress", "Stress intelligence", "monitor_heart", "03_stress_intelligence.py"),
        ("temporal", "Temporal analytics", "timeline", "04_temporal_analytics.py"),
    ],
    "Mapping & imagery": [
        ("zoning", "Field zoning", "grid_view", "05_field_zoning_gis.py"),
        ("spatial", "Spatial reconstruction", "view_in_ar", "06_spatial_reconstruction.py"),
        ("satellite", "Satellite analytics", "satellite_alt", "07_satellite_analytics.py"),
        ("landsat", "Landsat history", "history", "08_landsat_historical.py"),
    ],
    "Operations": [
        ("weather", "Weather & risk", "partly_cloudy_day", "09_weather_and_risk.py"),
        ("uav", "Live UAV control", "flight", "10_live_uav_control.py"),
        ("swarm", "Swarm operations", "hub", "11_swarm_operations.py"),
    ],
    "Intelligence": [
        ("twin", "Predictive digital twin", "model_training", "12_predictive_digital_twin.py"),
        ("optimizer", "Input optimizer", "tune", "13_ai_input_optimizer.py"),
        ("reports", "AI reports", "description", "14_ai_report.py"),
    ],
}

ITEMS = {item[0]: (group, *item[1:]) for group, items in GROUPS.items() for item in items}
