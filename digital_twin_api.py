"""
digital_twin_api.py
================================================================
FastAPI service for the paddy field digital twin (src/twin_core).

The twin core is the single source of truth. UE5 and the three.js
dashboard are views onto it; UE5 can also push rendered field values
back, which are assimilated as observations (source = "ue5").

Runs on http://127.0.0.1:8008 by default (see UAV_TWIN_API_PORT).

UE5-compatible routes (20 x 20 HISM grid, unchanged contract):
  GET    /                         Health check
  POST   /field/update             UE5 pushes per-instance state -> assimilated
  GET    /field/status             All 400 instances
  GET    /field/instance/{index}   Single instance
  GET    /field/summary            Aggregated statistics + recent history
  GET    /field/anomalies          Recent anomaly events
  GET    /field/heatmap            NDVI / stress (+ uncertainty) grids
  DELETE /field/reset              Reset the twin to its prior

Twin routes:
  GET    /twin/state               Full summary + per-zone values
  GET    /twin/layers/{layer}      One layer (mean or var) at any resolution
  POST   /twin/step                Advance the simulation N days
  POST   /twin/observe/point       Assimilate a geotagged reading
  POST   /twin/observe/grid        Assimilate a raster
  POST   /twin/what-if             Compare interventions on a forked twin
  POST   /twin/stage               Set the paddy growth stage
  GET    /twin/history             Daily history (mean + std per layer)
  GET    /twin/events              Recent twin events
  POST   /twin/ingest/rgb          RGB AI result + telemetry -> point observation
  POST   /twin/ingest/multispectral 4-band reflectance -> raster observations
  WS     /ws/twin                  Live event stream (UE5 / three.js)
  WS     /ws/field                 UE5 stream: coalesced 20x20 snapshots + drone tracks
                                   (see src/ue5_integration/README.md for the schema)

Simulation routes (hidden "physical" field + sensors, see virtual_field.py):
  POST   /sim/reset                New virtual field (hidden outbreak, dry patch)
  POST   /sim/advance              Advance truth and twin together
  POST   /sim/drone-pass           Serpentine drone RGB pass -> assimilated
  POST   /sim/survey               Multispectral survey -> assimilated
  GET    /sim/error                Twin vs truth: RMSE, bias, 95% coverage
  GET    /sim/truth/layers/{layer} Ground-truth layer (for debugging / UE5 A/B)
  POST   /sim/mission              Plan + fly a fleet mission (scout / spray / auto),
                                   in flight physics or over MAVLink (emulator / ArduPilot SITL)

Swarm routes (N-drone fleet, see swarm.py):
  GET    /swarm/state              Drone positions (dashboard /swarm format)
  POST   /swarm/drones             Add a drone
  DELETE /swarm/drones/{drone_id}  Remove a drone
  POST   /swarm/plan               Allocate scout / spray tasks without flying
  POST   /swarm/mission-files      Planned missions as QGC WPL 110 (.waypoints) per drone
  GET    /swarm/sitl-homes         Launch commands for one ArduCopter SITL instance per drone

Demo mode (paced closed loop for live UE5 presentations):
  POST   /demo/start               Run the story: outbreak -> scout -> decide -> spray -> recovery
  POST   /demo/stop                Stop it
  GET    /demo/status              Current phase
  (streams swarm.positions at 10 Hz and twin.status captions on /ws/field)

Live control (UE5 menu, one step at a time, animated like the demo):
  POST   /live/action              {"action": next_day|scout|spray|auto|outbreak|reset, ...}

Polling for clients without WebSockets (UE5 Python, VaRest):
  GET    /ue5/frame?since=<seq>    Latest caption + drone poses, and the field snapshot
                                   only when it changed since <seq>
================================================================
"""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.config import settings
from src.twin_core import LAYERS, DigitalTwin, FieldGrid
from src.twin_core.ingest import bands_to_layers, ingest_multispectral, ingest_rgb_result
from src.twin_core.swarm import Fleet, allocate, tasks_from_twin
from src.twin_core import mavlink_mission as mm
from src.twin_core.virtual_field import VirtualField, serpentine_waypoints
from src.twin_core.weather_feed import get_weather, synthetic_weather

FORECAST_DAYS = 7
BLANKET_DOSE_L_PER_HA = 25.0      # conventional whole-field fungicide spraying, per application


def expected_harvest_t_ha(twin: DigitalTwin, state=None) -> float:
    """
    Expected paddy harvest (t/ha) if the crop keeps its current relative
    health to maturity: each cell's NDVI relative to the stage's NDVI
    potential is scaled to a peak-canopy NDVI, then the repo's
    CropYieldPredictor turns that into mature biomass x harvest index x
    stress penalty (fungus counts through stress).
    """
    from src.ai_engine.yield_predictor import CropYieldPredictor
    from src.twin_core.paddy_model import ndvi_potential

    st = state or twin.state
    ndvi = np.clip(st.mean["ndvi"], 0.0, 1.0)
    if twin.crop_model == "paddy" and "gdd" in st.scalars:
        pot = max(ndvi_potential(twin.params, st.scalars["gdd"]), 1e-3)
        peak = twin.params.ndvi_max
    else:
        pot = peak = 0.86
    equiv = np.clip(ndvi / pot, 0.0, 1.0) * peak
    pred = CropYieldPredictor("Paddy Rice")
    biomass = pred.estimate_biomass(equiv, equiv, "Mature")
    return float(pred.predict_yield(biomass, st.mean["stress"], {}, "Mature").mean())

WeatherSource = Literal["manual", "synthetic", "openmeteo", "auto"]

UE5_GRID = settings.twin_ue5_grid
NUM_INSTANCES = UE5_GRID * UE5_GRID
ANOMALY_STRESS = 0.70
ANOMALY_NDVI_DROP = 0.15


# ----------------------------------------------------------------
# PYDANTIC MODELS
# ----------------------------------------------------------------

class InstancePayload(BaseModel):
    index:          int
    row:            int
    col:            int
    ndvi:           float = Field(..., ge=0.0, le=1.0)
    stress:         float = Field(..., ge=0.0, le=1.0)
    soil_moisture:  float = Field(..., ge=0.0, le=100.0)
    temperature_c:  float


class FieldUpdatePayload(BaseModel):
    day:       int
    instances: List[InstancePayload]


class Intervention(BaseModel):
    day:     int = Field(1, ge=1, description="Day relative to this request (1 = first simulated day)")
    zone_id: int
    type:    Literal["Fungicide Spray", "Nutrient Top-Dress", "Precision Irrigation"]
    cost:    float = 0.0


class StepRequest(BaseModel):
    days:          int = Field(1, ge=1, le=120)
    weather:       Optional[List[Dict[str, float]] | Dict[str, float]] = None
    weather_source: WeatherSource = Field(
        "manual", description="manual = use `weather` (or defaults); otherwise fetch/generate it")
    interventions: List[Intervention] = []


class RGBIngest(BaseModel):
    result:    Dict[str, Any] = Field(..., description="RGBInferenceResult.to_dict()")
    telemetry: Dict[str, Any] = Field(..., description="lat, lon, alt (m above canopy)")
    source:    str = "drone_rgb"
    frame_width:  int = Field(640, gt=0)
    frame_height: int = Field(480, gt=0)


class MultispectralIngest(BaseModel):
    green:    List[List[Optional[float]]]
    red:      List[List[Optional[float]]]
    red_edge: List[List[Optional[float]]]
    nir:      List[List[Optional[float]]]
    source:   str = "multispectral"


class SimReset(BaseModel):
    seed:   int = 42
    outbreaks: Optional[List[Dict[str, float]]] = Field(
        None, description="[{row, col, radius, load}] in twin grid cells; default = one hidden outbreak")
    reset_twin: bool = True
    simulated: bool = Field(False, description="Ignore an uploaded field and build a random virtual field")


class FieldFromImage(BaseModel):
    name: str = "Uploaded field"
    bands: Dict[str, List[List[Optional[float]]]] = Field(
        ..., description="green, red, red_edge, nir reflectance (0-1), any resolution; resampled onto the twin grid")
    center_lat: Optional[float] = Field(None, ge=-90, le=90)
    center_lon: Optional[float] = Field(None, ge=-180, le=180)
    width_m: Optional[float] = Field(None, gt=0, description="ground width of the image (from its georeference)")
    height_m: Optional[float] = Field(None, gt=0, description="ground height of the image")
    growth_stage: Optional[str] = None
    mode: Literal["new", "update"] = Field(
        "new", description="'new' field, or 'update': a new flight of the loaded field, assimilated by the twin")


class SimAdvance(BaseModel):
    days:           int = Field(1, ge=1, le=60)
    weather_source: Optional[Literal["synthetic", "openmeteo", "auto"]] = Field(
        None, description="default: the real forecast for an uploaded field ('auto'), synthetic otherwise")
    interventions:  List[Intervention] = []


class SimDronePass(BaseModel):
    altitude_m:     float = Field(10.0, gt=0, le=120)
    lane_spacing_m: float = Field(10.0, gt=0)
    step_m:         float = Field(5.0, gt=0)


class MissionRequest(BaseModel):
    kind: Literal["scout", "spray", "auto"] = Field(
        "scout", description="auto = scout, then spray zones the updated twin flags")
    fungus_threshold: float = Field(0.2, ge=0.0, le=1.0, description="cell-level action threshold")
    min_infected_frac: float = Field(0.1, gt=0.0, le=1.0,
                                     description="spray a zone when this share of it exceeds the threshold")
    dose_l_per_ha:    float = Field(25.0, gt=0.0, le=200.0)
    zones:            Optional[List[int]] = None
    include_track:    bool = False
    backend: Literal["physics", "mavlink"] = Field(
        "physics", description="physics = in-process 6-DOF fleet; mavlink = upload and fly over MAVLink "
                               "(UAV_TWIN_MAVLINK_MODE: emulated vehicles or ArduPilot SITL)")


class DemoRequest(BaseModel):
    seed:           int = 7
    days:           int = Field(15, ge=1, le=120)
    day_seconds:    float = Field(3.0, ge=0.0, le=120.0, description="wall seconds per simulated day")
    scout_every:    int = Field(3, ge=1, le=30, description="scout every N days (sprays follow if needed)")
    playback_speed: float = Field(6.0, gt=0.0, le=1000.0, description="simulated seconds per wall second while drones fly")
    backend:        Literal["physics", "mavlink"] = "physics"
    fungus_threshold: float = Field(0.15, ge=0.0, le=1.0)


class LiveAction(BaseModel):
    action: Literal["next_day", "scout", "spray", "auto", "outbreak", "reset"]
    row: Optional[float] = Field(None, description="outbreak: UE5-grid row (0-20, north edge = 0)")
    col: Optional[float] = Field(None, description="outbreak: UE5-grid col (0-20, west edge = 0)")
    radius: float = Field(2.5, gt=0.0, le=10.0, description="outbreak radius in UE5-grid cells (~12 m: big enough for 10 m-altitude scouting to resolve)")
    load: float = Field(0.5, gt=0.0, le=1.0, description="outbreak peak fungal load")
    seed: int = 7
    playback_speed: float = Field(6.0, gt=0.0, le=1000.0)
    backend: Literal["physics", "mavlink"] = "physics"
    fungus_threshold: float = Field(0.15, ge=0.0, le=1.0)


class AddDrone(BaseModel):
    drone_id: str = Field(..., min_length=1, max_length=40)


class SimSurvey(BaseModel):
    coverage: float = Field(0.9, gt=0, le=1.0)
    noise:    float = Field(0.01, ge=0, le=0.2)


class PointObservation(BaseModel):
    lat:     float
    lon:     float
    values:  Dict[str, float]
    source:  str = "phone_rgb"
    obs_var: Optional[Dict[str, float]] = None
    footprint_m:        float = Field(3.0, gt=0)
    influence_radius_m: float = Field(8.0, gt=0)


class GridObservation(BaseModel):
    layer:   str
    values:  List[List[Optional[float]]]
    source:  str = "multispectral"
    obs_var: Optional[float] = Field(None, gt=0)


class WhatIfRequest(BaseModel):
    days:          int = Field(7, ge=1, le=60)
    weather:       Optional[List[Dict[str, float]] | Dict[str, float]] = None
    interventions: List[Intervention] = []


class StageRequest(BaseModel):
    growth_stage: str


# ----------------------------------------------------------------
# APP FACTORY
# ----------------------------------------------------------------

def _default_twin() -> DigitalTwin:
    grid = FieldGrid.centered_on(
        settings.field_center_lat, settings.field_center_lon,
        rows=settings.twin_grid_rows, cols=settings.twin_grid_cols,
        cell_size_m=settings.twin_cell_size_m,
    )
    params = None
    if settings.twin_crop_model == "paddy" and settings.twin_params_file.exists():
        from src.twin_core.paddy_model import PaddyParams
        params = PaddyParams.from_json(settings.twin_params_file)
        print(f"[DigitalTwinAPI] Using calibrated paddy parameters from {settings.twin_params_file}")
    return DigitalTwin.load_or_create(
        settings.twin_field_id, settings.twin_storage_dir, grid=grid,
        propagation_model=settings.twin_propagation_model,
        crop_model=settings.twin_crop_model, params=params,
    )


def create_app(twin: Optional[DigitalTwin] = None) -> FastAPI:
    ctx: Dict[str, Any] = {"twin": twin, "anomalies": [], "ue5_snapshot": None, "virtual": None,
                           "fleet": None, "pending_interventions": [], "field_listeners": set(),
                           "last_track": None, "mavlink": None,
                           "demo": {"task": None, "phase": "idle", "text": "", "day": 0, "error": None},
                           "metrics": None, "user_field": None}

    def get_twin() -> DigitalTwin:
        if ctx["twin"] is None:
            ctx["twin"] = _default_twin()
        if not ctx.get("_versioned"):
            ctx["_versioned"] = True
            ctx["field_version"] = 1

            def bump(_event: Dict[str, Any]) -> None:
                ctx["field_version"] = ctx.get("field_version", 0) + 1
            ctx["twin"].subscribe(bump)
        return ctx["twin"]

    def get_virtual() -> VirtualField:
        if ctx["virtual"] is None:
            twin = get_twin()
            ctx["virtual"] = VirtualField(twin.state.grid, propagation_model=twin.propagation_model,
                                          zone_layout=twin.zone_layout, crop_model=twin.crop_model,
                                          params=twin.params)
        return ctx["virtual"]

    def get_fleet() -> Fleet:
        if ctx["fleet"] is None:
            ctx["fleet"] = Fleet.for_field(get_twin().state.grid, settings.twin_swarm_size)
        return ctx["fleet"]

    def drone_homes() -> Dict[str, tuple]:
        fleet = get_fleet()
        return {d: fleet.frame.to_latlon(*fleet.pads[d]) for d in fleet.drones}

    def mavlink_connections() -> Dict[str, str]:
        """Connection per drone; starts (or restarts) the emulator fleet when needed."""
        ids = list(get_fleet().drones)
        if settings.twin_mavlink_connections.strip():
            conns = [c.strip() for c in settings.twin_mavlink_connections.split(",") if c.strip()]
            return dict(zip(ids, conns))
        if settings.twin_mavlink_mode == "sitl":
            return {d: f"tcp:127.0.0.1:{settings.twin_mavlink_base_port + 10 * i}" for i, d in enumerate(ids)}
        emu = ctx["mavlink"]
        if emu is None or list(emu.connections) != ids:
            from src.twin_core.sim_vehicle import EmulatedFleet
            if emu is not None:
                emu.stop()
            emu = EmulatedFleet(drone_homes(), base_port=settings.twin_mavlink_base_port,
                                time_scale=settings.twin_mavlink_time_scale)
            ctx["mavlink"] = emu
        return dict(emu.connections)

    def plan_mission(kind: str, req: "MissionRequest"):
        fleet, twin = get_fleet(), get_twin()
        if req.zones:
            bad = [z for z in req.zones if z not in twin.zone_names]
            if bad:
                raise HTTPException(400, f"Unknown zone_id(s) {bad}. Valid: {sorted(twin.zone_names)}")
        tasks = tasks_from_twin(twin, kind, fleet.frame, fleet.spec,
                                fungus_threshold=req.fungus_threshold,
                                dose_l_per_ha=req.dose_l_per_ha, zones=req.zones,
                                min_infected_frac=req.min_infected_frac)
        return tasks, allocate(tasks, fleet.drone_info())

    def forecast(source: str, days: int, start_day: int) -> tuple:
        lat, lon = get_twin().state.grid.center_latlon()
        try:
            return get_weather(lat, lon, days, source=source, start_day=start_day)
        except Exception as e:
            raise HTTPException(502, f"Weather source '{source}' unavailable: {e}")

    # ---------------- UE5 stream helpers ----------------
    def notify_field_listeners(message: Optional[Dict[str, Any]] = None) -> None:
        for listener in list(ctx["field_listeners"]):
            listener(message)

    def publish_track(kind: str, track: Dict[str, List[List[float]]]) -> None:
        """Convert a mission track to UE5 grid coordinates and push it to /ws/field clients."""
        grid = get_twin().state.grid
        h_m, w_m = grid.rows * grid.cell_size_m, grid.cols * grid.cell_size_m
        drones = {}
        for d_id, samples in track.items():
            pts = []
            for t, lat, lon, alt, spraying in samples:
                east, south = grid.latlon_to_local_m(lat, lon)
                pts.append([t, round(south / h_m * UE5_GRID, 3), round(east / w_m * UE5_GRID, 3),
                            alt, spraying])
            drones[d_id] = pts
        msg = {"type": "swarm.track", "kind": kind, "sim_day": get_twin().state.sim_day,
               "fields": ["t_s", "row", "col", "alt_m", "spraying"], "drones": drones}
        ctx["last_track"] = msg
        notify_field_listeners(msg)

    def field_snapshot(seq: int) -> Dict[str, Any]:
        twin = get_twin()
        with twin._lock:
            s = twin.state
            g = ue5_grids()
            flat = lambda a: np.round(np.asarray(a, dtype=np.float64), 4).ravel().tolist()  # noqa: E731
            water = (s.resample_aux("water_mm", UE5_GRID, UE5_GRID)
                     if "water_mm" in s.aux else np.zeros((UE5_GRID, UE5_GRID)))
            zones = np.asarray(twin.zone_labels)
            r_idx = (np.arange(UE5_GRID) * s.grid.rows // UE5_GRID) + s.grid.rows // (2 * UE5_GRID)
            c_idx = (np.arange(UE5_GRID) * s.grid.cols // UE5_GRID) + s.grid.cols // (2 * UE5_GRID)
            anomalies = sorted({a["index"] for a in ctx["anomalies"] if a["day"] == s.sim_day})
            return {
                "type": "field.snapshot", "seq": seq, "field_id": twin.field_id,
                "sim_day": s.sim_day, "growth_stage": s.growth_stage,
                "gdd": round(float(s.scalars.get("gdd", 0.0)), 1),
                "grid": {"rows": UE5_GRID, "cols": UE5_GRID,
                         "cell_size_m": round(s.grid.rows * s.grid.cell_size_m / UE5_GRID, 3),
                         "origin_lat": s.grid.origin_lat, "origin_lon": s.grid.origin_lon},
                "ndvi": flat(np.clip(g["ndvi"], 0.0, 1.0)),
                "stress": flat(g["stress"]),
                "stress_std": flat(g["stress_std"]),
                "moisture": flat(g["moisture"]),
                "fungus": flat(s.resample("fungus", UE5_GRID, UE5_GRID)),
                "water_mm": flat(water),
                "zone": zones[np.ix_(r_idx, c_idx)].ravel().astype(int).tolist(),
                "anomalies": anomalies,
                "pending_sprays": [p["zone_id"] for p in ctx["pending_interventions"]],
                **truth_layers(flat),
                **forecast_layers(flat),
            }

    def new_metrics() -> Dict[str, Any]:
        return {"scout_missions": 0, "spray_missions": 0, "flight_s": 0.0, "fungicide_kg": 0.0,
                "blanket_kg": 0.0, "zones_sprayed": 0, "error_history": []}

    def metrics_state() -> Dict[str, Any]:
        if ctx["metrics"] is None:
            ctx["metrics"] = new_metrics()
        return ctx["metrics"]

    def record_mission(kind: str, report: Dict[str, Any]) -> None:
        m = metrics_state()
        m[f"{kind}_missions"] += 1
        m["flight_s"] += sum(float(d.get("flight_s", d.get("wall_s", 0.0)))
                             for d in report.get("drones", {}).values())
        if kind == "spray" and report.get("completed_sprays"):
            m["fungicide_kg"] += sum(float(p.get("applied_kg", p.get("planned_kg", 0.0)))
                                     for p in report["completed_sprays"])
            m["zones_sprayed"] += len(report["completed_sprays"])
            g = get_twin().state.grid
            m["blanket_kg"] += g.area_m2 / 10_000.0 * BLANKET_DOSE_L_PER_HA   # 1 L ~ 1 kg
        record_error()

    def record_error() -> None:
        if ctx["virtual"] is None:
            return
        m = metrics_state()
        e = ctx["virtual"].twin_error(get_twin())["fungus"]["rmse"]
        m["error_history"].append([get_twin().state.sim_day, round(float(e), 4)])
        del m["error_history"][:-200]

    def metrics_view() -> Dict[str, Any]:
        """Numbers for the UE5 results panel."""
        twin = get_twin()
        m = metrics_state()
        b = forecast_bundle()
        saved = (1.0 - m["fungicide_kg"] / m["blanket_kg"]) * 100.0 if m["blanket_kg"] > 0 else None
        hist = m["error_history"]
        return {
            "sim_day": twin.state.sim_day,
            "growth_stage": twin.state.growth_stage,
            "harvest_t_ha": {"now": round(b["harvest_now"], 2),
                             "in_7d_no_spray": round(b["harvest_no_spray"], 2),
                             "in_7d_if_sprayed": round(b["harvest_if_sprayed"], 2),
                             "spray_zones": b["spray_zones"]},
            "fungicide": {"used_kg": round(m["fungicide_kg"], 2), "blanket_kg": round(m["blanket_kg"], 2),
                          "saved_pct": None if saved is None else round(saved, 1),
                          "zones_sprayed": m["zones_sprayed"]},
            "missions": {"scout": m["scout_missions"], "spray": m["spray_missions"],
                         "flight_min": round(m["flight_s"] / 60.0, 1)},
            "twin_error": {"first": hist[0][1] if hist else None, "now": hist[-1][1] if hist else None,
                           "history": hist[-30:]},
        }

    def forecast_bundle() -> Dict[str, Any]:
        """
        The twin's prediction FORECAST_DAYS ahead, on forks (the live twin is
        untouched): doing nothing vs spraying the zones the twin flags now.
        Uses the same weather forecast /sim/advance feeds the twin. Cached
        until the twin changes.
        """
        version = int(ctx.get("field_version", 0))
        cached = ctx.get("_forecast")
        if cached is not None and cached["version"] == version:
            return cached
        twin = get_twin()
        if ctx.get("user_field"):          # an uploaded field: forecast with its real weather (synthetic offline)
            lat, lon = twin.state.grid.center_latlon()
            weather, _ = get_weather(lat, lon, FORECAST_DAYS, source="auto", start_day=twin.state.sim_day + 1)
        else:
            weather = synthetic_weather(FORECAST_DAYS, start_day=twin.state.sim_day + 1)
        fleet = get_fleet()
        spray_tasks = tasks_from_twin(twin, "spray", fleet.frame, fleet.spec, fungus_threshold=0.15)
        plan = [{"day": 1, "zone_id": t.zone_id, "type": "Fungicide Spray"} for t in spray_tasks]
        plan += [{"day": 1, "zone_id": p["zone_id"], "type": p["type"]} for p in ctx["pending_interventions"]]
        # "No spray": the fleet's sprays (flown or flagged) never happen, so the
        # gap to "if sprayed" is the value of acting, before or after flying.
        no_spray = twin.fork()
        no_spray.step(days=FORECAST_DAYS, weather=weather)
        sprayed = twin.fork()
        sprayed.step(days=FORECAST_DAYS, weather=weather, interventions=plan)

        def brief(t: DigitalTwin) -> Dict[str, float]:
            m = t.state.mean
            return {"fungus_mean": round(float(m["fungus"].mean()), 4),
                    "fungus_peak": round(float(m["fungus"].max()), 4),
                    "ndvi_mean": round(float(m["ndvi"].mean()), 4),
                    "stress_mean": round(float(m["stress"].mean()), 4)}

        bundle = {"version": version, "days": FORECAST_DAYS, "state": no_spray.state,
                  "harvest_now": expected_harvest_t_ha(twin),
                  "harvest_no_spray": expected_harvest_t_ha(no_spray),
                  "harvest_if_sprayed": expected_harvest_t_ha(sprayed),
                  "no_spray": brief(no_spray), "if_sprayed": brief(sprayed),
                  "spray_zones": sorted({p["zone_id"] for p in plan}),
                  "on_day": twin.state.sim_day + FORECAST_DAYS}
        ctx["_forecast"] = bundle
        return bundle

    def forecast_layers(flat) -> Dict[str, Any]:
        b = forecast_bundle()
        st = b["state"]
        return {"forecast_days": b["days"],
                "forecast_stress": flat(st.resample("stress", UE5_GRID, UE5_GRID)),
                "forecast_fungus": flat(st.resample("fungus", UE5_GRID, UE5_GRID)),
                "forecast_ndvi": flat(np.clip(st.resample("ndvi", UE5_GRID, UE5_GRID), 0.0, 1.0)),
                "forecast_std": flat(np.sqrt(st.resample("stress", UE5_GRID, UE5_GRID, which="var"))),
                "what_if": {"days": b["days"], "on_day": b["on_day"], "no_spray": b["no_spray"],
                            "if_sprayed": b["if_sprayed"], "spray_zones": b["spray_zones"]},
                "metrics": metrics_view()}

    def truth_layers(flat) -> Dict[str, Any]:
        """Hidden 'reality' of the virtual field, for twin-vs-reality views (sim only)."""
        vf = ctx["virtual"]
        if vf is None:
            return {}
        t = vf.truth.state
        return {"truth_stress": flat(t.resample("stress", UE5_GRID, UE5_GRID)),
                "truth_fungus": flat(t.resample("fungus", UE5_GRID, UE5_GRID)),
                "truth_ndvi": flat(np.clip(t.resample("ndvi", UE5_GRID, UE5_GRID), 0.0, 1.0))}

    # ---------------- UE5-resolution helpers ----------------
    def ue5_grids() -> Dict[str, np.ndarray]:
        s = get_twin().state
        return {
            "ndvi": s.resample("ndvi", UE5_GRID, UE5_GRID),
            "stress": s.resample("stress", UE5_GRID, UE5_GRID),
            "moisture": s.resample("moisture", UE5_GRID, UE5_GRID),
            "ndvi_std": np.sqrt(s.resample("ndvi", UE5_GRID, UE5_GRID, which="var")),
            "stress_std": np.sqrt(s.resample("stress", UE5_GRID, UE5_GRID, which="var")),
        }

    def detect_anomalies() -> int:
        """Flag UE5 cells with high stress or a sharp NDVI drop since the last check."""
        g = ue5_grids()
        prev = ctx["ue5_snapshot"]
        ctx["ue5_snapshot"] = g
        drop = (prev["ndvi"] - g["ndvi"]) if prev is not None else np.zeros_like(g["ndvi"])
        flagged = (g["stress"] > ANOMALY_STRESS) | (drop > ANOMALY_NDVI_DROP)
        day, t = get_twin().state.sim_day, time.time()
        new = []
        for r, c in zip(*np.nonzero(flagged)):
            new.append({
                "day": day, "index": int(r * UE5_GRID + c), "row": int(r), "col": int(c),
                "ndvi": round(float(g["ndvi"][r, c]), 4),
                "stress": round(float(g["stress"][r, c]), 4),
                "ndvi_drop": round(float(drop[r, c]), 4),
                "timestamp": t,
            })
        ctx["anomalies"].extend(new)
        del ctx["anomalies"][:-200]
        return len(new)

    def instance(idx: int, g: Dict[str, np.ndarray], t: float) -> Dict[str, Any]:
        r, c = divmod(idx, UE5_GRID)
        temp = get_twin().state.last_weather.get("temperature", 29.0)
        return {
            "index": idx, "row": r, "col": c,
            "ndvi": round(float(np.clip(g["ndvi"][r, c], 0.0, 1.0)), 4),
            "stress": round(float(g["stress"][r, c]), 4),
            "soil_moisture": round(float(g["moisture"][r, c]) * 100.0, 2),
            "temperature_c": float(temp),
            "ndvi_std": round(float(g["ndvi_std"][r, c]), 4),
            "stress_std": round(float(g["stress_std"][r, c]), 4),
            "last_updated": t,
        }

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        t = get_twin()
        ctx["ue5_snapshot"] = ue5_grids()
        print(f"[DigitalTwinAPI] field={t.field_id} grid={t.state.grid.rows}x{t.state.grid.cols} "
              f"day={t.state.sim_day} -> http://{settings.twin_api_host}:{settings.twin_api_port}/docs",
              flush=True)
        yield
        if ctx["mavlink"] is not None:
            ctx["mavlink"].stop()

    app = FastAPI(
        title="Paddy Field Digital Twin API",
        description="Single source of truth for the simulated paddy field. "
                    "UE5 and the three.js dashboard read from it; observations are assimilated.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.ctx = ctx

    # ============================================================
    # UE5-COMPATIBLE ROUTES
    # ============================================================

    @app.get("/", summary="Health check")
    def health():
        t = get_twin()
        return {
            "status": "ok",
            "service": "Paddy Field Digital Twin API",
            "version": "2.0.0",
            "field_id": t.field_id,
            "current_day": t.state.sim_day,
            "instances": NUM_INSTANCES,
            "twin_grid": [t.state.grid.rows, t.state.grid.cols],
        }

    @app.post("/field/update", summary="UE5 pushes field state; assimilated as observations")
    def field_update(payload: FieldUpdatePayload):
        if len(payload.instances) > NUM_INSTANCES:
            raise HTTPException(400, f"Too many instances: {len(payload.instances)} > {NUM_INSTANCES}")
        obs = {k: np.full((UE5_GRID, UE5_GRID), np.nan, dtype=np.float32)
               for k in ("ndvi", "moisture")}
        for inst in payload.instances:
            if not (0 <= inst.row < UE5_GRID and 0 <= inst.col < UE5_GRID):
                raise HTTPException(400, f"Instance {inst.index} row/col outside {UE5_GRID}x{UE5_GRID} grid")
            obs["ndvi"][inst.row, inst.col] = inst.ndvi
            obs["moisture"][inst.row, inst.col] = inst.soil_moisture / 100.0
        twin = get_twin()
        # Stress is derived from NDVI + fungus in the model, so assimilating
        # NDVI is enough; UE5's stress value is reported back unchanged.
        results = []
        for layer, arr in obs.items():
            results.extend(twin.observe_grid(layer, arr, source="ue5"))
        new_anomalies = detect_anomalies()
        s = twin.state.summary()["layers"]
        return {
            "accepted": len(payload.instances),
            "day": twin.state.sim_day,
            "ue5_day": payload.day,
            "new_anomalies": new_anomalies,
            "avg_ndvi": s["ndvi"]["mean"],
            "avg_stress": s["stress"]["mean"],
            "assimilation": results,
        }

    @app.get("/field/status", summary="Full current state — all 400 instances")
    def field_status():
        g, t = ue5_grids(), time.time()
        return {"day": get_twin().state.sim_day,
                "instances": [instance(i, g, t) for i in range(NUM_INSTANCES)]}

    @app.get("/field/instance/{index}", summary="Single instance record")
    def field_instance(index: int):
        if not 0 <= index < NUM_INSTANCES:
            raise HTTPException(404, f"Instance {index} not found.")
        return instance(index, ue5_grids(), time.time())

    @app.get("/field/summary", summary="Aggregated field statistics")
    def field_summary():
        twin = get_twin()
        g = ue5_grids()
        stress = g["stress"]
        n = stress.size
        hist = [h for h in twin.history if h["reason"] == "predict"][-14:]
        return {
            "day": twin.state.sim_day,
            "growth_stage": twin.state.growth_stage,
            "total_instances": n,
            "ndvi": {k: round(float(f(g["ndvi"])), 4) for k, f in
                     (("mean", np.mean), ("min", np.min), ("max", np.max))},
            "stress": {
                "mean": round(float(stress.mean()), 4),
                "high_count": int((stress > 0.5).sum()),
                "healthy_count": int((stress < 0.2).sum()),
                "pct_stressed": round(float((stress > 0.5).sum()) / n * 100, 1),
            },
            "soil_moisture": {k: round(float(f(g["moisture"])) * 100.0, 4) for k, f in
                              (("mean", np.mean), ("min", np.min), ("max", np.max))},
            "temperature_c": {"mean": twin.state.last_weather.get("temperature", 29.0)},
            "history_days": len(hist),
            "history": [{"day": h["sim_day"], "avg_ndvi": h["ndvi_mean"],
                         "avg_stress": h["stress_mean"], "timestamp": h["timestamp"]} for h in hist],
        }

    @app.get("/field/anomalies", summary="Recent anomaly events")
    def field_anomalies(limit: int = 50):
        a = ctx["anomalies"]
        return {"total": len(a), "anomalies": a[-limit:]}

    @app.get("/field/heatmap", summary="NDVI + stress (and their std) as UE5 grids")
    def field_heatmap():
        g = ue5_grids()
        r4 = lambda a: np.round(a.astype(np.float64), 4).tolist()  # noqa: E731
        return {
            "day": get_twin().state.sim_day,
            "grid_size": UE5_GRID,
            "ndvi_grid": r4(np.clip(g["ndvi"], 0.0, 1.0)),
            "stress_grid": r4(g["stress"]),
            "ndvi_std_grid": r4(g["ndvi_std"]),
            "stress_std_grid": r4(g["stress_std"]),
        }

    @app.delete("/field/reset", summary="Reset the twin to its prior state")
    def field_reset():
        get_twin().reset()
        ctx["anomalies"].clear()
        ctx["ue5_snapshot"] = ue5_grids()
        return {"status": "reset", "instances": NUM_INSTANCES}

    # ============================================================
    # TWIN ROUTES
    # ============================================================

    @app.get("/twin/state", summary="Twin summary and per-zone values")
    def twin_state():
        twin = get_twin()
        return {"field_id": twin.field_id, **twin.state.summary(), "zones": twin.zone_summary()}

    @app.get("/twin/layers/{layer}", summary="One layer at any resolution")
    def twin_layer(layer: str, rows: Optional[int] = None, cols: Optional[int] = None,
                   which: Literal["mean", "var", "std"] = "mean"):
        if layer not in LAYERS:
            raise HTTPException(404, f"Unknown layer '{layer}'. Expected one of {list(LAYERS)}.")
        s = get_twin().state
        rows, cols = rows or s.grid.rows, cols or s.grid.cols
        if not (1 <= rows <= 500 and 1 <= cols <= 500):
            raise HTTPException(400, "rows and cols must be between 1 and 500")
        arr = s.resample(layer, rows, cols, which="mean" if which == "mean" else "var")
        if which == "std":
            arr = np.sqrt(arr)
        return {"layer": layer, "which": which, "rows": rows, "cols": cols,
                "sim_day": s.sim_day, "values": np.round(arr.astype(np.float64), 5).tolist()}

    @app.post("/twin/step", summary="Advance the simulation")
    def twin_step(req: StepRequest):
        twin = get_twin()
        _check_zones(twin, req.interventions)
        weather, used = req.weather, "manual"
        if req.weather_source != "manual":
            weather, used = forecast(req.weather_source, req.days, twin.state.sim_day + 1)
        result = twin.step(days=req.days, weather=weather,
                           interventions=[i.model_dump() for i in req.interventions])
        result["new_anomalies"] = detect_anomalies()
        result["weather_source"] = used
        return result

    @app.post("/twin/ingest/rgb", summary="Assimilate an RGB AI result at the drone/phone position")
    def twin_ingest_rgb(req: RGBIngest):
        return ingest_rgb_result(get_twin(), req.result, req.telemetry, req.source,
                                 (req.frame_width, req.frame_height))

    @app.post("/twin/ingest/multispectral", summary="Assimilate 4-band reflectance over the field")
    def twin_ingest_multispectral(req: MultispectralIngest):
        bands = {k: np.array([[np.nan if v is None else v for v in row] for row in getattr(req, k)],
                             dtype=np.float32) for k in ("green", "red", "red_edge", "nir")}
        shapes = {b.shape for b in bands.values()}
        if len(shapes) != 1 or bands["nir"].ndim != 2 or bands["nir"].size == 0:
            raise HTTPException(400, "All four bands must be non-empty 2D lists of the same shape")
        return {"results": ingest_multispectral(get_twin(), bands, req.source)}

    # ============================================================
    # SIMULATION ROUTES (virtual physical field)
    # ============================================================

    @app.post("/sim/reset", summary="Create a new hidden virtual field")
    def sim_reset(req: SimReset):
        if ctx.get("user_field") and not req.simulated:
            # An uploaded field is loaded: start again from it (the demo story hides one outbreak in it).
            return build_user_field(seed=req.seed, outbreaks=req.outbreaks, reset_twin=req.reset_twin)
        twin = get_twin()
        if req.reset_twin:
            twin.reset()
            ctx["anomalies"].clear()
            ctx["ue5_snapshot"] = ue5_grids()
        ctx["pending_interventions"] = []
        ctx["metrics"] = new_metrics()
        ctx["virtual"] = VirtualField(twin.state.grid, seed=req.seed, outbreaks=req.outbreaks,
                                      propagation_model=twin.propagation_model,
                                      zone_layout=twin.zone_layout, crop_model=twin.crop_model,
                                      params=twin.params)
        return {"status": "ok", "seed": req.seed, "error": ctx["virtual"].twin_error(twin)}

    # ============================================================
    # USER FIELD (an uploaded multispectral image becomes the twin's field)
    # ============================================================

    def _resample_to_grid(arr: np.ndarray, shape) -> np.ndarray:
        """Area-average a (H, W) array onto the twin grid; NaNs (no-data) are ignored."""
        rows, cols = shape
        h, w = arr.shape
        ri = (np.arange(h) * rows // h)
        ci = (np.arange(w) * cols // w)
        valid = np.isfinite(arr)
        s = np.zeros(shape, np.float64)
        n = np.zeros(shape, np.float64)
        np.add.at(s, (ri[:, None].repeat(w, 1)[valid], ci[None, :].repeat(h, 0)[valid]), arr[valid])
        np.add.at(n, (ri[:, None].repeat(w, 1)[valid], ci[None, :].repeat(h, 0)[valid]), 1.0)
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        if np.isnan(out).any():                          # cells with no data at all: field mean
            out = np.where(np.isnan(out), np.nanmean(out) if np.isfinite(out).any() else 0.0, out)
        return out.astype(np.float32)

    # What a drone image reliably says about the field: NDVI. Not "moisture": over a green rice canopy the
    # image's NDWI (green/NIR) is strongly negative, which would read a flooded paddy as drying out. Not
    # "nitrogen" either: the red-edge proxy (CIre / 5) is uncalibrated for these sensors and drove the
    # forecast into false nitrogen deficiency. Both stay with the crop model until calibrated.
    IMAGE_LAYERS = ("ndvi",)

    def _assimilate_image(twin, layers: Dict[str, np.ndarray], source: str) -> None:
        for name in IMAGE_LAYERS:
            if name in layers:
                twin.observe_grid(name, layers[name], source=source)

    def build_user_field(seed: int = 42, outbreaks=None, reset_twin: bool = True) -> Dict[str, Any]:
        """Rebuild the twin and the 'real' field from the stored upload.

        REALITY = the field exactly as the image shows it (NDVI, moisture, nitrogen), evolving with the
        crop model; outbreaks=[] adds no disease, None hides the default one (demo story).
        TWIN    = its prior, then the image assimilated as a multispectral survey, so it starts close
        to the real field with honest uncertainty, and drone scouting refines it from there.
        """
        uf = ctx["user_field"]
        twin = get_twin()
        layers = uf["layers"]
        if reset_twin:
            twin.reset()
            _assimilate_image(twin, layers, source=f"upload:{uf['name']}")
            ctx["anomalies"].clear()
            ctx["ue5_snapshot"] = ue5_grids()
        ctx["pending_interventions"] = []
        ctx["metrics"] = new_metrics()
        vf = VirtualField(twin.state.grid, seed=seed, outbreaks=[], dry_patch={},
                          propagation_model=twin.propagation_model, zone_layout=twin.zone_layout,
                          crop_model=twin.crop_model, params=twin.params)
        m = vf.truth.state.mean
        for name in IMAGE_LAYERS:
            if name in layers and name in m:
                m[name] = np.clip(layers[name], -1.0 if name == "ndvi" else 0.0, 1.0).astype(np.float32)
        g = twin.state.grid
        if outbreaks is None:      # demo / reset story: one disease patch the twin has not been told about
            outbreaks = [{"row": g.rows * 0.3, "col": g.cols * 0.65, "radius": g.rows * 0.12, "load": 0.55}]
        for ob in outbreaks:
            vf.inject_outbreak(ob["row"], ob["col"], radius_cells=ob.get("radius", 3.0), load=ob.get("load", 0.5))
        vf.truth._refresh_stress()
        for name in LAYERS:
            vf.truth.state.var[name][:] = 0.0
        ctx["virtual"] = vf
        ctx["field_version"] = ctx.get("field_version", 0) + 1
        return {"status": "ok", "source": "upload", "name": uf["name"], "seed": seed,
                "error": vf.twin_error(twin)}

    @app.post("/field/from-image", summary="Make an uploaded multispectral image the twin's field")
    def field_from_image(req: FieldFromImage):
        need = ("green", "red", "red_edge", "nir")
        missing = [b for b in need if b not in req.bands]
        if missing:
            raise HTTPException(400, f"Missing bands {missing}; expected {list(need)}")
        twin = get_twin()
        if req.mode == "update":
            if not ctx.get("user_field"):
                raise HTTPException(409, "No uploaded field to update: send it with mode 'new' first")
            return field_update(req, need)
        g0 = twin.state.grid
        cell = settings.twin_cell_size_m
        size_note = None
        if req.width_m and req.height_m:
            side = max(req.width_m, req.height_m)
            if 40.0 <= side <= 3000.0:                    # a plausible field: the grid covers it exactly
                cell = side / g0.rows
            else:
                size_note = (f"The image covers {req.width_m:.0f} x {req.height_m:.0f} m, so it is shown on the "
                             f"default {g0.rows * settings.twin_cell_size_m:.0f} m grid.")
        g = twin.state.grid
        lat, lon = (req.center_lat, req.center_lon) if req.center_lat is not None and req.center_lon is not None \
            else g.center_latlon()
        grid = FieldGrid.centered_on(lat, lon, rows=g.rows, cols=g.cols, cell_size_m=cell)
        if (round(grid.origin_lat, 7), round(grid.origin_lon, 7), round(grid.cell_size_m, 4)) != (
                round(g.origin_lat, 7), round(g.origin_lon, 7), round(g.cell_size_m, 4)):
            # Same number of cells, anchored on (and sized to) the real field: a fresh twin, fleet and pads.
            # It keeps the old twin's storage (none in tests), so it never writes over another twin's files.
            ctx["twin"] = DigitalTwin(field_id=twin.field_id, grid=grid,
                                      storage_dir=twin.storage_dir.parent if twin.storage_dir else None,
                                      propagation_model=twin.propagation_model, crop_model=twin.crop_model,
                                      zone_layout=twin.zone_layout, params=twin.params)
            ctx["_versioned"] = False
            ctx["fleet"] = None
            twin = get_twin()
        if req.growth_stage:
            try:
                twin.set_growth_stage(req.growth_stage)
            except Exception:
                pass
        bands = _bands_on_grid(req, need, twin.state.grid.shape)
        layers = bands_to_layers(bands)
        means = {k: round(float(np.nanmean(v)), 4) for k, v in layers.items()}
        ctx["user_field"] = {"name": req.name, "bands": bands, "layers": layers,
                             "center": twin.state.grid.center_latlon(), "loaded_at": time.time(),
                             "flights": [{"name": req.name, "at": time.time(), "sim_day": 0, "means": means}]}
        out = build_user_field(seed=42, outbreaks=[])
        g = twin.state.grid
        out.update({"grid": {"rows": g.rows, "cols": g.cols, "cell_size_m": round(g.cell_size_m, 3),
                             "center": g.center_latlon(), "area_ha": round(g.area_m2 / 10000.0, 3)},
                    "means": means, "size_note": size_note})
        return out

    def _bands_on_grid(req, need, shape) -> Dict[str, np.ndarray]:
        bands = {}
        for b in need:
            arr = np.array([[np.nan if v is None else v for v in row] for row in req.bands[b]], dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                raise HTTPException(400, f"Band '{b}' must be a non-empty 2D list")
            if np.nanmax(arr) > 1.5:                      # 0-10000 / 0-65535 scaled reflectance
                arr = arr / (10000.0 if np.nanmax(arr) <= 10000 else 65535.0)
            bands[b] = _resample_to_grid(arr, shape)
        return bands

    def field_update(req, need) -> Dict[str, Any]:
        """A new flight of the loaded field: the twin assimilates it as a survey (its uncertainty
        shrinks), and the real field takes the observed values (its disease state is kept)."""
        twin, vf, uf = get_twin(), get_virtual(), ctx["user_field"]
        bands = _bands_on_grid(req, need, twin.state.grid.shape)
        layers = bands_to_layers(bands)
        before = vf.twin_error(twin)
        std_before = float(np.sqrt(twin.state.var["ndvi"]).mean())
        _assimilate_image(twin, layers, source=f"flight:{req.name}")
        m = vf.truth.state.mean
        for name in IMAGE_LAYERS:
            if name in layers and name in m:
                m[name] = np.clip(layers[name], -1.0 if name == "ndvi" else 0.0, 1.0).astype(np.float32)
        vf.truth._refresh_stress()
        for name in LAYERS:
            vf.truth.state.var[name][:] = 0.0
        means = {k: round(float(np.nanmean(v)), 4) for k, v in layers.items()}
        uf.update({"bands": bands, "layers": layers})
        uf.setdefault("flights", []).append({"name": req.name, "at": time.time(),
                                            "sim_day": twin.state.sim_day, "means": means})
        detect_anomalies()
        ctx["field_version"] = ctx.get("field_version", 0) + 1
        return {"status": "ok", "source": "upload", "mode": "update", "name": uf["name"],
                "flights": len(uf["flights"]), "means": means, "error_before": before,
                "error_after": vf.twin_error(twin),
                "ndvi_uncertainty": {"before": round(std_before, 4),
                                     "after": round(float(np.sqrt(twin.state.var["ndvi"]).mean()), 4)}}

    @app.get("/field/flights", summary="Flights (uploads) of the current field")
    def field_flights():
        uf = ctx.get("user_field")
        return {"name": uf["name"] if uf else None, "flights": (uf or {}).get("flights", [])}

    @app.delete("/field/user", summary="Forget the uploaded field and go back to the simulated one")
    def field_forget():
        ctx["user_field"] = None
        return sim_reset(SimReset(simulated=True))

    @app.get("/field/source", summary="Which field the twin is tracking")
    def field_source():
        return field_source_view()

    def field_source_view() -> Dict[str, Any]:
        uf = ctx.get("user_field")
        if uf:
            lat, lon = uf["center"]
            return {"source": "upload", "name": uf["name"], "center": [round(lat, 6), round(lon, 6)]}
        return {"source": "simulated", "name": "Field Alpha (simulated)", "center": None}

    @app.post("/sim/advance", summary="Advance the virtual field and the twin together")
    def sim_advance(req: SimAdvance):
        twin, vf = get_twin(), get_virtual()
        _check_zones(twin, req.interventions)
        source = req.weather_source or ("auto" if ctx.get("user_field") else "synthetic")
        weather, used = forecast(source, req.days, vf.sim_day + 1)
        plan = [i.model_dump() for i in req.interventions]
        # Sprays flown by the fleet since the last advance take effect on day 1.
        flown = [{"day": 1, "zone_id": p["zone_id"], "type": p["type"]}
                 for p in ctx["pending_interventions"]]
        ctx["pending_interventions"] = []
        plan += flown
        actual = vf.advance(req.days, weather, plan)        # truth gets the real weather
        result = twin.step(days=req.days, weather=weather, interventions=plan)  # twin gets the forecast
        record_error()
        return {"sim_day": twin.state.sim_day, "truth_day": vf.sim_day, "weather_source": used,
                "applied_fleet_interventions": flown,
                "forecast": weather, "actual_weather": actual,
                "new_anomalies": detect_anomalies(), "logs": result["logs"],
                "error": vf.twin_error(twin)}

    @app.post("/sim/drone-pass", summary="Fly a serpentine RGB pass and assimilate every frame")
    def sim_drone_pass(req: SimDronePass):
        twin, vf = get_twin(), get_virtual()
        before = vf.twin_error(twin)
        used = 0
        for lat, lon in serpentine_waypoints(twin.state.grid, req.lane_spacing_m, req.step_m):
            frame = vf.drone_camera(lat, lon, alt=req.altitude_m)
            if frame and ingest_rgb_result(twin, frame, {"lat": lat, "lon": lon,
                                                         "alt": req.altitude_m})["assimilated"]:
                used += 1
        detect_anomalies()
        return {"frames_assimilated": used, "error_before": before, "error_after": vf.twin_error(twin)}

    @app.post("/sim/survey", summary="Multispectral survey of the virtual field, assimilated")
    def sim_survey(req: SimSurvey):
        twin, vf = get_twin(), get_virtual()
        before = vf.twin_error(twin)
        ingest_multispectral(twin, vf.multispectral_survey(noise=req.noise, coverage=req.coverage))
        detect_anomalies()
        return {"error_before": before, "error_after": vf.twin_error(twin)}

    @app.post("/sim/mission", summary="Plan and fly a fleet mission over the virtual field")
    def sim_mission(req: MissionRequest):
        twin, vf, fleet = get_twin(), get_virtual(), get_fleet()
        if not fleet.drones:
            raise HTTPException(409, "Fleet has no drones")
        before = vf.twin_error(twin)
        assimilated = {"n": 0}

        def on_capture(drone_id: str, lat: float, lon: float, alt: float) -> None:
            frame = vf.drone_camera(lat, lon, alt)
            if frame and ingest_rgb_result(twin, frame, {"lat": lat, "lon": lon, "alt": alt})["assimilated"]:
                assimilated["n"] += 1

        wind = twin.state.last_weather or None
        reports = []
        for kind in (("scout", "spray") if req.kind == "auto" else (req.kind,)):
            tasks, plan = plan_mission(kind, req)
            if req.backend == "mavlink":
                report = mm.execute_plan_mavlink(
                    plan, fleet.frame, mavlink_connections(), homes=drone_homes(),
                    on_capture=on_capture if kind == "scout" else None,
                    capture_spacing_m=fleet.spec.capture_spacing_m,
                    lane_model=mm.lane_model_for(settings.twin_mavlink_mode, settings.twin_params_file.parent))
                if report["errors"] and not report["drones"]:
                    raise HTTPException(502, f"MAVLink execution failed: {report['errors']}")
            else:
                report = fleet.execute(plan, on_capture=on_capture if kind == "scout" else None, wind=wind)
            ctx["pending_interventions"].extend(report["completed_sprays"])
            record_mission(kind, report)
            publish_track(kind, report["track"])
            if not req.include_track:
                report.pop("track")
            reports.append({"kind": kind, "plan": plan.to_dict(), "report": report})
        detect_anomalies()
        summary = {"kind": req.kind, "backend": req.backend, "frames_assimilated": assimilated["n"],
                   "sprays_queued": [p["zone_id"] for p in ctx["pending_interventions"]]}
        twin._emit("swarm.mission", summary)
        return {**summary, "missions": reports, "error_before": before,
                "error_after": vf.twin_error(twin),
                "pending_interventions": ctx["pending_interventions"]}

    # ============================================================
    # SWARM ROUTES
    # ============================================================

    @app.get("/swarm/state", summary="Fleet positions and status")
    def swarm_state():
        return {**get_fleet().snapshot(), "pending_interventions": ctx["pending_interventions"]}

    @app.post("/swarm/drones", summary="Add a drone to the fleet")
    def swarm_add(req: AddDrone):
        try:
            get_fleet().add_drone(req.drone_id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"drones": sorted(get_fleet().drones)}

    @app.delete("/swarm/drones/{drone_id}", summary="Remove a drone from the fleet")
    def swarm_remove(drone_id: str):
        try:
            get_fleet().remove_drone(drone_id)
        except KeyError:
            raise HTTPException(404, f"Drone '{drone_id}' not found")
        return {"drones": sorted(get_fleet().drones)}

    @app.post("/swarm/plan", summary="Allocate tasks to the fleet without flying")
    def swarm_plan(req: MissionRequest):
        if req.kind == "auto":
            raise HTTPException(400, "Plan 'scout' or 'spray'; 'auto' is only for /sim/mission")
        tasks, plan = plan_mission(req.kind, req)
        return {"tasks": [t.to_dict() for t in tasks], **plan.to_dict()}

    @app.post("/swarm/mission-files", summary="Planned missions as .waypoints (QGC WPL 110) per drone")
    def swarm_mission_files(req: MissionRequest):
        if req.kind == "auto":
            raise HTTPException(400, "Export 'scout' or 'spray'; 'auto' depends on the scout results")
        fleet = get_fleet()
        _, plan = plan_mission(req.kind, req)
        homes = drone_homes()
        out = {}
        for k, (d_id, a) in enumerate(plan.assignments.items()):
            if a.tasks:
                mission = mm.compile_mission(
                    a, fleet.frame, transit_alt=14.0 + 2.0 * k, home=homes[d_id],
                    lane_model=mm.lane_model_for(settings.twin_mavlink_mode, settings.twin_params_file.parent))
                out[d_id] = {"items": len(mission.items), "pump_pwm": mission.spray_pwm,
                             "waypoints": mm.to_qgc_wpl(mission)}
        return {"kind": req.kind, "missions": out, "unassigned": [t.task_id for t in plan.unassigned]}

    @app.get("/swarm/sitl-homes", summary="One ArduCopter SITL instance per drone, at its launch pad")
    def swarm_sitl_homes():
        out = []
        for i, (d_id, (lat, lon)) in enumerate(drone_homes().items()):
            port = settings.twin_mavlink_base_port + 10 * i
            out.append({"drone_id": d_id, "instance": i, "home_lat": round(lat, 8), "home_lon": round(lon, 8),
                        "connection": f"tcp:127.0.0.1:{port}",
                        "command": f"sim_vehicle.py -v ArduCopter -I {i} --sysid {i + 1} "
                                   f"--custom-location={lat:.8f},{lon:.8f},0,0 --no-mavproxy"})
        return {"mode": settings.twin_mavlink_mode, "instances": out}

    # ============================================================
    # DEMO MODE
    # ============================================================

    def demo_status(phase: str, text: str) -> None:
        d = ctx["demo"]
        d.update(phase=phase, text=text, day=get_twin().state.sim_day)
        status_msg = {"type": "twin.status", "phase": phase, "text": text,
                      "sim_day": d["day"], "error": vf_error_brief()}
        ctx["latest_status"] = status_msg
        notify_field_listeners(status_msg)

    def vf_error_brief() -> Optional[Dict[str, float]]:
        if ctx["virtual"] is None:
            return None
        e = ctx["virtual"].twin_error(get_twin())
        return {k: e[k]["rmse"] for k in ("stress", "fungus", "ndvi")}

    def grid_rc(lat: float, lon: float) -> tuple:
        grid = get_twin().state.grid
        east, south = grid.latlon_to_local_m(lat, lon)
        return (south / (grid.rows * grid.cell_size_m) * UE5_GRID,
                east / (grid.cols * grid.cell_size_m) * UE5_GRID)

    async def replay(report: Dict[str, Any], speed: float, ingest: bool) -> None:
        """Re-fly a finished mission in real time: drone positions at 10 Hz, and
        each camera frame assimilated when the replay reaches it, so the field
        changes in UE5 as the drones pass over it."""
        tracks = {d: t for d, t in report.get("track", {}).items() if t}
        if not tracks:
            return
        captures = sorted(report.get("captures", []), key=lambda c: c[0]) if ingest else []
        end_t = max(t[-1][0] for t in tracks.values())
        twin, vf = get_twin(), get_virtual()
        loop = asyncio.get_running_loop()
        start = loop.time()
        ci = 0
        idx = {d: 0 for d in tracks}
        while True:
            sim_t = (loop.time() - start) * speed
            drones = {}
            for d, tr in tracks.items():
                i = idx[d]
                while i + 1 < len(tr) - 1 and tr[i + 1][0] <= sim_t:
                    i += 1
                idx[d] = i
                a, b = tr[i], tr[min(i + 1, len(tr) - 1)]
                f = 0.0 if b[0] <= a[0] else min(1.0, max(0.0, (sim_t - a[0]) / (b[0] - a[0])))
                lat, lon = a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
                row, col = grid_rc(lat, lon)
                r2, c2 = grid_rc(b[1], b[2])
                r1, c1 = grid_rc(a[1], a[2])
                yaw = math.degrees(math.atan2(c2 - c1, -(r2 - r1))) if (r2, c2) != (r1, c1) else None
                drones[d] = [round(row, 3), round(col, 3), round(a[3] + f * (b[3] - a[3]), 2),
                             int(a[4]), None if yaw is None else round(yaw % 360.0, 1)]
            pos_msg = {"type": "swarm.positions", "t_s": round(min(sim_t, end_t), 2),
                       "fields": ["row", "col", "alt_m", "spraying", "heading_deg"],
                       "drones": drones}
            ctx["latest_positions"] = pos_msg
            notify_field_listeners(pos_msg)
            batch = []
            while ci < len(captures) and captures[ci][0] <= sim_t:
                batch.append(captures[ci])
                ci += 1
            if batch:
                def _ingest(items=batch):
                    for _t, _d, lat, lon, alt in items:
                        frame = vf.drone_camera(lat, lon, alt)
                        if frame:
                            ingest_rgb_result(twin, frame, {"lat": lat, "lon": lon, "alt": alt})
                await asyncio.to_thread(_ingest)
            if sim_t >= end_t:
                return
            await asyncio.sleep(0.1)

    async def fly(kind: str, req: DemoRequest) -> Dict[str, Any]:
        mreq = MissionRequest(kind=kind, backend=req.backend, fungus_threshold=req.fungus_threshold)
        fleet = get_fleet()
        tasks, plan = plan_mission(kind, mreq)
        if not tasks:
            return {"tasks": 0}
        if req.backend == "mavlink":
            report = await asyncio.to_thread(
                mm.execute_plan_mavlink, plan, fleet.frame, mavlink_connections(), drone_homes(), None,
                fleet.spec.capture_spacing_m, 60.0, 1800.0,
                mm.lane_model_for(settings.twin_mavlink_mode, settings.twin_params_file.parent))
        else:
            report = await asyncio.to_thread(fleet.execute, plan, None, 0.05, 3600.0, 2.0, 0.5,
                                             get_twin().state.last_weather or None)
        await replay(report, req.playback_speed, ingest=(kind == "scout"))
        ctx["pending_interventions"].extend(report["completed_sprays"])
        record_mission(kind, report)
        publish_track(kind, report["track"])
        detect_anomalies()
        return {"tasks": len(tasks), "sprays": [p["zone_id"] for p in report["completed_sprays"]]}

    async def run_demo(req: DemoRequest) -> None:
        d = ctx["demo"]
        try:
            await asyncio.to_thread(sim_reset, SimReset(seed=req.seed))
            demo_status("start", "Day 0: the twin starts from its prior. A fungal outbreak is hidden "
                                 "somewhere in the real field.")
            await asyncio.sleep(req.day_seconds)
            for _ in range(req.days):
                adv = await asyncio.to_thread(sim_advance, SimAdvance(days=1))
                sprayed = [i["zone_id"] for i in adv["applied_fleet_interventions"]]
                day = adv["sim_day"]
                demo_status("advance", f"Day {day}: crop grows, disease spreads"
                            + (f"; fungicide takes effect in zone(s) {sprayed}" if sprayed else "") + ".")
                if day % req.scout_every == 0:
                    demo_status("scout", f"Day {day}: drones scouting the field; the twin updates as they fly.")
                    await fly("scout", req)
                    demo_status("decide", f"Day {day}: twin re-planned from what the drones saw.")
                    res = await fly("spray", req)
                    if res.get("tasks"):
                        demo_status("spray", f"Day {day}: sprayed zone(s) {res.get('sprays')}; "
                                             "effect from tomorrow.")
                    else:
                        demo_status("decide", f"Day {day}: no zone above the action threshold; no spraying.")
                await asyncio.sleep(req.day_seconds)
            demo_status("done", f"Demo finished on day {get_twin().state.sim_day}.")
        except asyncio.CancelledError:
            demo_status("stopped", "Demo stopped.")
            raise
        except Exception as e:  # report, do not kill the server
            d["error"] = f"{type(e).__name__}: {e}"
            demo_status("error", d["error"])
        finally:
            d["task"] = None

    def touch_field() -> None:
        """Truth changed without a twin event: bump the version and push a snapshot."""
        ctx["field_version"] = ctx.get("field_version", 0) + 1
        notify_field_listeners(None)

    async def run_action(req: LiveAction) -> None:
        d = ctx["demo"]
        demo_req = DemoRequest(playback_speed=req.playback_speed, backend=req.backend,
                               fungus_threshold=req.fungus_threshold)
        try:
            if req.action == "reset":
                await asyncio.to_thread(sim_reset, SimReset(seed=req.seed))
                demo_status("start", "Reset: the twin starts from its prior; a new outbreak is hidden "
                                     "in the real field.")
            elif req.action == "next_day":
                adv = await asyncio.to_thread(sim_advance, SimAdvance(days=1))
                sprayed = [i["zone_id"] for i in adv["applied_fleet_interventions"]]
                demo_status("advance", f"Day {adv['sim_day']}: crop grows, disease spreads"
                            + (f"; fungicide takes effect in zone(s) {sprayed}" if sprayed else "") + ".")
            elif req.action == "outbreak":
                vf = get_virtual()
                g = get_twin().state.grid
                row = req.row if req.row is not None else UE5_GRID / 2.0
                col = req.col if req.col is not None else UE5_GRID / 2.0
                scale = g.rows / UE5_GRID
                await asyncio.to_thread(vf.inject_outbreak, row * scale, col * scale,
                                        req.radius * scale, req.load)
                touch_field()
                demo_status("outbreak", f"Day {get_twin().state.sim_day}: a new outbreak started near "
                                        f"row {row:.0f}, col {col:.0f}. The twin does not know yet; "
                                        "scout to find it.")
            if req.action in ("scout", "auto"):
                day = get_twin().state.sim_day
                demo_status("scout", f"Day {day}: drones scouting the field; the twin updates as they fly.")
                res = await fly("scout", demo_req)
                demo_status("decide", f"Day {day}: twin updated from {res.get('tasks', 0)} scouted zones.")
            if req.action in ("spray", "auto"):
                day = get_twin().state.sim_day
                res = await fly("spray", demo_req)
                if res.get("tasks"):
                    demo_status("spray", f"Day {day}: sprayed zone(s) {res.get('sprays')}; effect from tomorrow.")
                else:
                    demo_status("decide", f"Day {day}: no zone above the action threshold; nothing to spray.")
        except asyncio.CancelledError:
            demo_status("stopped", "Stopped.")
            raise
        except Exception as e:
            d["error"] = f"{type(e).__name__}: {e}"
            demo_status("error", d["error"])
        finally:
            d["task"] = None

    @app.post("/live/action", summary="One live step (UE5 menu): next day, scout, spray, outbreak, reset")
    async def live_action(req: LiveAction):
        d = ctx["demo"]
        if d["task"] is not None and not d["task"].done():
            raise HTTPException(409, "Busy: a demo or action is running; wait or POST /demo/stop")
        d["error"] = None
        d["task"] = asyncio.get_running_loop().create_task(run_action(req))
        return {"status": "started", "action": req.action}

    @app.post("/demo/start", summary="Start the paced closed-loop demo for UE5")
    async def demo_start(req: DemoRequest):
        d = ctx["demo"]
        if d["task"] is not None and not d["task"].done():
            raise HTTPException(409, "A demo is already running; POST /demo/stop first")
        d["error"] = None
        d["task"] = asyncio.get_running_loop().create_task(run_demo(req))
        return {"status": "started", **req.model_dump()}

    @app.post("/demo/stop", summary="Stop the demo")
    async def demo_stop():
        task = ctx["demo"]["task"]
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return {"status": "stopped"}

    @app.get("/demo/status", summary="Demo phase and twin-vs-truth error")
    def demo_status_route():
        d = ctx["demo"]
        return {"running": d["task"] is not None and not d["task"].done(), "phase": d["phase"],
                "text": d["text"], "sim_day": d["day"], "error": d["error"],
                "twin_error": vf_error_brief()}

    @app.get("/ue5/frame", summary="Polling view for UE5 Python / VaRest clients")
    def ue5_frame(since: int = -1):
        """
        Everything a UE5 client needs per frame. ``field`` (a field.snapshot)
        is only included when the twin changed since ``since``; pass back the
        returned ``seq``. Drone poses and the caption are always included.
        """
        get_twin()
        version = int(ctx.get("field_version", 0))
        return {
            "seq": version,
            "status": ctx.get("latest_status"),
            "drones": ctx.get("latest_positions"),
            "demo_running": ctx["demo"]["task"] is not None and not ctx["demo"]["task"].done(),
            "field": field_snapshot(version) if version != since else None,
            "metrics": metrics_view(),
            "weather": frame_weather(),
            "field_source": field_source_view(),
        }

    def frame_weather():
        """Today's weather in the (virtual) real field, for rain effects in UE5."""
        vf = ctx.get("virtual")
        w = (vf.truth.state.last_weather if vf is not None else None) or get_twin().state.last_weather or {}
        return {"precipitation_mm": round(float(w.get("precipitation", 0.0)), 2),
                "temperature_c": round(float(w.get("temperature", 29.0)), 1),
                "humidity_pct": round(float(w.get("humidity", 75.0)), 1),
                "wind_speed_m_s": round(float(w.get("wind_speed", 2.0)), 2)}

    @app.get("/sim/error", summary="Twin vs ground truth")
    def sim_error():
        twin, vf = get_twin(), get_virtual()
        return {"sim_day": twin.state.sim_day, "truth_day": vf.sim_day, "error": vf.twin_error(twin)}

    @app.get("/sim/truth/layers/{layer}", summary="Ground-truth layer of the virtual field")
    def sim_truth_layer(layer: str, rows: Optional[int] = None, cols: Optional[int] = None):
        if layer not in LAYERS:
            raise HTTPException(404, f"Unknown layer '{layer}'. Expected one of {list(LAYERS)}.")
        s = get_virtual().truth.state
        rows, cols = rows or s.grid.rows, cols or s.grid.cols
        if not (1 <= rows <= 500 and 1 <= cols <= 500):
            raise HTTPException(400, "rows and cols must be between 1 and 500")
        arr = s.resample(layer, rows, cols)
        return {"layer": layer, "rows": rows, "cols": cols, "truth_day": s.sim_day,
                "values": np.round(arr.astype(np.float64), 5).tolist()}

    @app.post("/twin/observe/point", summary="Assimilate a geotagged reading")
    def twin_observe_point(obs: PointObservation):
        unknown = set(obs.values) - set(LAYERS)
        if unknown:
            raise HTTPException(400, f"Unknown layers {sorted(unknown)}. Expected {list(LAYERS)}.")
        try:
            results = get_twin().observe_point(
                obs.lat, obs.lon, obs.values, obs.source, obs.obs_var,
                obs.footprint_m, obs.influence_radius_m)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"results": results}

    @app.post("/twin/observe/grid", summary="Assimilate a raster observation")
    def twin_observe_grid(obs: GridObservation):
        if obs.layer not in LAYERS:
            raise HTTPException(400, f"Unknown layer '{obs.layer}'. Expected one of {list(LAYERS)}.")
        arr = np.array([[np.nan if v is None else v for v in row] for row in obs.values],
                       dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            raise HTTPException(400, "values must be a non-empty 2D list")
        return {"results": get_twin().observe_grid(obs.layer, arr, obs.source, obs.obs_var)}

    @app.post("/twin/what-if", summary="Compare interventions against doing nothing")
    def twin_what_if(req: WhatIfRequest):
        twin = get_twin()
        _check_zones(twin, req.interventions)
        return twin.what_if(days=req.days, weather=req.weather,
                            interventions=[i.model_dump() for i in req.interventions])

    @app.post("/twin/stage", summary="Set the paddy growth stage")
    def twin_stage(req: StageRequest):
        get_twin().set_growth_stage(req.growth_stage)
        return {"growth_stage": req.growth_stage}

    @app.get("/twin/history", summary="History of layer means and uncertainty")
    def twin_history(limit: int = 365):
        return {"history": get_twin().history[-limit:]}

    @app.get("/twin/events", summary="Recent twin events")
    def twin_events(limit: int = 50):
        return {"events": list(get_twin().events)[-limit:]}

    @app.websocket("/ws/twin")
    async def ws_twin(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        def push(event: Dict[str, Any]) -> None:
            def _put() -> None:
                if not queue.full():
                    queue.put_nowait(event)
            loop.call_soon_threadsafe(_put)

        twin = get_twin()
        unsubscribe = twin.subscribe(push)
        try:
            await ws.send_json({"type": "twin.hello", "field_id": twin.field_id,
                                **twin.state.summary()})
            while True:
                event = await queue.get()
                await ws.send_json(_jsonable(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            unsubscribe()

    @app.websocket("/ws/field")
    async def ws_field(ws: WebSocket, min_interval_s: float = 0.25):
        """
        UE5 stream. Sends a full field.snapshot on connect and whenever the
        twin changes (coalesced to at most one per ``min_interval_s``), and
        swarm.track after each fleet mission. The client may send
        {"type": "snapshot"} to request a snapshot immediately.
        """
        await ws.accept()
        loop = asyncio.get_running_loop()
        dirty = asyncio.Event()
        outbox: asyncio.Queue = asyncio.Queue(maxsize=16)

        def on_twin_event(_event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(dirty.set)

        def on_message(message: Optional[Dict[str, Any]]) -> None:
            def _put() -> None:
                if message is None:
                    dirty.set()
                elif not outbox.full():
                    outbox.put_nowait(message)
            loop.call_soon_threadsafe(_put)

        unsubscribe = get_twin().subscribe(on_twin_event)
        ctx["field_listeners"].add(on_message)
        seq = 0

        async def sender() -> None:
            nonlocal seq
            await ws.send_json(field_snapshot(seq))
            if ctx["last_track"] is not None:
                await ws.send_json(ctx["last_track"])
            while True:
                waiters = [asyncio.ensure_future(dirty.wait()), asyncio.ensure_future(outbox.get())]
                done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                for w in pending:
                    w.cancel()
                for w in done:
                    if w is waiters[1]:
                        await ws.send_json(_jsonable(w.result()))
                if dirty.is_set():
                    await asyncio.sleep(min_interval_s)   # coalesce bursts of updates
                    dirty.clear()
                    seq += 1
                    await ws.send_json(field_snapshot(seq))

        async def receiver() -> None:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "snapshot":
                    dirty.set()

        tasks = [asyncio.ensure_future(sender()), asyncio.ensure_future(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            unsubscribe()
            ctx["field_listeners"].discard(on_message)

    return app


def _check_zones(twin: DigitalTwin, interventions: List[Intervention]) -> None:
    for i in interventions:
        if i.zone_id not in twin.zone_names:
            raise HTTPException(400, f"Unknown zone_id {i.zone_id}. Valid: {sorted(twin.zone_names)}")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host=settings.twin_api_host, port=settings.twin_api_port, log_level="info")
