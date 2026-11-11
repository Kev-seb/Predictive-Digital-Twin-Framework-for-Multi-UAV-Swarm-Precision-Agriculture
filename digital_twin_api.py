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
================================================================
"""

from __future__ import annotations

import asyncio
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
from src.twin_core.ingest import ingest_multispectral, ingest_rgb_result
from src.twin_core.swarm import Fleet, allocate, tasks_from_twin
from src.twin_core import mavlink_mission as mm
from src.twin_core.virtual_field import VirtualField, serpentine_waypoints
from src.twin_core.weather_feed import get_weather

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


class SimAdvance(BaseModel):
    days:           int = Field(1, ge=1, le=60)
    weather_source: Literal["synthetic", "openmeteo", "auto"] = "synthetic"
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
                           "last_track": None, "mavlink": None}

    def get_twin() -> DigitalTwin:
        if ctx["twin"] is None:
            ctx["twin"] = _default_twin()
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
            }

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
        twin = get_twin()
        if req.reset_twin:
            twin.reset()
            ctx["anomalies"].clear()
            ctx["ue5_snapshot"] = ue5_grids()
        ctx["pending_interventions"] = []
        ctx["virtual"] = VirtualField(twin.state.grid, seed=req.seed, outbreaks=req.outbreaks,
                                      propagation_model=twin.propagation_model,
                                      zone_layout=twin.zone_layout, crop_model=twin.crop_model,
                                      params=twin.params)
        return {"status": "ok", "seed": req.seed, "error": ctx["virtual"].twin_error(twin)}

    @app.post("/sim/advance", summary="Advance the virtual field and the twin together")
    def sim_advance(req: SimAdvance):
        twin, vf = get_twin(), get_virtual()
        _check_zones(twin, req.interventions)
        weather, used = forecast(req.weather_source, req.days, vf.sim_day + 1)
        plan = [i.model_dump() for i in req.interventions]
        # Sprays flown by the fleet since the last advance take effect on day 1.
        flown = [{"day": 1, "zone_id": p["zone_id"], "type": p["type"]}
                 for p in ctx["pending_interventions"]]
        ctx["pending_interventions"] = []
        plan += flown
        actual = vf.advance(req.days, weather, plan)        # truth gets the real weather
        result = twin.step(days=req.days, weather=weather, interventions=plan)  # twin gets the forecast
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
                    capture_spacing_m=fleet.spec.capture_spacing_m)
                if report["errors"] and not report["drones"]:
                    raise HTTPException(502, f"MAVLink execution failed: {report['errors']}")
            else:
                report = fleet.execute(plan, on_capture=on_capture if kind == "scout" else None, wind=wind)
            ctx["pending_interventions"].extend(report["completed_sprays"])
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
                mission = mm.compile_mission(a, fleet.frame, transit_alt=14.0 + 2.0 * k, home=homes[d_id])
                out[d_id] = {"items": len(mission.items), "spray_speed_m_s": mission.spray_speed,
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
