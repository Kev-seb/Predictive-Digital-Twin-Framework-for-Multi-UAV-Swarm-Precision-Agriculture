"""
digital_twin_api.py
================================================================
FastAPI backend for the UE5 Paddy Field Digital Twin.
Runs on http://127.0.0.1:8000

Endpoints:
  GET  /                          Health check
  POST /field/update              Receive per-instance state from UE5
  GET  /field/status              Full current field state (all 400)
  GET  /field/instance/{index}    Single instance record
  GET  /field/summary             Aggregated field statistics
  GET  /field/anomalies           Instances with NDVI drop > threshold
  GET  /field/heatmap             NDVI grid as 2D list (20x20)
  DELETE /field/reset             Reset all state to defaults
  GET  /docs                      Auto-generated Swagger UI

HARD CONSTRAINTS:
  - This file is NEVER read or written by UE5 Python scripts.
    UE5 communicates only via HTTP.
  - Never touches Phase A lighting, BP_FieldDataController,
    HISM component, or M_RiceCrop material.
================================================================
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from contextlib import asynccontextmanager
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import __version__ as _fastapi_version
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
HOST      = "127.0.0.1"
PORT      = 8008
GRID_SIZE = 20
NUM_INSTANCES = GRID_SIZE * GRID_SIZE   # always 400

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


class InstanceRecord(BaseModel):
    index:          int
    row:            int
    col:            int
    ndvi:           float
    stress:         float
    soil_moisture:  float
    temperature_c:  float
    last_updated:   float   # Unix timestamp


# ----------------------------------------------------------------
# IN-MEMORY STATE  (survives for the lifetime of this process)
# ----------------------------------------------------------------
_field_state: Dict[int, InstanceRecord] = {}
_current_day: int = 0
_anomaly_log: List[dict] = []          # recent anomaly events
_history:     List[dict] = []          # one summary entry per simulated day


def _init_state():
    """Populate default state for 400 instances if empty."""
    if _field_state:
        return
    t = time.time()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            idx = row * GRID_SIZE + col
            _field_state[idx] = InstanceRecord(
                index=idx, row=row, col=col,
                ndvi=0.85, stress=0.0,
                soil_moisture=45.0, temperature_c=28.0,
                last_updated=t,
            )


@asynccontextmanager
async def lifespan(application: FastAPI):
    _init_state()
    sys.stdout.write(f"[DigitalTwinAPI] Listening on http://{HOST}:{PORT}\n")
    sys.stdout.write(f"[DigitalTwinAPI] Swagger UI -> http://{HOST}:{PORT}/docs\n")
    sys.stdout.write(f"[DigitalTwinAPI] {NUM_INSTANCES} field records initialized.\n")
    sys.stdout.flush()
    yield


# ----------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------
app = FastAPI(
    title="Paddy Field Digital Twin API",
    description=(
        "Real-time backend for the UE5 paddy field digital twin. "
        "UE5 pushes per-instance NDVI + Stress after each simulated day."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ----------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------

@app.get("/", summary="Health check")
async def health():
    return {
        "status":       "ok",
        "service":      "Paddy Field Digital Twin API",
        "version":      "1.0.0",
        "current_day":  _current_day,
        "instances":    len(_field_state),
    }


@app.post("/field/update", summary="Receive field state from UE5")
async def field_update(payload: FieldUpdatePayload):
    """
    Called by paddy_field_digital_twin.py after each simulated day.
    Accepts up to 400 instance records, updates in-memory state,
    and logs anomalies (stress > 0.7 or NDVI drop > 0.15).
    """
    global _current_day

    if len(payload.instances) > NUM_INSTANCES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many instances: {len(payload.instances)} > {NUM_INSTANCES}"
        )

    _current_day = payload.day
    t = time.time()
    updated_anomalies = []

    for inst in payload.instances:
        prev = _field_state.get(inst.index)

        # Anomaly: stress spike or rapid NDVI drop
        is_anomaly = False
        if prev:
            ndvi_drop = prev.ndvi - inst.ndvi
            if ndvi_drop > 0.15 or inst.stress > 0.70:
                is_anomaly = True
                updated_anomalies.append({
                    "day":        payload.day,
                    "index":      inst.index,
                    "row":        inst.row,
                    "col":        inst.col,
                    "ndvi":       inst.ndvi,
                    "stress":     inst.stress,
                    "ndvi_drop":  round(ndvi_drop, 4),
                    "timestamp":  t,
                })

        _field_state[inst.index] = InstanceRecord(
            index=inst.index, row=inst.row, col=inst.col,
            ndvi=inst.ndvi, stress=inst.stress,
            soil_moisture=inst.soil_moisture,
            temperature_c=inst.temperature_c,
            last_updated=t,
        )

    # Append new anomalies (keep last 200)
    _anomaly_log.extend(updated_anomalies)
    if len(_anomaly_log) > 200:
        _anomaly_log[:] = _anomaly_log[-200:]

    # Daily summary snapshot
    all_records = list(_field_state.values())
    avg_ndvi   = sum(r.ndvi   for r in all_records) / len(all_records)
    avg_stress = sum(r.stress for r in all_records) / len(all_records)
    _history.append({
        "day":        payload.day,
        "avg_ndvi":   round(avg_ndvi,   4),
        "avg_stress": round(avg_stress, 4),
        "anomalies":  len(updated_anomalies),
        "timestamp":  t,
    })
    if len(_history) > 365:
        _history[:] = _history[-365:]

    return {
        "accepted":        len(payload.instances),
        "day":             payload.day,
        "new_anomalies":   len(updated_anomalies),
        "avg_ndvi":        round(avg_ndvi,   4),
        "avg_stress":      round(avg_stress, 4),
    }


@app.get("/field/status", summary="Full current state — all 400 instances")
async def field_status():
    return {
        "day":       _current_day,
        "instances": [r.dict() for r in _field_state.values()],
    }


@app.get("/field/instance/{index}", summary="Single instance record")
async def field_instance(index: int):
    record = _field_state.get(index)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Instance {index} not found.")
    return record


@app.get("/field/summary", summary="Aggregated field statistics")
async def field_summary():
    if not _field_state:
        raise HTTPException(status_code=503, detail="No field data yet.")

    records = list(_field_state.values())
    n = len(records)

    ndvi_vals   = [r.ndvi   for r in records]
    stress_vals = [r.stress for r in records]
    moist_vals  = [r.soil_moisture for r in records]
    temp_vals   = [r.temperature_c for r in records]

    stressed = sum(1 for s in stress_vals if s > 0.5)
    healthy  = sum(1 for s in stress_vals if s < 0.2)

    return {
        "day":             _current_day,
        "total_instances": n,
        "ndvi": {
            "mean": round(sum(ndvi_vals) / n, 4),
            "min":  round(min(ndvi_vals), 4),
            "max":  round(max(ndvi_vals), 4),
        },
        "stress": {
            "mean":           round(sum(stress_vals) / n, 4),
            "high_count":     stressed,          # stress > 0.5
            "healthy_count":  healthy,           # stress < 0.2
            "pct_stressed":   round(stressed / n * 100, 1),
        },
        "soil_moisture": {
            "mean": round(sum(moist_vals) / n, 4),
            "min":  round(min(moist_vals), 4),
            "max":  round(max(moist_vals), 4),
        },
        "temperature_c": {
            "mean": round(sum(temp_vals) / n, 4),
        },
        "history_days": len(_history),
        "history":      _history[-14:],   # last 14 days
    }


@app.get("/field/anomalies", summary="Recent anomaly events")
async def field_anomalies(limit: int = 50):
    return {
        "total":     len(_anomaly_log),
        "anomalies": _anomaly_log[-limit:],
    }


@app.get("/field/heatmap", summary="NDVI + Stress as 20x20 grid")
async def field_heatmap():
    """
    Returns NDVI and Stress as 2D 20x20 lists.
    Row 0 = grid row 0 (top), Column 0 = grid col 0 (left).
    Suitable for heatmap visualisation in a browser dashboard.
    """
    ndvi_grid   = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
    stress_grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    for r in _field_state.values():
        ndvi_grid[r.row][r.col]   = round(r.ndvi,   4)
        stress_grid[r.row][r.col] = round(r.stress, 4)

    return {
        "day":         _current_day,
        "grid_size":   GRID_SIZE,
        "ndvi_grid":   ndvi_grid,
        "stress_grid": stress_grid,
    }


@app.delete("/field/reset", summary="Reset all state to defaults")
async def field_reset():
    global _current_day
    _field_state.clear()
    _anomaly_log.clear()
    _history.clear()
    _current_day = 0
    _init_state()
    return {"status": "reset", "instances": NUM_INSTANCES}


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    _init_state()
    uvicorn.run(
        "digital_twin_api:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
