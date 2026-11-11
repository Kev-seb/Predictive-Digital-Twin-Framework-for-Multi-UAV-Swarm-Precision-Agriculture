# Digital Twin Plan

Roadmap for turning the framework into a working predictive digital twin of a
multi-UAV paddy field operation.

## Scope decisions

| Decision | Choice |
|---|---|
| Physical side | **Simulation only**: ArduPilot SITL and the in-repo flight physics. No field hardware. |
| 3D front-end | **UE5** is the primary twin view. The three.js view in the Streamlit dashboard stays. |
| Crop | **Paddy rice** (Nursery → Tillering/Vegetative → Reproductive → Ripening). |

## Where the repo started

Most parts existed, but they weren't connected:

- **Three backends with separate state:**
  - `digital_twin_api.py` held an in-memory 20 × 20 grid with fixed defaults.
  - `mobile_api_server.py` read `FieldDigitalTwin`.
  - `dashboard.py` kept its own `shared_state` dict.
- **The twin forgot its state between runs.** `FieldDigitalTwin` kept a scalar EMA of stress, and the agronomic simulator was rebuilt on every scenario, so moisture, nitrogen and fungus were discarded.
- **No data assimilation.** Observations overwrote state instead of correcting it.
- **Uncalibrated models.** Simulator coefficients were hand-set, and the ConvLSTM fell back to synthetic history.
- **Fixed swarm.** Two drones, `drone_alpha` and `drone_beta`, were hardcoded.
- **Open loop.** Missions were generated but never executed or checked.

## Target architecture

```
 Simulated physical side            TWIN CORE (src/twin_core)                     Views
 ───────────────────────            ─────────────────────────                     ─────
 SITL UAV swarm (MAVLink) ──┐   ┌─ FieldState: georeferenced grid, mean + var  ┌─▶ UE5 (HISM 20x20, /ws/twin)
 Simulated drone camera ────┼──▶│  per layer, persisted, versioned by day      ├─▶ Streamlit + three.js
 Multispectral survey ──────┤   ├─ Drone registry (N UAVs)                     └─▶ Mobile PWA
 Open-Meteo weather ────────┘   ├─ Loop: PREDICT → ASSIMILATE → DECIDE → ACT
            ◀── missions ───────┴─ Event stream + fork() for what-if
```

## Phases

### Phase 0: Stabilize ✅ (done)
- [x] Dependencies install and the existing tests pass (17 → 35 with the new suite).
- [x] Ports, grid and field settings live in `src/config/config.py` (`UAV_TWIN_*`).

### Phase 1: Unified twin state ✅ (done)
- [x] `src/twin_core/field_state.py`: `FieldGrid` (lat/lon ↔ cell, zones) and `FieldState` (mean + variance per layer, resampling, save/load).
- [x] `src/twin_core/twin.py`: `DigitalTwin` keeps one `AgronomicDigitalTwinSimulator` alive, so state carries over between days. It also handles persistence, the event log, subscribers and `fork()`/`what_if()`.
- [x] `digital_twin_api.py` is rebuilt on the twin core:
  - The UE5 `/field/*` contract is unchanged, with uncertainty fields added.
  - New `/twin/*` routes and the `/ws/twin` stream.
- [ ] Point `mobile_api_server.py` (`/api/field/state`, `/api/field/scenario/state`) and the dashboard's twin tab at `DigitalTwin` instead of `FieldDigitalTwin`.

### Phase 2: Live sync and assimilation (in progress)
- [x] Per-cell Kalman filter (`src/twin_core/assimilation.py`): process noise per day, observation noise per sensor, and point observations with a distance-weighted footprint.
- [x] Stress is diagnostic, derived from NDVI and fungus. A stress reading is assimilated as the NDVI it implies, so the simulator doesn't discard it.
- [ ] Ingestion adapters:
  - GeoTIFF survey → indices → `observe_grid`
  - simulated drone camera / RGB AI result → `observe_point`
- [ ] Twin clock driven by Open-Meteo daily forecasts (`src/weather/openmeteo_client.py`).
- [ ] Upgrade to an ensemble Kalman filter over the simulator, so moisture and nitrogen get updated through their correlation with NDVI.

### Phase 3: Swarm as first-class assets
- [ ] `DroneRegistry` with N drones and one interface for `UAVFlightDynamicsSimulator` and SITL (pymavlink).
- [ ] Remove the `drone_alpha` / `drone_beta` hardcoding in `dashboard.py`.
- [ ] Zone-to-drone allocation that accounts for battery, spray payload and distance.

### Phase 4: Close the loop (SITL)
- [ ] Decision → mission → execution → verification:
  - treatment optimizer → QGC mission → MAVLink upload to ArduPilot SITL → fly
  - each flown cell becomes an observation; the treatment goes into the twin as an intervention
- [ ] Compare the outcome after the mission against the `what_if` prediction.

### Phase 5: Calibrate and validate
- [ ] Fit simulator coefficients to the flight time series in `data/manifests/flight_manifest.csv`. The evapotranspiration term is a known issue: it dries a flooded paddy from 0.65 to 0.38 moisture in 7 days without rain.
- [ ] Train the ConvLSTM on real flight sequences instead of synthetic history.
- [ ] Report forecast skill: RMSE of predicted vs. observed NDVI and stress at the next flight, plus calibration of the twin's uncertainty.

### Phase 6: Visualization
- [ ] UE5 subscribes to `/ws/twin` instead of polling; drive HISM material parameters from `ndvi`, `stress` and `stress_std`.
- [ ] three.js view reads drone positions from the drone registry and field layers from `/twin/layers/*`.

## Done when

Three SITL drones fly the field. The twin forecasts a stress hotspot 3 days ahead and dispatches sprayers to it. It then assimilates the flight after the mission, and UE5 shows the outcome next to the forecast error.

## Using the twin core today

```bash
python digital_twin_api.py            # http://127.0.0.1:8008/docs
```

```bash
# advance a humid week
curl -X POST localhost:8008/twin/step -H 'content-type: application/json' \
     -d '{"days": 7, "weather": {"temperature": 31, "humidity": 88}}'
```

```bash
# a drone camera saw a stressed patch
curl -X POST localhost:8008/twin/observe/point -H 'content-type: application/json' \
     -d '{"lat": 10.0001, "lon": 78.0001, "values": {"stress": 0.8}, "source": "drone_rgb"}'
```

```bash
# what does spraying zones 5 and 6 buy us over 5 days?
curl -X POST localhost:8008/twin/what-if -H 'content-type: application/json' \
     -d '{"days": 5, "interventions": [{"zone_id": 5, "type": "Fungicide Spray"},
                                       {"zone_id": 6, "type": "Fungicide Spray"}]}'
```
