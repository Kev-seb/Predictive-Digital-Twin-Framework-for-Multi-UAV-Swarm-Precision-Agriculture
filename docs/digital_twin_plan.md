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

### Phase 2: Live sync and assimilation ✅ (done, except the EnKF upgrade)
- [x] Per-cell Kalman filter (`src/twin_core/assimilation.py`): process noise per day, observation noise per sensor, and point observations with a distance-weighted footprint.
- [x] Stress is diagnostic, derived from NDVI and fungus. A stress reading is assimilated as the NDVI it implies, so the simulator doesn't discard it.
- [x] Ingestion adapters (`src/twin_core/ingest.py`):
  - RGB AI result + telemetry → point observation. Stress comes from `crop_stress_score`; fungus from disease-box coverage. The footprint is sized by altitude, and the variance by model confidence and image quality.
  - 4-band reflectance → NDVI / NDWI / CIre → ndvi, moisture, nitrogen.
- [x] Weather forcing (`src/twin_core/weather_feed.py`): Open-Meteo daily forecast, or a seeded monsoon generator offline. Selected per step with `weather_source`.
- [x] Virtual physical field (`src/twin_core/virtual_field.py`):
  - hidden ground truth with a fungal outbreak and a dry patch, driven by the *actual* weather (forecast plus error)
  - simulated drone camera and multispectral sensors
  - `twin_error()` scoring
- [x] API: `/twin/ingest/rgb`, `/twin/ingest/multispectral`, and `/sim/reset|advance|drone-pass|survey|error|truth/layers`.
- [x] Live pipeline: `LiveFieldController.set_twin_sink()` + `TwinForwarder` (`src/twin_core/twin_client.py`) forward every accepted, geotagged inference from the mobile API to the twin API. It's non-blocking and can be switched off with `UAV_TWIN_FORWARD_LIVE=false`.
- [ ] Upgrade to an ensemble Kalman filter over the simulator, so moisture and nitrogen get updated through their correlation with NDVI.
- [ ] Point `mobile_api_server.py` `/api/field/state` at the twin API instead of `FieldDigitalTwin` survey files.

#### Validation result (`python scripts/run_twin_osse.py --days 21`)

Setup: a hidden outbreak and a dry patch, a drone RGB pass every 3 days (10 m altitude, 200 frames), and a multispectral survey every 7 days with 90% coverage. Values are RMSE against the hidden truth on day 21:

| Layer | Model-only twin | Assimilating twin | 95% interval coverage |
|---|---|---|---|
| Stress | 0.083 | **0.017** | 0.97 |
| NDVI | 0.117 | **0.021** | 0.99 |
| Fungus | 0.075 | **0.030** | 0.98 |
| Nitrogen | 0.028 | **0.006** | 1.00 |
| Moisture | 0.003 | 0.001 | 1.00 (uninformative: see Phase 5) |

Coverage near 1.0 means the twin's uncertainty is honest, if slightly conservative. Process noise can be tuned down in Phase 5.

### Phase 3: Swarm as first-class assets ✅ (done, except the dashboard migration)
- [x] `Fleet` (`src/twin_core/swarm.py`): any number of `UAVFlightDynamicsSimulator` drones sharing one `shared_state`, so the existing potential-field collision avoidance works fleet-wide. Includes add/remove, launch pads, battery swap and tank refill.
- [x] Task generation from the twin:
  - **scout:** one lawn-mower task per zone; the most uncertain zones go first.
  - **spray:** zones above a fungal-load threshold; the dose scales with load.
- [x] Greedy min-makespan allocation that respects a 35% battery reserve and tank capacity. Infeasible tasks are returned as `unassigned`. Planned times match flown physics within ~5–10%.
- [x] Execution in 6-DOF physics at 20 Hz:
  - transit altitudes are layered per drone
  - camera captures every 5 m
  - spray flow is duty-cycled to apply the planned dose
  - the battery failsafe aborts the mission and flies the drone home
- [x] API: `/swarm/state`, `/swarm/drones`, `/swarm/plan`, `/sim/mission`.
- [ ] Remove the `drone_alpha` / `drone_beta` hardcoding in `dashboard.py`. The fleet can feed it: `/swarm/state` returns the same shape as the dashboard's `/swarm` endpoint.

### Phase 4: Close the loop (in progress)
- [x] Closed loop in simulation: scout → assimilate → spray the zones the twin flags → completed sprays become interventions on the next simulated day, for both the hidden field and the twin (`/sim/mission` + `/sim/advance`).
- [ ] Execute the same plans on ArduPilot SITL: QGC mission → MAVLink upload → fly, with telemetry replacing the in-process physics.
- [ ] Compare the outcome after the mission against the `what_if` prediction.

#### Closed-loop result (`python scripts/run_twin_osse.py --closed-loop --days 21 --fleet 3`)

A 3-drone fleet scouts every 3 days. The twin-driven run also sprays zones whose fungal load is above 0.2. It is compared with an identical run that only scouts.

| Day 21, hidden field | Scout only | Twin-driven spraying |
|---|---|---|
| Peak fungal load | 0.486 | **0.308** (−37%) |
| Mean fungal load | 0.081 | **0.073** |
| Mean NDVI | 0.586 | **0.592** |
| Fungicide used | 0 | 2.9 kg (2 zone sprays) |

Caveats, both for Phase 5:
- The outbreak shrinks on its own (0.61 → 0.49), so the disease model under-spreads and the value of spraying is understated.
- NDVI declines in both runs, which is unrealistic for a vegetative paddy.

### Phase 5: Calibrate and validate
- [ ] Fit simulator coefficients to the flight time series in `data/manifests/flight_manifest.csv`. Two water-balance problems are known:
  - Evapotranspiration dries a flooded paddy from 0.65 to 0.38 moisture in 7 days without rain.
  - Rain adds 0.06 per mm, so one monsoon spell saturates moisture to 1.0 field-wide. Moisture then carries no information (both twins score ~0 error).
  - A bunded paddy water-depth model (ponding, percolation, drainage) would fix both.
- [ ] Tune process noise per layer so the twin's 95% intervals contain the truth ~95% of the time, not 97–100%.
- [ ] Disease dynamics: the PDE outbreak decays under humid monsoon weather. Check the growth rate against blast/sheath-blight literature.
- [ ] Crop growth: NDVI falls ~0.11 in 21 vegetative days without stress. The nitrogen uptake and growth terms need fitting.
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
