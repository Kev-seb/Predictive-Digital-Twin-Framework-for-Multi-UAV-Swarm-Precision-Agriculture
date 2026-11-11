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

#### Validation result with the original generic crop model (superseded; see Phase 5)

Setup: a hidden outbreak and a dry patch, a drone RGB pass every 3 days (10 m altitude, 200 frames), and a multispectral survey every 7 days with 90% coverage. Values are RMSE against the hidden truth on day 21:

| Layer | Model-only twin | Assimilating twin | 95% interval coverage |
|---|---|---|---|
| Stress | 0.083 | **0.017** | 0.97 |
| NDVI | 0.117 | **0.021** | 0.99 |
| Fungus | 0.075 | **0.030** | 0.98 |
| Nitrogen | 0.028 | **0.006** | 1.00 |
| Moisture | 0.003 | 0.001 | 1.00 (uninformative: see Phase 5) |

Coverage near 1.0 means the twin's uncertainty is honest, if slightly conservative. Process noise can be tuned down in Phase 5.

### Phase 3: Swarm as first-class assets ✅ (done)
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
- [x] Dashboard migrated off the hardcoded `drone_alpha` / `drone_beta` (`src/digital_twin/swarm_guidance.py`).
  - The fleet size is set with `UAV_DASHBOARD_SWARM_SIZE` (default 2) and can be changed live from the Swarm tab, up to 12 drones.
  - Every mission pattern is written for drone *i* of *n* and reproduces the original two-drone paths exactly at *n* = 2 (tested).
  - The physics loop, `/swarm`, the three.js followers, fleet commands, the 3D plot and the telemetry deck all scale with the fleet.
  - Alpha stays the primary drone, linked to MAVLink telemetry and waypoint missions.
  - The two-operator multiplayer roles (`operator_alpha` / `operator_beta`) are unchanged.

### Phase 4: Close the loop ✅ (done: simulation, emulator and ArduCopter SITL)
- [x] Closed loop in simulation: scout → assimilate → spray the zones the twin flags → completed sprays become interventions on the next simulated day, for both the hidden field and the twin (`/sim/mission` + `/sim/advance`).
- [x] Execute the same plans over MAVLink (`src/twin_core/mavlink_mission.py`; guide in `docs/sitl.md`):
  - **Mission compiler:** takeoff, waypoints, and the sprayer pump on servo 9, on along lanes and off on headland turns. The pump PWM per zone sets a variable-rate dose from a fitted lane-time model; the report gives `applied_kg` measured from telemetry next to `planned_kg`.
  - **Ground-station link:** mission upload protocol with boot-time retries, telemetry stream requests, GUIDED → arm → takeoff → AUTO → RTL, land and disarm. Camera captures come from `GLOBAL_POSITION_INT`.
  - **In-repo MAVLink vehicle emulator** on the 6-DOF physics, for tests anywhere.
  - **API:** `/sim/mission` with `"backend": "mavlink"`, `/swarm/sitl-homes`, and `/swarm/mission-files` (`.waypoints`). `scripts/sitl_fleet.py` launches one ArduCopter SITL per drone; `scripts/calibrate_spray_timing.py` fits pump timing.
- [x] **Verified on ArduCopter SITL** (built from ArduPilot `master`), 3 instances driven over HTTP:
  - the scout fed 166 frames into the twin, and fungus RMSE fell from 0.076 to 0.044
  - the twin-triggered spray applied 2.75 kg against 2.72 kg planned
  - the next day's assimilated state reached fungus RMSE 0.023
- Real-ArduPilot issues found and fixed:
  - telemetry must be requested
  - `NO_SPACE` and `INVALID_SEQUENCE` uploads right after boot
  - jerk-limited lanes defeat dosing by speed, so dosing is by pump rate instead: overdose went from +21–61% to +1–3%
- [ ] Compare the outcome after the mission against the `what_if` prediction.

Results are reported under Phase 5, after model calibration.

### Phase 5: Calibrate and validate ✅ (done in simulation; real-data steps ready to run)

The generic `AgronomicDigitalTwinSimulator` was the limiting factor:
- It treated moisture > 0.85 as waterlogging damage, which is wrong for a flooded paddy.
- It dried the field by 0.03 a day and saturated it after any rain.
- Its outbreaks decayed, and NDVI fell during the vegetative stage.

It's replaced as the twin's default by a rice-specific process model. The legacy model is still available as `crop_model="legacy"`.

- [x] **`src/twin_core/paddy_model.py`**, with every coefficient in `PaddyParams` (JSON-serialisable):
  - **Water balance:** ponded depth and root-zone saturation per cell; crop ET = FAO-56 rice Kc × ET0; cell-varying percolation; 100 mm bund overflow; canal irrigation every 5 days. Drought stress starts only once standing water is gone.
  - **Phenology:** thermal time (GDD, base 10 °C); NDVI potential rises to a booting peak and declines through senescence.
  - **Nitrogen:** uptake, leaching and top-dressing.
  - **Disease:** logistic growth driven by temperature and leaf wetness, capped at a 0.6 maximum severity, with adult-plant resistance, spore diffusion and wind advection, and fungicide kill plus ~5-day half-life protection.
  - **Stress:** NDVI shortfall relative to the stage's potential, plus fungal load (`0.65·deficit + 0.35·fungus`).
- [x] **Phenology fitted to `flight_manifest.csv`** (`scripts/calibrate_paddy.py`): all 20 labelled flights over both fields are classified correctly. Stage onsets: Vegetative 252, Booting 1071, Flowering 1449, Mature 1854 GDD after transplanting.
- [x] **Seasonal behaviour** (tested):
  - A managed season peaks at NDVI ~0.85 at booting and ends at ~0.45.
  - An unsprayed monsoon epidemic plateaus at 30–50% severity.
  - Without irrigation, the field drains in ~1 week and then droughts.
  - Disease does not grow in dry weather.
- [x] **Twin uncertainty calibrated:**
  - Process and prior noise are tuned so normalised squared errors average ~1; the 95% intervals now contain the truth 95–97% of the time, up from 99–100%.
  - Adaptive covariance inflation (innovations > 3σ) lets the twin absorb surprises such as a hidden outbreak, which one scout pass now detects: 0.28 vs a true 0.34.
  - Observation noise is matched to the sensor's real error.
- [x] **Experiment fixes:**
  - The virtual field uses separate weather and sensor random streams, and flight turbulence is seeded, so paired runs see identical weather.
  - Spraying triggers on infected *area* (≥10% of a zone above the action threshold) instead of the zone mean.
- [ ] **On your machine with the raw imagery:**
  1. Run `python scripts/extract_flight_ndvi.py --raw <paddy_dataset>`, which writes `data/manifests/flight_ndvi.csv`.
  2. Run `python scripts/calibrate_paddy.py` to refit phenology with Open-Meteo archive temperatures and fit the NDVI curve. This writes `outputs/calibration/paddy_params.json`, which the twin API loads automatically.
- [ ] **Re-tune sensor noise on real data:** estimate `SENSOR_NOISE` from ground-truth plots, then process noise from real innovation statistics. The current values are only self-consistent inside the simulation.

#### Results with the calibrated model

**Tracking** (`python scripts/run_twin_osse.py --days 21`). RMSE against the hidden truth on day 21:

| Layer | Model-only twin | Assimilating twin | 95% coverage |
|---|---|---|---|
| Stress | 0.062 | **0.017** | 0.94 |
| NDVI | 0.045 | **0.014** | 0.97 |
| Fungus | 0.080 | **0.027** | 0.92 |
| Moisture | 0.032 | **0.019** | 1.00 |
| Nitrogen | 0.033 | **0.026** | 0.98 |

Moisture is now informative: the hidden leaky patch drains between canal turns.

**Closed loop** (`python scripts/run_twin_osse.py --closed-loop --days 21 --fleet 3`). Paired runs with identical weather:

| Day 21, hidden field | Scout only | Twin-driven spraying |
|---|---|---|
| Mean fungal load | 0.072 | **0.050** (−31%) |
| Peak fungal load | 0.375 | **0.190** (−49%) |
| Mean NDVI | 0.788 | **0.799** |
| Fungicide | 0 | 3.7 kg, 2 zones, day 3 |

### Phase 6: Visualization (in progress)
- [x] `/ws/field`: a UE5-oriented push stream.
  - It sends coalesced 20 × 20 snapshots (≤ 4 per second) as flat arrays: NDVI, stress, stress std, moisture, fungus, water depth, zone, today's anomalies and pending sprays.
  - It sends drone mission tracks in grid coordinates; clients that join late get the last track.
  - Covered by tests; `scripts/twin_stream_client.py` is a reference client.
- [x] `TwinStream` UE5 plugin (`src/ue5_integration/`): `UTwinStreamComponent` auto-reconnects, parses messages into Blueprint structs and events, writes HISM per-instance custom data (slots 0–5), and provides `SampleTrack` / `GridToLocal` helpers for replaying drone tracks. **Not yet compiled in a UE5 project.**
- [ ] Build the plugin in the UE5 project, then wire `M_RiceCrop` to the custom-data slots and the drone actors to `OnSwarmTrack`.
- [ ] three.js view: read drone positions from `/swarm/state` and field layers from `/twin/layers/*`.

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
