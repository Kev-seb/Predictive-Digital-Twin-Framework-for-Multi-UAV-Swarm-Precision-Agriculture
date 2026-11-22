# UE5 integration

The `TwinStream` C++ plugin (sections 1–7) connects a UE5 level to the digital twin API (`digital_twin_api.py`) over the `/ws/field` WebSocket. It replaces polling `/field/status` every 500 ms. The server pushes a snapshot whenever the twin changes, and a flight track after every drone mission.

> **Status:** written against long-stable UE 5.x APIs (WebSockets, Json, instanced static mesh custom data), but **not yet compiled in a UE5 project**. The server side and the message schema are covered by tests (`tests/test_twin_core.py::test_ws_field_*`). If your engine version reports a compile error, it's most likely a small API-signature difference in a `TryGet…Field` call.

## Quick start: no C++ (recommended)

`python/twin_live.py` runs inside the UE5 editor using UE5's built-in Python. There's nothing to compile and nothing to buy.

1. **Enable Python in UE5 (once):** *Edit → Plugins*, search for **Python Editor Script Plugin**, tick it, and restart the editor.
2. **Start the twin** on the same PC. On Windows, double-click **`start_demo.bat`** in the repository root. It starts the twin, waits until it's ready, and prints the next step. To have it open UE5 with the script already running, open `start_demo.bat` in a text editor once and set `UE_PROJECT` to your `.uproject` path. Then step 3 happens by itself.

   From a terminal, the same thing is:

   ```bash
   python scripts/start_demo.py
   ```

   Options: `--ue-project path/to/Game.uproject` opens UE5 with the script running, `--demo` starts the story at once, and `--fresh` resets the field to day 0. To start the twin on its own, run `python digital_twin_api.py`.

3. **Run the script in UE5:** *File → Execute Python Script…* and pick `src/ue5_integration/python/twin_live.py`.
   - **Empty level:** the script builds three paddy fields side by side, each 100 m square, on farmland.
     - Each field is 20 × 20 patches of rice, planted in rows and standing in water inside mud bunds.
     - Plant height follows NDVI, from 30 cm seedlings to a 1 m canopy.
     - Leaf colour follows health: green when healthy, yellow when stressed, red-brown when diseased, grey where the twin is unsure.
     - The water is glassy while the field is flooded and turns to brown mud as the soil dries.

     On the first run, the script generates the rice mesh and the leaf and water materials into `/Game/TwinDemo`; this takes about a minute. Later runs reuse them. The leaves sway in the wind.

     Set `FIELD_LOOK = "blocks"` for the old coloured cubes. Those also appear automatically if the assets can't be built, and the Output Log says why. Your own `CROP_MESH` takes priority over the generated rice.
     - **TWIN:** what the twin believes now.
     - **REALITY:** the hidden truth.
     - **FORECAST +7 DAYS:** what the twin predicts a week ahead if nothing is done. It's faded where the prediction is uncertain. This is the "predictive" part of a predictive digital twin.

     A **results panel** shows what the twin is worth. For example:

     ```text
     DAY 3  |  Vegetative
     Expected harvest:  5.51 t/ha
       in 7 days, no spray:  5.37 t/ha
       in 7 days, sprayed:   5.41 t/ha  (zones 6)
     Worst disease in 7 days:  36%  ->  16% if sprayed
     Fungicide:  3.0 kg  vs  25 kg whole-field  (88% saved)
     Twin vs reality error:  0.076  ->  0.031
     Missions:  1 scout  |  1 spray  |  9.3 min flown
     ```

     Expected harvest uses the repo's yield model (mature biomass × harvest index × stress penalty), fed with each cell's health relative to its growth stage. "Whole-field" is conventional blanket spraying of the field at 25 L/ha per application.

     The results panel hangs above the fields and the caption sits in front of them like a subtitle. All text stands on dark boards so it stays readable against a bright sky. Set `SHOW_BOARDS = False` to remove the boards. In the twin, an outbreak shows red even while the twin is still unsure of its edges; uncertainty only greys the healthy colour.

     Set `SHOW_FORECAST = False` or `SHOW_PANEL = False` at the top of the script to hide the forecast field or the panel. In a level with your own field, an actor labelled "Forecast" gets the prediction.
   - **Level with your rice field** (a 400-instance HISM): the script drives your field instead, through per-instance custom data slots 0–5, which you read in `M_RiceCrop` with *PerInstanceCustomData* nodes. Label a second field actor with "Reality" to get the hidden truth on it.
4. **Start the story:**

   ```bash
   curl -X POST localhost:8008/demo/start -H 'content-type: application/json' -d '{}'
   ```

   Quadcopters (hub, X arms, spinning rotors, spray tank) take off, scout, and spray, with a blue cone under them while they spray. The twin field changes under them as they fly. A caption above the field narrates each step and shows the twin-vs-reality error.
5. **Controls:** **Digital Twin buttons on the toolbar above the viewport:** Demo, Stop, Next day, Scout, Spray, Outbreak, Reset, then the camera views Overview, Close-up and Follow drone. Hover a button to see what it does. The same actions are in the **Digital Twin** menu on the editor's menu bar, so you can run the show live without a browser or terminal.

   | Menu entry | What happens |
   |---|---|
   | Start demo (full story) / Stop | runs or stops the paced story |
   | Next day | the crop grows and the disease spreads one day |
   | Scout now / Spray now / Scout + spray | the drones fly that mission, animated |
   | **Inject outbreak** | plants a new disease patch in the hidden **Reality** field, at the field block you've selected, or else where the centre of your view meets a field. The twin isn't told, so "Scout now" shows it finding the patch. |
   | Reset field | starts again with a new hidden outbreak |
   | Overview camera | all three fields |
   | Close-up of the crop | low over the twin's rice, to see the plants and water |
   | Follow a drone | the camera chases a drone (the one spraying, if any) while it flies; Overview stops it |

   Only one step runs at a time. A second click while drones are flying logs "busy" in the Output Log.

6. **Camera:** on start, the script moves the editor camera to a raised view over both fields, with TWIN on the left and REALITY on the right. The caption and labels always turn to face you. If you fly away, type `twin_live_camera()` in the Output Log's Python console to come back.
7. **Stop or restart:** run the script again, or type `twin_live_stop()` in the same console.

Settings are at the top of the script: API address, field mode, and where to build the blocks.

### Realistic look with your own assets

The script runs in any project with engine shapes only. To use better-looking assets, put their paths in these settings at the top of `twin_live.py`. To get a path, right-click the asset in the Content Browser, choose **Copy Reference**, and keep the part between the quotes, for example `/Game/Drones/SM_AgriDrone.SM_AgriDrone`.

| Setting | What it replaces | Notes |
|---|---|---|
| `DRONE_MESH` | the engine-shape quadcopter | Static mesh, nose along +X, about 1 m across. Use `DRONE_MESH_SCALE` to resize it. If its material has a `Color` parameter, each drone gets its own colour. |
| `SPRAY_NIAGARA` | the blue spray cone | A Niagara system that emits downward. It runs only while the drone sprays. |
| `RAIN_NIAGARA` | the falling blue streaks | A rain Niagara system, placed above the fields. |
| `CROP_MESH` | the cubes (blocks mode) | A rice clump with its pivot at the base, about 1 m wide. It grows taller with NDVI. Health colour goes to the material's `CROP_COLOR_PARAM` vector parameter (default `Color`). |

Rain shows on days when the real field gets more than `RAIN_MM` (1 mm) of rain. `/ue5/frame` reports that day's weather in `weather`. Set `SHOW_RAIN = False` to turn rain off. If a path is wrong, the Output Log warns once and the built-in look is used.

Where to get them: a Megascans grass or plant clump from Fab (inside UE5) for the crop, and any quadcopter mesh from Fab for the drone. For spray, make a Niagara system from the **Fountain** template and point it down. For rain, make a sprite emitter that spawns in a wide box and falls fast.

This path is tested against the real API with a stand-in for UE5's Python module (`tests/test_ue5_python.py`), but it hasn't been run inside UE5 yet. If a call doesn't exist in your engine version, the Output Log names it. Send me that line.

**Want the C++ plugin instead?** Your project can compile C++ if its folder has a `Source/` folder and a `.sln` file next to the `.uproject`, and Visual Studio with the *Game development with C++* workload is installed. If not, use the Python path above; it does the same job.

## 1. Install

1. Copy `src/ue5_integration/TwinStream/` to `<YourProject>/Plugins/TwinStream/`.
2. Right-click the `.uproject` → *Generate Visual Studio project files*, then build. The project needs to be a C++ project; *Tools → New C++ Class* converts a Blueprint-only one.
3. Check that **Twin Stream** is enabled under *Edit → Plugins*.

## 2. Wire it to the field

1. Start the twin:

   ```bash
   python digital_twin_api.py
   ```

   It listens on `127.0.0.1:8008`; `UAV_TWIN_API_HOST` and `UAV_TWIN_API_PORT` change that.
2. On the actor that owns the field's HISM, e.g. `BP_FieldDataController`:
   - add a **Twin Stream** component
   - in *BeginPlay*, set `TargetInstances` to the rice HISM component
3. Order the HISM instances so that **instance index = row × 20 + col**. Row 0 is the north edge and col 0 the west edge, the same convention as `/field/status`. If your instances were created in another order, leave `TargetInstances` empty. Instead, bind `OnFieldSnapshot` and apply the values with your own index mapping.
4. Press Play. The output log shows `LogTwinStream: Connected to ws://…`, and the component reconnects automatically if the API restarts.

## 3. Material (`M_RiceCrop`)

For every snapshot, the component writes six per-instance custom data floats. Read them with **PerInstanceCustomData** nodes:

| Index | Value | Suggested use |
|---|---|---|
| 0 | NDVI 0–1 | base green ↔ yellow lerp |
| 1 | Stress 0–1 | tint toward brown |
| 2 | Stress std (twin uncertainty) | desaturate or add a hatch pattern where the twin is unsure |
| 3 | Moisture 0–1 | water-surface mask: ~0.9 ponded, 0.6 drained, < 0.6 drying |
| 4 | Fungus 0–1 | lesion texture mask |
| 5 | Anomaly 0/1 | emissive pulse on flagged cells |

Showing uncertainty (index 2) is what makes this a twin rather than a heat map: viewers can see which parts of the field the system actually knows.

## 4. Drones

`OnSwarmTrack(MissionKind, SimDay, Tracks)` fires after every `/sim/mission` (scout or spray). Each `FTwinDroneTrack` holds 1 Hz samples with `TimeS`, `Row`, `Col` (continuous grid coordinates), `AltM` and `bSpraying`. To replay a mission:

1. Spawn or reuse one drone actor per `DroneId`.
2. Drive a timeline or a tick from 0 to the last `TimeS`, scaled as you like (the missions are 1–3 simulated minutes).
3. Each frame, call `SampleTrack(Track, T)`, then `GridToLocal(Row, Col, AltM, Snapshot.CellSizeM)`. Add the field actor's location, set the drone's location, and toggle a spray Niagara system from `bSpraying`.

`GridToLocal` returns centimetres with X = east (col), Y = south (row) and Z = up. Place the field mesh so the grid's north-west corner sits at the actor origin. The centre of instance (r, c) is `GridToLocal(r + 0.5, c + 0.5, 0, CellSizeM)`.

## 5. Demo for presentations

Demo mode runs the whole digital-twin story automatically, paced for an audience. With the default settings (15 days, 3 s per day, drones replayed at 6×), it takes about 6–8 minutes.

1. **Day 0:** the twin starts from its prior, and a fungal outbreak is hidden in the "real" field.
2. **Days 1–3:** the crop grows and the disease spreads unseen.
3. **Day 3, scouting:** the drones take off and fly over the field. **The twin's field changes colour under the drones as they pass**, because each camera frame is assimilated live, and the outbreak appears.
4. **Decision:** the twin flags the infected zones.
5. **Spraying:** the drones fly back over those zones with their spray on.
6. **Following days:** the fungicide takes effect, and later scouting confirms the recovery.

Start the API, then the demo:

```bash
python digital_twin_api.py
```

```bash
curl -X POST localhost:8008/demo/start -H 'content-type: application/json' -d '{}'
```

Optional settings: `{"days": 15, "day_seconds": 3, "scout_every": 3, "playback_speed": 6, "seed": 7}`. Add `"backend": "mavlink"` to fly it on ArduPilot SITL. `POST /demo/stop` stops it; `GET /demo/status` shows the current phase.

### Blueprint setup (one level)

1. **Twin field:** your rice HISM (20 × 20 instances, `M_RiceCrop`) on an actor with a **Twin Stream** component. In *BeginPlay*, set `TargetInstances` to the HISM.
2. **Reality field (optional, recommended):** duplicate the field actor a few metres to the side, then set the component's `TruthInstances` to that copy's HISM. Both fields use the same material slots. Audiences see the twin start "clean", then match reality once the drones have flown. That convergence is what makes it a digital twin rather than an animation.
3. **Drones:** bind **OnDronePositions(TimeS, Drones)**. For each `FTwinDronePosition`:
   - find or spawn a drone actor keyed by `DroneId` (keep a `Map<String, Actor>`)
   - `SetActorLocation(FieldOrigin + GridToLocal(Row, Col, AltM, Snapshot.CellSizeM))`
   - if `bHasHeading`, `SetActorRotation(Yaw = HeadingToYaw(HeadingDeg))`
   - activate or deactivate a spray Niagara system from `bSpraying`

   Updates arrive at about 10 Hz. Use `VInterpTo` or `RInterpTo` in Tick for smooth motion.
4. **Captions:** bind **OnStatus(Phase, Text, SimDay)** to a UMG text block. You can also switch camera shots on `Phase`:
   - `scout`: follow the drones
   - `decide`: top-down view of the field
   - `spray`: close-up
5. **Uncertainty:** in `M_RiceCrop`, use custom-data slot 2 (stress std) to desaturate or overlay a hatch pattern. On day 0 the whole twin field looks uncertain, and it sharpens where the drones have flown.

`GridToLocal` places row 0 / col 0 at the field actor's origin, which should be the north-west corner. The grid is 20 cells of 5 m (100 m × 100 m). The launch pads sit just south of the field, at row ≈ 21.

## 6. Message schema (`/ws/field`)

Messages are JSON text frames.

**`field.snapshot`** is sent on connect, then whenever the twin changes, at most once every 0.25 s:

```jsonc
{
  "type": "field.snapshot", "seq": 12, "field_id": "field_alpha",
  "sim_day": 4, "growth_stage": "Vegetative", "gdd": 743.3,
  "grid": {"rows": 20, "cols": 20, "cell_size_m": 5.0, "origin_lat": 10.00045, "origin_lon": 77.99954},
  "ndvi": [400 floats], "stress": [...], "stress_std": [...], "moisture": [...],
  "fungus": [...], "water_mm": [...], "zone": [400 ints],
  "anomalies": [instance indices flagged today], "pending_sprays": [zone ids]
}
```

**`swarm.track`** is sent after each mission. The most recent one is also sent to clients that connect later:

```jsonc
{"type": "swarm.track", "kind": "scout", "sim_day": 4,
 "fields": ["t_s", "row", "col", "alt_m", "spraying"],
 "drones": {"drone_01": [[0.0, 20.9, 10.0, 0.5, 0], ...], ...}}
```

**`swarm.positions`** is sent at about 10 Hz while a demo replays a mission:

```jsonc
{"type": "swarm.positions", "t_s": 42.5, "fields": ["row", "col", "alt_m", "spraying", "heading_deg"],
 "drones": {"drone_01": [6.12, 11.03, 10.0, 0, 90.0], ...}}
```

`heading_deg` is `null` while a drone hovers.

**`twin.status`** is sent on each demo step:

```jsonc
{"type": "twin.status", "phase": "scout", "text": "Day 3: drones scouting the field; ...",
 "sim_day": 3, "error": {"stress": 0.06, "fungus": 0.078, "ndvi": 0.04}}
```

`error` is the twin's RMSE against the hidden reality.

In simulation, **`field.snapshot`** also carries `truth_ndvi`, `truth_stress` and `truth_fungus`, each 400 floats.

Every snapshot also carries the twin's 7-day forecast, computed on a copy of the twin so the live twin is untouched:
- **`forecast_ndvi`, `forecast_stress`, `forecast_fungus`, `forecast_std`:** the predicted field if nothing is done.
- **`what_if`:** `{days, on_day, no_spray, if_sprayed, spray_zones}`. "No spray" assumes none of the fleet's sprays happen; "if sprayed" includes the sprays already flown plus the zones the twin flags now. Each scenario gives mean and peak fungus, mean NDVI and mean stress.

`/ue5/frame` and every snapshot also carry **`metrics`**, the numbers behind the results panel: expected harvest (now, and in 7 days with and without spraying), fungicide used against blanket spraying, twin-vs-reality error history, and missions and flight time.

**Client → server:** `{"type": "snapshot"}` requests a snapshot immediately.

To check the stream without UE5:

```bash
python scripts/twin_stream_client.py --heatmap
```

## 7. Driving the twin from UE5

The component is read-only. To advance or act on the twin from UE5 (UI buttons, a sequencer), call the REST API with the HTTP module or the VaRest plugin:

| Action | Call |
|---|---|
| Next day | `POST /sim/advance {"days": 1}` |
| Scout, then spray flagged zones | `POST /sim/mission {"kind": "auto"}` |
| Reset to a new hidden field | `POST /sim/reset {"seed": 7}` |

Every call that changes the twin is pushed back to UE5 as a new snapshot.
