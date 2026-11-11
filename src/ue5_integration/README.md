# UE5 integration: TwinStream plugin

`TwinStream` connects a UE5 level to the digital twin API (`digital_twin_api.py`) over the `/ws/field` WebSocket. It replaces polling `/field/status` every 500 ms. The server pushes a snapshot whenever the twin changes, and a flight track after every drone mission.

> **Status:** written against long-stable UE 5.x APIs (WebSockets, Json, instanced static mesh custom data), but **not yet compiled in a UE5 project**. The server side and the message schema are covered by tests (`tests/test_twin_core.py::test_ws_field_*`). If your engine version reports a compile error, it's most likely a small API-signature difference in a `TryGet…Field` call.

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

## 5. Message schema (`/ws/field`)

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

**Client → server:** `{"type": "snapshot"}` requests a snapshot immediately.

To check the stream without UE5:

```bash
python scripts/twin_stream_client.py --heatmap
```

## 6. Driving the twin from UE5

The component is read-only. To advance or act on the twin from UE5 (UI buttons, a sequencer), call the REST API with the HTTP module or the VaRest plugin:

| Action | Call |
|---|---|
| Next day | `POST /sim/advance {"days": 1}` |
| Scout, then spray flagged zones | `POST /sim/mission {"kind": "auto"}` |
| Reset to a new hidden field | `POST /sim/reset {"seed": 7}` |

Every call that changes the twin is pushed back to UE5 as a new snapshot.
