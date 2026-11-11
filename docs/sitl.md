# Flying the twin's missions on ArduPilot SITL

The twin plans fleet missions (`src/twin_core/swarm.py`) and executes them over MAVLink (`src/twin_core/mavlink_mission.py`). The ground-station code is the same for three targets:

| `UAV_TWIN_MAVLINK_MODE` | Vehicles | Needs |
|---|---|---|
| `emulated` (default) | In-process MAVLink emulators on the repo's 6-DOF flight physics (`sim_vehicle.py` in `src/twin_core`) | nothing |
| `sitl` | ArduCopter SITL, one instance per drone | an ArduPilot build |
| (any) + `UAV_TWIN_MAVLINK_CONNECTIONS` | Whatever the connection strings point at, e.g. `udp:…` to real aircraft | your own safety process |

The emulator is not ArduPilot. It implements the parts of the ArduCopter MAVLink behaviour the twin uses: mission upload, GUIDED/AUTO/RTL/LAND, arming, takeoff, `DO_SET_SERVO`, `DO_CHANGE_SPEED`, and the telemetry. That makes the pipeline testable anywhere. Validate against real SITL before trusting results.

## 1. Build ArduCopter SITL (once)

Follow ArduPilot's *Setting up the Build Environment* guide for your OS, then:

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
```

```bash
cd ardupilot && ./waf configure --board sitl && ./waf copter
```

```bash
export PATH=$PATH:$(pwd)/Tools/autotest
```

## 2. Start the twin in SITL mode

```bash
UAV_TWIN_MAVLINK_MODE=sitl python digital_twin_api.py
```

## 3. Launch one SITL instance per drone

```bash
python scripts/sitl_fleet.py --launch
```

This asks the API for each drone's launch pad (`GET /swarm/sitl-homes`) and runs, for drone *i*:

```bash
sim_vehicle.py -v ArduCopter -I <i> --sysid <i+1> --custom-location=<lat>,<lon>,0,0 --no-mavproxy
```

Instance *i* listens on `tcp:127.0.0.1:5760 + 10*i`, which is where the twin connects. Wait about a minute for GPS/EKF; the twin also retries arming for up to 2 minutes.

## 4. Fly

```bash
curl -X POST localhost:8008/sim/reset -H 'content-type: application/json' -d '{"seed": 7}'
```

```bash
curl -X POST localhost:8008/sim/advance -H 'content-type: application/json' -d '{"days": 3}'
```

```bash
curl -X POST localhost:8008/sim/mission -H 'content-type: application/json' -d '{"kind": "auto", "backend": "mavlink"}'
```

Each drone is set to GUIDED, armed, takes off, switches to AUTO, flies its mission, then RTLs, lands and disarms. During scouting, positions from `GLOBAL_POSITION_INT` trigger simulated camera frames, which are assimilated into the twin. Completed spray zones become interventions on the next `/sim/advance`. The flight tracks stream to UE5 over `/ws/field`.

To fly the same plans by hand in Mission Planner or QGroundControl, export them:

```bash
curl -X POST localhost:8008/swarm/mission-files -H 'content-type: application/json' -d '{"kind": "spray"}'
```

The result has one QGC WPL 110 `.waypoints` text per drone.

## Mission format and spraying

- Item 0 is home, then `NAV_TAKEOFF`, waypoints, and `NAV_RETURN_TO_LAUNCH` at the end. Each drone transits at its own altitude (14 m + 2 m per drone).
- **Pump:** `DO_SET_SERVO` on **servo 9** (1900 µs on, 1100 µs off), switched on along each spray lane and off on the headland turns. In ArduPilot, `DO_SET_SERVO` only drives an output whose `SERVO9_FUNCTION` is 0 or 1, which is the SITL default.
- **Variable-rate dose:** with a fixed-flow pump, the dose is set by ground speed (`DO_CHANGE_SPEED`), using v = L / (D/F − n·t_v):
  - L: total lane length
  - D: planned dose
  - F: pump flow, 0.12 kg/s
  - n: number of lanes
  - t_v: settling time per lane end, `VERTEX_DWELL_S`

  The mission report gives `planned_kg` next to `applied_kg`, which is measured from telemetry as pump-on time × flow. On the emulator they agree within about 10%. Re-fit `VERTEX_DWELL_S` from your SITL runs.
- **ArduPilot's own sprayer:** if you use AC_Sprayer instead (`SPRAY_ENABLE=1`, `SERVO9_FUNCTION=22`), its flow already scales with ground speed, so the dose per area is set by `SPRAY_PUMP_RATE` rather than by speed. Per-zone dose changes would then need `PARAM_SET` between zones, which the twin does not do yet.
