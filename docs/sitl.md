# Flying the twin's missions on ArduPilot SITL

The twin plans fleet missions (`src/twin_core/swarm.py`) and executes them over MAVLink (`src/twin_core/mavlink_mission.py`). The ground-station code is the same for three targets:

| `UAV_TWIN_MAVLINK_MODE` | Vehicles | Needs |
|---|---|---|
| `emulated` (default) | In-process MAVLink emulators on the repo's 6-DOF flight physics (`sim_vehicle.py` in `src/twin_core`) | nothing |
| `sitl` | ArduCopter SITL, one instance per drone | an ArduPilot build |
| (any) + `UAV_TWIN_MAVLINK_CONNECTIONS` | Whatever the connection strings point at, e.g. `udp:…` to real aircraft | your own safety process |

The emulator is not ArduPilot. It implements the parts of the ArduCopter MAVLink behaviour the twin uses: mission upload, GUIDED/AUTO/RTL/LAND, arming, takeoff, `DO_SET_SERVO`, `DO_CHANGE_SPEED`, and the telemetry. That makes the pipeline testable anywhere.

## Verified on ArduCopter SITL

The pipeline below was run end to end against ArduCopter SITL built from ArduPilot `master` (stock `copter.parm`, `--speedup 5`), with 3 instances driven over HTTP:

| Step | Result |
|---|---|
| Scout (`/sim/mission {"kind": "auto", "backend": "mavlink"}`) | 3 drones flew 16 zone tasks, all landed and disarmed; 166 frames assimilated; twin fungus RMSE 0.076 → 0.044 |
| Spray | the twin flagged zone 6; pump PWM 1682; applied 2.75 kg vs 2.72 kg planned (+1.3%) |
| Next `/sim/advance` | spray applied to the field; fungus RMSE → 0.023 |

Real ArduPilot behaviour the emulator did not show, all handled now:
- telemetry has to be requested
- right after boot, ArduPilot answers mission uploads with `NO_SPACE` and, when retried, `INVALID_SEQUENCE`
- commanded spray speed is not reached on short lanes, which led to the pump-rate dosing below

## 1. Build ArduCopter SITL (once)

Follow ArduPilot's *Setting up the Build Environment* guide for your OS, then:

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/ardupilot
```

```bash
cd ~/ardupilot && ./waf configure --board sitl && ./waf copter
```

The build needs `empy==3.3.4` (`pip install empy==3.3.4 future pexpect`).

## 2. Start the twin in SITL mode

```bash
UAV_TWIN_MAVLINK_MODE=sitl python digital_twin_api.py
```

## 3. Launch one SITL instance per drone

```bash
python scripts/sitl_fleet.py --ardupilot ~/ardupilot --launch --speedup 5
```

This asks the API for each drone's launch pad (`GET /swarm/sitl-homes`) and runs the built SITL binary for each drone *i*, with its own working directory and log under `sitl_fleet/`:

```bash
build/sitl/bin/arducopter --model + --defaults Tools/autotest/default_params/copter.parm -I <i> --sysid <i+1> --home <lat>,<lon>,0,0 -w
```

The launcher runs the binary directly because `sim_vehicle.py --no-mavproxy` exits as soon as it has started the vehicle, and all its instances log to the same file. If you prefer MAVProxy windows, `/swarm/sitl-homes` also lists equivalent `sim_vehicle.py` commands.

Instance *i* listens on `tcp:127.0.0.1:5760 + 10*i`, which is where the twin connects. You can fly right away: the twin retries mission upload and arming while ArduPilot boots and its GPS/EKF settle.

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
- **Pump:** `DO_SET_SERVO` on **servo 9**, switched on along each spray lane and off on the headland turns. The PWM sets the flow: F = 0.12 kg/s × (PWM − 1100)/800. In ArduPilot, `DO_SET_SERVO` only drives an output whose `SERVO9_FUNCTION` is 0 or 1, which is the SITL default.
- **Variable-rate dose by pump rate:** each zone's PWM is chosen so that flow × predicted pump-on time = planned dose. Pump-on time per lane comes from a fitted lane-time model, t = L / v_eff + t₀. Lanes are flown at 5 m/s (`DO_CHANGE_SPEED`).
  - Dosing by ground speed was tried first and does not work on ArduPilot. On 20 m lanes ArduCopter averaged ~3.1 m/s (4.4 m/s peak) whatever speed was commanded, because waypoint navigation is jerk-limited and stops at each lane end. It overdosed by 21–61%.
- **Calibrate the lane-time model** on your target:

  ```bash
  python scripts/calibrate_spray_timing.py --mode sitl
  ```

  This flies 8 serpentine lanes of 10–60 m and writes `outputs/calibration/spray_timing_sitl.json`, which the API loads.

  | Target | v_eff | t₀ | Fit RMSE |
  |---|---|---|---|
  | ArduCopter SITL | 5.20 m/s | 2.37 s | 0.16 s |
  | Emulator | 5.13 m/s | 1.32 s | 0.23 s |

  Calibrate with the same geometry as real missions. Lanes with long diagonal turns gave t₀ = 1.85 s and a consistent +12% overdose.
- **Measured dose:** the mission report gives `applied_kg` (pump-on time × flow, from the PWM in `SERVO_OUTPUT_RAW`) next to `planned_kg`, plus `pump_pwm` and `pump_on_s`. On SITL the twin's spray zones came out at +1% to +3% of plan.
- **ArduPilot's own sprayer:** if you use AC_Sprayer instead (`SPRAY_ENABLE=1`, `SERVO9_FUNCTION=22`), its flow already scales with ground speed, so the dose per area is set by `SPRAY_PUMP_RATE` rather than by speed. Per-zone dose changes would then need `PARAM_SET` between zones, which the twin does not do yet.
