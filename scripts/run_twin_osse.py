"""
run_twin_osse.py
----------------
Observing-system simulation experiment for the paddy digital twin.

A hidden VirtualField (with a fungal outbreak and a dry patch the twin does
not know about) plays the physical field. Two twins track it with the same
weather forecast:

  * open-loop : model only, never observes
  * assimilating : drone RGB passes + periodic multispectral surveys

The script reports RMSE vs truth and 95%-interval coverage per layer and
day, and writes the result to outputs/twin_osse/.

    python scripts/run_twin_osse.py --days 21 --drone-every 3 --survey-every 7

``--closed-loop`` instead flies an N-drone fleet in flight physics: every
``--drone-every`` days it scouts the field, and (in the twin-driven run)
sprays the zones the updated twin flags. It compares the hidden field's
health against an identical run that scouts but never sprays.

    python scripts/run_twin_osse.py --closed-loop --days 21 --fleet 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.twin_core import LAYERS, DigitalTwin, FieldGrid  # noqa: E402
from src.twin_core.ingest import ingest_multispectral, ingest_rgb_result  # noqa: E402
from src.twin_core.virtual_field import VirtualField, serpentine_waypoints  # noqa: E402
from src.twin_core.swarm import Fleet, allocate, tasks_from_twin  # noqa: E402
from src.twin_core.weather_feed import synthetic_weather  # noqa: E402


def run(days: int, drone_every: int, survey_every: int, altitude: float,
        lane_spacing: float, seed: int, propagation_model: str) -> dict:
    grid = FieldGrid.centered_on(10.0, 78.0)
    truth = VirtualField(grid, seed=seed, propagation_model=propagation_model)
    open_loop = DigitalTwin("open_loop", grid=grid, propagation_model=propagation_model)
    assim = DigitalTwin("assimilating", grid=grid, propagation_model=propagation_model)
    waypoints = serpentine_waypoints(grid, lane_spacing_m=lane_spacing)

    rows = []
    for day in range(1, days + 1):
        forecast = synthetic_weather(1, seed=seed, start_day=day)
        actual = truth.advance(1, forecast)
        open_loop.step(1, forecast)
        assim.step(1, forecast)

        observed = []
        if drone_every and day % drone_every == 0:
            n = 0
            for lat, lon in waypoints:
                frame = truth.drone_camera(lat, lon, alt=altitude)
                if frame and ingest_rgb_result(
                        assim, frame, {"lat": lat, "lon": lon, "alt": altitude})["assimilated"]:
                    n += 1
            observed.append(f"drone:{n}")
        if survey_every and day % survey_every == 0:
            ingest_multispectral(assim, truth.multispectral_survey(coverage=0.9))
            observed.append("survey")

        rows.append({
            "day": day,
            "observed": observed,
            "rain_mm": round(actual[0]["precipitation"], 1),
            "open_loop": truth.twin_error(open_loop),
            "assimilating": truth.twin_error(assim),
        })
    return {
        "config": {"days": days, "drone_every": drone_every, "survey_every": survey_every,
                   "altitude_m": altitude, "lane_spacing_m": lane_spacing, "seed": seed,
                   "propagation_model": propagation_model, "waypoints": len(waypoints),
                   "grid": grid.to_dict()},
        "daily": rows,
    }


def closed_loop(days: int, every: int, n_drones: int, threshold: float, seed: int,
                propagation_model: str, spray: bool) -> dict:
    import random
    random.seed(seed)   # flight-physics turbulence: identical across paired runs
    grid = FieldGrid.centered_on(10.0, 78.0)
    truth = VirtualField(grid, seed=seed, propagation_model=propagation_model)
    twin = DigitalTwin("closed_loop", grid=grid, propagation_model=propagation_model)
    fleet = Fleet.for_field(grid, n_drones)

    def capture(_d, lat, lon, alt):
        frame = truth.drone_camera(lat, lon, alt)
        if frame:
            ingest_rgb_result(twin, frame, {"lat": lat, "lon": lon, "alt": alt})

    pending, daily = [], []
    totals = {"spray_kg": 0.0, "flight_s": 0.0, "zone_sprays": 0, "captures": 0}
    for day in range(1, days + 1):
        forecast = synthetic_weather(1, seed=seed, start_day=day)
        acts = [{"day": 1, "zone_id": p["zone_id"], "type": "Fungicide Spray"} for p in pending]
        pending = []
        truth.advance(1, forecast, acts)
        twin.step(1, forecast, acts)
        flew = []
        if every and day % every == 0:
            kinds = ("scout", "spray") if spray else ("scout",)
            for kind in kinds:
                tasks = tasks_from_twin(twin, kind, fleet.frame, fleet.spec, fungus_threshold=threshold)
                if not tasks:
                    continue
                rep_ = fleet.execute(allocate(tasks, fleet.drone_info()),
                                     on_capture=capture if kind == "scout" else None)
                pending.extend(rep_["completed_sprays"])
                for st in rep_["drones"].values():
                    totals["spray_kg"] += st["spray_used_kg"]
                    totals["flight_s"] += st["flight_s"]
                    totals["captures"] += st["captures"]
                totals["zone_sprays"] += len(rep_["completed_sprays"])
                flew.append(f"{kind}:{len(tasks)}")
        m = truth.truth.state.mean
        daily.append({"day": day, "flew": flew,
                      "truth_fungus": round(float(m["fungus"].mean()), 4),
                      "truth_fungus_max": round(float(m["fungus"].max()), 4),
                      "truth_ndvi": round(float(m["ndvi"].mean()), 4),
                      "twin_fungus_rmse": truth.twin_error(twin)["fungus"]["rmse"]})
    return {"spray": spray, "daily": daily,
            "totals": {k: round(v, 2) for k, v in totals.items()}}


def table(result: dict, layers=("stress", "fungus", "ndvi", "moisture", "nitrogen")) -> str:
    head = "| Day | Obs | " + " | ".join(f"{k} open / DA (cov95)" for k in layers) + " |"
    sep = "|" + "---|" * (len(layers) + 2)
    lines = [head, sep]
    for r in result["daily"]:
        cells = [f"{r['open_loop'][k]['rmse']:.3f} / **{r['assimilating'][k]['rmse']:.3f}** "
                 f"({r['assimilating'][k]['coverage_95']:.2f})" for k in layers]
        lines.append(f"| {r['day']} | {', '.join(r['observed']) or '-'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--drone-every", type=int, default=3, help="days between drone passes (0 = never)")
    ap.add_argument("--survey-every", type=int, default=7, help="days between surveys (0 = never)")
    ap.add_argument("--altitude", type=float, default=10.0, help="drone height above canopy (m)")
    ap.add_argument("--lane-spacing", type=float, default=10.0, help="survey lane spacing (m)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="pde", choices=["baseline", "pde", "hybrid", "gnn"])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "twin_osse")
    ap.add_argument("--closed-loop", action="store_true", help="fleet scouting + twin-driven spraying")
    ap.add_argument("--fleet", type=int, default=3, help="drones in the fleet (closed loop)")
    ap.add_argument("--threshold", type=float, default=0.2, help="zone fungal load that triggers spraying")
    args = ap.parse_args()

    if args.closed_loop:
        t0 = time.time()
        runs = {name: closed_loop(args.days, args.drone_every, args.fleet, args.threshold,
                                  args.seed, args.model, spray)
                for name, spray in (("scout_only", False), ("twin_driven", True))}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "closed_loop_result.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
        a, b = runs["scout_only"]["daily"], runs["twin_driven"]["daily"]
        print("| Day | Flights | Truth fungus: scout-only / twin-driven | Max fungus | Truth NDVI |")
        print("|---|---|---|---|---|")
        for x, y in zip(a, b):
            print(f"| {x['day']} | {', '.join(y['flew']) or '-'} | {x['truth_fungus']:.3f} / "
                  f"**{y['truth_fungus']:.3f}** | {x['truth_fungus_max']:.3f} / "
                  f"**{y['truth_fungus_max']:.3f}** | {x['truth_ndvi']:.3f} / **{y['truth_ndvi']:.3f}** |")
        print("\nTotals (twin-driven):", runs["twin_driven"]["totals"])
        print(f"Wrote {args.out / 'closed_loop_result.json'} in {time.time() - t0:.1f}s")
        return

    t0 = time.time()
    result = run(args.days, args.drone_every, args.survey_every, args.altitude,
                 args.lane_spacing, args.seed, args.model)
    md = table(result)
    final = result["daily"][-1]
    summary = {k: {"open_loop_rmse": final["open_loop"][k]["rmse"],
                   "assimilating_rmse": final["assimilating"][k]["rmse"],
                   "coverage_95": final["assimilating"][k]["coverage_95"]} for k in LAYERS}
    result["final"] = summary

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "osse_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "osse_table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nFinal day {final['day']}:")
    for k, v in summary.items():
        print(f"  {k:9s} open-loop {v['open_loop_rmse']:.4f} -> assimilating "
              f"{v['assimilating_rmse']:.4f}  (95% coverage {v['coverage_95']:.2f})")
    print(f"\nWrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
