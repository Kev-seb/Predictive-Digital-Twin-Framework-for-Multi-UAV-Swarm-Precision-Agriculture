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
    args = ap.parse_args()

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
