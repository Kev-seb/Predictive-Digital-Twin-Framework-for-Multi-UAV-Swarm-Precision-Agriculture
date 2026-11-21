"""
validate_cross_layer.py
-----------------------
OSSE check of the twin's cross-layer correction (src/twin_core/ensemble.py).

Two twins track the same hidden VirtualField (fungal outbreak + a leaky,
draining soil patch the twins don't know about), with the same forecasts and
the same observations (drone RGB passes every 3 days, multispectral survey
weekly). One uses the per-layer Kalman filter only, the other adds the
cross-layer ensemble. Reported, averaged over seeds:

  * RMSE and 95% coverage per layer after ``--days`` days (nowcast)
  * how well each twin knows the leaky patch's percolation rate
  * the 7-day moisture forecast error over the leaky patch, made from day
    ``--days`` with no further observations (the predictive part)

    python scripts/validate_cross_layer.py --days 21 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.twin_core import LAYERS, DigitalTwin, FieldGrid  # noqa: E402
from src.twin_core.ingest import ingest_multispectral, ingest_rgb_result  # noqa: E402
from src.twin_core.virtual_field import VirtualField, serpentine_waypoints  # noqa: E402
from src.twin_core.weather_feed import synthetic_weather  # noqa: E402


def run(seed: int, days: int, drone_every: int, survey_every: int, horizon: int) -> dict:
    grid = FieldGrid.centered_on(10.0, 78.0)
    truth = VirtualField(grid, seed=seed)
    twins = {"per_layer": DigitalTwin("per_layer", grid=grid, cross_layer=False),
             "cross_layer": DigitalTwin("cross_layer", grid=grid, cross_layer=True)}
    waypoints = serpentine_waypoints(grid, lane_spacing_m=10.0)
    leaky = truth.truth.state.aux["percolation_mm_day"] > 3.0 * float(np.median(
        truth.truth.state.aux["percolation_mm_day"]))
    for day in range(1, days + 1):
        forecast = synthetic_weather(1, seed=seed, start_day=day)
        truth.advance(1, forecast)
        for t in twins.values():
            t.step(1, forecast)
        if drone_every and day % drone_every == 0:
            frames = [(truth.drone_camera(lat, lon, alt=12.0), lat, lon) for lat, lon in waypoints]
            for t in twins.values():
                for f, lat, lon in frames:
                    if f:
                        ingest_rgb_result(t, f, {"lat": lat, "lon": lon, "alt": 12.0})
        if survey_every and day % survey_every == 0:
            survey = truth.multispectral_survey(coverage=0.9)
            for t in twins.values():
                ingest_multispectral(t, survey)

    out = {}
    true_perc = truth.truth.state.aux["percolation_mm_day"][leaky]
    # Forecast: everyone runs `horizon` more days on the same forecast, no observations.
    weather = synthetic_weather(horizon, seed=seed, start_day=days + 1)
    future = truth.truth.fork()
    future.step(horizon, weather)
    for name, t in twins.items():
        err = truth.twin_error(t)
        est = t.state.aux["percolation_mm_day"][leaky]
        f = t.fork()
        f.step(horizon, weather)
        fe = f.state.mean["moisture"][leaky] - future.state.mean["moisture"][leaky]
        out[name] = {
            "rmse": {k: err[k]["rmse"] for k in LAYERS},
            "coverage_95": {k: err[k]["coverage_95"] for k in LAYERS},
            "leak_rate_factor_error": float(np.exp(np.abs(np.log(est / true_perc)).mean())),
            "forecast_moisture_rmse_leaky": float(np.sqrt(np.mean(fe ** 2))),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--drone-every", type=int, default=3)
    ap.add_argument("--survey-every", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "cross_layer" / "result.json"))
    a = ap.parse_args(argv)

    runs = [run(s, a.days, a.drone_every, a.survey_every, a.horizon) for s in a.seeds]
    summary = {}
    for name in ("per_layer", "cross_layer"):
        rs = [r[name] for r in runs]
        summary[name] = {
            "rmse": {k: round(float(np.mean([r["rmse"][k] for r in rs])), 4) for k in LAYERS},
            "coverage_95": {k: round(float(np.mean([r["coverage_95"][k] for r in rs])), 3) for k in LAYERS},
            "leak_rate_factor_error": round(float(np.mean([r["leak_rate_factor_error"] for r in rs])), 2),
            "forecast_moisture_rmse_leaky": round(float(np.mean(
                [r["forecast_moisture_rmse_leaky"] for r in rs])), 4),
        }
    p, c = summary["per_layer"], summary["cross_layer"]
    print(f"OSSE, {len(a.seeds)} seeds, {a.days} days, drone every {a.drone_every} d, "
          f"survey every {a.survey_every} d\n")
    print(f"{'':34s}{'per-layer':>11s}{'cross-layer':>13s}{'change':>9s}")
    for k in LAYERS:
        ch = (c["rmse"][k] / p["rmse"][k] - 1.0) * 100 if p["rmse"][k] else 0.0
        print(f"  {k + ' RMSE (95% cover)':32s}{p['rmse'][k]:>7.4f} ({p['coverage_95'][k]:.2f})"
              f"{c['rmse'][k]:>7.4f} ({c['coverage_95'][k]:.2f}){ch:>+7.0f}%")
    print(f"  {'leaky soil: leak rate off by':32s}{'x' + format(p['leak_rate_factor_error'], '.2f'):>11s}"
          f"{'x' + format(c['leak_rate_factor_error'], '.2f'):>13s}")
    ch = (c["forecast_moisture_rmse_leaky"] / p["forecast_moisture_rmse_leaky"] - 1.0) * 100
    print(f"  {f'{a.horizon}-day moisture forecast (leaky)':32s}{p['forecast_moisture_rmse_leaky']:>11.4f}"
          f"{c['forecast_moisture_rmse_leaky']:>13.4f}{ch:>+8.0f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "summary": summary, "runs": runs}, indent=2))
    print(f"\nwritten to {out}")
    return summary


if __name__ == "__main__":
    main()
