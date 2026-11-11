"""
calibrate_paddy.py
------------------
Fit the paddy crop model (src/twin_core/paddy_model.py) to the flight data.

    python scripts/calibrate_paddy.py                      # Open-Meteo archive temps, else 28 C
    python scripts/calibrate_paddy.py --offline            # constant-temperature climatology
    python scripts/calibrate_paddy.py --lat 10.8 --lon 79.1

Inputs
    data/manifests/flight_manifest.csv   stage-labelled flight dates (in repo)
    data/manifests/flight_ndvi.csv       per-flight NDVI (optional; create it with
                                         scripts/extract_flight_ndvi.py on the raw imagery)
Outputs
    outputs/calibration/paddy_params.json   loaded by the twin API (UAV_TWIN_PARAMS_FILE)
    outputs/calibration/calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.config import settings  # noqa: E402
from src.twin_core.calibration import (  # noqa: E402
    calibrate, constant_temperature, openmeteo_archive_temperature,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "flight_manifest.csv")
    ap.add_argument("--ndvi", type=Path, default=ROOT / "data" / "manifests" / "flight_ndvi.csv")
    ap.add_argument("--lat", type=float, default=settings.field_center_lat)
    ap.add_argument("--lon", type=float, default=settings.field_center_lon)
    ap.add_argument("--offline", action="store_true", help="skip Open-Meteo, use --mean-temp")
    ap.add_argument("--mean-temp", type=float, default=28.0)
    ap.add_argument("--out", type=Path, default=settings.twin_params_file.parent)
    args = ap.parse_args()

    temps, temp_source = constant_temperature(args.mean_temp), f"constant {args.mean_temp} C"
    if not args.offline:
        archive = openmeteo_archive_temperature(args.lat, args.lon)
        try:
            import datetime as dt
            archive(dt.date(2025, 7, 1), dt.date(2025, 7, 2))
            temps, temp_source = archive, f"Open-Meteo archive ({args.lat}, {args.lon})"
        except Exception as e:
            print(f"[calibrate] Open-Meteo archive unavailable ({type(e).__name__}); "
                  f"using constant {args.mean_temp} C")

    params, report = calibrate(args.manifest, temps, args.ndvi if args.ndvi.exists() else None)
    report["temperature_source"] = temp_source
    report["ndvi_source"] = str(args.ndvi) if args.ndvi.exists() else None

    args.out.mkdir(parents=True, exist_ok=True)
    params.to_json(args.out / "paddy_params.json")
    (args.out / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    phen = report["phenology"]
    print(f"Temperatures: {temp_source}")
    print(f"Flights: {report['n_flights']} over {report['n_fields']} field(s); "
          f"classified correctly after fit: {phen['flights_classified_correctly']}")
    print("Stage onset (GDD after transplanting, base 10 C):")
    for stage, g in phen["onset"].items():
        b = phen["onset_bounds"].get(stage, {})
        print(f"  {stage:10s} {g:7.1f}   (between {b.get('after_gdd')} and {b.get('before_gdd')})")
    print(f"  harvest    {phen['gdd_harvest']:7.1f}")
    nd = report["ndvi_curve"]
    print("NDVI curve:", nd if not nd.get("fitted") else f"fitted on {nd['n_points']} flights, "
          f"RMSE {nd['rmse']}: {nd['params']}")
    print(f"Wrote {args.out / 'paddy_params.json'}")


if __name__ == "__main__":
    main()
