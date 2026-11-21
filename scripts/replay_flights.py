"""
replay_flights.py
-----------------
Replay your real drone flights through the digital twin, oldest first.

Run it where the raw dataset lives (it is not in the repository):

    python scripts/calibrate_paddy.py                      # once: fit the model to the flights
    python scripts/replay_flights.py --raw data/raw/paddy_dataset --field F1

For each field-stage flight the twin runs forward to the flight date on the
real weather of those days (Open-Meteo archive; --offline for a synthetic
monsoon), predicts the flight's NDVI, and then assimilates it. The table
shows how close the twin's prediction was, against the model alone.

Outputs, in outputs/replay/<field>/:
  report.json            per-flight numbers
  flight_<id>.png        twin NDVI before and after each flight, and what the drone saw
and, with --save-twin, the replayed twin as field "real_<field>" so the API
and UE5 show it:

    UAV_TWIN_FIELD_ID=real_F1 python digital_twin_api.py      (Windows: set UAV_TWIN_FIELD_ID=real_F1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.config import settings  # noqa: E402
from src.twin_core import DigitalTwin  # noqa: E402
from src.twin_core.flight_replay import archive_weather, default_weather, find_flights, replay  # noqa: E402
from src.twin_core.paddy_model import PaddyParams  # noqa: E402


def save_map(path: Path, row, before, after, observed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    vmin, vmax = np.nanpercentile(np.concatenate([before.ravel(), after.ravel()]), [2, 98])
    for a, img, title in zip(ax, (before, observed, after),
                             ("Twin predicted", "Drone measured", "Twin after update")):
        im = a.imshow(img, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
    fig.suptitle(f"{row['flight_id']}  {row['date']}  ({row['stage_labelled']})")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "paddy_dataset")
    ap.add_argument("--field", default="F1", help="flight-name prefix of the field (F1, F2)")
    ap.add_argument("--tiles", type=int, default=3, help="split each image into tiles x tiles patches")
    ap.add_argument("--max-images", type=int, default=150, help="images sampled per flight (0 = all)")
    ap.add_argument("--obs-var", type=float, default=0.0025, help="NDVI observation variance")
    ap.add_argument("--params", type=Path, default=settings.twin_params_file,
                    help="calibrated paddy parameters (default: scripts/calibrate_paddy.py output)")
    ap.add_argument("--offline", action="store_true", help="synthetic weather instead of Open-Meteo")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--save-twin", action="store_true", help="save the twin as field real_<field>")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    if not a.raw.exists():
        sys.exit(f"Raw dataset not found at {a.raw}. Pass --raw <path to paddy_dataset>.")
    flights = find_flights(a.raw, a.field)
    if not flights:
        sys.exit(f"No flight folders for field {a.field} under {a.raw}")
    params = PaddyParams.from_json(a.params) if a.params and Path(a.params).exists() else PaddyParams()
    print(f"{len(flights)} flights for {a.field}; parameters: "
          f"{a.params if a.params and Path(a.params).exists() else 'defaults (run calibrate_paddy.py first)'}")
    out = a.out or ROOT / "outputs" / "replay" / a.field
    out.mkdir(parents=True, exist_ok=True)

    weather = {"fn": default_weather, "source": "synthetic"}

    def make_twin(grid):
        return DigitalTwin(f"real_{a.field}", grid=grid, storage_dir=None, params=params)

    def weather_fn(start, end):
        if not a.offline and weather["source"] != "synthetic (archive unavailable)":
            try:
                w = archive_weather(*_center[0])(start, end)
                if len(w) == (end - start).days:
                    weather["source"] = "Open-Meteo archive"
                    return w
            except Exception as e:
                print(f"  Open-Meteo archive unavailable ({type(e).__name__}); using synthetic weather")
            weather["source"] = "synthetic (archive unavailable)"
        return default_weather(start, end)

    _center = [None]
    maps = {}

    def on_flight(row, twin, before, measured):
        if measured is not None:
            maps[row["flight_id"]] = (before, measured, twin.state.mean["ndvi"].copy())

    # The weather location: the first image with a position.
    from src.twin_core.flight_replay import read_pose
    for f in flights:
        for pre in f.prefixes[:1]:
            pose = read_pose(pre.with_name(pre.name + "_MS_R.TIF"))
            if pose:
                _center[0] = (pose.lat, pose.lon)
        if _center[0]:
            break
    if _center[0] is None:
        sys.exit("No image had a GPS position (DJI XMP or EXIF).")

    result = replay(flights, make_twin, weather_fn, tiles=a.tiles,
                    max_images=a.max_images or None, obs_var=a.obs_var, on_flight=on_flight)

    rows = result["flights"]
    print(f"\nWeather: {weather['source']}.  Transplanting: {result['transplant']}.  "
          f"Grid {result['grid']['rows']} x {result['grid']['cols']} cells of {result['grid']['cell_size_m']:.1f} m\n")
    print(f"{'flight':8s}{'date':12s}{'stage':12s}{'gap d':>6s}{'tiles':>7s}{'measured':>10s}"
          f"{'twin pred':>11s}{'model':>8s}{'twin err':>10s}{'model err':>10s}")
    for r in rows:
        if "observed_ndvi" not in r:
            print(f"{r['flight_id']:8s}{r['date']:12s}{r['stage_labelled']:12s}{r['days_since_last']:>6d}"
                  f"{r['tiles']:>7d}   (no usable images)")
            continue
        print(f"{r['flight_id']:8s}{r['date']:12s}{r['stage_labelled']:12s}{r['days_since_last']:>6d}"
              f"{r['tiles']:>7d}{r['observed_ndvi']:>10.3f}{r['twin_predicted_ndvi']:>11.3f}"
              f"{r['model_only_ndvi']:>8.3f}{r['twin_rmse']:>10.3f}{r['model_only_rmse']:>10.3f}")
    scored = [r for r in rows[1:] if "twin_rmse" in r]      # the first flight has nothing to learn from
    if scored:
        tw = float(np.mean([r["twin_rmse"] for r in scored]))
        mo = float(np.mean([r["model_only_rmse"] for r in scored]))
        print(f"\nPredicting the next flight (flights 2..{len(rows)}): twin RMSE {tw:.3f}, model alone {mo:.3f}"
              + (f"  ({(1 - tw / mo) * 100:.0f}% better)" if mo > 0 else ""))

    if not a.no_maps:
        for r in rows:
            if r["flight_id"] in maps:
                before, measured, after = maps[r["flight_id"]]
                save_map(out / f"flight_{r['flight_id']}.png", r, before, after, measured)
    (out / "report.json").write_text(json.dumps(
        {"field": a.field, "weather": weather["source"], "transplant": result["transplant"],
         "grid": result["grid"], "flights": rows}, indent=2))
    print(f"\nReport and maps in {out}")
    if a.save_twin:
        twin = result["twin"]
        twin.storage_dir = Path(settings.twin_storage_dir) / twin.field_id
        twin.save()
        print(f"Saved twin '{twin.field_id}' to {twin.storage_dir}. Show it in UE5 with:\n"
              f"  UAV_TWIN_FIELD_ID={twin.field_id} python digital_twin_api.py")
    return rows


if __name__ == "__main__":
    main()
