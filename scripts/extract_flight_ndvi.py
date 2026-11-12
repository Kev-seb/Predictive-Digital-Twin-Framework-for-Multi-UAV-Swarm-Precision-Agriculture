"""
extract_flight_ndvi.py
----------------------
Per-flight canopy NDVI from the raw multispectral imagery, for calibrating
the paddy model's NDVI curve (scripts/calibrate_paddy.py).

Run this where the raw dataset lives (it is not in the repository):

    python scripts/extract_flight_ndvi.py --raw data/raw/paddy_dataset --sets 40

For every flight directory matched by research/01_build_data_manifest.py
(e.g. F1_03_2025_08_16_1609_Vegetative_Stage) it samples up to --sets image
sets (<prefix>_MS_R.TIF + <prefix>_MS_NIR.TIF), computes NDVI per pixel and
records the median of each image, then the median across images.

NDVI from uncalibrated digital numbers is approximate (no reflectance-panel
or sunlight-sensor correction), but the seasonal shape is what the
calibration fits, and the per-flight ratio is robust to a common gain.

Output: data/manifests/flight_ndvi.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_RE = re.compile(r"^(F[12])_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\(1\))?_([A-Za-z]+)_[Ss]tage$")


def read_band(path: Path) -> np.ndarray:
    try:
        import tifffile
        return tifffile.imread(str(path)).astype(np.float32)
    except ImportError:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def image_ndvi(red: np.ndarray, nir: np.ndarray) -> float:
    valid = (red > 0) & (nir > 0)
    if valid.sum() < 100:
        return float("nan")
    ndvi = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    return float(np.median(ndvi))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "paddy_dataset")
    ap.add_argument("--sets", type=int, default=40, help="image sets sampled per flight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifests" / "flight_ndvi.csv")
    args = ap.parse_args()

    if not args.raw.exists():
        sys.exit(f"Raw dataset not found at {args.raw}. Pass --raw <path to paddy_dataset>.")
    rng = random.Random(args.seed)
    rows = []
    for d in sorted(p for p in args.raw.rglob("*") if p.is_dir() and FLIGHT_RE.match(p.name)):
        if "(1)" in d.name:
            continue  # duplicate directory, skipped by the manifest as well
        field_code, seq, yr, mo, da, _, stage = FLIGHT_RE.match(d.name).groups()
        prefixes = sorted({re.sub(r"_MS_R\.TIF$", "", f.name) for f in d.rglob("*_MS_R.TIF")})
        rng.shuffle(prefixes)
        values = []
        for pre in prefixes[: args.sets]:
            red = next(d.rglob(f"{pre}_MS_R.TIF"), None)
            nir = next(d.rglob(f"{pre}_MS_NIR.TIF"), None)
            if red and nir:
                v = image_ndvi(read_band(red), read_band(nir))
                if np.isfinite(v):
                    values.append(v)
        if not values:
            print(f"  {d.name}: no usable R/NIR pairs")
            continue
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        rows.append({"flight_id": f"{field_code}_{seq}",
                     "field_id": "Field001" if field_code == "F1" else "Field002",
                     "date": f"{yr}-{mo}-{da}", "stage": stage.capitalize(),
                     "ndvi_median": round(float(med), 4), "ndvi_p25": round(float(q25), 4),
                     "ndvi_p75": round(float(q75), 4), "n_images": len(values)})
        print(f"  {rows[-1]['flight_id']} {rows[-1]['date']} {rows[-1]['stage']:10s} "
              f"NDVI {med:.3f} ({len(values)} images)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["flight_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} flights to {args.out}")


if __name__ == "__main__":
    main()
