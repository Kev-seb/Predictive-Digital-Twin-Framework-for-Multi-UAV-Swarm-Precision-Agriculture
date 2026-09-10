"""
02_extract_patches_with_provenance.py  — P0: DATA INTEGRITY & DISK-EFFICIENT SAMPLING
Extracts 224x224 patches from raw TIF flight directories with provenance tracking.

Disk-Aware Sampling Strategy:
  - Max 10 image sets per flight (sampled uniformly across session time)
  - Non-overlapping stride (224px)
  - Near-uniform patch filtering (std < 0.01)
  - Max 250 patches per flight session -> ~4,500-5,000 patches total (~1.8 GB disk footprint)
  - Stores as float16 to halve disk size while maintaining precision

Outputs:
  data/processed/research/patches/{stage}/{patch_id}.npy (float16)
  data/manifests/patch_manifest.csv
  data/manifests/patch_summary.json
"""
from __future__ import annotations
import csv
import json
import logging
import re
import datetime
from pathlib import Path

import numpy as np
import tifffile

ROOT       = Path(__file__).resolve().parents[1]
RAW        = ROOT / "data" / "raw" / "paddy_dataset"
OUT_PATCH  = ROOT / "data" / "processed" / "research" / "patches"
MANIFEST   = ROOT / "data" / "manifests"
MANIFEST.mkdir(parents=True, exist_ok=True)

PATCH_SIZE          = 224
STRIDE              = 224   # non-overlapping
MAX_SETS_PER_FLIGHT = 10    # uniform frame subsampling per flight
MAX_PATCHES_PER_FLIGHT = 250

FLIGHT_RE = re.compile(
    r"^(F[12])_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\(1\))?_([A-Za-z]+)_[Ss]tage$"
)

logging.basicConfig(
    filename=str(MANIFEST / "extraction_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

def parse_flight_dir(d: Path):
    m = FLIGHT_RE.match(d.name)
    if not m:
        return None
    field_code, seq, yr, mo, da, time_str, stage = m.groups()
    return {
        "flight_id":  f"{field_code}_{seq}",
        "field_id":   "Field001" if field_code == "F1" else "Field002",
        "date":       f"{yr}-{mo}-{da}",
        "time":       f"{time_str[:2]}:{time_str[2:]}",
        "raw_stage":  stage.capitalize(),
    }

def relative_normalize(band: np.ndarray) -> np.ndarray:
    """Per-band 1st–99th percentile clipping -> [0, 1]."""
    p1  = float(np.percentile(band, 1))
    p99 = float(np.percentile(band, 99))
    if p99 <= p1:
        return np.zeros_like(band, dtype=np.float32)
    normed = (band.astype(np.float32) - p1) / (p99 - p1)
    return np.clip(normed, 0.0, 1.0)

def extract_flight(flight_dir: Path, meta: dict, out_dir: Path, writer, patch_counter: list):
    """Extract non-overlapping patches from sampled image sets of a single flight."""
    stage     = meta["raw_stage"]
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Collect all complete image-set prefixes
    tif_map: dict[str, dict] = {}
    for f in sorted(flight_dir.glob("*.TIF")):
        m = re.search(r"_MS_(G|R|RE|NIR)\.TIF$", f.name)
        if not m:
            continue
        prefix = f.name[:m.start()]
        if prefix not in tif_map:
            tif_map[prefix] = {}
        tif_map[prefix][m.group(1)] = f

    all_prefixes = sorted([p for p, b in tif_map.items() if all(k in b for k in ["G", "R", "RE", "NIR"])])
    if not all_prefixes:
        return 0

    # Subsample image sets uniformly across flight duration
    if len(all_prefixes) > MAX_SETS_PER_FLIGHT:
        indices = np.linspace(0, len(all_prefixes) - 1, MAX_SETS_PER_FLIGHT, dtype=int)
        sampled_prefixes = [all_prefixes[i] for i in indices]
    else:
        sampled_prefixes = all_prefixes

    n_patches = 0
    for prefix in sampled_prefixes:
        if n_patches >= MAX_PATCHES_PER_FLIGHT:
            break
        band_files = tif_map[prefix]
        try:
            bands_raw = [
                tifffile.imread(str(band_files[b])).astype(np.float32)
                for b in ["G", "R", "RE", "NIR"]
            ]
        except Exception as e:
            logging.error("Read error: %s | %s", prefix, e)
            continue

        shapes = [b.shape for b in bands_raw]
        if len(set(shapes)) > 1:
            logging.warning("Band shape mismatch: %s shapes=%s", prefix, shapes)
            continue

        H, W = bands_raw[0].shape
        if H < PATCH_SIZE or W < PATCH_SIZE:
            continue

        # Per-band relative normalization
        bands_norm = np.stack([relative_normalize(b) for b in bands_raw], axis=0)  # (4, H, W)

        # Patch extraction
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            if n_patches >= MAX_PATCHES_PER_FLIGHT:
                break
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                if n_patches >= MAX_PATCHES_PER_FLIGHT:
                    break
                patch = bands_norm[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # Skip near-uniform background/soil/sky patches
                mean_std = float(np.mean([patch[i].std() for i in range(4)]))
                if mean_std < 0.01:
                    continue

                pid = patch_counter[0]
                patch_counter[0] += 1
                fname = f"patch_{pid:06d}.npy"
                # Store as float16 to save disk space
                np.save(str(stage_dir / fname), patch.astype(np.float16))

                writer.writerow({
                    "patch_id":        pid,
                    "patch_file":      str(stage_dir / fname),
                    "source_file":     prefix,
                    "flight_id":       meta["flight_id"],
                    "field_id":        meta["field_id"],
                    "date":            meta["date"],
                    "time":            meta["time"],
                    "stage":           stage,
                    "x":               x,
                    "y":               y,
                    "band_config":     "G,R,RE,NIR",
                    "norm_method":     "relative_p1p99",
                    "split":           "",
                })
                n_patches += 1

    return n_patches

def main():
    print("[PATCH] Starting disk-efficient patch extraction with provenance tracking...")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}, max per flight: {MAX_PATCHES_PER_FLIGHT}")
    print(f"  Frame sampling: max {MAX_SETS_PER_FLIGHT} frames per flight session")
    print("  Storage: float16 .npy format for disk efficiency")

    field_roots = {
        "Field001": RAW / "Field001_3acres" / "Field001_3acres",
        "Field002": RAW / "Field002_2acres",
    }

    manifest_path = MANIFEST / "patch_manifest.csv"
    fieldnames = [
        "patch_id", "patch_file", "source_file", "flight_id", "field_id",
        "date", "time", "stage", "x", "y", "band_config", "norm_method", "split"
    ]

    patch_counter = [0]
    flight_summary = []

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fld, fld_root in field_roots.items():
            if not fld_root.exists():
                continue
            for d in sorted(fld_root.iterdir()):
                if not d.is_dir() or "(1)" in d.name:
                    continue
                meta = parse_flight_dir(d)
                if meta is None:
                    continue
                n = extract_flight(d, meta, OUT_PATCH, writer, patch_counter)
                flight_summary.append({**meta, "n_patches": n})
                print(f"  {meta['flight_id']} | {meta['date']} | {meta['raw_stage']:12s} | patches={n}")

    total = patch_counter[0]
    print(f"\n[PATCH] Extraction complete. Total patches extracted: {total}")
    print(f"[PATCH] Manifest written to: {manifest_path}")

    stage_dist = {}
    for fs in flight_summary:
        s = fs["raw_stage"]
        stage_dist[s] = stage_dist.get(s, 0) + fs["n_patches"]
    print(f"  Distribution by stage: {stage_dist}")

    summary = {
        "generated_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_patches":      total,
        "patch_size":         PATCH_SIZE,
        "stride":             STRIDE,
        "max_sets_per_flight": MAX_SETS_PER_FLIGHT,
        "max_patches_per_flight": MAX_PATCHES_PER_FLIGHT,
        "dtype":              "float16",
        "stage_distribution": stage_dist,
        "flights":            flight_summary,
    }
    with open(MANIFEST / "patch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return total, stage_dist

if __name__ == "__main__":
    main()
