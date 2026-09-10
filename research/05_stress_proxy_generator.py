"""
05_stress_proxy_generator.py  — PHASE 4: DEFENSIBLE STRESS PROXY & LABELS

Solves Fatal Flaw SF-04 / SF-26 / PB-04 / HI-08:
  - Generates mathematically sound crop stress labels based on per-stage NDVI Z-score anomalies:
      Z = (NDVI - mean_NDVI_stage) / std_NDVI_stage
  - Assigns 4 stress severity levels:
      0: Healthy / No Stress (Z >= -0.5)
      1: Low Stress         (-1.5 <= Z < -0.5)
      2: Moderate Stress    (-2.5 <= Z < -1.5)
      3: High Stress        (Z < -2.5)

Outputs:
  data/manifests/stress_patch_manifest.csv
  data/manifests/stress_distribution.json
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"

def generate_stress_labels():
    patch_manifest_path = MANIFEST_DIR / "patch_manifest.csv"
    if not patch_manifest_path.exists():
        print(f"[ERROR] {patch_manifest_path} not found.")
        return

    records = []
    with open(patch_manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"[STRESS] Computing per-stage NDVI statistics across {len(records)} patches...")

    # Stage -> list of mean NDVI values per patch
    stage_ndvis: dict[str, list[float]] = {}
    patch_ndvis: list[float] = []

    for rec in records:
        patch = np.load(rec["patch_file"]).astype(np.float32)  # (4, 224, 224)
        r = patch[1, :, :]
        nir = patch[3, :, :]
        ndvi = (nir - r) / (nir + r + 1e-6)
        mean_ndvi = float(np.mean(ndvi))
        rec["mean_ndvi"] = mean_ndvi
        patch_ndvis.append(mean_ndvi)

        stage = rec["stage"]
        if stage not in stage_ndvis:
            stage_ndvis[stage] = []
        stage_ndvis[stage].append(mean_ndvi)

    stage_stats = {}
    for stage, ndvi_list in stage_ndvis.items():
        mu = float(np.mean(ndvi_list))
        sigma = float(np.std(ndvi_list))
        stage_stats[stage] = {"mean": mu, "std": sigma}
        print(f"  Stage: {stage:12s} | Mean NDVI: {mu:.4f} ± {sigma:.4f}")

    # Calculate Z-score and stress category per patch
    stress_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for rec in records:
        stage = rec["stage"]
        mu = stage_stats[stage]["mean"]
        sigma = max(stage_stats[stage]["std"], 1e-5)
        z = (rec["mean_ndvi"] - mu) / sigma
        rec["ndvi_z_score"] = round(z, 4)

        if z >= -0.5:
            rec["stress_level"] = 0
            rec["stress_label"] = "No Stress"
        elif z >= -1.5:
            rec["stress_level"] = 1
            rec["stress_label"] = "Low Stress"
        elif z >= -2.5:
            rec["stress_level"] = 2
            rec["stress_label"] = "Moderate Stress"
        else:
            rec["stress_level"] = 3
            rec["stress_label"] = "High Stress"

        stress_counts[rec["stress_level"]] += 1

    out_csv = MANIFEST_DIR / "stress_patch_manifest.csv"
    fieldnames = list(records[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"\n[STRESS] Generated stress proxy labels saved to {out_csv}")
    print(f"  Stress Level Distribution: {stress_counts}")

    summary = {
        "stage_statistics": stage_stats,
        "stress_distribution": stress_counts,
        "methodology": "Per-stage NDVI Z-score spectral anomaly proxy (disclosed as proxy label)",
    }
    with open(MANIFEST_DIR / "stress_distribution.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    generate_stress_labels()
