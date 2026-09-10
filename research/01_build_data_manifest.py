"""
build_data_manifest.py  — P0: DATA INTEGRITY
Crawls the raw UAV dataset, counts images per flight, extracts
metadata from directory names, and writes:
  data/manifests/flight_manifest.csv
  data/manifests/dataset_statistics.json

Does NOT extract patches — that comes later with provenance tracking.
"""
from __future__ import annotations
import json
import csv
import re
import datetime
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
RAW    = ROOT / "data" / "raw" / "paddy_dataset"
OUT    = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

# ── directory name parser ──────────────────────────────────────────────────
FLIGHT_RE = re.compile(
    r"^(F[12])_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\(1\))?_([A-Za-z]+)_[Ss]tage$"
)

STAGE_MAP_4 = {
    "Nursery":    "Nursery",
    "Vegetative": "Vegetative",
    "Booting":    "Vegetative",   # legacy merge — we track original below
    "Flowering":  "Flowering",
    "Mature":     "Mature",
}
STAGE_MAP_5 = {
    "Nursery":    "Nursery",
    "Vegetative": "Vegetative",
    "Booting":    "Booting",
    "Flowering":  "Flowering",
    "Mature":     "Mature",
}

BANDS = ["MS_G", "MS_R", "MS_RE", "MS_NIR"]

def parse_flight_dir(d: Path):
    name = d.name
    m = FLIGHT_RE.match(name)
    if not m:
        return None
    field_code, seq, yr, mo, da, time_str, stage = m.groups()
    field_id   = "Field001" if field_code == "F1" else "Field002"
    date_str   = f"{yr}-{mo}-{da}"
    time_str   = f"{time_str[:2]}:{time_str[2:]}"
    flight_id  = f"{field_code}_{seq}"
    is_dup     = "(1)" in name          # duplicate marker
    return {
        "flight_id":      flight_id,
        "field_id":       field_id,
        "date":           date_str,
        "time":           time_str,
        "raw_stage":      stage.capitalize(),
        "stage_4class":   STAGE_MAP_4.get(stage.capitalize(), stage.capitalize()),
        "stage_5class":   STAGE_MAP_5.get(stage.capitalize(), stage.capitalize()),
        "dir_name":       name,
        "is_duplicate":   is_dup,
    }

def count_images(flight_dir: Path):
    tif_sets  = {}
    rgb_count = 0
    for f in flight_dir.rglob("*.TIF"):
        prefix = re.sub(r"_MS_(G|R|RE|NIR)\.TIF$", "", f.name)
        if prefix not in tif_sets:
            tif_sets[prefix] = set()
        band = re.search(r"_MS_(G|R|RE|NIR)\.TIF$", f.name)
        if band:
            tif_sets[prefix].add(band.group(1))
    for f in flight_dir.rglob("*_D.JPG"):
        rgb_count += 1
    # complete sets = all 4 bands present
    complete = sum(1 for v in tif_sets.values() if len(v) == 4)
    partial  = sum(1 for v in tif_sets.values() if 0 < len(v) < 4)
    return {
        "image_sets_total":    len(tif_sets),
        "image_sets_complete": complete,
        "image_sets_partial":  partial,
        "rgb_jpg_count":       rgb_count,
        "available_bands":     sorted(set(b for v in tif_sets.values() for b in v)),
    }

def main():
    print("[MANIFEST] Scanning raw dataset...")
    rows = []
    field_dirs = {
        "Field001": RAW / "Field001_3acres" / "Field001_3acres",
        "Field002": RAW / "Field002_2acres",
    }

    seen_flights = {}   # flight_id → first occurrence (deduplicate)

    for fld, fld_dir in field_dirs.items():
        if not fld_dir.exists():
            print(f"  [WARN] {fld_dir} not found — skipping")
            continue
        for d in sorted(fld_dir.iterdir()):
            if not d.is_dir():
                continue
            meta = parse_flight_dir(d)
            if meta is None:
                print(f"  [SKIP] Unrecognised dir: {d.name}")
                continue
            fid = meta["flight_id"]
            if fid in seen_flights and not meta["is_duplicate"]:
                print(f"  [WARN] Duplicate flight dir ignored: {d.name}")
                continue
            if meta["is_duplicate"]:
                print(f"  [INFO] Skipping duplicate (1) dir: {d.name}")
                continue
            seen_flights[fid] = d
            counts = count_images(d)
            row = {**meta, **counts, "path": str(d)}
            rows.append(row)
            print(f"  {fid} | {meta['date']} {meta['time']} | "
                  f"stage={meta['raw_stage']:12s} | "
                  f"complete_sets={counts['image_sets_complete']}")

    # Sort by field + date + time
    rows.sort(key=lambda r: (r["field_id"], r["date"], r["time"]))

    # Write flight_manifest.csv
    fm_path = OUT / "flight_manifest.csv"
    fieldnames = [
        "flight_id", "field_id", "date", "time", "raw_stage",
        "stage_4class", "stage_5class",
        "image_sets_total", "image_sets_complete", "image_sets_partial",
        "rgb_jpg_count", "available_bands", "path"
    ]
    with open(fm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r2 = {k: r.get(k, "") for k in fieldnames}
            r2["available_bands"] = str(r.get("available_bands", []))
            w.writerow(r2)
    print(f"\n[MANIFEST] Wrote {fm_path}")

    # Aggregate statistics
    total_sets      = sum(r["image_sets_complete"] for r in rows)
    by_field        = {}
    by_stage4       = {}
    by_stage5       = {}
    for r in rows:
        fld = r["field_id"]
        s4  = r["stage_4class"]
        s5  = r["stage_5class"]
        by_field[fld]  = by_field.get(fld, 0)  + r["image_sets_complete"]
        by_stage4[s4]  = by_stage4.get(s4, 0)  + r["image_sets_complete"]
        by_stage5[s5]  = by_stage5.get(s5, 0)  + r["image_sets_complete"]

    stats = {
        "generated_at":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_flights":         len(rows),
        "total_complete_image_sets": total_sets,
        "fields":                sorted(by_field.keys()),
        "image_sets_by_field":   by_field,
        "image_sets_by_stage_4class": by_stage4,
        "image_sets_by_stage_5class": by_stage5,
        "flights_by_field": {
            fld: [r["flight_id"] for r in rows if r["field_id"] == fld]
            for fld in sorted(by_field.keys())
        },
        "note_booting_flights": [
            r["flight_id"] for r in rows if r["raw_stage"] == "Booting"
        ],
        "note_duplicate_dirs_skipped": [
            d.name for d in list(RAW.rglob("*"))
            if d.is_dir() and "(1)" in d.name
        ],
    }
    stats_path = OUT / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[MANIFEST] Wrote {stats_path}")
    print(f"\n  Total flights        : {stats['total_flights']}")
    print(f"  Total complete sets  : {stats['total_complete_image_sets']}")
    print(f"  By stage (4-class)   : {by_stage4}")
    print(f"  By stage (5-class)   : {by_stage5}")
    print(f"  Booting flights      : {stats['note_booting_flights']}")

    return rows, stats

if __name__ == "__main__":
    main()
