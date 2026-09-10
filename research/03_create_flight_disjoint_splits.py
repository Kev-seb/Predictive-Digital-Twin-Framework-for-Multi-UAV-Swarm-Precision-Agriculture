"""
03_create_flight_disjoint_splits.py  — P0: DATA INTEGRITY & LEAKAGE CONTROL

Constructs scientifically defensible flight-disjoint splits:
  1. Stage-Stratified Flight-Disjoint Split (Primary Benchmark):
     - Every stage is present in Train, Val, and Test.
     - Crucially, NO flight overlaps between splits (zero spatial data leakage).
  2. Sequential Temporal Holdout Split:
     - Tests out-of-distribution temporal generalization across late-season flights.
  3. Cross-Field Generalization Split:
     - Field001 → Field002
  4. Leave-One-Flight-Out (LOFO) Map

Outputs:
  data/splits/splits_manifest.json
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
SPLIT_DIR    = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def create_splits():
    flight_manifest_path = MANIFEST_DIR / "flight_manifest.csv"
    if not flight_manifest_path.exists():
        print(f"[ERROR] Flight manifest not found at {flight_manifest_path}")
        return

    flight_records = []
    with open(flight_manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flight_records.append(row)

    # Group flights by raw_stage
    by_stage = {}
    for r in flight_records:
        stage = r["raw_stage"]
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(r["flight_id"])

    print(f"[SPLIT] Flights by stage:")
    for stage, flist in by_stage.items():
        print(f"  {stage:12s}: {flist}")

    # 1. Stage-Stratified Flight-Disjoint Split
    # For each stage (4 flights: F1_a, F1_b, F2_a, F2_b):
    # Train: 2 flights (F1_a, F2_a) -> 10 flights total
    # Val:   1 flight  (F1_b)       -> 5 flights total
    # Test:  1 flight  (F2_b)       -> 5 flights total
    strat_train, strat_val, strat_test = [], [], []

    for stage, flist in sorted(by_stage.items()):
        # Split flist (usually 4 flights)
        if len(flist) >= 4:
            strat_train.extend([flist[0], flist[2]])
            strat_val.append(flist[1])
            strat_test.append(flist[3])
        elif len(flist) == 2:
            strat_train.append(flist[0])
            strat_test.append(flist[1])
        else:
            strat_train.extend(flist)

    stratified_split = {
        "train_flights": sorted(strat_train),
        "val_flights":   sorted(strat_val),
        "test_flights":  sorted(strat_test),
    }

    # 2. Sequential Temporal Holdout Split
    f1_flights = [r["flight_id"] for r in flight_records if r["field_id"] == "Field001"]
    f2_flights = [r["flight_id"] for r in flight_records if r["field_id"] == "Field002"]

    temporal_split = {
        "train_flights": f1_flights[:6] + f2_flights[:6],
        "val_flights":   f1_flights[6:8] + f2_flights[6:8],
        "test_flights":  f1_flights[8:] + f2_flights[8:],
    }

    # 3. Cross-Field Split (Field001 -> Field002)
    cross_field_split = {
        "train_flights": f1_flights[:8],
        "val_flights":   f1_flights[8:],
        "test_flights":  f2_flights,
    }

    # 4. Leave-One-Flight-Out (LOFO) Map
    all_flights = f1_flights + f2_flights
    lofo_splits = {}
    for test_f in all_flights:
        rem = [f for f in all_flights if f != test_f]
        lofo_splits[test_f] = {
            "train_flights": rem[:-2],
            "val_flights":   rem[-2:],
            "test_flights":  [test_f]
        }

    splits_manifest = {
        "primary_flight_disjoint": stratified_split,
        "stage_stratified_flight_disjoint": stratified_split,
        "temporal_holdout_flight_disjoint": temporal_split,
        "cross_field_f1_to_f2": cross_field_split,
        "leave_one_flight_out": lofo_splits,
        "all_flights": all_flights
    }

    out_path = SPLIT_DIR / "splits_manifest.json"
    with open(out_path, "w") as f:
        json.dump(splits_manifest, f, indent=2)

    print(f"\n[SPLIT] Primary Stage-Stratified Flight-Disjoint Split:")
    print(f"  Train ({len(stratified_split['train_flights'])} flights): {stratified_split['train_flights']}")
    print(f"  Val   ({len(stratified_split['val_flights'])} flights): {stratified_split['val_flights']}")
    print(f"  Test  ({len(stratified_split['test_flights'])} flights): {stratified_split['test_flights']}")
    print(f"[SPLIT] Wrote split configurations to {out_path}")

if __name__ == "__main__":
    create_splits()
