"""
08_evaluate_stage_model.py - honest accuracy of a growth-stage checkpoint on flights it never saw.

    python research/08_evaluate_stage_model.py [--ckpt models/classifiers/efficientnet_stage_best.pt]
                                               [--split stage_stratified_flight_disjoint]

Uses the extracted research patches (data/manifests/patch_manifest.csv) of the split's TEST flights,
labels them with the flight's 4-class stage (data/manifests/flight_manifest.csv), and reports accuracy,
macro F1 and the confusion matrix. Writes results/stage_model_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification.stress_classifier import STAGE_CLASSES, MultispectralEfficientNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "classifiers" / "efficientnet_stage_best.pt")
    ap.add_argument("--split", default="stage_stratified_flight_disjoint")
    ap.add_argument("--part", default="test_flights")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stage_model_eval.json")
    a = ap.parse_args()

    splits = json.loads((ROOT / "data" / "splits" / "splits_manifest.json").read_text())
    flights = set(splits[a.split][a.part])
    fm = pd.read_csv(ROOT / "data" / "manifests" / "flight_manifest.csv").set_index("flight_id")
    pm = pd.read_csv(ROOT / "data" / "manifests" / "patch_manifest.csv")
    pm = pm[pm.flight_id.isin(flights)].copy()
    name_to_idx = {v: k for k, v in STAGE_CLASSES.items()}
    pm["label"] = pm.flight_id.map(lambda f: name_to_idx.get(fm.loc[f, "stage_4class"], -1))
    pm = pm[pm.label >= 0]
    if pm.empty:
        sys.exit("No patches for the chosen flights")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultispectralEfficientNet(num_classes=len(STAGE_CLASSES), in_channels=4, pretrained=False)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.to(dev).eval()

    preds, labels = [], []
    batch = []
    with torch.no_grad():
        for path, lab in zip(pm.patch_file, pm.label):
            x = np.load(path).astype(np.float32)
            if x.shape[0] != 4 and x.shape[-1] == 4:
                x = np.transpose(x, (2, 0, 1))
            batch.append(x)
            labels.append(int(lab))
            if len(batch) == 64:
                preds += model(torch.from_numpy(np.stack(batch)).to(dev)).argmax(1).cpu().tolist()
                batch = []
        if batch:
            preds += model(torch.from_numpy(np.stack(batch)).to(dev)).argmax(1).cpu().tolist()

    y, p = np.array(labels), np.array(preds)
    k = len(STAGE_CLASSES)
    cm = np.zeros((k, k), int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    f1s = []
    for c in range(k):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        if cm[c, :].sum():
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    res = {"checkpoint": str(a.ckpt.relative_to(ROOT)), "checkpoint_val_acc": ck.get("val_acc"),
           "split": a.split, "flights": sorted(flights), "patches": int(len(y)),
           "accuracy": float((y == p).mean()), "macro_f1": float(np.mean(f1s)),
           "chance": 1.0 / len(set(y.tolist())), "classes": [STAGE_CLASSES[i] for i in range(k)],
           "confusion": cm.tolist(), "device": dev}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({kk: res[kk] for kk in ("patches", "accuracy", "macro_f1", "chance", "checkpoint_val_acc")}))
    print("confusion (rows = true):", res["classes"])
    for name, row in zip(res["classes"], cm):
        print(f"  {name:11s} {row.tolist()}")


if __name__ == "__main__":
    main()
