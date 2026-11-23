"""
09_train_stage_model.py - growth-stage classifier trained and tested on DIFFERENT flights (GPU).

    python research/09_train_stage_model.py [--split stage_stratified_flight_disjoint] [--epochs 25]
                                            [--norm instance|flight|both]

Why: the earlier checkpoint scored 96.5% on patches from its own training flights but ~30% on unseen
flights (research/08). Here every number is flight-disjoint:
  train flights -> fit, validation flights -> pick the epoch, test flights -> the reported result.

Radiometric robustness:
  * normalisation: 'instance' (each patch, each band to zero mean / unit variance) removes per-flight
    brightness; 'flight' standardises each band with that flight's statistics.
  * augmentation: flips, 90-degree rotations, random crops, and per-band gain/offset jitter (simulated
    calibration error), so the model cannot key on one flight's absolute reflectance.

Outputs: models/classifiers/stage_flightdisjoint.pt (+ .json model card), results/stage_training.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification.stress_classifier import STAGE_CLASSES, MultispectralEfficientNet  # noqa: E402

NAME_TO_IDX = {v: k for k, v in STAGE_CLASSES.items()}


def load_split(split: str):
    splits = json.loads((ROOT / "data" / "splits" / "splits_manifest.json").read_text())[split]
    fm = pd.read_csv(ROOT / "data" / "manifests" / "flight_manifest.csv").set_index("flight_id")
    pm = pd.read_csv(ROOT / "data" / "manifests" / "patch_manifest.csv")
    pm["label"] = pm.flight_id.map(lambda f: NAME_TO_IDX.get(fm.loc[f, "stage_4class"], -1))
    pm = pm[pm.label >= 0]
    parts = {k: pm[pm.flight_id.isin(splits[f"{k}_flights"])] for k in ("train", "val", "test")}
    return parts


def load_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.stack([np.load(p).astype(np.float32) for p in df.patch_file])
    return xs, df.label.to_numpy(np.int64), df.flight_id.to_numpy()


def normalise(x: np.ndarray, flights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "instance":
        m = x.mean(axis=(2, 3), keepdims=True)
        s = x.std(axis=(2, 3), keepdims=True) + 1e-4
        return (x - m) / s
    out = np.empty_like(x)
    for f in np.unique(flights):               # 'flight': one mean/std per band per flight
        sel = flights == f
        m = x[sel].mean(axis=(0, 2, 3), keepdims=True)
        s = x[sel].std(axis=(0, 2, 3), keepdims=True) + 1e-4
        out[sel] = (x[sel] - m) / s
    return out


class Patches(Dataset):
    def __init__(self, x, y, train: bool, crop: int = 192):
        self.x, self.y, self.train, self.crop = x, y, train, crop

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
            h = x.shape[1]
            r, c = np.random.randint(0, h - self.crop + 1, 2)
            x = x[:, r:r + self.crop, c:c + self.crop]
            gain = np.random.uniform(0.8, 1.2, (4, 1, 1)).astype(np.float32)     # simulated calibration error
            off = np.random.uniform(-0.2, 0.2, (4, 1, 1)).astype(np.float32)
            x = x * gain + off
        else:
            o = (x.shape[1] - self.crop) // 2
            x = x[:, o:o + self.crop, o:o + self.crop]
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.y[i])


def evaluate(model, loader, dev, k):
    model.eval()
    cm = np.zeros((k, k), int)
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for x, y in loader:
            p = model(x.to(dev)).argmax(1).cpu().numpy()
            for t, q in zip(y.numpy(), p):
                cm[t, q] += 1
    acc = np.trace(cm) / max(1, cm.sum())
    f1 = [2 * cm[c, c] / max(1, 2 * cm[c, c] + cm[:, c].sum() - cm[c, c] + cm[c, :].sum() - cm[c, c])
          for c in range(k) if cm[c, :].sum()]
    return float(acc), float(np.mean(f1)), cm


def train_one(parts, norm, epochs, dev, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = {}
    for k, df in parts.items():
        x, y, f = load_arrays(df)
        data[k] = (normalise(x, f, norm), y)
    k = len(STAGE_CLASSES)
    loaders = {"train": DataLoader(Patches(*data["train"], train=True), batch_size=48, shuffle=True, num_workers=0),
               "val": DataLoader(Patches(*data["val"], train=False), batch_size=96),
               "test": DataLoader(Patches(*data["test"], train=False), batch_size=96)}
    model = MultispectralEfficientNet(num_classes=k, in_channels=4, pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(loaders["train"]))
    counts = np.bincount(data["train"][1], minlength=k).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / k, device=dev)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = {"val_f1": -1}
    for ep in range(epochs):
        model.train()
        t = time.time()
        for x, y in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x.to(dev)), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        va, vf, _ = evaluate(model, loaders["val"], dev, k)
        print(f"  [{norm}] epoch {ep + 1:2d}/{epochs}  loss {loss.item():.3f}  val acc {va:.3f} f1 {vf:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        if vf > best["val_f1"]:
            best = {"val_f1": vf, "val_acc": va, "epoch": ep + 1,
                    "state": {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}}
    model.load_state_dict(best.pop("state"))
    ta, tf, cm = evaluate(model, loaders["test"], dev, k)
    best.update({"norm": norm, "test_acc": ta, "test_f1": tf, "test_confusion": cm.tolist(),
                 "train_patches": len(data["train"][1]), "val_patches": len(data["val"][1]),
                 "test_patches": len(data["test"][1])})
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="stage_stratified_flight_disjoint")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--norm", default="both", choices=["instance", "flight", "both"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts = load_split(a.split)
    print(f"device {dev}; split {a.split}: " + ", ".join(f"{k} {len(v)} patches / {v.flight_id.nunique()} flights"
                                                          for k, v in parts.items()), flush=True)
    runs = []
    best_model, best_run = None, None
    for norm in (["instance", "flight"] if a.norm == "both" else [a.norm]):
        model, run = train_one(parts, norm, a.epochs, dev)
        print(f"  [{norm}] best epoch {run['epoch']}: val f1 {run['val_f1']:.3f} | TEST acc {run['test_acc']:.3f} "
              f"f1 {run['test_f1']:.3f}", flush=True)
        runs.append(run)
        if best_run is None or run["val_f1"] > best_run["val_f1"]:        # chosen on validation, not test
            best_model, best_run = model, run
    out = ROOT / "models" / "classifiers" / "stage_flightdisjoint.pt"
    torch.save({"model_state_dict": best_model.state_dict(), "norm": best_run["norm"], "crop": 192,
                "classes": [STAGE_CLASSES[i] for i in range(len(STAGE_CLASSES))], "in_channels": 4}, out)
    card = {"model": "EfficientNet-B0, 4-band (G, R, RE, NIR) input, ImageNet-initialised",
            "task": "paddy growth stage from a 224 px multispectral drone patch",
            "classes": [STAGE_CLASSES[i] for i in range(len(STAGE_CLASSES))],
            "training_data": "data/manifests/patch_manifest.csv (2 fields, 20 flights), flight-disjoint split "
                             f"'{a.split}'", "normalisation": best_run["norm"],
            "evaluation": {kk: best_run[kk] for kk in ("test_acc", "test_f1", "val_f1", "epoch",
                                                     "train_patches", "val_patches", "test_patches")},
            "test_flights_unseen": True, "chance_accuracy": 1.0 / len(STAGE_CLASSES),
            "limitations": "Two fields, one season, one sensor. Stage labels are per flight, not per patch. "
                           "Accuracy on other farms, sensors or seasons is unknown until tested there.",
            "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": dev}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "stage_training.json").write_text(json.dumps({"split": a.split, "runs": runs}, indent=2))
    print(f"saved {out.relative_to(ROOT)} (norm={best_run['norm']}, test acc {best_run['test_acc']:.3f})")


if __name__ == "__main__":
    main()
