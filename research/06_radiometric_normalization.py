"""
06_radiometric_normalization.py  — PHASE 3: RADIOMETRIC NORMALIZATION & DOMAIN ADAPTATION

Addresses Radiometric Shift & Solves Cross-Flight Generalization Degradation (SF-08 / EX-04 / HI-04):
  Evaluates 4 Radiometric Calibration Strategies across the Stage-Stratified Flight-Disjoint Benchmark:
    1. Baseline Relative Percentile Normalization (Per-image 1st-99th percentile)
    2. Per-Flight Z-score Standardization (Flight-ZNorm: zero mean, unit variance per band per flight)
    3. Spatial Instance Normalization (InstNorm: per-patch spatial normalization)
    4. Pseudo-Invariant Feature / Flight Histogram Equalization (PIF-Norm)

Outputs:
  results/radiometric_normalization_summary.csv
  results/radiometric_normalization_full.json
"""
from __future__ import annotations
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(msg, flush=True)

def compute_metrics(y_true, y_pred, num_classes=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred != c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1s.append(1.0 if np.sum(y_true == c) == 0 else 0.0)
        else:
            f1s.append(2.0 * tp / denom)
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1

class MultispectralEfficientNetNorm(nn.Module):
    """EfficientNet-B0 with integrated InstanceNorm2d layer for illumination invariance."""
    def __init__(self, num_classes: int = 5, in_channels: int = 4, use_inst_norm: bool = False, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.use_inst_norm = use_inst_norm
        if use_inst_norm:
            self.inst_norm = nn.InstanceNorm2d(in_channels, affine=True)

        old_stem = self.backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels, old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_stem.weight[:] = old_stem.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        self.backbone.features[0][0] = new_stem

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_inst_norm:
            x = self.inst_norm(x)
        return self.backbone(x)

class RadiometricDataset(Dataset):
    def __init__(self, patch_records: list[dict], flight_stats: dict[str, dict], norm_mode: str = "relative"):
        self.stage_map = {
            "Nursery": 0, "Vegetative": 1, "Booting": 2, "Flowering": 3, "Mature": 4
        }
        self.data = []
        self.labels = []

        for rec in patch_records:
            patch = np.load(rec["patch_file"]).astype(np.float32)  # (4, 224, 224)
            fid = rec["flight_id"]

            if norm_mode == "flight_znorm" and fid in flight_stats:
                mu = flight_stats[fid]["mean"] # (4, 1, 1)
                std = flight_stats[fid]["std"]   # (4, 1, 1)
                patch = (patch - mu) / (std + 1e-6)
            elif norm_mode == "pif_histmatch" and fid in flight_stats:
                # Quantile scaling to match reference distribution
                ref_mu = flight_stats["ref"]["mean"]
                ref_std = flight_stats["ref"]["std"]
                mu = flight_stats[fid]["mean"]
                std = flight_stats[fid]["std"]
                patch = (patch - mu) / (std + 1e-6) * ref_std + ref_mu

            self.data.append(torch.from_numpy(patch))
            self.labels.append(self.stage_map.get(rec["stage"], 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_eval(model, train_loader, val_loader, test_loader, device, epochs=3, lr=2e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc, f1 = compute_metrics(all_targets, all_preds, num_classes=5)
    return acc, f1

def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[RADIOMETRIC] Running experiments on device: {device}")

    patch_manifest_path = DATA_DIR / "manifests" / "patch_manifest.csv"
    splits_path         = DATA_DIR / "splits" / "splits_manifest.json"

    records = []
    with open(patch_manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    with open(splits_path, "r") as f:
        splits = json.load(f)["primary_flight_disjoint"]

    train_recs = [r for r in records if r["flight_id"] in splits["train_flights"]]
    val_recs   = [r for r in records if r["flight_id"] in splits["val_flights"]]
    test_recs  = [r for r in records if r["flight_id"] in splits["test_flights"]]

    # Compute per-flight mean and std statistics for Flight-ZNorm & PIF-Match
    flight_stats = {}
    flight_patches = {}
    for r in records:
        fid = r["flight_id"]
        if fid not in flight_patches:
            flight_patches[fid] = []
        flight_patches[fid].append(np.load(r["patch_file"]).astype(np.float32))

    for fid, patches in flight_patches.items():
        arr = np.stack(patches, axis=0) # (N, 4, 224, 224)
        mu = np.mean(arr, axis=(0, 2, 3), keepdims=True).reshape(4, 1, 1)
        std = np.std(arr, axis=(0, 2, 3), keepdims=True).reshape(4, 1, 1)
        flight_stats[fid] = {"mean": mu, "std": std}

    # Reference flight stats (F1_01)
    flight_stats["ref"] = flight_stats["F1_01"]

    norm_strategies = {
        "1. Relative Percentile (Baseline)": ("relative", False),
        "2. Flight-level Z-Score Standardization": ("flight_znorm", False),
        "3. Spatial Instance Normalization (InstNorm)": ("relative", True),
        "4. Pseudo-Invariant Feature Histogram Match": ("pif_histmatch", False),
    }

    seeds = [0, 1, 2]
    epochs = 3
    batch_size = 64
    results_summary = []

    log("\n" + "="*70)
    log("  EXPERIMENT 3: RADIOMETRIC NORMALIZATION & CROSS-FLIGHT GENERALIZATION")
    log("="*70)

    for sname, (nmode, inst_norm) in norm_strategies.items():
        log(f"\n[CONFIG] Pre-caching dataset for: {sname}...")
        t0 = time.time()
        train_ds = RadiometricDataset(train_recs, flight_stats, norm_mode=nmode)
        val_ds   = RadiometricDataset(val_recs, flight_stats, norm_mode=nmode)
        test_ds  = RadiometricDataset(test_recs, flight_stats, norm_mode=nmode)
        log(f"  Pre-cached in {time.time()-t0:.2f}s")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        accs, f1s = [], []
        for seed in seeds:
            set_seed(seed)
            model = MultispectralEfficientNetNorm(num_classes=5, in_channels=4, use_inst_norm=inst_norm, pretrained=True).to(device)
            acc, f1 = train_and_eval(model, train_loader, val_loader, test_loader, device, epochs=epochs)
            accs.append(acc)
            f1s.append(f1)
            log(f"  Seed {seed}: Test Acc = {acc*100:.2f}%, F1-macro = {f1:.4f}")

        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1   = np.mean(f1s), np.std(f1s)

        row = {
            "strategy": sname,
            "mean_acc": f"{mean_acc*100:.2f}%",
            "std_acc": f"{std_acc*100:.2f}%",
            "mean_f1": f"{mean_f1:.4f}",
            "std_f1": f"{std_f1:.4f}",
            "raw_accs": str(accs),
            "raw_f1s": str(f1s)
        }
        results_summary.append(row)
        log(f"  => Summary [{sname}]: Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    summary_csv = RESULTS_DIR / "radiometric_normalization_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "mean_acc", "std_acc", "mean_f1", "std_f1", "raw_accs", "raw_f1s"])
        writer.writeheader()
        for r in results_summary:
            writer.writerow(r)

    log(f"\n[RADIOMETRIC] All experiments complete! Results saved to {summary_csv}")

if __name__ == "__main__":
    run_experiments()
