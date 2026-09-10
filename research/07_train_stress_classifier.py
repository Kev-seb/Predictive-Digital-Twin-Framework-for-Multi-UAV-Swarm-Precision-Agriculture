"""
07_train_stress_classifier.py  — PHASE 4: STRESS PROXY CLASSIFIER

Trains & evaluates crop stress level classifier using the per-stage NDVI Z-score proxy labels:
  - 4 classes: 0 (No Stress), 1 (Low Stress), 2 (Moderate Stress), 3 (High Stress)
  - Uses fixed weight-inflated EfficientNet-B0 with Instance Normalization
  - Evaluated on Stage-Stratified Flight-Disjoint split (unseen flights)
  - Evaluates across 3 independent seeds

Outputs:
  results/stress_classifier_summary.csv
  results/stress_classifier_full.json
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

def compute_metrics(y_true, y_pred, num_classes=4):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1s.append(1.0 if np.sum(y_true == c) == 0 else 0.0)
        else:
            f1s.append(2.0 * tp / denom)
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1

class StressEfficientNet(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 4, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
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
        x = self.inst_norm(x)
        return self.backbone(x)

class StressDataset(Dataset):
    def __init__(self, patch_records: list[dict]):
        self.data = []
        self.labels = []
        for rec in patch_records:
            patch = np.load(rec["patch_file"]).astype(np.float32)  # (4, 224, 224)
            stress_level = int(rec.get("stress_level", 0))
            self.data.append(torch.from_numpy(patch))
            self.labels.append(stress_level)

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

    acc, f1 = compute_metrics(all_targets, all_preds, num_classes=4)
    return acc, f1

def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[STRESS MODEL] Running training on device: {device}")

    stress_manifest_path = DATA_DIR / "manifests" / "stress_patch_manifest.csv"
    splits_path          = DATA_DIR / "splits" / "splits_manifest.json"

    if not stress_manifest_path.exists():
        log("[INFO] Stress manifest not found. Generating stress proxy labels first...")
        from research.05_stress_proxy_generator import generate_stress_labels
        generate_stress_labels()

    records = []
    with open(stress_manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    with open(splits_path, "r") as f:
        splits = json.load(f)["primary_flight_disjoint"]

    train_recs = [r for r in records if r["flight_id"] in splits["train_flights"]]
    val_recs   = [r for r in records if r["flight_id"] in splits["val_flights"]]
    test_recs  = [r for r in records if r["flight_id"] in splits["test_flights"]]

    log(f"[DATA] Stress Classifier — Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    log("\n[CONFIG] Pre-caching memory dataset for Stress Proxy Classifier...")
    t0 = time.time()
    train_ds = StressDataset(train_recs)
    val_ds   = StressDataset(val_recs)
    test_ds  = StressDataset(test_recs)
    log(f"  Pre-cached in {time.time()-t0:.2f}s")

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    seeds = [0, 1, 2]
    epochs = 3
    accs, f1s = [], []

    for seed in seeds:
        set_seed(seed)
        model = StressEfficientNet(num_classes=4, in_channels=4, pretrained=True).to(device)
        acc, f1 = train_and_eval(model, train_loader, val_loader, test_loader, device, epochs=epochs)
        accs.append(acc)
        f1s.append(f1)
        log(f"  Seed {seed}: Test Acc = {acc*100:.2f}%, F1-macro = {f1:.4f}")

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1, std_f1   = np.mean(f1s), np.std(f1s)

    summary_csv = RESULTS_DIR / "stress_classifier_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "mean_acc", "std_acc", "mean_f1", "std_f1", "raw_accs", "raw_f1s"])
        writer.writeheader()
        writer.writerow({
            "task": "Crop Stress Proxy Classification (4-class)",
            "mean_acc": f"{mean_acc*100:.2f}%",
            "std_acc": f"{std_acc*100:.2f}%",
            "mean_f1": f"{mean_f1:.4f}",
            "std_f1": f"{std_f1:.4f}",
            "raw_accs": str(accs),
            "raw_f1s": str(f1s)
        })

    log(f"\n[STRESS MODEL] Results saved to {summary_csv}")
    log(f"  => Summary: Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, F1 = {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    run_experiments()
