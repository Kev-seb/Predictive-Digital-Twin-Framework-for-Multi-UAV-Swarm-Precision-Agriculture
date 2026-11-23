"""
10_train_disease_model.py - rice leaf disease classifier from public CC BY 4.0 photo datasets (GPU).

    python research/10_train_disease_model.py [--epochs 12]

Data (download with the links in data/external/rice_disease/*/SOURCE.txt):
  data/external/rice_disease/sethy_2020/images/Rice Leaf Disease Images/<class>/...
  data/external/rice_disease/riceleafdisease_bd5/images/RiceLeafBD/<class>/...      (field photos)

Honest evaluation:
  * near-duplicates (the same leaf rotated / flipped / re-saved) are grouped by a perceptual hash that is
    invariant to the 8 flips/rotations, and a whole group goes to train, validation OR test, never across;
  * the test score is also reported on the field photos alone (BD5), the closest to a farmer's phone photo.

Outputs: models/disease/rice_disease_effb0.pt and .json (model card with dataset credits),
         results/disease_training.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "rice_disease"
CACHE = DATA / "cache_288"
OUT = ROOT / "models" / "disease"

CLASSES = ["Healthy", "Bacterial leaf blight", "Blast", "Brown spot", "Narrow brown spot", "Sheath blight", "Tungro"]
FOLDER_TO_CLASS = {
    "bacterialblight": "Bacterial leaf blight", "blast": "Blast", "brownspot": "Brown spot", "tungro": "Tungro",
    "narrow_brown_spot": "Narrow brown spot", "normal_leaf": "Healthy", "sheath_blight": "Sheath blight",
}
SOURCES = {
    "sethy_2020": ("Rice Leaf Disease Image Samples (Sethy, Mendeley Data, CC BY 4.0)",
                   DATA / "sethy_2020" / "images" / "Rice Leaf Disease Images"),
    "riceleafdisease_bd5": ("RiceLeafDisease-BD5 (Mendeley Data, DOI 10.17632/sjkptpgv79.2, CC BY 4.0)",
                            DATA / "riceleafdisease_bd5" / "images" / "RiceLeafBD"),
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect():
    items = []
    for src, (_, root) in SOURCES.items():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            cls = FOLDER_TO_CLASS.get(d.name.lower())
            if cls is None:
                continue
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EXTS:
                    items.append({"path": f, "cls": CLASSES.index(cls), "source": src})
    return items


def _cache_one(item):
    import hashlib
    out = CACHE / item["source"] / f"{hashlib.md5(str(item['path']).encode()).hexdigest()[:16]}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(item["path"]).convert("RGB")
        im.thumbnail((288, 288 * 4)) if im.width < im.height else im.thumbnail((288 * 4, 288))
        s = min(im.size)
        if s > 288:
            im = im.resize((round(im.width * 288 / s), round(im.height * 288 / s)))
        im.save(out, quality=92)
    small = np.asarray(Image.open(out).convert("L").resize((9, 9)), dtype=np.float32)
    return out, small


def dihedral_hash(small: np.ndarray) -> int:
    """Difference hash, minimised over the 8 rotations/flips: the same leaf gets the same key."""
    keys = []
    for k in range(4):
        for flip in (False, True):
            a = np.rot90(small, k)
            a = a[:, ::-1] if flip else a
            bits = (a[:, 1:] > a[:, :-1]).flatten()[:64]
            keys.append(int("".join("1" if b else "0" for b in bits), 2))
    return min(keys)


def split_groups(items, seed=0, val=0.15, test=0.15):
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[(it["source"], it["cls"], it["hash"])].append(i)
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    by_cls = defaultdict(list)
    for key, idx in groups.items():
        by_cls[(key[0], key[1])].append(idx)
    for _, glist in by_cls.items():                    # stratified by source and class
        order = rng.permutation(len(glist))
        n = len(glist)
        n_te, n_va = max(1, round(n * test)), max(1, round(n * val))
        for j, gi in enumerate(order):
            part = "test" if j < n_te else ("val" if j < n_te + n_va else "train")
            parts[part].extend(glist[gi])
    return parts, len(groups)


class Leaves(Dataset):
    def __init__(self, items, train):
        import torchvision.transforms as T
        self.items = items
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                             T.RandomVerticalFlip(), T.RandomRotation(20),
                             T.ColorJitter(0.3, 0.3, 0.3, 0.03), T.ToTensor(), norm]) if train else \
            T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return self.tf(Image.open(it["cache"]).convert("RGB")), it["cls"]


def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for x, y in loader:
            ps += model(x.to(dev)).argmax(1).cpu().tolist()
            ys += y.tolist()
    return np.array(ys), np.array(ps)


def scores(y, p):
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    present = [c for c in range(k) if cm[c].sum()]
    f1 = [2 * cm[c, c] / max(1, 2 * cm[c, c] + cm[:, c].sum() - cm[c, c] + cm[c].sum() - cm[c, c]) for c in present]
    return {"accuracy": float((y == p).mean()) if len(y) else None, "macro_f1": float(np.mean(f1)) if f1 else None,
            "per_class_recall": {CLASSES[c]: float(cm[c, c] / cm[c].sum()) for c in present},
            "n": int(len(y)), "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    import torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = collect()
    print(f"{len(items)} images; caching at 288 px and hashing...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for it, (cache, small) in zip(items, ex.map(_cache_one, items)):
            it["cache"], it["hash"] = cache, dihedral_hash(small)
    parts, n_groups = split_groups(items)
    print(f"{n_groups} distinct leaves after grouping near-duplicates ({len(items) - n_groups} duplicates); "
          + ", ".join(f"{k} {len(v)}" for k, v in parts.items()), flush=True)
    sets = {k: [items[i] for i in v] for k, v in parts.items()}
    counts = Counter(it["cls"] for it in sets["train"])
    weights = torch.tensor([len(sets["train"]) / max(1, counts[c]) / len(CLASSES) for c in range(len(CLASSES))],
                           device=dev)
    loaders = {"train": DataLoader(Leaves(sets["train"], True), batch_size=48, shuffle=True, num_workers=4,
                                   persistent_workers=True),
               "val": DataLoader(Leaves(sets["val"], False), batch_size=96, num_workers=2),
               "test": DataLoader(Leaves(sets["test"], False), batch_size=96, num_workers=2)}
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * len(loaders["train"]))
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best_f1, best_state, best_ep = -1, None, 0
    for ep in range(a.epochs):
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
        v = scores(*evaluate(model, loaders["val"], dev))
        print(f"  epoch {ep + 1:2d}/{a.epochs} loss {loss.item():.3f}  val acc {v['accuracy']:.3f} "
              f"f1 {v['macro_f1']:.3f} ({time.time() - t:.0f}s)", flush=True)
        if v["macro_f1"] > best_f1:
            best_f1, best_ep = v["macro_f1"], ep + 1
            best_state = {k: t_.detach().cpu().clone() for k, t_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    y, p = evaluate(model, loaders["test"], dev)
    src = np.array([it["source"] for it in sets["test"]])
    res = {"test_all": scores(y, p), "test_field_photos_bd5": scores(y[src == "riceleafdisease_bd5"],
                                                                     p[src == "riceleafdisease_bd5"]),
           "test_sethy": scores(y[src == "sethy_2020"], p[src == "sethy_2020"]),
           "best_epoch": best_ep, "val_macro_f1": best_f1, "images": len(items), "distinct_leaves": n_groups,
           "split_sizes": {k: len(v) for k, v in parts.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES, "arch": "efficientnet_b0",
                "input": 224, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
               OUT / "rice_disease_effb0.pt")
    card = {"model": "EfficientNet-B0 (ImageNet-initialised), RGB leaf photo, 224 px",
            "task": "rice leaf disease from a close-up photo (phone or camera), not from drone imagery",
            "classes": CLASSES,
            "evaluation": {"test_accuracy": res["test_all"]["accuracy"], "test_macro_f1": res["test_all"]["macro_f1"],
                           "field_photos_accuracy": res["test_field_photos_bd5"]["accuracy"],
                           "field_photos_macro_f1": res["test_field_photos_bd5"]["macro_f1"],
                           "per_class_recall": res["test_all"]["per_class_recall"],
                           "split": "near-duplicate leaves grouped; whole groups in train, validation or test"},
            "training_data": [v[0] for v in SOURCES.values()],
            "credits": "Trained on 'Rice Leaf Disease Image Samples' (P. K. Sethy, Mendeley Data) and "
                       "'RiceLeafDisease-BD5' (Mendeley Data, DOI 10.17632/sjkptpgv79.2), both CC BY 4.0.",
            "limitations": "Photos from Bangladesh and India; lab-style and field photos of single leaves. Not tested "
                           "on other varieties, cameras or whole-plant photos. Narrow brown spot and sheath blight "
                           "come from one source only. A diagnosis should be confirmed by an agronomist.",
            "trained_at": time.strftime("%Y-%m-%d %H:%M"), "device": dev}
    (OUT / "rice_disease_effb0.json").write_text(json.dumps(card, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "disease_training.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"test_acc": res["test_all"]["accuracy"], "test_f1": res["test_all"]["macro_f1"],
                      "field_photos_acc": res["test_field_photos_bd5"]["accuracy"],
                      "per_class_recall": res["test_all"]["per_class_recall"]}, indent=1))


if __name__ == "__main__":
    main()
