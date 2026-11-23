"""
leaf_disease.py - rice leaf disease from a close-up photo (research/10_train_disease_model.py).

predict(image) -> top classes with probabilities. Trained on two CC BY 4.0 datasets (credits in the
model card, models/disease/rice_disease_effb0.json). A photo is not a diagnosis: the app shows the
model's tested accuracy next to every answer and suggests confirming with an agronomist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "disease" / "rice_disease_effb0.pt"

# What each class means for the twin's "fungus" layer, and a short, cautious note for the farmer.
INFO = {
    "Healthy": {"fungal": 0.0, "kind": "healthy",
                "note": "No disease signs found on this leaf."},
    "Blast": {"fungal": 1.0, "kind": "fungal (Magnaporthe oryzae)",
              "note": "Diamond-shaped lesions with grey centres. Spreads fast in humid, cool nights; check nearby "
                      "plants and ask about a blast fungicide."},
    "Brown spot": {"fungal": 1.0, "kind": "fungal (Bipolaris oryzae)",
                   "note": "Round brown spots; often linked to nutrient-poor or water-stressed soil."},
    "Narrow brown spot": {"fungal": 1.0, "kind": "fungal (Cercospora janseniana)",
                          "note": "Short, narrow brown streaks along the leaf; usually late in the season."},
    "Sheath blight": {"fungal": 1.0, "kind": "fungal (Rhizoctonia solani)",
                      "note": "Oval greenish-grey lesions on the sheath near the water line; favoured by dense "
                              "canopy and high nitrogen."},
    "Bacterial leaf blight": {"fungal": 0.0, "kind": "bacterial (Xanthomonas oryzae)",
                              "note": "Yellow-to-white lesions from the leaf tip; fungicides do not help. Drain "
                                      "the field if possible and avoid extra nitrogen."},
    "Tungro": {"fungal": 0.0, "kind": "viral, spread by green leafhoppers",
               "note": "Yellow-orange leaves and stunting; control the leafhoppers and remove infected plants."},
}

_M: Dict[str, Any] = {}


def available() -> bool:
    return MODEL.exists()


def card() -> Optional[Dict[str, Any]]:
    p = MODEL.with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else None


def _load():
    if "m" not in _M:
        import torch
        import torch.nn as nn
        import torchvision
        ck = torch.load(MODEL, map_location="cpu", weights_only=False)
        m = torchvision.models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(ck["classes"]))
        m.load_state_dict(ck["model_state_dict"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _M.update(m=m.to(dev).eval(), ck=ck, dev=dev)
    return _M["m"], _M["ck"], _M["dev"]


def predict(image, top: int = 3) -> List[Dict[str, Any]]:
    """image: a PIL image (or path). Averages the photo and its mirror for a steadier answer."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    m, ck, dev = _load()
    img = Image.open(image).convert("RGB") if not hasattr(image, "convert") else image.convert("RGB")
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(ck["normalize"]["mean"], ck["normalize"]["std"])])
    x = tf(img)[None]
    x = torch.cat([x, torch.flip(x, dims=[3])]).to(dev)
    with torch.no_grad():
        prob = torch.softmax(m(x).float(), 1).mean(0).cpu().numpy()
    order = prob.argsort()[::-1][:top]
    return [{"disease": ck["classes"][i], "probability": float(prob[i]), **INFO.get(ck["classes"][i], {})}
            for i in order]


def explain(image, class_name: Optional[str] = None):
    """Grad-CAM: where in the photo the model looked for its answer. Returns an RGB PIL overlay."""
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    m, ck, dev = _load()
    img = Image.open(image).convert("RGB") if not hasattr(image, "convert") else image.convert("RGB")
    view = T.Compose([T.Resize(256), T.CenterCrop(224)])(img)
    x = T.Compose([T.ToTensor(), T.Normalize(ck["normalize"]["mean"], ck["normalize"]["std"])])(view)[None].to(dev)
    acts, grads = {}, {}
    layer = m.features[-1]
    h1 = layer.register_forward_hook(lambda _m, _i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: grads.__setitem__("g", go[0]))
    try:
        x.requires_grad_(True)
        out = m(x)
        k = ck["classes"].index(class_name) if class_name in ck["classes"] else int(out.argmax(1))
        m.zero_grad()
        out[0, k].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1))[0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-6)
    heat = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(view.size, Image.BILINEAR), dtype=np.float32) / 255
    base = np.asarray(view, dtype=np.float32) / 255
    red = np.stack([np.ones_like(heat), 0.55 * (1 - heat), 0.1 * (1 - heat)], -1)    # yellow -> red
    blend = base * (1 - 0.55 * heat[..., None]) + red * 0.55 * heat[..., None]
    return Image.fromarray(np.uint8(np.clip(blend, 0, 1) * 255))


def fungal_probability(results: List[Dict[str, Any]]) -> float:
    """Probability mass on fungal classes among the returned results (a lower bound)."""
    return float(sum(r["probability"] * r.get("fungal", 0.0) for r in results))
