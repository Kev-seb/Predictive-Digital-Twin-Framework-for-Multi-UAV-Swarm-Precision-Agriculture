"""
crop_stage.py - paddy growth stage, two ways, with honest reliability.

1. From the transplanting date (primary): the field's real daily temperatures since transplanting
   (Open-Meteo archive, then forecast; a 28 C climatology if offline) give growing degree-days, and the
   calibrated thresholds (outputs/calibration/paddy_params.json, scripts/calibrate_paddy.py) give the
   stage. On the project's 20 real flights this classified every flight correctly.
2. From the image (secondary): the flight-disjoint EfficientNet (research/09_train_stage_model.py)
   on 224 px patches of the uploaded image, averaged. Its tested accuracy on unseen flights is read
   from the model card and always shown next to its answer.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "outputs" / "calibration" / "paddy_params.json"
STAGE_MODEL = ROOT / "models" / "classifiers" / "stage_flightdisjoint.pt"


def _params():
    from src.twin_core.paddy_model import PaddyParams
    return PaddyParams.from_json(PARAMS) if PARAMS.exists() else PaddyParams()


def _daily_temps(lat: float, lon: float, start: dt.date, end: dt.date) -> tuple[list[float], str]:
    """Mean daily temperature for days start+1 .. end."""
    n = (end - start).days
    if n <= 0:
        return [], "none"
    try:
        from src.twin_core.flight_replay import archive_weather
        w = archive_weather(lat, lon)(start, end)
        temps = [float(d.get("temperature", 28.0)) for d in w]
        if len(temps) >= n - 7:                      # the archive lags a few days: fill with the forecast
            missing = n - len(temps)
            if missing > 0:
                from src.twin_core.weather_feed import get_weather
                fc, _ = get_weather(lat, lon, missing, source="auto")
                temps += [float(d.get("temperature", 28.0)) for d in fc]
            return temps[:n], "Open-Meteo archive"
    except Exception:
        pass
    return [28.0] * n, "28 °C climatology (offline)"


def stage_from_transplanting(transplanted: dt.date, lat: float, lon: float,
                             today: Optional[dt.date] = None) -> Dict[str, Any]:
    from src.twin_core.paddy_model import STAGES, daily_gdd, stage_for_gdd
    p = _params()
    today = today or dt.date.today()
    temps, source = _daily_temps(lat, lon, transplanted, today)
    gdd = float(sum(daily_gdd(p, t) for t in temps))
    stage = stage_for_gdd(p, gdd)
    out = {"stage": stage, "gdd": gdd, "days_after_transplanting": (today - transplanted).days,
           "weather": source, "calibrated": PARAMS.exists()}
    i = STAGES.index(stage)
    rate = float(np.mean([daily_gdd(p, t) for t in temps[-14:]])) if temps else daily_gdd(p, 28.0)
    if i + 1 < len(STAGES):
        nxt = STAGES[i + 1]
        out.update({"next_stage": nxt, "days_to_next": max(0, round((p.gdd_onset[nxt] - gdd) / max(rate, 1e-3)))})
    else:
        out.update({"next_stage": "Harvest", "days_to_next": max(0, round((p.gdd_harvest - gdd) / max(rate, 1e-3)))})
    return out


# ------------------------------------------------------------------ image model
_MODEL = {}


def stage_model_card() -> Optional[Dict[str, Any]]:
    card = STAGE_MODEL.with_suffix(".json")
    return json.loads(card.read_text()) if card.exists() else None


def _load_model():
    if "m" not in _MODEL:
        import torch
        from src.classification.stress_classifier import MultispectralEfficientNet
        ck = torch.load(STAGE_MODEL, map_location="cpu", weights_only=False)
        m = MultispectralEfficientNet(num_classes=len(ck["classes"]), in_channels=ck.get("in_channels", 4),
                                      pretrained=False)
        m.load_state_dict(ck["model_state_dict"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.update(m=m.to(dev).eval(), ck=ck, dev=dev)
    return _MODEL["m"], _MODEL["ck"], _MODEL["dev"]


def stage_from_image(bands: Dict[str, np.ndarray], max_patches: int = 64) -> Optional[Dict[str, Any]]:
    """Average stage probabilities over up to max_patches 224 px patches of the image."""
    if not STAGE_MODEL.exists():
        return None
    import torch
    m, ck, dev = _load_model()
    x = np.stack([np.asarray(bands[k], dtype=np.float32) for k in ("green", "red", "red_edge", "nir")])
    x = np.nan_to_num(x, nan=0.0)
    lo, hi = np.percentile(x, 1, axis=(1, 2), keepdims=True), np.percentile(x, 99, axis=(1, 2), keepdims=True)
    x = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1)          # as the training patches (p1-p99)
    size = 224
    h, w = x.shape[1:]
    if h < size or w < size:                                         # small image: upscale to one patch
        import torch.nn.functional as F
        x = F.interpolate(torch.from_numpy(x)[None], size=(max(h, size), max(w, size)), mode="bilinear")[0].numpy()
        h, w = x.shape[1:]
    rows, cols = range(0, h - size + 1, size), range(0, w - size + 1, size)
    patches = [x[:, r:r + size, c:c + size] for r in rows for c in cols]
    if len(patches) > max_patches:
        idx = np.random.default_rng(0).choice(len(patches), max_patches, replace=False)
        patches = [patches[i] for i in sorted(idx)]
    p = np.stack(patches)
    mu, sd = p.mean(axis=(0, 2, 3), keepdims=True), p.std(axis=(0, 2, 3), keepdims=True) + 1e-4
    p = (p - mu) / sd                                               # 'flight' normalisation: this image's stats
    crop = ck.get("crop", 192)
    o = (size - crop) // 2
    p = p[:, :, o:o + crop, o:o + crop]
    with torch.no_grad():
        prob = torch.softmax(m(torch.from_numpy(np.ascontiguousarray(p)).to(dev)).float(), 1).mean(0).cpu().numpy()
    k = int(prob.argmax())
    return {"stage": ck["classes"][k], "confidence": float(prob[k]), "patches": len(patches),
            "probabilities": {c: float(v) for c, v in zip(ck["classes"], prob)}}
