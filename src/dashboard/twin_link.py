"""
twin_link.py - connect the dashboard's uploaded field to the live twin API (digital_twin_api.py).

Upload -> POST /field/from-image: the twin, the "real" field, the drones and the UE5 level all
switch to the uploaded field. Remove -> DELETE /field/user: back to the simulated field.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

API = f"http://{os.getenv('UAV_TWIN_API_HOST', '127.0.0.1')}:{os.getenv('UAV_TWIN_API_PORT', '8008')}"
MAX_SIDE = 96          # bands are downsampled to at most 96 x 96 before sending (the twin grid is 40 x 40)


def _request(method: str, path: str, body=None, timeout: float = 30.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _downsample(a: np.ndarray) -> list:
    a = np.asarray(a, dtype=np.float32)
    step = max(1, int(np.ceil(max(a.shape) / MAX_SIDE)))
    h, w = (a.shape[0] // step) * step, (a.shape[1] // step) * step
    if step > 1:
        a = np.nanmean(a[:h, :w].reshape(h // step, step, w // step, step), axis=(1, 3))
    a = np.round(a, 4)
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a]


def footprint_m(img) -> tuple[float, float] | None:
    """Ground (width, height) of the image in metres from its georeference, or None."""
    t, crs = getattr(img, "transform", None), getattr(img, "crs", None)
    if t is None or not crs:
        return None
    w_units, h_units = abs(t.a) * img.width, abs(t.e) * img.height
    if "4326" in str(crs):                       # degrees -> metres at the field's latitude
        lat = t.f + t.e * img.height / 2
        return w_units * 111_320.0 * float(np.cos(np.radians(lat))), h_units * 110_540.0
    return w_units, h_units                      # projected CRS: already metres


def push_field(img, name: str, geo=None, growth_stage: str | None = None, mode: str = "new") -> tuple[bool, str]:
    """Make the uploaded image the twin's field ('new'), or a new flight of it ('update'). (ok, message)."""
    body = {"name": name, "bands": {k: _downsample(img.bands[k]) for k in ("green", "red", "red_edge", "nir")},
            "growth_stage": growth_stage, "mode": mode}
    if geo:
        body["center_lat"], body["center_lon"] = float(geo[0]), float(geo[1])
    fp = footprint_m(img)
    if fp:
        body["width_m"], body["height_m"] = float(fp[0]), float(fp[1])
    try:
        r = _request("POST", "/field/from-image", body)
    except urllib.error.HTTPError as e:
        if e.code == 409 and mode == "update":
            return push_field(img, name, geo, growth_stage, "new")      # the API restarted: start the field again
        return False, f"The twin API rejected the field (HTTP {e.code}): {e.read().decode()[:200]}"
    except Exception:
        return False, "The live twin is not running, so it still shows the simulated field. Start it with " \
                      "start_demo.bat, then press 'Send to live twin' on the Live twin & drones page."
    if r.get("mode") == "update":
        u = r.get("ndvi_uncertainty", {})
        return True, (f"Flight {r.get('flights')} of {r.get('name')} assimilated by the live twin "
                      f"(twin uncertainty {u.get('before', 0):.3f} → {u.get('after', 0):.3f}).")
    g = r.get("grid", {})
    size = f" ({g['area_ha']:.2f} ha grid, {g['cell_size_m']:.1f} m cells)" if g.get("area_ha") else ""
    note = f" {r['size_note']}" if r.get("size_note") else ""
    return True, (f"The live twin now tracks {r.get('name', name)}{size}: the twin, the drones and the Unreal "
                  f"scene use this field.{note}")


def forget_field() -> None:
    try:
        _request("DELETE", "/field/user")
    except Exception:
        pass


def twin_source() -> dict | None:
    try:
        return _request("GET", "/field/source", timeout=3.0)
    except Exception:
        return None
