"""
field_image.py - load a multispectral field GeoTIFF robustly, keeping true reflectance.

The older loader (multispectral_loader._load_with_rasterio) stretched every band to 0-1 on its own
percentiles, which changes NIR/Red ratios and so every vegetation index. This one:
  * keeps reflectance: 0-1 data as is; 0-10,000 or 0-65,535 integers rescaled (and flagged);
  * masks no-data (the file's nodata value, the dataset mask, all-zero pixels) and fills those pixels with
    the band median so downstream maths stays finite; the mask is kept on image.valid_mask;
  * downsamples very large images on read (max_side), scaling the georeference to match;
  * checks the band count and reports what it found: load_field_image() -> (image, report).
Band order: G, R, RE, NIR (bands 1-4), unless the file's band descriptions name them.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.core.multispectral_loader import MultispectralImage

NAMES = {"green": ("green", "g"), "red": ("red", "r"), "red_edge": ("rededge", "red edge", "red_edge", "re"),
         "nir": ("nir", "near infrared", "near-infrared")}


def _band_order(descriptions) -> list:
    found = {}
    for i, d in enumerate(descriptions or []):
        key = (d or "").strip().lower()
        for name, aliases in NAMES.items():
            if key in aliases and name not in found:
                found[name] = i
    if len(found) == 4:
        return [found[k] for k in ("green", "red", "red_edge", "nir")]
    return [0, 1, 2, 3]


def load_field_image(path, max_side: int = 2048) -> Tuple[MultispectralImage, Dict[str, Any]]:
    import rasterio
    from rasterio.enums import Resampling

    warnings = []
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"This image has {src.count} band(s); a 4-band Green/Red/Red-edge/NIR GeoTIFF is needed.")
        h, w = src.height, src.width
        scale = max(1.0, max(h, w) / max_side)
        out_h, out_w = int(round(h / scale)), int(round(w / scale))
        order = _band_order(src.descriptions)
        idx = [i + 1 for i in order]
        data = src.read(idx, out_shape=(4, out_h, out_w), resampling=Resampling.average).astype(np.float32)
        mask = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
        if src.nodata is not None:
            mask &= ~np.any(data == src.nodata, axis=0)
        transform = src.transform * src.transform.scale(w / out_w, h / out_h)
        crs = str(src.crs) if src.crs else None
        dtype = src.dtypes[0]
    mask &= np.any(data != 0, axis=0) & np.all(np.isfinite(data), axis=0)
    if scale > 1.0:
        warnings.append(f"Large image ({w} × {h} px) read at {out_w} × {out_h} px to keep the app fast.")
    valid = data[:, mask]
    if valid.size == 0:
        raise ValueError("The image has no valid pixels (everything is no-data).")
    top = float(np.percentile(valid, 99.5))
    if top <= 1.5:
        units = "reflectance (0-1)"
    elif top <= 12_000:
        data /= 10_000.0
        units = "reflectance × 10,000 (rescaled)"
    else:
        data /= 65_535.0
        units = f"raw digital numbers ({dtype}, rescaled to 0-1)"
        warnings.append("The values look like raw sensor numbers, not calibrated reflectance. Indices will be "
                        "relative only; calibrate the image (reflectance panel / sunlight sensor) for true values.")
    nodata_pct = float(100.0 * (1 - mask.mean()))
    for b in range(4):                                      # fill no-data with the band median
        med = float(np.median(data[b][mask]))
        data[b][~mask] = med
    bands = {"green": data[0], "red": data[1], "red_edge": data[2], "nir": data[3]}
    img = MultispectralImage(bands, crs=crs, transform=transform, source_path=str(path))
    img.valid_mask = mask
    nir, red = data[3][mask], data[1][mask]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    p95 = float(np.percentile(ndvi, 95))
    if p95 < 0.4:
        warnings.append(f"Even the greenest part of the image has NDVI {p95:.2f} (healthy paddy is usually 0.6-0.85). "
                        "Either the field has little green canopy, or the image is not radiometrically "
                        "calibrated: check the reflectance calibration before trusting stress results.")
    if nodata_pct > 1:
        warnings.append(f"{nodata_pct:.0f}% of the image is outside the field / no-data; it is excluded.")
    report = {"original_px": [w, h], "used_px": [out_w, out_h], "downsampled": scale > 1.0, "units": units,
              "nodata_pct": round(nodata_pct, 2), "band_order": ["green", "red", "red_edge", "nir"],
              "band_indexes": idx, "ndvi_p5": float(np.percentile(ndvi, 5)), "ndvi_p95": p95,
              "warnings": warnings}
    return img, report
