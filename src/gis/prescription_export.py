"""
prescription_export.py - variable-rate prescription maps that field machines can load.

From the treatment plan's zone prescriptions and the zone label image (same size as the uploaded
GeoTIFF), each zone is traced as polygons with the image's own georeference and given numeric rates:

    ZONE, NAME, N_KG_HA, P_KG_HA, K_KG_HA, IRR_MM, FUNG_L_HA, RISK

Formats (all in EPSG:4326):
    GeoJSON                    GIS tools, web maps
    Shapefile (.zip)           DJI Terra / Agras, John Deere Operations Center, most VRA controllers
    KML                        Google Earth (zones coloured by fungicide rate)
    CSV                        spreadsheets
    ISOXML TASKDATA (.zip)     ISOBUS terminals; a basic ISO 11783-10 task with one treatment zone per
                               management zone (fungicide L/ha and nitrogen kg/ha set points). Test it on
                               your terminal before field use.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np

RATE_FIELDS = ("N_KG_HA", "P_KG_HA", "K_KG_HA", "IRR_MM", "FUNG_L_HA")


def _num(text: str, unit_pattern: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*" + unit_pattern, text or "", re.I)
    return float(m.group(1)) if m else 0.0


def zone_rates(prescriptions) -> Dict[int, Dict[str, Any]]:
    """Numeric rates per zone from ZonePrescription objects (dosages are text like '120 kg/ha N')."""
    out = {}
    for zp in prescriptions:
        rec = {r.category: r.dosage_rate for r in zp.recommendations}
        nut = rec.get("Nutrient", "")
        out[int(zp.zone_id)] = {
            "ZONE": int(zp.zone_id), "NAME": zp.zone_name[:40],
            "N_KG_HA": _num(nut, r"kg/ha\s*N\b"), "P_KG_HA": _num(nut, r"kg/ha\s*P"),
            "K_KG_HA": _num(nut, r"kg/ha\s*K"), "IRR_MM": _num(rec.get("Irrigation", ""), r"mm"),
            "FUNG_L_HA": _num(rec.get("Fungicide", ""), r"L/ha"),
            "RISK": round(float(zp.fungal_risk_prob), 3), "AREA_PCT": round(float(zp.area_pct), 1),
        }
    return out


def _block_majority(labels: np.ndarray, block: int) -> np.ndarray:
    """Each block x block cell takes its most common zone: clean zones a machine can follow."""
    if block <= 1:
        return labels
    h, w = labels.shape
    hb, wb = -(-h // block), -(-w // block)
    out = np.zeros((hb, wb), np.int32)
    for i in range(hb):
        for j in range(wb):
            cell = labels[i * block:(i + 1) * block, j * block:(j + 1) * block].ravel()
            out[i, j] = np.bincount(cell[cell >= 0]).argmax() if (cell >= 0).any() else -1
    return out


def zones_gdf(zone_labels: np.ndarray, rates: Dict[int, Dict[str, Any]], image=None,
              center: Optional[tuple] = None, pixel_m: float = 0.05, cell_m: float = 2.0, min_cells: int = 1):
    """Polygons of each zone in EPSG:4326, traced with the image's transform (or a centred grid).

    cell_m: prescription cell size on the ground (each cell takes its majority zone). Sprayers and VRA
    controllers work at metre scale, not at the drone's centimetre pixels.
    """
    import geopandas as gpd
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    labels = np.asarray(zone_labels).astype(np.int32)
    t = getattr(image, "transform", None) if image is not None else None
    crs = getattr(image, "crs", None) if image is not None else None
    if t is None or not crs:                                   # not georeferenced: a grid centred on `center`
        lat, lon = center or (0.0, 0.0)
        h, w = labels.shape
        dlat, dlon = pixel_m / 110_540.0, pixel_m / (111_320.0 * np.cos(np.radians(lat)))
        t = Affine(dlon, 0, lon - w / 2 * dlon, 0, -dlat, lat + h / 2 * dlat)
        crs = "EPSG:4326"
    # ground size of one pixel, to choose the block size
    px_m = abs(t.a) * (111_320.0 * np.cos(np.radians(t.f)) if "4326" in str(crs) else 1.0)
    block = max(1, int(round(cell_m / max(px_m, 1e-9))))
    block = min(block, max(1, min(labels.shape) // 4))        # keep at least ~4 cells across a small image
    labels = _block_majority(labels, block)
    t = t * Affine.scale(block)
    min_px = min_cells
    polys: Dict[int, list] = {}
    for geom, val in rasterio.features.shapes(labels, transform=t):
        polys.setdefault(int(val), []).append(shape(geom))
    rows, geoms = [], []
    px_area = abs(t.a * t.e)
    for z, parts in sorted(polys.items()):
        if z not in rates:
            continue
        g = unary_union(parts)
        if g.area < px_area * min_px:
            continue
        rows.append(rates[z])
        geoms.append(g.simplify(abs(t.a) * 0.5))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=str(crs))
    return gdf.to_crs("EPSG:4326") if str(crs) != "EPSG:4326" else gdf


def to_geojson(gdf) -> bytes:
    return gdf.to_json().encode("utf-8")


def to_shapefile_zip(gdf, name: str = "prescription") -> bytes:
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{name}.shp"
        gdf.to_file(p, driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(d).iterdir():
                z.write(f, f.name)
    return buf.getvalue()


def to_csv(gdf) -> bytes:
    return gdf.drop(columns="geometry").to_csv(index=False).encode("utf-8")


def _rings(geom):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    return [(list(p.exterior.coords), [list(i.coords) for i in p.interiors]) for p in polys]


def to_kml(gdf, title: str = "Prescription") -> bytes:
    maxf = max(1e-6, float(gdf["FUNG_L_HA"].max()) if len(gdf) else 1.0)
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"<name>{title}</name>"]
    for _, r in gdf.iterrows():
        f = float(r["FUNG_L_HA"]) / maxf                       # green (none) -> red (highest fungicide rate)
        col = f"99{0:02x}{int(255 * (1 - f)):02x}{int(255 * f):02x}"   # KML colour is aabbggrr
        desc = "<br/>".join(f"{k}: {r[k]}" for k in ("N_KG_HA", "P_KG_HA", "K_KG_HA", "IRR_MM", "FUNG_L_HA", "RISK"))
        out.append(f"<Placemark><name>Zone {r['ZONE']}: {r['NAME']}</name><description><![CDATA[{desc}]]></description>"
                   f"<Style><PolyStyle><color>{col}</color></PolyStyle><LineStyle><color>ff000000</color></LineStyle>"
                   f"</Style><MultiGeometry>")
        for ext, holes in _rings(r.geometry):
            def ring(c):
                return " ".join(f"{x:.7f},{y:.7f},0" for x, y in c)
            out.append("<Polygon><outerBoundaryIs><LinearRing><coordinates>" + ring(ext) +
                       "</coordinates></LinearRing></outerBoundaryIs>" +
                       "".join("<innerBoundaryIs><LinearRing><coordinates>" + ring(h) +
                               "</coordinates></LinearRing></innerBoundaryIs>" for h in holes) + "</Polygon>")
        out.append("</MultiGeometry></Placemark>")
    out.append("</Document></kml>")
    return "\n".join(out).encode("utf-8")


def to_isoxml_zip(gdf, task_name: str = "Paddy prescription", field_name: str = "Field") -> bytes:
    """Basic ISO 11783-10 (v4) TASKDATA: one task, one treatment zone per management zone.

    Set points: DDI 0x0001 Setpoint Volume Per Area (mm3/m2; 1 L/ha = 100) for fungicide and
    DDI 0x0006 Setpoint Mass Per Area (mg/m2; 1 kg/ha = 100) for nitrogen.
    """
    from xml.sax.saxutils import quoteattr
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<ISO11783_TaskData VersionMajor="4" VersionMinor="2" ManagementSoftwareManufacturer="PaddyTwin" '
             'ManagementSoftwareVersion="1.0" DataTransferOrigin="1">',
             '<PDT A="PDT1" B="Fungicide"/>', '<PDT A="PDT2" B="Nitrogen"/>',
             f'<PFD A="PFD1" C={quoteattr(field_name)} D="{int(sum(g.area for g in gdf.to_crs(3857).geometry))}">',
             "</PFD>",
             f'<TSK A="TSK1" B={quoteattr(task_name)} E="PFD1" G="1" I="1">']
    pnt = 0
    for i, (_, r) in enumerate(gdf.iterrows(), start=1):
        lines.append(f'<TZN A="{i}" B={quoteattr(str(r["NAME"]))}>')
        lines.append(f'<PDV A="0001" B="{int(round(float(r["FUNG_L_HA"]) * 100))}" C="PDT1"/>')
        lines.append(f'<PDV A="0006" B="{int(round(float(r["N_KG_HA"]) * 100))}" C="PDT2"/>')
        for ext, holes in _rings(r.geometry):
            lines.append('<PLN A="6">')
            for kind, ring in [("1", ext)] + [("2", h) for h in holes]:
                lines.append(f'<LSG A="{kind}">')
                for x, y in ring:
                    pnt += 1
                    lines.append(f'<PNT A="2" C="{y:.9f}" D="{x:.9f}"/>')
                lines.append("</LSG>")
            lines.append("</PLN>")
        lines.append("</TZN>")
    lines += ['<TIM A="2000-01-01T00:00:00" D="1"/>', "</TSK>", "</ISO11783_TaskData>"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("TASKDATA/TASKDATA.XML", "\n".join(lines))
    return buf.getvalue()


def summary(gdf) -> Dict[str, float]:
    """Area-weighted totals: what this prescription applies over the whole field."""
    area_ha = gdf.to_crs(3857).geometry.area / 10_000.0
    out = {"area_ha": float(area_ha.sum())}
    for k in RATE_FIELDS:
        out[k.replace("_HA", "_total").replace("_MM", "_mm_ha_mean")] = float((gdf[k] * area_ha).sum())
    out["FUNG_area_ha"] = float(area_ha[gdf["FUNG_L_HA"] > 0].sum())
    return out
