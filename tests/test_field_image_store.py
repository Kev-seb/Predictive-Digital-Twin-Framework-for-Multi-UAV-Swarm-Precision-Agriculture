"""Robust GeoTIFF loading (src/core/field_image.py) and the field store (src/storage/field_store.py)."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.core.field_image import load_field_image
from src.storage import field_store as fs


def _tif(path, scale=10_000.0, dtype="uint16", size=(600, 400), nodata=0, raw_dn=False):
    h, w = size
    g = np.full((h, w), 0.08); r = np.full((h, w), 0.05); re = np.full((h, w), 0.3); nir = np.full((h, w), 0.6)
    data = np.stack([g, r, re, nir]) * scale
    data[:, :50, :] = nodata                                   # a no-data strip outside the field
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4, dtype=dtype, crs="EPSG:4326",
                       transform=from_origin(79.0, 11.0, 1e-6, 1e-6), nodata=nodata) as dst:
        dst.write(data.astype(dtype))
    return path


def test_keeps_true_reflectance_from_scaled_integers(tmp_path):
    img, rep = load_field_image(_tif(tmp_path / "a.tif"), max_side=4096)
    ndvi = (img.bands["nir"] - img.bands["red"]) / (img.bands["nir"] + img.bands["red"])
    assert abs(float(np.median(ndvi[img.valid_mask])) - (0.6 - 0.05) / 0.65) < 1e-3   # not stretched per band
    assert "10,000" in rep["units"]
    assert abs(rep["nodata_pct"] - 50 / 600 * 100) < 0.5 and not np.isnan(img.bands["nir"]).any()


def test_large_image_is_downsampled_with_matching_georeference(tmp_path):
    img, rep = load_field_image(_tif(tmp_path / "b.tif"), max_side=150)
    assert rep["downsampled"] and max(rep["used_px"]) == 150
    assert abs(img.transform.a * img.width - 1e-6 * 400) < 1e-9   # same ground extent


def test_raw_numbers_and_low_ndvi_are_flagged(tmp_path):
    _, rep = load_field_image(_tif(tmp_path / "c.tif", scale=60_000.0, nodata=0), max_side=4096)
    assert any("raw sensor numbers" in w for w in rep["warnings"])
    p = tmp_path / "d.tif"
    with rasterio.open(p, "w", driver="GTiff", height=100, width=100, count=4, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(79, 11, 1e-6, 1e-6)) as dst:
        dst.write(np.stack([np.full((100, 100), v, np.float32) for v in (0.1, 0.2, 0.25, 0.25)]))
    _, rep = load_field_image(p)
    assert any("not radiometrically" in w or "calibrated" in w for w in rep["warnings"])


def test_needs_four_bands(tmp_path):
    p = tmp_path / "e.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=3, dtype="float32") as dst:
        dst.write(np.ones((3, 10, 10), np.float32))
    with pytest.raises(ValueError, match="4-band"):
        load_field_image(p)


def test_store_fields_flights_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "DB", tmp_path / "t.db")
    monkeypatch.setattr(fs, "FILES", tmp_path / "files")
    src = _tif(tmp_path / "f.tif")
    fid = fs.create_field("Plot 7", 11.0, 79.0, 100.0, 80.0)
    fs.add_flight(fid, "week1.tif", src, 0.7, 0.1)
    fs.add_flight(fid, "week2.tif", src, 0.75, 0.08)
    assert fs.list_fields()[0]["flights"] == 2
    paths = [f["path"] for f in fs.flights(fid)]
    assert all((tmp_path / "files" / str(fid)).joinpath(p.split("\\")[-1].split("/")[-1]).exists() for p in paths)
    fs.save_tasks(fid, [{"Zone": "Z1", "Action": "Fungicide Spray", "Dose": "1.5 L/ha", "Planned day": "Day 1",
                         "Priority": 5, "Status": "Done", "Done on": "2026-11-23", "Note": "north end"}])
    assert fs.load_tasks(fid)[0]["Status"] == "Done"
    fs.delete_field(fid)
    assert fs.list_fields() == [] and not (tmp_path / "files" / str(fid)).exists()
