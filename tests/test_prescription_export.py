"""Prescription maps for machines (src/gis/prescription_export.py) and costs (src/ai_engine/economics.py)."""
import io
import xml.dom.minidom as md
import zipfile
from types import SimpleNamespace

import numpy as np
from affine import Affine

from src.ai_engine import economics as econ
from src.gis import prescription_export as pe


def _prescriptions():
    rec = lambda cat, dose: SimpleNamespace(category=cat, dosage_rate=dose)
    return [SimpleNamespace(zone_id=0, zone_name="Healthy", area_pct=60.0, fungal_risk_prob=0.05,
                            recommendations=[rec("Nutrient", "40 kg/ha N | 0 kg/ha P2O5 | 15 kg/ha K2O"),
                                             rec("Irrigation", "0 mm"), rec("Fungicide", "0.0 L/ha")]),
            SimpleNamespace(zone_id=1, zone_name="Blast patch", area_pct=40.0, fungal_risk_prob=0.8,
                            recommendations=[rec("Nutrient", "120 kg/ha N | 45 kg/ha P2O5 | 60 kg/ha K2O"),
                                             rec("Irrigation", "25 mm (Deep wetting)"), rec("Fungicide", "1.5 L/ha")])]


def _field(noise=0.2):
    """100 m x 100 m field at 5 cm pixels; zone 1 in the east 40%, with salt noise."""
    lab = np.zeros((2000, 2000), int)
    lab[:, 1200:] = 1
    lab = np.where(np.random.default_rng(0).random(lab.shape) < noise, 1 - lab, lab)
    img = SimpleNamespace(crs="EPSG:4326",
                          transform=Affine(0.05 / (111_320 * np.cos(np.radians(11))), 0, 79.0, 0, -0.05 / 110_540, 11.0))
    return lab, img


def test_rates_are_numbers_parsed_from_the_dosage_text():
    r = pe.zone_rates(_prescriptions())
    assert (r[1]["N_KG_HA"], r[1]["P_KG_HA"], r[1]["K_KG_HA"], r[1]["IRR_MM"], r[1]["FUNG_L_HA"]) == (120, 45, 60, 25, 1.5)
    assert r[0]["FUNG_L_HA"] == 0 and r[0]["N_KG_HA"] == 40


def test_zones_are_clean_georeferenced_polygons():
    lab, img = _field()
    g = pe.zones_gdf(lab, pe.zone_rates(_prescriptions()), image=img, cell_m=2.0)
    assert len(g) == 2 and str(g.crs) == "EPSG:4326"
    s = pe.summary(g)
    assert abs(s["area_ha"] - 1.0) < 0.05                      # the 100 m field is ~1 ha
    assert abs(s["FUNG_area_ha"] - 0.4) < 0.05                 # only the blast patch is sprayed
    assert abs(s["FUNG_L_total"] - 1.5 * 0.4) < 0.1
    minx, miny, maxx, maxy = g.total_bounds
    assert 78.99 < minx < maxx < 79.01 and 10.99 < miny < maxy < 11.01


def test_every_format_is_valid():
    lab, img = _field()
    g = pe.zones_gdf(lab, pe.zone_rates(_prescriptions()), image=img)
    shp = zipfile.ZipFile(io.BytesIO(pe.to_shapefile_zip(g)))
    assert {n.rsplit(".", 1)[1] for n in shp.namelist()} >= {"shp", "shx", "dbf", "prj"}
    md.parseString(pe.to_kml(g))
    x = md.parseString(zipfile.ZipFile(io.BytesIO(pe.to_isoxml_zip(g))).read("TASKDATA/TASKDATA.XML"))
    pdv = {(p.getAttribute("A"), p.getAttribute("B")) for p in x.getElementsByTagName("PDV")}
    assert ("0001", "150") in pdv and ("0006", "12000") in pdv  # 1.5 L/ha and 120 kg/ha in ISO units
    assert len(x.getElementsByTagName("PNT")) < 5000             # metre-scale cells, not pixel zigzags
    assert pe.to_csv(g).decode().splitlines()[0].startswith("ZONE,NAME,N_KG_HA")


def test_targeted_spraying_costs_less_than_blanket():
    lab, img = _field(noise=0.0)
    g = pe.zones_gdf(lab, pe.zone_rates(_prescriptions()), image=img)
    p = econ.Prices(fungicide_rs_l=1000.0, drone_rs_acre=400.0)
    e = econ.plan_economics(pe.summary(g), float(g["FUNG_L_HA"].max()), p, yield_gain_t_ha=0.3)
    assert abs(e["fungicide_saved_pct"] - 60.0) < 3             # 40% of the field sprayed
    assert e["spray_saving_rs"] > 0
    assert abs(e["yield_value_rs"] - 0.3 * 1.0 * 10 * p.paddy_rs_quintal) < 0.05 * e["yield_value_rs"]
    assert abs(p.rs_kg_n - 266.5 / 45 / 0.46) < 1e-9
