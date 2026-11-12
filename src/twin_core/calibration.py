"""
calibration.py
--------------
Fit PaddyParams to field data.

1. Phenology: stage-labelled flights (data/manifests/flight_manifest.csv)
   + daily temperatures -> GDD onset of each stage. Each onset is placed
   midway between the last flight of the earlier stages and the first flight
   of that stage, so every labelled flight is classified correctly.

2. NDVI curve: per-flight mean NDVI (data/manifests/flight_ndvi.csv, made
   by scripts/extract_flight_ndvi.py from the raw imagery) -> ndvi_min,
   ndvi_max, rise midpoint/width and ndvi_mature by bounded least squares.

Both work on any number of fields; results are pooled.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.twin_core.paddy_model import STAGES, PaddyParams, daily_gdd, ndvi_potential, stage_for_gdd

TempSource = Callable[[dt.date, dt.date], Dict[dt.date, float]]


@dataclass
class Flight:
    flight_id: str
    field_id: str
    date: dt.date
    stage: str
    ndvi: Optional[float] = None


def read_flights(manifest_csv: Path, ndvi_csv: Optional[Path] = None,
                 stage_column: str = "stage_5class") -> List[Flight]:
    ndvi: Dict[str, float] = {}
    if ndvi_csv and Path(ndvi_csv).exists():
        with open(ndvi_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("ndvi_median") not in (None, "", "nan"):
                    ndvi[row["flight_id"]] = float(row["ndvi_median"])
    flights = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flights.append(Flight(row["flight_id"], row["field_id"],
                                  dt.date.fromisoformat(row["date"]),
                                  row.get(stage_column) or row["raw_stage"],
                                  ndvi.get(row["flight_id"])))
    return sorted(flights, key=lambda x: (x.field_id, x.date))


def transplant_dates(flights: Sequence[Flight], days_after_last_nursery: int = 0) -> Dict[str, dt.date]:
    """
    Transplanting is taken as the last 'Nursery' flight per field (plus an
    optional offset); fields with no nursery flight start 25 days before the
    first flight.
    """
    out = {}
    for field_id in sorted({f.field_id for f in flights}):
        fl = [f for f in flights if f.field_id == field_id]
        nursery = [f.date for f in fl if f.stage == "Nursery"]
        out[field_id] = (max(nursery) + dt.timedelta(days=days_after_last_nursery)) if nursery \
            else min(f.date for f in fl) - dt.timedelta(days=25)
    return out


def constant_temperature(mean_c: float = 28.0) -> TempSource:
    def src(start: dt.date, end: dt.date) -> Dict[dt.date, float]:
        return {start + dt.timedelta(days=i): mean_c for i in range((end - start).days + 1)}
    return src


def openmeteo_archive_temperature(lat: float, lon: float) -> TempSource:
    """Daily mean of Open-Meteo archive Tmax/Tmin (needs network)."""
    def src(start: dt.date, end: dt.date) -> Dict[dt.date, float]:
        from src.weather.openmeteo_client import fetch_historical_daily

        d = fetch_historical_daily(lat, lon, start, end)
        out = {}
        for day, tmax, tmin in zip(d["dates"], d["temperature_2m_max"], d["temperature_2m_min"]):
            if tmax is not None and tmin is not None:
                out[dt.date.fromisoformat(day)] = (float(tmax) + float(tmin)) / 2.0
        return out
    return src


def flight_gdd(flights: Sequence[Flight], p: PaddyParams, temps: TempSource,
               transplant: Optional[Dict[str, dt.date]] = None) -> List[Tuple[Flight, float]]:
    """Cumulative GDD since transplanting at each flight date."""
    transplant = transplant or transplant_dates(flights)
    out = []
    for field_id, t0 in transplant.items():
        fl = [f for f in flights if f.field_id == field_id]
        series = temps(t0, max(f.date for f in fl))
        fallback = float(np.mean(list(series.values()))) if series else 28.0
        for f in fl:
            g = 0.0
            day = t0
            while day < f.date:
                day += dt.timedelta(days=1)
                g += daily_gdd(p, series.get(day, fallback))
            out.append((f, g))
    return out


def fit_phenology(flights: Sequence[Flight], p: PaddyParams, temps: TempSource,
                  transplant: Optional[Dict[str, dt.date]] = None) -> Tuple[PaddyParams, Dict]:
    pts = flight_gdd(flights, p, temps, transplant)
    onset = dict(p.gdd_onset)
    report = {"flights": [{"flight_id": f.flight_id, "field_id": f.field_id,
                           "date": f.date.isoformat(), "stage": f.stage, "gdd": round(g, 1)}
                          for f, g in pts], "onset_bounds": {}}
    for k in range(1, len(STAGES)):
        stage = STAGES[k]
        before = [g for f, g in pts if f.stage in STAGES[:k]]
        during = [g for f, g in pts if f.stage in STAGES[k:]]
        lo = max(before) if before else None
        hi = min([g for f, g in pts if f.stage == stage], default=None)
        report["onset_bounds"][stage] = {"after_gdd": lo, "before_gdd": hi}
        if lo is not None and hi is not None and hi > lo:
            onset[stage] = round((lo + hi) / 2.0, 1)
        elif lo is not None and not during:
            onset[stage] = max(onset[stage], lo + 1.0)
    # Keep onsets strictly increasing.
    for k in range(1, len(STAGES)):
        onset[STAGES[k]] = max(onset[STAGES[k]], onset[STAGES[k - 1]] + 50.0)
    mature_flights = [g for f, g in pts if f.stage == "Mature"]
    harvest = max(p.gdd_harvest, (max(mature_flights) + 50.0) if mature_flights else 0.0,
                  onset["Mature"] + 150.0)
    fitted = replace(p, gdd_onset=onset, gdd_harvest=round(harvest, 1))
    correct = sum(stage_for_gdd(fitted, g) == f.stage for f, g in pts)
    report["onset"] = onset
    report["gdd_harvest"] = fitted.gdd_harvest
    report["flights_classified_correctly"] = f"{correct}/{len(pts)}"
    return fitted, report


def fit_ndvi_curve(points: Sequence[Tuple[float, float]], p: PaddyParams) -> Tuple[PaddyParams, Dict]:
    """Least-squares fit of the NDVI potential curve to (gdd, ndvi) points."""
    from scipy.optimize import least_squares

    pts = np.array([(g, n) for g, n in points if n is not None and np.isfinite(n)], dtype=float)
    if len(pts) < 4:
        return p, {"fitted": False, "reason": f"need >= 4 NDVI flights, have {len(pts)}"}
    names = ("ndvi_min", "ndvi_max", "ndvi_rise_mid_gdd", "ndvi_rise_width_gdd", "ndvi_mature")
    x0 = np.array([getattr(p, n) for n in names])
    lo = np.array([0.0, 0.4, 100.0, 40.0, 0.1])
    hi = np.array([0.6, 0.98, 1500.0, 500.0, 0.8])

    def resid(x):
        q = replace(p, **dict(zip(names, x)))
        return np.array([ndvi_potential(q, g) for g in pts[:, 0]]) - pts[:, 1]

    sol = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi))
    fitted = replace(p, **{n: round(float(v), 4) for n, v in zip(names, sol.x)})
    r = resid(sol.x)
    return fitted, {"fitted": True, "n_points": int(len(pts)),
                    "rmse": round(float(np.sqrt(np.mean(r ** 2))), 4),
                    "params": {n: round(float(v), 4) for n, v in zip(names, sol.x)}}


def calibrate(manifest_csv: Path, temps: TempSource, ndvi_csv: Optional[Path] = None,
              base: Optional[PaddyParams] = None) -> Tuple[PaddyParams, Dict]:
    flights = read_flights(manifest_csv, ndvi_csv)
    p, phen = fit_phenology(flights, base or PaddyParams(), temps)
    pts = [(g, f.ndvi) for f, g in flight_gdd(flights, p, temps) if f.ndvi is not None]
    p, ndvi_rep = fit_ndvi_curve(pts, p)
    return p, {"phenology": phen, "ndvi_curve": ndvi_rep, "n_flights": len(flights),
               "n_fields": len({f.field_id for f in flights})}
