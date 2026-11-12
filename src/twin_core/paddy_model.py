"""
paddy_model.py
--------------
Process model of a bunded, transplanted paddy field, used as the twin's
default crop model.

The generic AgronomicDigitalTwinSimulator treats moisture > 0.85 as
waterlogging damage and loses 0.03 moisture per day, so a flooded paddy (the
normal state of rice) was modelled as drying and "waterlogged" at the same
time. This module replaces it with rice-specific processes:

Water (per cell, mm)
    ponded depth h in [0, bund_mm] and root-zone saturation S in [0, 1]
    in:   rain, canal irrigation on a rotation, precision irrigation
    out:  crop ET = Kc(stage) * ET0 (FAO-56 rice Kc), percolation (cell-varying)
    h > bund_mm overflows. When h = 0 the root zone dries and drought stress
    starts below S ~ 0.85 (rice is very drought sensitive).

Phenology (field-wide)
    thermal time, GDD with base 10 C. Stage onsets are calibrated from the
    flight manifest (see calibration.py); NDVI potential follows a logistic
    rise to a booting/flowering peak and a linear senescence to maturity.

Crop vigour
    NDVI relaxes toward potential * f_N * f_W * (1 - fungal damage).

Disease (fungal load, fraction of tissue)
    logistic growth r(T, RH, stage) P (1 - P / P_max), diffusion + wind advection of
    spores. Fungicide kills a fraction of the load and gives residual
    protection that decays over ~1-2 weeks.

Nitrogen (relative soil N)
    uptake during vegetative growth, small leaching with percolation,
    top-dressing adds N.

Observable moisture (the twin's ``moisture`` layer, comparable to
(NDWI + 1) / 2) is  m = 0.6 * S + 0.4 * min(h / 60 mm, 1),
so a ponded field reads ~0.9, a drained but saturated one 0.6, and a drying
one < 0.6. ``sync_from_moisture`` inverts this after assimilation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

STAGES = ("Nursery", "Vegetative", "Booting", "Flowering", "Mature")


@dataclass
class PaddyParams:
    # --- phenology: GDD (base 10 C) at the onset of each stage after transplanting.
    # Defaults fitted to data/manifests/flight_manifest.csv (2 fields, 20 flights,
    # transplanting = last nursery flight, 28 C climatology) with
    # scripts/calibrate_paddy.py --offline; all 20 flights classified correctly.
    t_base_c: float = 10.0
    t_cap_c: float = 35.0
    gdd_onset: Dict[str, float] = field(default_factory=lambda: {
        "Nursery": 0.0, "Vegetative": 252.0, "Booting": 1071.0,
        "Flowering": 1449.0, "Mature": 1854.0})
    gdd_harvest: float = 2300.0

    # --- NDVI potential curve
    ndvi_min: float = 0.25          # just after transplanting (water background)
    ndvi_max: float = 0.86          # peak canopy
    ndvi_rise_mid_gdd: float = 500.0
    ndvi_rise_width_gdd: float = 150.0
    ndvi_mature: float = 0.45       # at harvest
    ndvi_relax_per_day: float = 0.15

    # --- water balance
    bund_mm: float = 100.0
    root_zone_mm: float = 120.0
    percolation_mm_day: float = 2.0  # puddled clay-loam, typical 1-5 mm/day
    kc: Dict[str, float] = field(default_factory=lambda: {   # FAO-56 rice
        "Nursery": 1.05, "Vegetative": 1.10, "Booting": 1.20, "Flowering": 1.20, "Mature": 0.90})
    et0_default_mm: float = 4.8      # Hargreaves estimate, 10 N, monsoon
    irrigation_interval_days: int = 5   # canal rotation
    irrigation_refill_mm: float = 60.0  # target depth after a canal turn
    precision_irrigation_mm: float = 50.0
    drought_s_onset: float = 0.85    # f_W < 1 below this saturation
    drought_s_wilt: float = 0.55     # f_W = 0 at this saturation
    moisture_full_depth_mm: float = 60.0

    # --- nitrogen
    n_uptake_per_day: float = 0.006
    n_leach_per_mm_perc: float = 0.0004
    n_sufficient: float = 0.35
    n_topdress: float = 0.25

    # --- disease
    r_max_per_day: float = 0.20      # apparent infection rate under ideal conditions
    max_severity: float = 0.6        # plateau for a moderately resistant variety
    t_opt_c: float = 26.0
    t_sd_c: float = 4.0
    rh_onset: float = 75.0           # no leaf wetness below this mean RH
    rh_full: float = 95.0
    spore_diffusion_m2_day: float = 1.0
    wind_advection_m_day_per_kmh: float = 0.05
    stage_susceptibility: Dict[str, float] = field(default_factory=lambda: {
        # Adult-plant resistance lowers susceptibility after flowering.
        "Nursery": 0.6, "Vegetative": 0.9, "Booting": 1.0, "Flowering": 0.8, "Mature": 0.3})
    fungicide_kill: float = 0.7
    fungicide_protection_halflife_days: float = 5.0
    fungus_damage_weight: float = 0.7

    # --- stress definition
    stress_fungus_weight: float = 0.35

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "PaddyParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ===================================================================== helpers
def stage_for_gdd(p: PaddyParams, gdd: float) -> str:
    current = STAGES[0]
    for s in STAGES:
        if gdd >= p.gdd_onset[s]:
            current = s
    return current


def stage_midpoint_gdd(p: PaddyParams, stage: str) -> float:
    """Thermal time in the middle of ``stage`` (unknown names -> mid Vegetative)."""
    if stage not in p.gdd_onset:
        stage = "Vegetative"
    i = STAGES.index(stage)
    end = p.gdd_onset[STAGES[i + 1]] if i + 1 < len(STAGES) else p.gdd_harvest
    return float((p.gdd_onset[stage] + end) / 2.0)


def ndvi_potential(p: PaddyParams, gdd: float) -> float:
    rise = 1.0 / (1.0 + math.exp(-(gdd - p.ndvi_rise_mid_gdd) / p.ndvi_rise_width_gdd))
    pot = p.ndvi_min + (p.ndvi_max - p.ndvi_min) * rise
    sen_start = p.gdd_onset["Flowering"]
    if gdd > sen_start:
        frac = min(1.0, (gdd - sen_start) / max(p.gdd_harvest - sen_start, 1.0))
        pot = pot - frac * (pot - p.ndvi_mature)
    return float(pot)


def daily_gdd(p: PaddyParams, temp_c: float) -> float:
    return float(max(0.0, min(temp_c, p.t_cap_c) - p.t_base_c))


def gdd_for_dat(p: PaddyParams, days: int, mean_temp_c: float = 28.0) -> float:
    return days * daily_gdd(p, mean_temp_c)


def moisture_from_water(p: PaddyParams, h: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.clip(0.6 * s + 0.4 * np.minimum(h / p.moisture_full_depth_mm, 1.0), 0.0, 1.0)


def water_from_moisture(p: PaddyParams, m: np.ndarray):
    m = np.clip(m, 0.0, 1.0)
    ponded = m > 0.6
    h = np.where(ponded, (m - 0.6) / 0.4 * p.moisture_full_depth_mm, 0.0)
    s = np.where(ponded, 1.0, m / 0.6)
    return h.astype(np.float32), s.astype(np.float32)


# =================================================================== simulator
class PaddyFieldSimulator:
    """
    Operates in place on a FieldState: mean layers (ndvi, moisture, nitrogen,
    fungus, stress) plus auxiliary process state in ``state.aux``:

        water_mm, soil_sat, protection, percolation_mm_day   (per cell)
        gdd, days_since_irrigation                           (scalars, in state.scalars)
    """

    AUX = ("water_mm", "soil_sat", "protection", "percolation_mm_day")

    def __init__(self, state, zone_labels: np.ndarray, zone_names: Dict[int, str],
                 params: Optional[PaddyParams] = None):
        self.p = params or PaddyParams()
        self.state = state
        self.zone_labels = zone_labels
        self.zone_names = zone_names
        self._ensure_aux()

    # ------------------------------------------------------------- state
    def _ensure_aux(self) -> None:
        st, p = self.state, self.p
        shape = st.grid.shape
        if "water_mm" not in st.aux or "soil_sat" not in st.aux:
            h, s = water_from_moisture(p, st.mean["moisture"])
            st.aux["water_mm"], st.aux["soil_sat"] = h, s
        st.aux.setdefault("protection", np.zeros(shape, dtype=np.float32))
        st.aux.setdefault("percolation_mm_day",
                          np.full(shape, p.percolation_mm_day, dtype=np.float32))
        if "gdd" not in st.scalars:
            st.scalars["gdd"] = stage_midpoint_gdd(p, st.growth_stage)
        st.scalars.setdefault("days_since_irrigation", 0.0)
        st.growth_stage = stage_for_gdd(p, st.scalars["gdd"])

    def sync_from_moisture(self, tol: float = 1e-3) -> None:
        """After assimilation, move water state to agree with the observed moisture."""
        st = self.state
        modelled = moisture_from_water(self.p, st.aux["water_mm"], st.aux["soil_sat"])
        changed = np.abs(modelled - st.mean["moisture"]) > tol
        if np.any(changed):
            h, s = water_from_moisture(self.p, st.mean["moisture"])
            st.aux["water_mm"] = np.where(changed, h, st.aux["water_mm"]).astype(np.float32)
            st.aux["soil_sat"] = np.where(changed, s, st.aux["soil_sat"]).astype(np.float32)

    def set_stage(self, stage: str) -> None:
        if stage in self.p.gdd_onset:
            self.state.scalars["gdd"] = float(self.p.gdd_onset[stage])
        self.state.growth_stage = stage_for_gdd(self.p, self.state.scalars["gdd"])

    # ------------------------------------------------------ stress relation
    @property
    def ndvi_pot(self) -> float:
        return ndvi_potential(self.p, self.state.scalars["gdd"])

    def stress(self, ndvi: np.ndarray, fungus: np.ndarray) -> np.ndarray:
        w = self.p.stress_fungus_weight
        deficit = np.clip(1.0 - ndvi / max(self.ndvi_pot, 1e-3), 0.0, 1.0)
        return np.clip((1.0 - w) * deficit + w * fungus, 0.0, 1.0).astype(np.float32)

    def ndvi_from_stress(self, stress: np.ndarray, fungus: np.ndarray) -> np.ndarray:
        w = self.p.stress_fungus_weight
        deficit = np.clip((stress - w * fungus) / (1.0 - w), 0.0, 1.0)
        return np.clip(self.ndvi_pot * (1.0 - deficit), -1.0, 1.0).astype(np.float32)

    def ndvi_per_stress(self) -> float:
        """|d ndvi / d stress| for scaling observation variance."""
        return self.ndvi_pot / (1.0 - self.p.stress_fungus_weight)

    # ----------------------------------------------------------- one day
    def simulate_day(self, weather: Dict[str, Any], interventions: List[Dict[str, Any]],
                     **_: Any) -> List[str]:
        st, p, logs = self.state, self.p, []
        m = st.mean
        temp = float(weather.get("temperature", 29.0))
        rh = float(weather.get("humidity", 78.0))
        rain = float(weather.get("precipitation", 0.0))
        wind = float(weather.get("wind_speed", 8.0))
        wind_dir = float(weather.get("wind_direction", 225.0))

        stage = stage_for_gdd(p, st.scalars["gdd"])
        zone_mask = lambda kind: self._zones_mask(interventions, kind)  # noqa: E731

        # ---------------- water balance
        h, s = st.aux["water_mm"], st.aux["soil_sat"]
        inflow = np.full(h.shape, rain, dtype=np.float32)
        st.scalars["days_since_irrigation"] += 1.0
        if p.irrigation_interval_days > 0 and \
                st.scalars["days_since_irrigation"] >= p.irrigation_interval_days and stage != "Mature":
            inflow += np.maximum(p.irrigation_refill_mm - h, 0.0)
            st.scalars["days_since_irrigation"] = 0.0
            logs.append(f"Canal irrigation turn: field topped up to {p.irrigation_refill_mm:.0f} mm.")
        pi = zone_mask("Precision Irrigation")
        if pi.any():
            inflow += pi * p.precision_irrigation_mm
            logs.append(f"Precision irrigation applied to {self._zone_list(interventions, 'Precision Irrigation')}.")

        et0 = float(weather.get("et0", p.et0_default_mm * (0.7 if rain > 1.0 else 1.0)))
        f_w_prev = self._f_water(h, s)
        et = p.kc.get(stage, 1.1) * et0 * np.where(h > 0, 1.0, np.maximum(f_w_prev, 0.15))
        perc = st.aux["percolation_mm_day"] * np.where(h > 0, 1.0, s)
        # Fill the root zone first, then pond.
        deficit = (1.0 - s) * p.root_zone_mm
        to_soil = np.minimum(inflow, deficit)
        s = s + to_soil / p.root_zone_mm
        h = h + (inflow - to_soil) - et - perc
        # Water demand not met by ponded water comes out of the root zone.
        short = np.clip(-h, 0.0, None)
        h = np.clip(h, 0.0, p.bund_mm)
        s = np.clip(s - short / p.root_zone_mm, 0.0, 1.0)
        st.aux["water_mm"], st.aux["soil_sat"] = h.astype(np.float32), s.astype(np.float32)
        m["moisture"] = moisture_from_water(p, h, s).astype(np.float32)
        f_w = self._f_water(h, s)
        dry_frac = float((f_w < 0.9).mean())
        if dry_frac > 0.02:
            logs.append(f"Drought stress on {dry_frac * 100:.1f}% of the field (no standing water).")

        # ---------------- nitrogen
        veg = 1.0 if stage in ("Vegetative", "Booting") else 0.3
        m["nitrogen"] = np.clip(m["nitrogen"] - p.n_uptake_per_day * veg * f_w
                                - p.n_leach_per_mm_perc * perc, 0.0, 1.0).astype(np.float32)
        nt = zone_mask("Nutrient Top-Dress")
        if nt.any():
            m["nitrogen"] = np.clip(m["nitrogen"] + nt * p.n_topdress, 0.0, 1.0).astype(np.float32)
            logs.append(f"Nitrogen top-dress applied to {self._zone_list(interventions, 'Nutrient Top-Dress')}.")
        f_n = np.clip(m["nitrogen"] / p.n_sufficient, 0.4, 1.0)

        # ---------------- disease
        fung = m["fungus"].astype(np.float64)
        prot = st.aux["protection"]
        spray = zone_mask("Fungicide Spray")
        if spray.any():
            fung = fung * (1.0 - p.fungicide_kill * spray)
            prot = np.maximum(prot, spray)
            logs.append(f"Fungicide applied to {self._zone_list(interventions, 'Fungicide Spray')}.")
        suit = math.exp(-((temp - p.t_opt_c) ** 2) / (2 * p.t_sd_c ** 2)) * \
            float(np.clip((rh - p.rh_onset) / (p.rh_full - p.rh_onset), 0.0, 1.0))
        r = p.r_max_per_day * suit * p.stage_susceptibility.get(stage, 0.8) * (1.0 - 0.9 * prot)
        fung = fung + r * fung * (1.0 - fung / p.max_severity)
        fung = self._spread(fung, wind, wind_dir)
        m["fungus"] = np.clip(fung, 0.0, 1.0).astype(np.float32)
        st.aux["protection"] = (prot * 0.5 ** (1.0 / p.fungicide_protection_halflife_days)).astype(np.float32)
        for z_id, name in self.zone_names.items():
            mask = self.zone_labels == z_id
            if mask.any() and float(m["fungus"][mask].mean()) > 0.5:
                logs.append(f"WARNING: heavy fungal infection in {name} "
                            f"({float(m['fungus'][mask].mean()) * 100:.0f}% tissue).")

        # ---------------- phenology and canopy
        st.scalars["gdd"] += daily_gdd(p, temp)
        new_stage = stage_for_gdd(p, st.scalars["gdd"])
        if new_stage != stage:
            logs.append(f"Crop entered {new_stage} stage (GDD {st.scalars['gdd']:.0f}).")
        st.growth_stage = new_stage
        pot = self.ndvi_pot
        vigor = f_n * f_w * (1.0 - p.fungus_damage_weight * m["fungus"])
        target = pot * vigor
        m["ndvi"] = np.clip(m["ndvi"] + p.ndvi_relax_per_day * (target - m["ndvi"]),
                            -1.0, 1.0).astype(np.float32)
        m["stress"] = self.stress(m["ndvi"], m["fungus"])
        return logs

    # ------------------------------------------------------------ helpers
    def _f_water(self, h: np.ndarray, s: np.ndarray) -> np.ndarray:
        p = self.p
        f = np.clip((s - p.drought_s_wilt) / (p.drought_s_onset - p.drought_s_wilt), 0.0, 1.0)
        return np.where(h > 0, 1.0, f).astype(np.float32)

    def _zones_mask(self, interventions: List[Dict[str, Any]], kind: str) -> np.ndarray:
        mask = np.zeros(self.zone_labels.shape, dtype=np.float32)
        for iv in interventions:
            if iv.get("type") == kind:
                mask[self.zone_labels == int(iv["zone_id"])] = 1.0
        return mask

    def _zone_list(self, interventions, kind) -> str:
        return ", ".join(self.zone_names.get(int(i["zone_id"]), str(i["zone_id"]))
                         for i in interventions if i.get("type") == kind)

    def _spread(self, fung: np.ndarray, wind_kmh: float, wind_dir_deg: float) -> np.ndarray:
        """Spore diffusion + upwind advection, explicit with stable sub-steps."""
        cell = self.state.grid.cell_size_m
        d = self.p.spore_diffusion_m2_day * (1.0 + wind_kmh / 15.0) / cell ** 2
        v = self.p.wind_advection_m_day_per_kmh * wind_kmh / cell
        # Wind direction is where the wind comes FROM; spores move downwind.
        theta = math.radians(wind_dir_deg + 180.0)
        vx, vy = v * math.sin(theta), v * math.cos(theta)   # east, north (cells/day)
        n = max(1, int(math.ceil(max(4 * d, 2 * (abs(vx) + abs(vy))))))
        dt = 1.0 / n
        P = fung
        for _ in range(n):
            pad = np.pad(P, 1, mode="edge")
            lap = pad[1:-1, 2:] + pad[1:-1, :-2] + pad[2:, 1:-1] + pad[:-2, 1:-1] - 4 * P
            # Rows increase southwards: northward wind (vy > 0) moves mass to lower rows.
            gx = (P - pad[1:-1, :-2]) if vx > 0 else (pad[1:-1, 2:] - P)
            gy = (pad[2:, 1:-1] - P) if vy > 0 else (P - pad[:-2, 1:-1])
            P = P + dt * (d * lap - vx * gx + vy * gy)
        return P
