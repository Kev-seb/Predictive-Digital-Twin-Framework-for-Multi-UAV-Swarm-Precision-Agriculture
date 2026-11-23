"""
Decisions from the twin's belief, not just its mean.

* ``zone_risk`` / ``scouting_targets``: where the twin is unsure whether a zone
  crosses the spray threshold. The value of scouting a cell is its expected
  "surprise" at the threshold, sd * phi((threshold - mean) / sd): high where the
  estimate is close to the threshold and uncertain, near zero where the twin is
  confident either way (a well-known healthy cell, or a cell clearly infected).
* ``spray_options``: spray now, spray in N days, or do not spray. Each option is
  run forward on forks of the twin under the weather forecast, for several
  draws of the disease state from the twin's uncertainty (the same draws for
  every option, so the comparison is paired). The result is the gain over not
  spraying in rupees (harvest value minus product and drone hire), its spread,
  and the chance that the option beats not spraying.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from src.ai_engine.economics import ACRE_HA, Prices
from src.twin_core.swarm import SPRAY_WIND_MAX_M_S

RAIN_SPRAY_MAX_MM = 5.0          # do not plan a spray into a wet day (wash-off)
WEATHER_WIND_KMH = 3.6           # forecasts carry wind in km/h


def zone_risk(twin, threshold: float = 0.2) -> List[Dict[str, Any]]:
    """Per zone: mean fungal load, its uncertainty, expected infected share and scouting value."""
    mu = twin.state.mean["fungus"].astype(np.float64)
    sd = np.sqrt(np.maximum(twin.state.var["fungus"].astype(np.float64), 1e-8))
    zsc = (threshold - mu) / sd
    p_exceed = 1.0 - norm.cdf(zsc)
    voi = sd * norm.pdf(zsc)
    out = []
    for z_id, name in sorted(twin.zone_names.items()):
        m = twin.zone_labels == z_id
        if not m.any():
            continue
        out.append({"zone_id": int(z_id), "zone": name,
                    "fungus_mean": round(float(mu[m].mean()), 4),
                    "fungus_sd": round(float(sd[m].mean()), 4),
                    "infected_share": round(float(p_exceed[m].mean()), 4),
                    "scout_value": round(float(voi[m].mean()), 5)})
    return out


def scouting_targets(twin, threshold: float = 0.2, keep: float = 0.25,
                     min_value: float = 1e-3) -> List[int]:
    """Zones worth flying, best first: within ``keep`` x of the best zone's scouting value."""
    risk = sorted(zone_risk(twin, threshold), key=lambda r: -r["scout_value"])
    if not risk or risk[0]["scout_value"] < min_value:
        return []
    top = risk[0]["scout_value"]
    return [r["zone_id"] for r in risk if r["scout_value"] >= keep * top]


def _weather_on(weather: Sequence[Dict[str, Any]], day: int) -> Dict[str, Any]:
    return dict(weather[min(day, len(weather)) - 1]) if weather else {}


def spray_options(
    twin,
    harvest_fn: Callable[[Any], float],
    weather: Sequence[Dict[str, Any]],
    zones: Optional[List[int]] = None,
    threshold: float = 0.2,
    min_infected_frac: float = 0.1,
    delays: Sequence[int] = (0, 2),
    horizon_days: int = 21,
    samples: int = 8,
    prices: Prices = Prices(),
    product_l_ha: float = 1.0,
) -> Dict[str, Any]:
    """
    Compare spraying ``zones`` after each delay in ``delays`` (0 = tomorrow,
    the first forecast day) against not spraying. ``harvest_fn(twin)`` gives
    the expected harvest in t/ha; ``weather`` is the forecast, day 1 first.
    """
    risk = zone_risk(twin, threshold)
    if zones is None:
        zones = [r["zone_id"] for r in risk if r["infected_share"] >= min_infected_frac]
    zones = [int(z) for z in zones]
    grid = twin.state.grid
    cell_ha = grid.cell_size_m ** 2 / 10_000.0
    field_ha = grid.rows * grid.cols * cell_ha
    spray_ha = float(np.isin(twin.zone_labels, zones).sum()) * cell_ha if zones else 0.0
    cost = spray_ha * (product_l_ha * prices.fungicide_rs_l + prices.drone_rs_acre / ACRE_HA)
    rs_per_t = 10.0 * prices.paddy_rs_quintal
    weather = list(weather)[:horizon_days]
    base = {"zones": zones, "zone_risk": risk, "spray_area_ha": round(spray_ha, 3),
            "field_area_ha": round(field_ha, 3), "horizon_days": horizon_days, "samples": samples,
            "spray_cost_rs": round(cost, 0)}
    if not zones:
        return {**base, "options": [], "scout_first": False,
                "recommendation": {"option": "no spray", "text": "No zone is likely above the spray "
                                   "threshold. Do not spray; keep scouting on schedule."}}

    # Paired draws of the disease state: quantiles of one shared shift (in sd units)
    # per draw, so every option sees the same plausible fields.
    shifts = norm.ppf((np.arange(samples) + 0.5) / samples)
    mu = twin.state.mean["fungus"].astype(np.float64)
    sd = np.sqrt(np.maximum(twin.state.var["fungus"].astype(np.float64), 0.0))

    def run(day: Optional[int], z: float) -> float:
        f = twin.fork()
        f.state.mean["fungus"] = np.clip(mu + z * sd, 0.0, 1.0).astype(np.float32)
        plan = [] if day is None else [{"day": day, "zone_id": zid, "type": "Fungicide Spray"} for zid in zones]
        f.step(days=horizon_days, weather=weather or None, interventions=plan)
        return harvest_fn(f)

    none_h = np.array([run(None, z) for z in shifts])
    options = []
    for delay in delays:
        day = int(delay) + 1
        w = _weather_on(weather, day)
        rain = float(w.get("precipitation", 0.0))
        wind = float(w.get("wind_speed", 0.0)) / WEATHER_WIND_KMH
        notes = []
        if rain > RAIN_SPRAY_MAX_MM:
            notes.append(f"{rain:.0f} mm of rain forecast that day (wash-off)")
        if wind > SPRAY_WIND_MAX_M_S:
            notes.append(f"wind {wind:.1f} m/s is above the {SPRAY_WIND_MAX_M_S:.0f} m/s spray limit")
        h = np.array([run(day, z) for z in shifts])
        gain = (h - none_h) * field_ha * rs_per_t - cost
        label = "spray tomorrow" if delay == 0 else f"spray in {delay + 1} days"
        options.append({"option": label, "day": day, "feasible": not notes, "notes": notes,
                        "rain_mm": round(rain, 1), "wind_m_s": round(wind, 1),
                        "harvest_t_ha": round(float(h.mean()), 3),
                        "gain_rs": round(float(gain.mean()), 0),
                        "gain_rs_p10": round(float(np.percentile(gain, 10)), 0),
                        "gain_rs_p90": round(float(np.percentile(gain, 90)), 0),
                        "p_better_than_none": round(float((gain > 0).mean()), 3)})
    no_spray = {"option": "no spray", "harvest_t_ha": round(float(none_h.mean()), 3), "gain_rs": 0.0}

    feasible = [o for o in options if o["feasible"]]
    best = max(feasible, key=lambda o: o["gain_rs"], default=None)
    scout_first = False
    if best is None:
        rec = {"option": "wait", "text": "No spray day in the forecast window is flyable: "
               + "; ".join(n for o in options for n in o["notes"]) + "."}
    elif best["gain_rs"] <= 0:
        rec = {"option": "no spray", "text": f"Spraying costs about Rs {cost:,.0f} and is not expected to pay back "
               f"(best case {best['option']}: Rs {best['gain_rs']:,.0f})."}
    else:
        rec = {"option": best["option"], "text": f"{best['option'].capitalize()}: expected gain Rs {best['gain_rs']:,.0f} "
               f"over not spraying (range Rs {best['gain_rs_p10']:,.0f} to {best['gain_rs_p90']:,.0f}; "
               f"better than not spraying in {best['p_better_than_none'] * 100:.0f}% of plausible fields)."}
        others = [o for o in feasible if o is not best]
        if others and best["day"] == min(o["day"] for o in feasible):
            later = max(others, key=lambda o: o["gain_rs"])
            rec["text"] += f" Waiting ({later['option']}) loses about Rs {best['gain_rs'] - later['gain_rs']:,.0f}."
        if best["p_better_than_none"] < 0.7:
            scout_first = True
            rec["text"] += " The twin is unsure: scout the flagged zones first."
    for o in options:
        if not o["feasible"]:
            o["notes"] = "; ".join(o["notes"])
        else:
            o["notes"] = ""
    return {**base, "options": [no_spray, *options], "recommendation": rec, "scout_first": scout_first}
