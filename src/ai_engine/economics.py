"""
economics.py - what a treatment plan costs and returns, in rupees.

Default prices are typical Indian figures for a starting point; every one is editable in the app.
    paddy:      Minimum Support Price, common grade (Rs/quintal)
    urea:       Rs 266.50 per 45 kg bag (46% N)       -> Rs per kg N
    DAP:        Rs 1,350 per 50 kg bag (46% P2O5)      -> Rs per kg P2O5 (its N counted as a bonus, ignored)
    MOP:        Rs 1,700 per 50 kg bag (60% K2O)       -> Rs per kg K2O
    fungicide:  Rs per litre of formulated product
    drone hire: Rs per acre sprayed
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

ACRE_HA = 0.404686


@dataclass
class Prices:
    paddy_rs_quintal: float = 2369.0
    urea_rs_bag45: float = 266.5
    dap_rs_bag50: float = 1350.0
    mop_rs_bag50: float = 1700.0
    fungicide_rs_l: float = 1200.0
    drone_rs_acre: float = 450.0

    @property
    def rs_kg_n(self) -> float:
        return self.urea_rs_bag45 / 45.0 / 0.46

    @property
    def rs_kg_p2o5(self) -> float:
        return self.dap_rs_bag50 / 50.0 / 0.46

    @property
    def rs_kg_k2o(self) -> float:
        return self.mop_rs_bag50 / 50.0 / 0.60

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def plan_economics(summary: Dict[str, float], max_fung_l_ha: float, prices: Prices,
                   yield_gain_t_ha: Optional[float] = None) -> Dict[str, Any]:
    """summary: prescription_export.summary() of the plan (area, totals, sprayed area)."""
    area = summary["area_ha"]
    fert = (summary["N_KG_total"] * prices.rs_kg_n + summary["P_KG_total"] * prices.rs_kg_p2o5
            + summary["K_KG_total"] * prices.rs_kg_k2o)
    fung_l = summary["FUNG_L_total"]
    sprayed_ha = summary["FUNG_area_ha"]
    plan_spray = fung_l * prices.fungicide_rs_l + sprayed_ha / ACRE_HA * prices.drone_rs_acre
    blanket_l = max_fung_l_ha * area
    blanket_spray = blanket_l * prices.fungicide_rs_l + (area / ACRE_HA * prices.drone_rs_acre if max_fung_l_ha else 0.0)
    out = {"area_ha": area, "fertiliser_rs": fert, "spray_rs": plan_spray, "plan_total_rs": fert + plan_spray,
           "fungicide_l": fung_l, "blanket_fungicide_l": blanket_l, "blanket_spray_rs": blanket_spray,
           "spray_saving_rs": blanket_spray - plan_spray,
           "fungicide_saved_pct": (1 - fung_l / blanket_l) * 100 if blanket_l > 0 else 0.0}
    if yield_gain_t_ha is not None:
        value = yield_gain_t_ha * area * 10.0 * prices.paddy_rs_quintal          # 1 t = 10 quintal
        out.update({"yield_protected_t": yield_gain_t_ha * area, "yield_value_rs": value,
                    "net_benefit_rs": value - (fert + plan_spray),
                    "return_per_rupee": value / (fert + plan_spray) if fert + plan_spray > 0 else None})
    return out
