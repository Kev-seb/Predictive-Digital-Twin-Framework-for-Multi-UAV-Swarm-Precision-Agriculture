"""
twin.py
-------
DigitalTwin: the single source of truth for a simulated paddy field.

It owns
  * a :class:`FieldState` (mean + variance per layer, georeferenced),
  * one long-lived :class:`AgronomicDigitalTwinSimulator` whose internal
    moisture / nitrogen / fungus layers carry over from day to day,
  * an event log and in-process subscribers (used by the API's WebSocket).

The crop model is pluggable: ``crop_model="paddy"`` (default) uses the
rice-specific PaddyFieldSimulator (flooded water balance, thermal-time
phenology, weather-driven disease); ``"legacy"`` wraps the original
AgronomicDigitalTwinSimulator.

The core loop is PREDICT -> ASSIMILATE:

  step()            advances the simulator by N days and grows uncertainty
  observe_grid()    corrects the state with a raster (survey, UE5 render)
  observe_point()   corrects the state with a geotagged scout reading

``fork()`` returns an independent copy for what-if scenarios, so planning
never disturbs the live twin.
"""

from __future__ import annotations

import collections
import copy
import datetime as _dt
import json
import threading
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from src.twin_core import assimilation as da
from src.twin_core.field_state import LAYERS, FieldGrid, FieldState
from src.twin_core.paddy_model import PaddyFieldSimulator, PaddyParams

# Default daily weather for a monsoon-season paddy in Tamil Nadu
# (matches the default field centre in src/config/config.py).
DEFAULT_WEATHER: Dict[str, float] = {
    "temperature": 29.0,
    "humidity": 78.0,
    "precipitation": 0.0,
    "wind_speed": 8.0,
    "wind_direction": 225.0,
    "precipitation_probability": 20.0,
}

# Paddy phenology names used in the dataset -> simulator susceptibility keys.
_STAGE_TO_SIM = {
    "nursery": "Emergence",
    "emergence": "Emergence",
    "transplanting": "Emergence",
    "tillering": "Vegetative",
    "vegetative": "Vegetative",
    "booting": "Flowering",
    "heading": "Flowering",
    "reproductive": "Flowering",
    "flowering": "Flowering",
    "ripening": "Senescence",
    "maturity": "Senescence",
    "senescence": "Senescence",
}

# Fraction of the stress signal explained by fungal load in the simulator:
#   stress = (1 - ndvi) / 2 + STRESS_FUNGUS_WEIGHT * fungus
STRESS_FUNGUS_WEIGHT = 0.35

Event = Dict[str, Any]
Subscriber = Callable[[Event], None]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sim_stage(stage: str) -> str:
    return _STAGE_TO_SIM.get(stage.strip().lower(), "Vegetative")


class LegacyCropModel:
    """Adapter giving AgronomicDigitalTwinSimulator the crop-model interface."""

    def __init__(self, state: FieldState, zone_labels, zone_names, propagation_model: str):
        from src.digital_twin.simulator import AgronomicDigitalTwinSimulator

        m = state.mean
        self.state = state
        self.propagation_model = propagation_model
        self.sim = AgronomicDigitalTwinSimulator(
            ndvi=m["ndvi"], stress=m["stress"], ndwi=m["moisture"] * 2.0 - 1.0,
            cire=m["nitrogen"] * 5.0, zone_labels=zone_labels, zone_names=zone_names)

    def sync_from_moisture(self) -> None:
        pass

    def set_stage(self, stage: str) -> None:
        self.state.growth_stage = stage

    def simulate_day(self, weather, interventions, **_):
        for name in LAYERS:
            setattr(self.sim, name, self.state.mean[name].astype(np.float32).copy())
        logs = self.sim.simulate_day(weather=weather, interventions=interventions,
                                     propagation_model=self.propagation_model,
                                     growth_stage=sim_stage(self.state.growth_stage))
        for name in LAYERS:
            self.state.mean[name] = np.asarray(getattr(self.sim, name), dtype=np.float32).copy()
        return logs

    @staticmethod
    def stress(ndvi, fungus):
        return np.clip((1.0 - ndvi) / 2.0 + STRESS_FUNGUS_WEIGHT * fungus, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def ndvi_from_stress(stress, fungus):
        return np.clip(1.0 - 2.0 * (stress - STRESS_FUNGUS_WEIGHT * fungus), -1.0, 1.0).astype(np.float32)

    @staticmethod
    def ndvi_per_stress() -> float:
        return 2.0


class DigitalTwin:
    def __init__(
        self,
        field_id: str = "field_alpha",
        grid: Optional[FieldGrid] = None,
        state: Optional[FieldState] = None,
        storage_dir: Optional[Path] = None,
        propagation_model: str = "pde",
        zone_layout: tuple = (4, 4),
        max_events: int = 500,
        crop_model: str = "paddy",
        params: Optional[PaddyParams] = None,
    ):
        if crop_model not in ("paddy", "legacy"):
            raise ValueError("crop_model must be 'paddy' or 'legacy'")
        if state is None:
            if grid is None:
                grid = FieldGrid.centered_on(10.0, 78.0)
            state = FieldState.initial(grid)
        self.field_id = field_id
        self.state = state
        self.propagation_model = propagation_model
        self.zone_layout = tuple(zone_layout)
        self.zone_labels, self.zone_names = state.grid.block_zones(*self.zone_layout)
        self.storage_dir = Path(storage_dir) / field_id if storage_dir else None

        self.events: Deque[Event] = collections.deque(maxlen=max_events)
        self.history: List[Dict[str, Any]] = []
        self._subscribers: List[Subscriber] = []
        self._lock = threading.RLock()
        self.crop_model = crop_model
        self.params = params or PaddyParams()
        self._sim = None        # built lazily (the legacy model imports torch)
        self._sim_state = None  # the FieldState object the model is bound to
        self._model()           # initialise process state (water, GDD, stage)

    # ================================================================ setup
    @classmethod
    def load_or_create(
        cls,
        field_id: str,
        storage_dir: Path,
        grid: Optional[FieldGrid] = None,
        **kwargs: Any,
    ) -> "DigitalTwin":
        state_path = Path(storage_dir) / field_id / "state.npz"
        if state_path.exists():
            state = FieldState.load(state_path)
            twin = cls(field_id=field_id, state=state, storage_dir=storage_dir, **kwargs)
            hist_path = state_path.with_name("history.json")
            if hist_path.exists():
                twin.history = json.loads(hist_path.read_text(encoding="utf-8"))
            twin._emit("twin.loaded", {"sim_day": state.sim_day})
            return twin
        twin = cls(field_id=field_id, grid=grid, storage_dir=storage_dir, **kwargs)
        twin._record_history("init")
        twin._emit("twin.created", {"grid": twin.state.grid.to_dict()})
        twin.save()
        return twin

    def reset(self, values: Optional[Dict[str, float]] = None, seed: Optional[int] = 0) -> None:
        with self._lock:
            self.state = FieldState.initial(self.state.grid, values=values,
                                            growth_stage=self.state.growth_stage, seed=seed)
            self._sim = None
            self._model()
            self._refresh_stress()
            self.history = []
            self._record_history("reset")
            self._emit("twin.reset", {})
            self.save()

    # ============================================================ crop model
    def _model(self):
        if self._sim is None or self._sim_state is not self.state:
            if self.crop_model == "paddy":
                self._sim = PaddyFieldSimulator(self.state, self.zone_labels, self.zone_names,
                                                self.params)
            else:
                self._sim = LegacyCropModel(self.state, self.zone_labels, self.zone_names,
                                            self.propagation_model)
            self._sim_state = self.state
        return self._sim

    # ============================================================== predict
    def step(
        self,
        days: int = 1,
        weather: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None,
        interventions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Advance the twin ``days`` days. ``weather`` may be one dict (reused
        every day) or a per-day list. ``interventions`` follow the simulator
        format ``{"day": 1, "zone_id": 3, "type": "Fungicide Spray"}`` where
        ``day`` is relative to this call (1 = first simulated day).
        """
        if days < 1:
            raise ValueError("days must be >= 1")
        if isinstance(weather, dict) or weather is None:
            weather_seq = [{**DEFAULT_WEATHER, **(weather or {})}] * days
        else:
            weather_seq = [{**DEFAULT_WEATHER, **w} for w in weather]
            if not weather_seq:
                weather_seq = [dict(DEFAULT_WEATHER)]
            while len(weather_seq) < days:
                weather_seq.append(weather_seq[-1])
        interventions = interventions or []

        logs: List[str] = []
        with self._lock:
            model = self._model()
            for d in range(1, days + 1):
                todays = [i for i in interventions if int(i.get("day", 1)) == d]
                model.sync_from_moisture()   # respect assimilated moisture
                day_logs = model.simulate_day(weather=weather_seq[d - 1], interventions=todays)
                self.state.clip()
                da.inflate_process_noise(self.state, days=1.0)
                self._refresh_stress()
                self.state.sim_day += 1
                self.state.last_weather = {k: float(v) for k, v in weather_seq[d - 1].items()
                                           if isinstance(v, (int, float))}
                logs.extend(f"Day {self.state.sim_day}: {line}" for line in day_logs)
                self._record_history("predict")
            summary = self.state.summary()
            self._emit("twin.stepped", {"days": days, "logs": logs[-50:], "summary": summary})
            self.save()
        return {"sim_day": self.state.sim_day, "logs": logs, "summary": summary}

    # =========================================================== assimilate
    def observe_grid(
        self,
        layer: str,
        values: np.ndarray,
        source: str = "multispectral",
        obs_var: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Correct the twin with a raster observation of one layer."""
        with self._lock:
            if layer == "stress":
                # Stress is diagnostic: assimilate the NDVI it implies.
                implied = self._ndvi_from_stress(np.asarray(values, dtype=np.float32))
                r = da.observation_variance(source, "stress", obs_var) * self._model().ndvi_per_stress() ** 2
                result = da.assimilate_grid(self.state, "ndvi", implied, source, r)
                result.layer = "stress->ndvi"
            else:
                result = da.assimilate_grid(self.state, layer, values, source, obs_var)
            self._refresh_stress()
            out = [result.to_dict()]
            self._record_history(f"observe:{source}")
            self._emit("twin.observed", {"kind": "grid", "source": source, "results": out})
            self.save()
        return out

    def observe_point(
        self,
        lat: float,
        lon: float,
        values: Dict[str, float],
        source: str = "phone_rgb",
        obs_var: Optional[Dict[str, float]] = None,
        footprint_m: float = 3.0,
        influence_radius_m: float = 8.0,
    ) -> List[Dict[str, Any]]:
        """Correct the twin with a geotagged reading of one or more layers."""
        obs_var = obs_var or {}
        if self.state.grid.latlon_to_cell(lat, lon) is None:
            raise ValueError(f"Point ({lat:.6f}, {lon:.6f}) lies outside field '{self.field_id}'.")
        with self._lock:
            results = []
            # Fungus first, so a stress reading is converted using the
            # updated fungal load.
            ordered = sorted(values.items(), key=lambda kv: kv[0] != "fungus")
            for layer, value in ordered:
                if layer not in LAYERS or value is None:
                    continue
                if layer == "stress":
                    if "ndvi" in values:
                        continue  # a direct NDVI reading is more informative
                    cell = self.state.grid.latlon_to_cell(lat, lon)
                    fungus = np.float32(self.state.mean["fungus"][cell])
                    model = self._model()
                    implied = float(model.ndvi_from_stress(np.float32(value), fungus))
                    r = da.observation_variance(source, "stress", obs_var.get("stress")) \
                        * model.ndvi_per_stress() ** 2
                    res = da.assimilate_point(self.state, "ndvi", implied, lat, lon, source,
                                              r, footprint_m, influence_radius_m)
                    res.layer = "stress->ndvi"
                else:
                    res = da.assimilate_point(self.state, layer, float(value), lat, lon, source,
                                              obs_var.get(layer), footprint_m, influence_radius_m)
                results.append(res.to_dict())
            self._refresh_stress()
            self._record_history(f"observe:{source}")
            self._emit("twin.observed", {"kind": "point", "source": source, "lat": lat,
                                         "lon": lon, "results": results})
            self.save()
        return results

    def _ndvi_from_stress(self, stress_obs: np.ndarray) -> np.ndarray:
        """
        Invert the crop model's stress relation, so a stress observation is
        assimilated as the NDVI it implies. The observation variance is scaled
        by (d ndvi / d stress)^2 = ndvi_per_stress()^2.
        """
        if stress_obs.shape != self.state.grid.shape:
            stress_obs = da._resample_to(stress_obs, self.state.grid.shape)
        out = self._model().ndvi_from_stress(stress_obs, self.state.mean["fungus"])
        return np.where(np.isfinite(stress_obs), out, np.nan).astype(np.float32)

    def _refresh_stress(self) -> None:
        """
        Recompute the diagnostic stress layer from NDVI and fungus, exactly as
        the crop model does, and propagate their variances (first order).
        """
        model = self._model()
        m, v = self.state.mean, self.state.var
        w = getattr(getattr(model, "p", None), "stress_fungus_weight", STRESS_FUNGUS_WEIGHT)
        m["stress"] = model.stress(m["ndvi"], m["fungus"])
        g = model.ndvi_per_stress()
        v["stress"] = (v["ndvi"] / g ** 2 + (w ** 2) * v["fungus"]).astype(np.float32)
        self.state.clip()

    def set_growth_stage(self, stage: str) -> None:
        with self._lock:
            self._model().set_stage(stage)
            self._refresh_stress()
            self._emit("twin.stage", {"growth_stage": self.state.growth_stage,
                                      "gdd": self.state.scalars.get("gdd")})
            self.save()

    # ============================================================ scenarios
    def fork(self) -> "DigitalTwin":
        """Independent in-memory copy for what-if analysis (never persisted)."""
        with self._lock:
            twin = DigitalTwin(
                field_id=f"{self.field_id}__fork",
                state=self.state.copy(),
                storage_dir=None,
                propagation_model=self.propagation_model,
                zone_layout=self.zone_layout,
                crop_model=self.crop_model,
                params=self.params,
            )
            twin.history = copy.deepcopy(self.history[-30:])
            return twin

    def what_if(
        self,
        days: int,
        interventions: Optional[List[Dict[str, Any]]] = None,
        weather: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare doing nothing against a set of interventions over ``days``."""
        baseline = self.fork().step(days=days, weather=weather)
        scenario = self.fork().step(days=days, weather=weather, interventions=interventions)
        b, s = baseline["summary"]["layers"], scenario["summary"]["layers"]
        return {
            "days": days,
            "interventions": interventions or [],
            "baseline": baseline["summary"],
            "scenario": scenario["summary"],
            "delta": {k: round(s[k]["mean"] - b[k]["mean"], 4) for k in LAYERS},
            "scenario_logs": scenario["logs"],
        }

    # ========================================================== events/IO
    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subscribers.append(fn)

        def _unsubscribe() -> None:
            if fn in self._subscribers:
                self._subscribers.remove(fn)
        return _unsubscribe

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        event = {"type": kind, "field_id": self.field_id, "timestamp": _now(),
                 "sim_day": self.state.sim_day, **payload}
        self.events.append(event)
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.storage_dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=float) + "\n")
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:
                pass

    def _record_history(self, reason: str) -> None:
        s = self.state
        self.history.append({
            "sim_day": s.sim_day,
            "reason": reason,
            "timestamp": _now(),
            **{f"{k}_mean": round(float(s.mean[k].mean()), 4) for k in LAYERS},
            **{f"{k}_std": round(float(np.sqrt(s.var[k]).mean()), 4) for k in LAYERS},
        })
        if len(self.history) > 2000:
            self.history = self.history[-2000:]

    def save(self) -> None:
        if self.storage_dir is None:
            return
        self.state.save(self.storage_dir / "state.npz")
        (self.storage_dir / "history.json").write_text(json.dumps(self.history), encoding="utf-8")

    # ============================================================== views
    def zone_summary(self) -> List[Dict[str, Any]]:
        out = []
        for z_id, name in self.zone_names.items():
            mask = self.zone_labels == z_id
            out.append({
                "zone_id": int(z_id),
                "name": name,
                **{k: round(float(self.state.mean[k][mask].mean()), 4) for k in LAYERS},
            })
        return out
