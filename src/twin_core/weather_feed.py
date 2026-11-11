"""
weather_feed.py
---------------
Daily weather forcing for the twin.

Two sources share one output format (a list of per-day dicts accepted by
``DigitalTwin.step``):

* ``openmeteo``  live forecast via src/weather/openmeteo_client.py
* ``synthetic``  seeded monsoon-season generator for offline simulation

``get_weather(..., source="auto")`` tries Open-Meteo and falls back to the
synthetic generator, reporting which one was used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.twin_core.twin import DEFAULT_WEATHER

Weather = Dict[str, float]


def openmeteo_daily_to_twin(daily: Dict[str, List[Any]]) -> List[Weather]:
    """Convert ``fetch_daily_forecast`` output into twin weather dicts."""
    dates = daily.get("dates", [])

    def col(name: str, i: int, default: float) -> float:
        values = daily.get(name) or []
        v = values[i] if i < len(values) else None
        return float(v) if v is not None else default

    out: List[Weather] = []
    for i in range(len(dates)):
        tmax = col("temperature_2m_max", i, DEFAULT_WEATHER["temperature"] + 4)
        tmin = col("temperature_2m_min", i, DEFAULT_WEATHER["temperature"] - 4)
        rh_max = col("relative_humidity_2m_max", i, DEFAULT_WEATHER["humidity"] + 10)
        rh_min = col("relative_humidity_2m_min", i, DEFAULT_WEATHER["humidity"] - 10)
        rain = col("precipitation_sum", i, 0.0)
        out.append({
            "temperature": round((tmax + tmin) / 2.0, 2),
            "humidity": round((rh_max + rh_min) / 2.0, 1),
            "precipitation": round(rain, 2),
            "wind_speed": round(col("windspeed_10m_max", i, DEFAULT_WEATHER["wind_speed"]), 1),
            "wind_direction": DEFAULT_WEATHER["wind_direction"],
            "precipitation_probability": 90.0 if rain > 1.0 else 15.0,
            "et0": round(col("et0_fao_evapotranspiration", i, 4.5), 2),
        })
    return out


def synthetic_weather(days: int, seed: Optional[int] = 0, start_day: int = 0) -> List[Weather]:
    """
    Plausible south-west/north-east monsoon weather for a Tamil Nadu paddy:
    warm, humid, with clustered rain spells. Deterministic for a given seed
    and ``start_day``, so a truth field and its twin can share a weather
    stream (or perturb it on purpose).
    """
    rng = np.random.default_rng(None if seed is None else seed + 7919 * start_day)
    out: List[Weather] = []
    wet = bool(rng.random() < 0.4)
    for _ in range(days):
        # Two-state Markov chain gives realistic multi-day rain spells.
        wet = rng.random() < (0.65 if wet else 0.25)
        rain = float(rng.gamma(1.6, 9.0)) if wet else 0.0
        temp = float(rng.normal(28.0 if wet else 30.5, 1.2))
        hum = float(np.clip(rng.normal(88.0 if wet else 74.0, 4.0), 45.0, 100.0))
        out.append({
            "temperature": round(temp, 2),
            "humidity": round(hum, 1),
            "precipitation": round(rain, 2),
            "wind_speed": round(float(np.clip(rng.normal(11.0, 4.0), 0.0, 40.0)), 1),
            "wind_direction": round(float(rng.normal(225.0, 30.0)) % 360.0, 1),
            "precipitation_probability": 85.0 if wet else 20.0,
        })
    return out


def _fetch_openmeteo_daily(lat: float, lon: float, days: int, timeout: float = 5.0) -> Dict[str, List[Any]]:
    """
    Single short request (no retry loop), so ``source="auto"`` falls back to
    synthetic weather quickly when offline. Uses the same variables as
    src/weather/openmeteo_client.py.
    """
    import requests

    from src.weather.openmeteo_client import BASE_URL, DAILY_VARS

    resp = requests.get(BASE_URL, timeout=timeout, params={
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "forecast_days": min(16, max(1, days)),
        "timezone": "auto",
    })
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    return {"dates": daily.get("time", []), **{v: daily.get(v, []) for v in DAILY_VARS}}


def get_weather(
    lat: float,
    lon: float,
    days: int,
    source: str = "auto",
    seed: Optional[int] = 0,
    start_day: int = 0,
) -> Tuple[List[Weather], str]:
    """Return (per-day weather, source actually used)."""
    if source not in ("auto", "openmeteo", "synthetic"):
        raise ValueError(f"Unknown weather source '{source}'")
    if source in ("auto", "openmeteo"):
        try:
            weather = openmeteo_daily_to_twin(_fetch_openmeteo_daily(lat, lon, days))
            if weather:
                while len(weather) < days:
                    weather.append(weather[-1])
                return weather[:days], "openmeteo"
        except Exception:
            if source == "openmeteo":
                raise
    return synthetic_weather(days, seed=seed, start_day=start_day), "synthetic"
