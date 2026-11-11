"""
twin_core
---------
Unified, uncertainty-aware digital twin of a simulated paddy field.

    from src.twin_core import DigitalTwin, FieldGrid
    twin = DigitalTwin.load_or_create("field_alpha", storage_dir="outputs/twin_core")
    twin.step(days=3)
    twin.observe_point(lat, lon, {"stress": 0.6}, source="drone_rgb")
"""

from src.twin_core.field_state import LAYERS, FieldGrid, FieldState
from src.twin_core.twin import DEFAULT_WEATHER, DigitalTwin

__all__ = ["DigitalTwin", "FieldGrid", "FieldState", "LAYERS", "DEFAULT_WEATHER"]
