"""
config.py
---------
Centralised project configuration using Pydantic BaseSettings.

Settings are loaded from environment variables or a .env file,
with sensible defaults for local development.

Usage
-----
    from src.config.config import settings

    print(settings.data_root)
    print(settings.model_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]
    _PYDANTIC_V2 = False


from pydantic import BaseModel

class SegmentationSettings(BaseModel):
    checkpoint_dir: Path = Path(__file__).resolve().parents[2] / "models" / "segmentation" / "checkpoints"
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    num_classes: int = 5
    encoder: str = "resnet50"
    encoder_weights: Optional[str] = None
    patch_size: int = 256


class AppSettings(BaseSettings):
    """
    All configurable parameters for the UAV Crop Stress Intelligence system.
    Override via environment variables (prefixed UAV_) or a .env file.
    """

    # --- Paths ---
    project_root: Path  = Path(__file__).resolve().parents[2]
    data_root:    Path  = Path(__file__).resolve().parents[2] / "data"
    model_dir:    Path  = Path(__file__).resolve().parents[2] / "models"
    output_dir:   Path  = Path(__file__).resolve().parents[2] / "outputs"
    log_dir:      Path  = Path(__file__).resolve().parents[2] / "outputs" / "logs"

    # --- Multispectral loader ---
    image_size:   int   = 224
    in_channels:  int   = 4          # Green, Red, RedEdge, NIR
    normalize_reflectance: bool = True

    # --- Segmentation ---
    segmentation: SegmentationSettings = SegmentationSettings()

    # --- Training ---
    batch_size:     int   = 16
    num_epochs:     int   = 50
    learning_rate:  float = 1e-4
    weight_decay:   float = 1e-4
    num_workers:    int   = 4
    device:         str   = "cpu"    # "cuda" | "mps" | "cpu"

    # --- GIS ---
    default_crs:    str   = "EPSG:4326"
    gsd_meters:     float = 0.05
    grid_rows:      int   = 5
    grid_cols:      int   = 5
    field_center_lat: float = 10.0
    field_center_lon: float = 78.0

    # --- Weather / Open-Meteo ---
    openmeteo_base_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_forecast_days: int = 7
    weather_lat: float = 10.0
    weather_lon: float = 78.0

    # --- Database ---
    db_url: str = "sqlite:///outputs/crop_stress.db"

    # --- Dashboard ---
    dashboard_host: str  = "127.0.0.1"
    dashboard_port: int  = 8501
    debug_mode:     bool = False

    # --- Digital twin core (src/twin_core) ---
    twin_api_host:    str   = "127.0.0.1"
    twin_api_port:    int   = 8008     # digital_twin_api.py (UE5 + /twin/*)
    mobile_api_port:  int   = 8080     # src/mobile_api/mobile_api_server.py
    twin_field_id:    str   = "field_alpha"
    twin_grid_rows:   int   = 40
    twin_grid_cols:   int   = 40
    twin_cell_size_m: float = 2.5      # 40 x 2.5 m = 100 m, ~1 ha
    twin_ue5_grid:    int   = 20       # UE5 HISM grid is 20 x 20
    twin_propagation_model: Literal["baseline", "pde", "hybrid", "gnn"] = "pde"
    twin_storage_dir: Path  = Path(__file__).resolve().parents[2] / "outputs" / "twin_core"

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    if _PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_prefix="UAV_",
            env_file=".env",
            env_file_encoding="utf-8",
        )
    else:
        class Config:
            env_prefix = "UAV_"
            env_file = ".env"
            env_file_encoding = "utf-8"

    # ── convenience ──────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.data_root, self.model_dir, self.output_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
settings = AppSettings()
