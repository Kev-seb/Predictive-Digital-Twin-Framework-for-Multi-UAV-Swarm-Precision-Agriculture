"""Every dashboard page renders without an exception, and the upload-first workflow reaches the report.

Needs the full dashboard dependencies (requirements.txt); skipped where they are missing (the minimal CI).
Run locally:  python -m pytest tests/test_dashboard_pages.py -q
"""
import importlib.util
from pathlib import Path

import pytest

DEPS = ("streamlit", "cv2", "torch", "torchvision", "rasterio", "geopandas", "plotly", "segmentation_models_pytorch")
missing = [m for m in DEPS if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(missing), reason=f"dashboard dependencies missing: {missing}")

APP = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "dashboard.py")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from src.storage import field_store as fs
    monkeypatch.setattr(fs, "DB", tmp_path / "t.db")
    monkeypatch.setattr(fs, "FILES", tmp_path / "files")
    return fs


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=900)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_every_page_renders(store):
    from src.dashboard.ui_shell import PAGES
    at = _app()
    at.run()
    _button(at, "Use a synthetic demo field").click()
    at.run()
    for key in PAGES:
        if key in ("satellite", "landsat", "livefield"):     # need cloud credentials / live hardware
            continue
        at.session_state["pt_page"] = key
        at.run()
        assert not at.exception, f"{key}: {[str(e.value)[:300] for e in at.exception]}"


def test_workflow_from_upload_to_report(store):
    at = _app()
    at.run()
    assert at.session_state["pt_page"] == "upload"
    _button(at, "Use the sample GeoTIFF").click()
    at.run()
    assert "ms_image" in at.session_state and at.session_state["field_geo"] is not None
    assert len(store.list_fields()) == 1                      # the field was saved
    for page, action in (("vegetation", None), ("stress", None), ("zoning", None),
                         ("weather", "Fetch Weather Data"), ("twin", "Run Scenario Simulation Model"),
                         ("optimizer", "Run AI Optimization Solver"), ("report", "Build the field report (PDF)")):
        at.session_state["pt_page"] = page
        at.run()
        if action:
            _button(at, action).click()
            at.run()
        assert not at.exception, f"{page}: {[str(e.value)[:300] for e in at.exception]}"
    assert at.session_state["field_report_pdf"][:4] == b"%PDF"


def test_farmer_view_in_tamil(store):
    at = _app()
    at.run()
    at.sidebar.selectbox(key="pt_role").set_value("farmer")
    at.run()
    at.sidebar.selectbox(key="pt_lang").set_value("ta")
    at.run()
    assert not at.exception
    nav = [b.label for b in at.sidebar.button if b.key and b.key.startswith("nav_")]
    assert nav[0] == "வயல் முகப்பு" and len(nav) == 5
