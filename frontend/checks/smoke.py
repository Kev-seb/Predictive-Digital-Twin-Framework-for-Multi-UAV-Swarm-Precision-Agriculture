"""Integration smoke checks for navigation, shared state, and demo analysis."""
from pathlib import Path
import json
import hashlib
import tempfile
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
temp_root = ROOT / "frontend/runtime/test_temp"
temp_root.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_root)
os.environ["GARUDA_TWIN_MEMORY_DIR"] = str(ROOT / "frontend/runtime/smoke_twin_history")
sys.path.insert(0, str(ROOT / "frontend"))
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash
from navigation import GROUPS

app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=90).run()

def verify(name):
    errors = [e.message for e in app.exception]
    print(name, "PASS" if not errors else errors, flush=True)
    assert not errors, f"{name}: {errors}"

verify("Overview without a survey")
for group, items in GROUPS.items():
    for route, title, icon, file in items:
        if route in {"overview", "upload"}:
            continue
        app._page_hash = calc_hash(route)
        app.run()
        verify(title + " initial screen")
app._page_hash = calc_hash("upload")  # AppTest.switch_page hashes filenames, not custom URL paths.
app.run()
verify("Upload")
assert len(app.toggle) == 1, "Upload route did not render"
app.toggle[0].set_value(True).run()
verify("Demo survey processing")
assert app.session_state["indices"]["ndvi"].shape == (256, 256)
app.selectbox(key="crop_stage").set_value("Flowering").run()
for slug, route in [
    ("Vegetation", "vegetation"),
    ("Field zoning", "zoning"),
    ("Temporal", "temporal"),
    ("Treatment optimizer", "optimizer"),
    ("Digital twin", "twin"),
    ("Reports", "reports"),
    ("Overview with demo", "overview"),
]:
    app._page_hash = calc_hash(route)
    app.run()
    verify(slug)
    assert app.session_state["crop_stage"] == "Flowering", "Field settings were lost during navigation"

for entry in json.loads((ROOT / "frontend/checks/original_ui_hashes.json").read_text(encoding="utf-8-sig")):
    assert hashlib.sha256((ROOT / entry["Path"]).read_bytes()).hexdigest().upper() == entry["Hash"], entry["Path"]
print("Original dashboard hashes PASS", flush=True)
