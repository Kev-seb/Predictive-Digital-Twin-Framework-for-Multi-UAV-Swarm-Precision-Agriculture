"""
Runs the real UE5 editor script (src/ue5_integration/python/twin_live.py)
against the real twin API, with a fake `unreal` module standing in for the
editor (tests/fakes/unreal.py).
"""

import importlib.util
import io
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "ue5_integration" / "python" / "twin_live.py"


@pytest.fixture
def fake_unreal(monkeypatch):
    sys.path.insert(0, str(ROOT / "tests" / "fakes"))
    import unreal
    unreal._reset()
    monkeypatch.setitem(sys.modules, "unreal", unreal)
    yield unreal
    unreal._reset()
    sys.path.remove(str(ROOT / "tests" / "fakes"))


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Serve the twin API in-process and route the script's HTTP calls to it."""
    from fastapi.testclient import TestClient

    from digital_twin_api import create_app
    from src.twin_core import DigitalTwin, FieldGrid

    twin = DigitalTwin.load_or_create("ue5_field", tmp_path, grid=FieldGrid.centered_on(10.0, 78.0))
    client = TestClient(create_app(twin)).__enter__()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        path = url.split("8008", 1)[1]
        return _Resp(client.get(path).content)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    yield client
    client.__exit__(None, None, None)


def _run_script():
    spec = importlib.util.spec_from_file_location("twin_live", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import builtins
    return mod, builtins._twin_live


def _pump_until(unreal, live, cond, seconds=60.0):
    end = time.time() + seconds
    while time.time() < end:
        unreal._tick_all()
        if cond():
            return True
        time.sleep(0.03)
    return False


def _demo(api):
    r = api.post("/demo/start", json={"days": 3, "scout_every": 3, "day_seconds": 0.2,
                                      "playback_speed": 60, "seed": 7})
    assert r.status_code == 200


def test_blocks_mode_shows_field_drones_and_story(fake_unreal, api):
    mod, live = _run_script()
    try:
        assert isinstance(live.twin_field, mod.BlocksField) and len(live.twin_field.cells) == 400
        assert len(live.truth_field.cells) == 400
        assert _pump_until(fake_unreal, live, lambda: live.seq > 0)            # first field applied
        _demo(api)
        seen_spray = {"v": False}
        captions = set()

        def done():
            if any(not d["cone"].hidden for d in live.drones.values()):
                seen_spray["v"] = True
            captions.add(live.caption.text_render.text)
            return "Demo finished" in live.caption.text_render.text
        assert _pump_until(fake_unreal, live, done), live.caption.text_render.text
        assert len(live.drones) == 3 and seen_spray["v"]
        # Drones flew over the field (grid 20 x 20 cells of 500 cm from the origin).
        xs = [d["pos"].x for d in live.drones.values()]
        assert all(-500 <= x <= 20 * 500 + 500 for x in xs)
        # Twin and reality fields were coloured from the stream and differ from the default green.
        colours = [mid.params["Color"] for _, mid, _ in live.truth_field.cells]
        assert max(c.r for c in colours) > 0.2                                # outbreak visible in reality
        heights = {round(a.scale.z, 3) for a, _, _ in live.twin_field.cells}
        assert len(heights) > 10                                              # NDVI drives crop height
        assert any("drones scouting" in c and "twin vs reality error" in c for c in captions), captions
    finally:
        mod.twin_live_stop()
    assert not fake_unreal._LEVEL and not fake_unreal._TICKS                 # stop() cleans up


def test_hism_mode_writes_custom_data_and_maps_grid(fake_unreal, api):
    unreal = fake_unreal
    def field(label, x0):
        owner = unreal.Actor()
        owner.set_actor_label(label)
        unreal._LEVEL.append(owner)
        centers = [unreal.Vector(x0 + (c + 0.5) * 300.0, (r + 0.5) * 300.0, 50.0)
                   for r in range(20) for c in range(20)]
        return unreal.HierarchicalInstancedStaticMeshComponent(owner, centers)

    twin_comp = field("BP_FieldDataController", 0.0)
    real_comp = field("Reality_Field", 8000.0)
    mod, live = _run_script()
    try:
        assert isinstance(live.twin_field, mod.HismField) and live.twin_field.comp is twin_comp
        assert live.truth_field.comp is real_comp
        corner = live.twin_field.grid_to_world(0.0, 0.0, 0.0)
        assert (round(corner.x), round(corner.y)) == (0, 0)
        far = live.twin_field.grid_to_world(20.0, 20.0, 10.0)
        assert (round(far.x), round(far.y)) == (6000, 6000)
        assert far.z == pytest.approx(50.0 + 10.0 * 60.0)                     # 300 cm cells = 60 cm per metre
        _demo(api)
        assert _pump_until(unreal, live, lambda: "Demo finished" in live.caption.text_render.text)
        assert twin_comp.num_custom_data_floats == 6 and len(twin_comp.custom) == 400
        assert all(len(v) == 6 for v in twin_comp.custom.values()) and twin_comp.dirty_marks > 0
        assert len(real_comp.custom) == 400
        assert all(v[2] == 0.0 for v in real_comp.custom.values())            # reality has no uncertainty
        assert any(v[2] > 0.0 for v in twin_comp.custom.values())             # the twin does
    finally:
        mod.twin_live_stop()
