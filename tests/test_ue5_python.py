"""
Runs the real UE5 editor script (src/ue5_integration/python/twin_live.py)
against the real twin API, with a fake `unreal` module standing in for the
editor (tests/fakes/unreal.py).
"""

import importlib.util
import math
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
        if isinstance(url, urllib.request.Request):           # POST from the menu actions
            path = url.full_url.split("8008", 1)[1]
            r = client.post(path, content=url.data, headers={"Content-Type": "application/json"})
            if r.status_code >= 400:
                raise urllib.error.HTTPError(url.full_url, r.status_code, r.text, {}, None)
            return _Resp(r.content)
        path = url.split("8008", 1)[1]
        return _Resp(client.get(path).content)

    import urllib.error
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    yield client
    client.__exit__(None, None, None)


def _run_script():
    spec = importlib.util.spec_from_file_location("twin_live", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.twin_live_start()        # importing only defines it; UE's File > Execute runs it as __main__
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
        drone = next(iter(live.drones.values()))
        assert drone["body"].scale.x == pytest.approx(4.0 * 0.32)       # hub of a 4 m quadcopter
        assert len(drone["parts"]) == 11                                  # 2 arms, 4 motors, 4 rotors, tank
        rotors = [a for a, _, _, spins in drone["parts"] if spins]
        arm = (rotors[0].location - drone["body"].location)
        assert math.hypot(arm.x, arm.y) == pytest.approx(0.36 * 400, rel=0.01)   # rotors on the arm tips
        assert live.caption.text_render.size > 300 and live.caption.text_render.color.r == 255
        # Drones flew over the field (grid 20 x 20 cells of 500 cm from the origin).
        xs = [d["pos"].x for d in live.drones.values()]
        assert all(-500 <= x <= 20 * 500 + 500 for x in xs)
        # Twin and reality fields were coloured from the stream and differ from the default green.
        colours = [mid.params["Color"] for _, mid, _ in live.truth_field.cells]
        assert max(c.r for c in colours) > 0.2                                # outbreak visible in reality
        heights = {round(a.scale.z, 3) for a, _, _ in live.twin_field.cells}
        assert len(heights) > 10                                              # NDVI drives crop height
        forecast = [mid.params["Color"] for _, mid, _ in live.forecast_field.cells]
        assert len({(round(c.r, 3), round(c.g, 3)) for c in forecast}) > 10   # forecast field coloured
        assert any("drones scouting" in c and "twin vs reality error" in c for c in captions), captions
        panel = live.panel.text_render.text.split("\n")
        assert panel[0].startswith("DAY ") and "Expected harvest" in panel[1]
        assert any(l.startswith("Fungicide:") and "saved" in l for l in panel), panel
        assert any(l.startswith("Twin vs reality error:") for l in panel)
        assert all(ord(ch) < 128 for ch in live.panel.text_render.text)      # default UE font safe
    finally:
        mod.twin_live_stop()
    assert not fake_unreal._LEVEL and not fake_unreal._TICKS                 # stop() cleans up


def test_camera_framed_and_texts_face_it(fake_unreal, api):
    import math
    mod, live = _run_script()
    try:
        cam = fake_unreal.UnrealEditorSubsystem.camera
        assert cam is not None, "camera should be framed on start"
        loc, rot = cam
        assert loc.z > live.center.z and loc.y > live.center.y          # raised, south of the fields
        assert -60.0 < rot.pitch < -15.0                                  # looking down at the fields
        assert [t.text_render.text for t in live.labels] == ["TWIN", "REALITY", "FORECAST +7 DAYS"]
        fake_unreal._tick_all()
        for t in live.texts:                                              # readable side (+X) faces camera
            d = loc - t.location
            assert abs(math.remainder(t.rotation.yaw - math.degrees(math.atan2(d.y, d.x)), 360.0)) < 1e-6
        # Left to right as seen from the camera (looking north, -Y): twin, reality, forecast.
        assert live.labels[0].location.x < live.labels[1].location.x < live.labels[2].location.x
        # Every text has a dark board just behind it (away from the camera), sized to the text.
        assert len(live.boards) == len(live.texts)
        cam_loc = fake_unreal.UnrealEditorSubsystem.camera[0]
        for t in live.texts:
            board = mod._TEXTS[t.get_actor_label()]["board"]
            d_text = (t.location - cam_loc).length()
            d_board = (board.location - cam_loc).length()
            assert d_board > d_text and board.rotation.yaw == pytest.approx(t.rotation.yaw)
        before = mod._TEXTS["TwinResults"]["board"].scale
        mod._set_text(live.panel, "line one\nline two that is longer\nthree\nfour")
        after = mod._TEXTS["TwinResults"]["board"].scale
        assert after.z > before.z * 1.5 and after.y > 0                       # grows with the text
        # The caption sits in front of the fields, nearer the camera than any field.
        assert live.caption.location.y > live.twin_field.grid_to_world(20.0, 0.0, 0.0).y
        # An outbreak reads red even where the twin is still unsure.
        r, g, b = mod.crop_color(stress=0.5, fungus=0.45, std=0.2)
        assert r > 0.4 and r > 2 * g and r > 2 * b
        fake_unreal.UnrealEditorSubsystem.camera = None
        mod.twin_live_camera()
        assert fake_unreal.UnrealEditorSubsystem.camera is not None
    finally:
        mod.twin_live_stop()
    assert not fake_unreal._LEVEL


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


def test_menu_registered_and_actions_drive_the_twin(fake_unreal, api):
    import builtins
    mod, live = _run_script()
    try:
        menu = fake_unreal.ToolMenus.get().find_menu("LevelEditor.MainMenu.DigitalTwin")
        assert menu is not None and menu.label == "Digital Twin"
        labels = [e.label for _, e in menu.entries]
        assert "Scout now" in labels and "Inject outbreak (selected block / view centre)" in labels
        assert all(e.command[1].startswith("twin_live_") for _, e in menu.entries)
        api.post("/sim/reset", json={"seed": 5})

        def wait_idle():
            end = time.time() + 30
            while time.time() < end:
                fake_unreal._tick_all()
                if not api.get("/demo/status").json()["running"]:
                    return
                time.sleep(0.05)

        # Inject an outbreak at the selected Reality block (row 15, col 4).
        block = next(a for a, _, _ in live.truth_field.cells if a.label == "Reality_15_04")
        fake_unreal.EditorActorSubsystem.selected = [block]
        assert mod._target_cell(live) == (15.5, 4.5)
        before = api.get("/sim/truth/layers/fungus?rows=20&cols=20").json()["values"][15][4]
        exec(menu.entries[[e.label for _, e in menu.entries].index(
            "Inject outbreak (selected block / view centre)")][1].command[1], vars(builtins))
        time.sleep(0.3)
        wait_idle()
        after = api.get("/sim/truth/layers/fungus?rows=20&cols=20").json()["values"][15][4]
        assert after > before + 0.3

        # Without a selection, the view centre picks the cell: aim at the twin field's (5, 12).
        fake_unreal.EditorActorSubsystem.selected = []
        target = live.twin_field.grid_to_world(5.0, 12.0, 0.0)
        cam = target + fake_unreal.Vector(0.0, 3000.0, 3000.0)
        fake_unreal.UnrealEditorSubsystem.camera = (cam, mod._look_rotation(cam, target))
        r, c = mod._target_cell(live)
        assert abs(r - 5.0) < 0.05 and abs(c - 12.0) < 0.05

        # "Scout now" runs a scouting flight that the script animates.
        builtins.twin_live_action("scout")
        time.sleep(0.3)
        wait_idle()
        assert len(live.drones) >= 1
    finally:
        mod.twin_live_stop()


def test_project_assets_and_rain(fake_unreal, api):
    """Asset paths swap in your meshes / Niagara; rain follows the real field's weather."""
    mod, live = _run_script()
    mod.twin_live_stop()
    fake_unreal.MISSING_ASSETS.add("/Game/FX/NS_Missing")
    mod.DRONE_MESH = "/Game/Drones/SM_AgriDrone"
    mod.CROP_MESH = "/Game/Crops/SM_Rice"
    mod.SPRAY_NIAGARA = "/Game/FX/NS_Spray"
    mod.RAIN_NIAGARA = "/Game/FX/NS_Missing"           # missing -> falling-streak fallback
    mod.twin_live_start()
    import builtins
    live = builtins._twin_live
    try:
        actor, mid, loc = live.twin_field.cells[0]
        assert live.twin_field.crop and actor.static_mesh_component.mesh == "/Game/Crops/SM_Rice"
        assert any("NS_Missing" in m and "WARNING" in m for m in fake_unreal.LOG)
        assert live.rain_streaks and all(a.hidden for a in live.rain)
        live._apply_drones({"d1": [5.0, 5.0, 4.0, True, 90.0]})
        d = live.drones["d1"]
        assert d["body"].static_mesh_component.mesh == "/Game/Drones/SM_AgriDrone" and not d["parts"]
        assert isinstance(d["cone"], fake_unreal.NiagaraActor)
        fake_unreal._tick_all()
        assert d["cone"].niagara_component.active and not d["cone"].hidden     # spraying
        live._apply_drones({"d1": [5.0, 6.0, 4.0, False, 90.0]})
        fake_unreal._tick_all()
        assert not d["cone"].niagara_component.active
        # Rain: shown on a wet day, streaks fall, hidden again when dry.
        with live.lock:
            live.frame = {"seq": live.seq, "weather": {"precipitation_mm": 12.0}}
        z0 = [p[2] for p in live.rain_streaks]
        fake_unreal._tick_all()
        assert all(not a.hidden for a in live.rain)
        assert [p[2] for p in live.rain_streaks] != z0
        with live.lock:
            live.frame = {"seq": live.seq, "weather": {"precipitation_mm": 0.0}}
        fake_unreal._tick_all()
        assert all(a.hidden for a in live.rain)
        # The API reports the weather each frame.
        w = api.get("/ue5/frame").json()["weather"]
        assert set(w) == {"precipitation_mm", "temperature_c", "humidity_pct", "wind_speed_m_s"}
    finally:
        mod.twin_live_stop()
    assert not fake_unreal._LEVEL
