"""
twin_live.py - show the paddy-field digital twin live in the UE5 editor.

No C++ and no extra plugins: runs in UE5's built-in Python
(Edit > Plugins > "Python Editor Script Plugin", then restart the editor).

  1. Start the twin:            python digital_twin_api.py
  2. In UE5: File > Execute Python Script... > twin_live.py
     (or in the Output Log, Python mode:  exec(open(r"C:/path/to/twin_live.py").read()) )
  3. Start the story:           curl -X POST localhost:8008/demo/start -H 'content-type: application/json' -d '{}'
  4. Stop the script:           run it again, or in the Python console: twin_live_stop()
     Re-frame the camera:       twin_live_camera()

The script also adds a "Digital Twin" menu to the editor's main menu bar:
Start demo, Next day, Scout now, Spray now, Scout + spray, Inject outbreak
(at the selected field block, or where the view centre meets the field),
Reset, Stop, Re-frame camera.

Field modes (chosen automatically, see FIELD_MODE):
  "hism"   your field: an actor with an (H)ISM component of 20 x 20 = 400
           instances, instance index = row * 20 + col (row 0 = north edge).
           Per-instance custom data slots 0-5 are written for the crop
           material: NDVI, stress, stress std, moisture, fungus, anomaly.
           A second 400-instance field whose actor label contains "Reality"
           gets the hidden truth.
  "blocks" no setup at all: builds TWIN, REALITY and FORECAST fields.
           FIELD_LOOK = "paddy" (default): rice patches in water with mud
           bunds; the rice mesh and materials are generated on the first run
           into ASSET_DIR. Plant height = NDVI, leaf colour = stress /
           fungus (grey where the twin is unsure), water = soil moisture.
           FIELD_LOOK = "blocks": coloured engine cubes.

Controls: Digital Twin buttons on the level editor toolbar (Demo, Stop,
Next day, Scout, Spray, Outbreak, Reset, Overview, Close-up, Follow drone),
and the same in the Digital Twin menu. In the Python console:
twin_live_action('scout'), twin_live_view('closeup').

Drones are spheres (spray cone under them while spraying); a caption shows
the demo's narration and the twin-vs-reality error. On start the editor
camera is framed in a raised three-quarter view over the fields, and the
caption and the TWIN / REALITY labels always turn to face the camera.
"""

import json
import math
import threading
import time
import urllib.error
import urllib.request

import unreal

# ------------------------------------------------------------------ settings
API = "http://127.0.0.1:8008"
POLL_HZ = 10.0
FIELD_MODE = "auto"              # "auto" | "hism" | "blocks"
TWIN_ACTOR_LABEL = None          # hism mode: label of your field actor (None = first 400-instance ISM)
REALITY_LABEL_HINT = "Reality"   # hism mode: label substring of the optional reality field
FORECAST_LABEL_HINT = "Forecast" # hism mode: label substring of the optional forecast field
SHOW_FORECAST = True             # blocks mode: third field with the twin's 7-day prediction
SHOW_PANEL = True                # results panel: harvest, fungicide saved, twin accuracy
BLOCKS_ORIGIN = unreal.Vector(0.0, 0.0, 0.0)   # blocks mode: north-west corner of the twin field (cm)
BLOCKS_GAP_CELLS = 4             # blocks mode: gap between twin and reality fields
CELL_CM = 500.0                  # blocks mode: 5 m cells, like the twin's 20 x 20 grid
GRID = 20
SMOOTHING = 0.35                 # drone motion smoothing per editor tick (0..1)
AUTO_FRAME_CAMERA = True         # frame the editor camera over the fields on start
FIELD_LOOK = "paddy"              # blocks mode: "paddy" (rice plants in water) or "blocks" (coloured cubes)
ASSET_DIR = "/Game/TwinDemo"     # where the generated rice mesh and materials are saved
SHOW_LEGEND = True               # colour key under the caption
LEGEND_TEXT = ("Leaves: green healthy  |  yellow stressed  |  red-brown diseased  |  grey = twin unsure\n"
               "Water: glassy = flooded  |  brown = drying out")
SHOW_TOOLBAR = True              # Digital Twin buttons on the toolbar above the viewport
SHOW_BOARDS = True               # dark boards behind caption, labels and panel (readable over a bright sky)
DRONE_SIZE_M = 4.0               # drawn drone size; real ~1.6 m drones vanish over a 100 m field

# --- Realistic look: paths of assets in YOUR project (right-click asset > Copy Reference,
# keep the part between the quotes). None = built-in look, so the script runs in any project.
DRONE_MESH = None        # static mesh, nose along +X, e.g. "/Game/Drones/SM_AgriDrone.SM_AgriDrone"
DRONE_MESH_SCALE = 1.0   # extra scale for DRONE_MESH (1.0 = sized to DRONE_SIZE_M if the mesh is ~1 m)
SPRAY_NIAGARA = None     # Niagara system under each spraying drone, e.g. "/Game/FX/NS_Spray.NS_Spray"
RAIN_NIAGARA = None      # Niagara rain over the fields on rainy days, e.g. "/Game/FX/NS_Rain.NS_Rain"
CROP_MESH = None         # blocks mode: crop mesh per cell instead of cubes, e.g. a rice clump
CROP_COLOR_PARAM = "Color"   # vector parameter on CROP_MESH's material that takes the health colour
SHOW_RAIN = True         # rain on days the real field gets more than RAIN_MM of rain
RAIN_MM = 1.0

_SPHERE = "/Engine/BasicShapes/Sphere"
_CUBE = "/Engine/BasicShapes/Cube"
_CONE = "/Engine/BasicShapes/Cone"
_CYLINDER = "/Engine/BasicShapes/Cylinder"
_SHAPE_MAT = "/Engine/BasicShapes/BasicShapeMaterial"   # has a "Color" vector parameter
_DRONE_COLORS = [(0.22, 0.74, 0.97), (0.93, 0.28, 0.6), (0.64, 0.9, 0.21), (0.96, 0.62, 0.04),
                 (0.65, 0.55, 0.98), (0.18, 0.83, 0.75)]


def _actors():
    try:
        return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        return unreal.EditorLevelLibrary.get_all_level_actors()


def _spawn(cls, location, label):
    try:
        actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
            cls, location, unreal.Rotator(0.0, 0.0, 0.0))
    except Exception:
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(cls, location, unreal.Rotator(0.0, 0.0, 0.0))
    actor.set_actor_label(label)
    return actor


def _destroy(actor):
    try:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
    except Exception:
        unreal.EditorLevelLibrary.destroy_actor(actor)


_ASSETS = {}


def _asset(path):
    """Load an asset once; None (with one warning) when the path is unset or missing."""
    if not path:
        return None
    if path not in _ASSETS:
        try:
            _ASSETS[path] = unreal.load_asset(path)
        except Exception:
            _ASSETS[path] = None
        if _ASSETS[path] is None:
            unreal.log_warning(f"[twin_live] asset '{path}' not found; using the built-in look")
    return _ASSETS[path]


def _mesh_actor(mesh_path, location, scale, color, label, own_material=False, color_param="Color",
                material=None, rotation=None):
    """Static mesh actor tinted through a dynamic material instance.

    own_material=True keeps the mesh's own material (your crop or drone mesh)
    and only sets ``color_param`` on it; engine shapes get BasicShapeMaterial.
    ``mesh_path`` may also be a loaded mesh, and ``material`` a loaded material.
    """
    actor = _spawn(unreal.StaticMeshActor, location, label)
    comp = actor.static_mesh_component
    if not isinstance(mesh_path, str):
        comp.set_static_mesh(mesh_path)
    else:
        comp.set_static_mesh(_asset(mesh_path) if own_material else unreal.load_asset(mesh_path))
    if rotation is not None:
        actor.set_actor_rotation(rotation, True)
    actor.set_actor_scale3d(scale)
    try:
        if material is not None:
            comp.set_material(0, material)
            mid = comp.create_dynamic_material_instance(0, material)
        else:
            mid = comp.create_dynamic_material_instance(0, None if own_material else unreal.load_asset(_SHAPE_MAT))
    except Exception:
        mid = None
    if color is not None and mid is not None:
        mid.set_vector_parameter_value(color_param, unreal.LinearColor(color[0], color[1], color[2], 1.0))
    return actor, mid


def _niagara_actor(system_path, location, label):
    """A Niagara actor running ``system_path``, or None if unset/unavailable."""
    system = _asset(system_path)
    if system is None:
        return None
    try:
        actor = _spawn(unreal.NiagaraActor, location, label)
        comp = actor.niagara_component
        comp.set_asset(system)
        comp.activate(True)
        return actor
    except Exception as e:
        unreal.log_warning(f"[twin_live] could not start Niagara '{system_path}' ({e})")
        return None


def _set_active(actor, on):
    """Show/hide a mesh or Niagara actor (Niagara also stops emitting)."""
    actor.set_is_temporarily_hidden_in_editor(not on)
    comp = getattr(actor, "niagara_component", None)
    if comp is not None:
        try:
            if on:
                comp.activate(False)
            else:
                comp.deactivate()
        except Exception:
            pass


def _editor():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)


def _get_camera():
    """(location, rotation) of the active level viewport, or None if unavailable."""
    try:
        info = _editor().get_level_viewport_camera_info()
    except Exception:
        return None
    if not info:
        return None
    loc = next((x for x in info if isinstance(x, unreal.Vector)), None)
    return (loc, next((x for x in info if isinstance(x, unreal.Rotator)), None)) if loc else None


def _look_rotation(frm, to):
    d = to - frm
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y)))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


def _face_camera(actor, cam_loc):
    """Turn a TextRender actor so its readable (+X) side faces the camera; its board sits just behind."""
    loc = actor.get_actor_location()
    yaw = math.degrees(math.atan2(cam_loc.y - loc.y, cam_loc.x - loc.x))
    rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw)
    actor.set_actor_rotation(rot, True)
    m = _TEXTS.get(actor.get_actor_label())
    board = m["board"] if m else None
    if board is not None:
        back = m["size"] * 0.3
        y = math.radians(yaw)
        board.set_actor_location(loc - unreal.Vector(math.cos(y) * back, math.sin(y) * back, 0.0), False, True)
        board.set_actor_rotation(rot, True)


def _text_actor(location, text, size, label, color=None):
    actor = _spawn(unreal.TextRenderActor, location, label)
    tr = actor.text_render
    try:   # (UE's enum really is spelled "HorizTextAligment")
        tr.set_horizontal_alignment(unreal.HorizTextAligment.EHTA_CENTER)
    except Exception:
        pass
    try:   # centre multi-line text on the actor, so its board can sit centred behind it
        tr.set_vertical_alignment(unreal.VerticalTextAligment.EVRTA_TEXT_CENTER)
    except Exception:
        pass
    tr.set_world_size(size)
    _TEXTS[label] = {"size": size, "text": text, "board": None}
    if color is not None:
        try:
            tr.set_text_render_color(unreal.Color(*color, 255))
        except Exception:
            pass
    tr.set_text(text)
    return actor


# Text bookkeeping on the Python side: UE objects don't accept new Python attributes.
_TEXTS = {}     # actor label -> {"size": world size, "text": current text, "board": board actor or None}


def _meta(actor):
    return _TEXTS.setdefault(actor.get_actor_label(), {"size": 100.0, "text": "", "board": None})


def _board(text_actor, color=(0.02, 0.03, 0.04)):
    """A dark board behind a text, so it stays readable over a bright sky or field."""
    board, _ = _mesh_actor(_CUBE, text_actor.get_actor_location(), unreal.Vector(0.01, 1.0, 1.0), color,
                           text_actor.get_actor_label() + "_Board")
    _meta(text_actor)["board"] = board
    _fit_board(text_actor)
    return board


def _fit_board(text_actor):
    m = _meta(text_actor)
    if m["board"] is None:
        return
    lines = m["text"].split("\n")
    width = (max(len(l) for l in lines) * 0.62 + 1.5) * m["size"]      # default font ~0.6 em per char
    height = (len(lines) * 1.05 + 0.6) * m["size"]
    m["board"].set_actor_scale3d(unreal.Vector(0.02, width / 100.0, height / 100.0))


def _set_text(actor, text):
    m = _meta(actor)
    if m["text"] == text:
        return
    actor.text_render.set_text(text)
    m["text"] = text
    _fit_board(actor)


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def crop_color(stress, fungus, std=0.0):
    """Healthy green -> yellow (stress) -> brown, darkened by fungus, faded by uncertainty."""
    green, yellow, brown, grey = (0.10, 0.55, 0.12), (0.85, 0.78, 0.20), (0.42, 0.24, 0.10), (0.5, 0.5, 0.5)
    s = max(0.0, min(1.0, stress * 2.5))
    c = _lerp(green, yellow, min(1.0, s * 2.0)) if s < 0.5 else _lerp(yellow, brown, (s - 0.5) * 2.0)
    # Uncertainty fades the health colour only (at most halfway), then disease is painted on top,
    # so an outbreak the twin has found reads red even while it is still unsure of its extent.
    c = _lerp(c, grey, max(0.0, min(0.5, std * 6.0)))
    return _lerp(c, (0.55, 0.06, 0.04), max(0.0, min(1.0, fungus * 1.8)))


# ============================================================ paddy assets
# A realistic field without downloads: the script writes a rice-patch mesh as
# an OBJ file, imports it, and builds two materials with the Material Editing
# Library. Everything is saved under ASSET_DIR and reused on later runs.
PATCH_CM = 500.0          # the patch mesh covers one 5 m twin cell
ASSET_VERSION = 1


def rice_patch_obj(seed=7, hills=10, blades=7, segments=2, height_cm=95.0, size_cm=PATCH_CM):
    """
    OBJ text of a 5 x 5 m patch of transplanted rice: ``hills`` x ``hills``
    hills in rows, each a tuft of ``blades`` curved, tapering leaves. Z-up
    centimetres are written in OBJ's Y-up convention (as Blender exports it),
    base at height 0, centred on the origin. UV u runs 0 at the leaf base to 1
    at the tip, so the material can darken the base and sway the tips.
    """
    import random
    rnd = random.Random(seed)
    verts, uvs, norms, faces = [], [], [], []
    step = size_cm / hills
    for i in range(hills):
        for j in range(hills):
            hx = -size_cm / 2 + (j + 0.5) * step + rnd.uniform(-0.12, 0.12) * step
            hy = -size_cm / 2 + (i + 0.5) * step + rnd.uniform(-0.12, 0.12) * step
            for b in range(blades):
                az = 2 * math.pi * (b + rnd.random() * 0.8) / blades
                lean = math.radians(rnd.uniform(10.0, 38.0))
                h = height_cm * rnd.uniform(0.7, 1.0)
                w0 = rnd.uniform(3.5, 5.5)            # ~3x a real leaf, so the canopy reads from 100 m
                dx, dy = math.cos(az), math.sin(az)
                wx, wy = -dy, dx                      # across the leaf
                base = len(verts)
                for k in range(segments + 1):
                    t = k / segments
                    out = math.sin(lean) * h * t + 0.35 * h * t * t * math.sin(lean)   # arches outwards
                    z = math.cos(lean) * h * t - 0.12 * h * t * t
                    half = 0.5 * w0 * (1.0 - 0.85 * t)
                    cx, cy = hx + dx * out, hy + dy * out
                    for side in (-1, 1):
                        verts.append((cx + wx * half * side, cy + wy * half * side, z))
                        uvs.append((t, 0.0 if side < 0 else 1.0))
                    n = (dx * math.cos(lean), dy * math.cos(lean), -math.sin(lean))
                    norms.append((-n[0], -n[1], -n[2]))
                for k in range(segments):
                    a, b2, c, d = base + 2 * k, base + 2 * k + 1, base + 2 * k + 3, base + 2 * k + 2
                    faces.append((a, b2, c, k))
                    faces.append((a, c, d, k))
    lines = [f"# rice patch, {hills}x{hills} hills, {blades} leaves each, cm, Y-up"]
    lines += [f"v {x:.2f} {z:.2f} {-y:.2f}" for x, y, z in verts]          # Z-up -> OBJ Y-up
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in uvs]
    lines += [f"vn {nx:.4f} {nz:.4f} {-ny:.4f}" for nx, ny, nz in norms]
    per_blade = segments + 1
    for a, b2, c, k in faces:
        blade = a // (2 * per_blade)
        nrm = blade * per_blade + k + 1
        lines.append("f " + " ".join(f"{v + 1}/{v + 1}/{nrm}" for v in (a, b2, c)))
    return "\n".join(lines) + "\n"


def _asset_tools():
    return unreal.AssetToolsHelpers.get_asset_tools()


def _import_mesh(obj_text, name):
    """Import an OBJ as a static mesh in ASSET_DIR; returns the mesh or None."""
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"{name}.obj")
    with open(path, "w") as f:
        f.write(obj_text)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", ASSET_DIR)
    task.set_editor_property("destination_name", name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    try:
        opts = unreal.FbxImportUI()
        opts.set_editor_property("import_mesh", True)
        opts.set_editor_property("import_materials", False)
        opts.set_editor_property("import_textures", False)
        opts.set_editor_property("import_as_skeletal", False)
        task.set_editor_property("options", opts)
    except Exception:
        pass                                  # Interchange importer: defaults are fine
    _asset_tools().import_asset_tasks([task])
    return unreal.load_asset(f"{ASSET_DIR}/{name}")


def _mat_expr(mat, cls, x, y, **props):
    e = unreal.MaterialEditingLibrary.create_material_expression(mat, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def _connect(a, b, pin=""):
    lib = unreal.MaterialEditingLibrary
    if not lib.connect_material_expressions(a, "", b, pin) and pin == "":
        lib.connect_material_expressions(a, "", b, "Input")


def _new_material(name):
    path = f"{ASSET_DIR}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    return _asset_tools().create_asset(name, ASSET_DIR, unreal.Material, unreal.MaterialFactoryNew())


def _build_rice_material(name):
    """Two-sided leaves: Color parameter, darker at the base, tips sway in the wind."""
    L, M = unreal.MaterialEditingLibrary, unreal.MaterialProperty
    mat = _new_material(name)
    mat.set_editor_property("two_sided", True)
    color = _mat_expr(mat, unreal.MaterialExpressionVectorParameter, -900, -200,
                      parameter_name="Color", default_value=unreal.LinearColor(0.12, 0.5, 0.1, 1.0))
    uv = _mat_expr(mat, unreal.MaterialExpressionTextureCoordinate, -1100, 100)
    u = _mat_expr(mat, unreal.MaterialExpressionComponentMask, -900, 100, r=True, g=False, b=False, a=False)
    _connect(uv, u)
    dark = _mat_expr(mat, unreal.MaterialExpressionMultiply, -650, -250, const_b=0.35)
    _connect(color, dark, "A")
    shade = _mat_expr(mat, unreal.MaterialExpressionLinearInterpolate, -400, -150)
    _connect(dark, shade, "A")
    _connect(color, shade, "B")
    _connect(u, shade, "Alpha")
    L.connect_material_property(shade, "", M.MP_BASE_COLOR)
    rough = _mat_expr(mat, unreal.MaterialExpressionConstant, -400, 50, r=0.55)
    L.connect_material_property(rough, "", M.MP_ROUGHNESS)
    # Wind: offset = (1, 0.4, 0) * 7 cm * sin(1.7 t + 0.004 x) * u^2
    time_ = _mat_expr(mat, unreal.MaterialExpressionTime, -1100, 350)
    t1 = _mat_expr(mat, unreal.MaterialExpressionMultiply, -900, 350, const_b=1.7)
    _connect(time_, t1, "A")
    wp = _mat_expr(mat, unreal.MaterialExpressionWorldPosition, -1100, 500)
    wx = _mat_expr(mat, unreal.MaterialExpressionComponentMask, -900, 500, r=True, g=False, b=False, a=False)
    _connect(wp, wx)
    x1 = _mat_expr(mat, unreal.MaterialExpressionMultiply, -750, 500, const_b=0.004)
    _connect(wx, x1, "A")
    phase = _mat_expr(mat, unreal.MaterialExpressionAdd, -600, 420)
    _connect(t1, phase, "A")
    _connect(x1, phase, "B")
    sine = _mat_expr(mat, unreal.MaterialExpressionSine, -450, 420)
    _connect(phase, sine)
    u2 = _mat_expr(mat, unreal.MaterialExpressionMultiply, -600, 250)
    _connect(u, u2, "A")
    _connect(u, u2, "B")
    amp = _mat_expr(mat, unreal.MaterialExpressionMultiply, -300, 350)
    _connect(sine, amp, "A")
    _connect(u2, amp, "B")
    direction = _mat_expr(mat, unreal.MaterialExpressionConstant3Vector, -300, 520,
                          constant=unreal.LinearColor(7.0, 2.8, 0.0, 0.0))
    wpo = _mat_expr(mat, unreal.MaterialExpressionMultiply, -100, 420)
    _connect(direction, wpo, "A")
    _connect(amp, wpo, "B")
    L.connect_material_property(wpo, "", M.MP_WORLD_POSITION_OFFSET)
    L.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def _build_water_material(name):
    """Standing water: Color and Roughness parameters (glassy when flooded, matte mud when dry)."""
    L, M = unreal.MaterialEditingLibrary, unreal.MaterialProperty
    mat = _new_material(name)
    color = _mat_expr(mat, unreal.MaterialExpressionVectorParameter, -500, 0,
                      parameter_name="Color", default_value=unreal.LinearColor(0.04, 0.06, 0.05, 1.0))
    rough = _mat_expr(mat, unreal.MaterialExpressionScalarParameter, -500, 200,
                      parameter_name="Roughness", default_value=0.05)
    spec = _mat_expr(mat, unreal.MaterialExpressionConstant, -500, 350, r=1.0)
    L.connect_material_property(color, "", M.MP_BASE_COLOR)
    L.connect_material_property(rough, "", M.MP_ROUGHNESS)
    L.connect_material_property(spec, "", M.MP_SPECULAR)
    L.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def _upright_rotation(mesh):
    """Rotation that stands the imported patch upright (importers differ on OBJ's up axis)."""
    try:
        box = mesh.get_bounding_box()
        lo, hi = box.min, box.max
    except Exception:
        return unreal.Rotator(0.0, 0.0, 0.0), 1.0
    ext = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z]
    up = min(range(3), key=lambda i: ext[i])                  # the patch is 5 m wide, ~1 m tall
    span = max(ext)
    # The OBJ is in cm; an importer that read it as metres made it 100x larger.
    scale = 0.01 if span > 5 * PATCH_CM else (100.0 if 0 < span < PATCH_CM / 20 else 1.0)
    sign = 1.0 if abs((hi.x, hi.y, hi.z)[up]) >= abs((lo.x, lo.y, lo.z)[up]) else -1.0
    axis = unreal.Vector(*[sign if i == up else 0.0 for i in range(3)])
    for rot in (unreal.Rotator(0.0, 0.0, 0.0), unreal.Rotator(roll=90.0), unreal.Rotator(roll=-90.0),
                unreal.Rotator(pitch=90.0), unreal.Rotator(pitch=-90.0), unreal.Rotator(roll=180.0)):
        v = rot.rotate_vector(axis)
        if v.z > 0.99:
            return rot, scale
    return unreal.Rotator(0.0, 0.0, 0.0), scale


_PADDY = {}


def paddy_assets():
    """Rice mesh + materials, built once and saved in ASSET_DIR. None if the editor can't build them."""
    if "assets" in _PADDY:
        return _PADDY["assets"]
    v = ASSET_VERSION
    names = {"mesh": f"SM_RicePatch_v{v}", "rice": f"M_TwinRice_v{v}", "water": f"M_PaddyWater_v{v}"}
    try:
        lib = unreal.EditorAssetLibrary
        have = all(lib.does_asset_exist(f"{ASSET_DIR}/{n}") for n in names.values())
        if have:
            mesh, rice, water = (unreal.load_asset(f"{ASSET_DIR}/{names[k]}") for k in ("mesh", "rice", "water"))
        else:
            unreal.log("[twin_live] building the paddy assets (rice mesh, materials); about a minute, once")
            mesh = _import_mesh(rice_patch_obj(), names["mesh"])
            rice = water = None
            try:
                rice = _build_rice_material(names["rice"])
                water = _build_water_material(names["water"])
            except Exception as e:     # keep the rice mesh; colour it through the engine's basic material
                unreal.log_warning(f"[twin_live] could not build the paddy materials ({e}); using basic ones")
            rice = rice or unreal.load_asset(_SHAPE_MAT)
            water = water or unreal.load_asset(_SHAPE_MAT)
            try:
                mesh.set_material(0, rice)
                lib.save_loaded_asset(mesh)
            except Exception:
                pass
        if mesh is None:
            raise RuntimeError("the rice mesh did not import")
        rot, scale = _upright_rotation(mesh)
        _PADDY["assets"] = {"mesh": mesh, "rice": rice, "water": water, "rotation": rot, "scale": scale}
    except Exception as e:
        unreal.log_warning(f"[twin_live] could not build the paddy assets ({e}); using blocks instead")
        _PADDY["assets"] = None
    return _PADDY["assets"]


# ======================================================================= fields
class BlocksField:
    """A 20 x 20 field of cubes: height = NDVI, colour = stress / fungus / uncertainty."""

    def __init__(self, origin, label):
        self.origin, self.cells = origin, []
        # CROP_MESH: your crop clump per cell (pivot at its base, ~1 m wide), tinted via CROP_COLOR_PARAM.
        self.crop = _asset(CROP_MESH) is not None
        for r in range(GRID):
            for c in range(GRID):
                loc = unreal.Vector(origin.x + (c + 0.5) * CELL_CM, origin.y + (r + 0.5) * CELL_CM, origin.z)
                if self.crop:
                    actor, mid = _mesh_actor(CROP_MESH, loc, self._scale(0.5), (0.1, 0.55, 0.12),
                                             f"{label}_{r:02d}_{c:02d}", own_material=True,
                                             color_param=CROP_COLOR_PARAM)
                else:
                    actor, mid = _mesh_actor(_CUBE, loc, self._scale(0.3), (0.1, 0.55, 0.12),
                                             f"{label}_{r:02d}_{c:02d}")
                self.cells.append((actor, mid, loc))

    def _scale(self, height):
        if self.crop:   # keep the clump's shape: grow it with NDVI rather than stretching a cube
            k = CELL_CM / 100.0
            return unreal.Vector(k, k, k * height)
        return unreal.Vector(CELL_CM / 105.0, CELL_CM / 105.0, height)

    def apply(self, ndvi, stress, fungus, std):
        for i, (actor, mid, loc) in enumerate(self.cells):
            if self.crop:
                height = 0.25 + 1.0 * max(0.0, ndvi[i])
                actor.set_actor_scale3d(self._scale(height))     # pivot at the base: stays on the ground
            else:
                height = 0.15 + 1.6 * max(0.0, ndvi[i])
                actor.set_actor_scale3d(self._scale(height))
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, loc.z + height * 50.0), False, True)
            if mid is not None:
                r, g, b = crop_color(stress[i], fungus[i], std[i] if std else 0.0)
                mid.set_vector_parameter_value(CROP_COLOR_PARAM if self.crop else "Color",
                                               unreal.LinearColor(r, g, b, 1.0))

    def grid_to_world(self, row, col, alt_m):
        return unreal.Vector(self.origin.x + col * CELL_CM, self.origin.y + row * CELL_CM,
                             self.origin.z + alt_m * 100.0)

    def size_cm(self):
        return GRID * CELL_CM

    def world_to_grid(self, p):
        return (p.y - self.origin.y) / CELL_CM, (p.x - self.origin.x) / CELL_CM

    def destroy(self):
        for actor, _, _ in self.cells:
            _destroy(actor)


_PLANE = "/Engine/BasicShapes/Plane"
_MUD = (0.20, 0.14, 0.08)


class PaddyField(BlocksField):
    """
    A paddy field of generated rice patches standing in water, with mud bunds.
    Plant height follows NDVI, leaf colour follows health, and the water
    turns from glassy (flooded) to matte mud as the soil dries.
    """

    def __init__(self, origin, label, assets):
        self.origin, self.cells, self.crop, self.assets = origin, [], True, assets
        self.label, self.extras = label, []
        rot = assets["rotation"]
        # Which local axis ends up vertical (the importer may have left the patch on its side).
        self.up = 2 if abs(rot.roll) < 1 and abs(rot.pitch) < 1 else (1 if abs(rot.roll) > 1 else 0)
        self.k = assets["scale"] * CELL_CM / PATCH_CM
        size = GRID * CELL_CM
        mid_pt = unreal.Vector(origin.x + size / 2, origin.y + size / 2, origin.z)
        # Standing water over the whole field, a few cm deep around the plant bases.
        self.water, self.water_mid = _mesh_actor(_PLANE, mid_pt + unreal.Vector(0.0, 0.0, 6.0),
                                                 unreal.Vector(size / 100.0, size / 100.0, 1.0), None,
                                                 f"{label}_Water", material=assets["water"])
        self.extras.append(self.water)
        # Mud bunds (levees) around the field.
        w, h = 60.0, 35.0
        for i, (dx, dy, sx, sy) in enumerate(((0, -size / 2 - w / 2, size + 2 * w, w), (0, size / 2 + w / 2, size + 2 * w, w),
                                              (-size / 2 - w / 2, 0, w, size), (size / 2 + w / 2, 0, w, size))):
            bund, _ = _mesh_actor(_CUBE, mid_pt + unreal.Vector(dx, dy, h / 2), unreal.Vector(sx / 100, sy / 100, h / 100),
                                  _MUD, f"{label}_Bund{i}")
            self.extras.append(bund)
        for r in range(GRID):
            for c in range(GRID):
                loc = unreal.Vector(origin.x + (c + 0.5) * CELL_CM, origin.y + (r + 0.5) * CELL_CM, origin.z)
                actor, mid = _mesh_actor(assets["mesh"], loc, self._scale(0.8), (0.12, 0.5, 0.1),
                                         f"{label}_{r:02d}_{c:02d}", material=assets["rice"], rotation=rot)
                self.cells.append((actor, mid, loc))
        self.wet = None

    def _scale(self, height):
        v = [self.k, self.k, self.k]
        v[self.up] = self.k * height
        return unreal.Vector(*v)

    def apply(self, ndvi, stress, fungus, std, moisture=None):
        for i, (actor, mid, loc) in enumerate(self.cells):
            # Seedlings ~30 cm, full canopy ~1 m (NDVI 0.85).
            actor.set_actor_scale3d(self._scale(0.35 + 0.8 * max(0.0, min(1.0, ndvi[i]))))
            if mid is not None:
                r, g, b = crop_color(stress[i], fungus[i], std[i] if std else 0.0)
                mid.set_vector_parameter_value("Color", unreal.LinearColor(r, g, b, 1.0))
        if moisture and self.water_mid is not None:
            wet = max(0.0, min(1.0, (sum(moisture) / len(moisture) - 0.6) / 0.3))   # 0.9+ = standing water
            if self.wet is None or abs(wet - self.wet) > 0.02:
                self.wet = wet
                c = _lerp(_MUD, (0.03, 0.05, 0.045), wet)
                self.water_mid.set_vector_parameter_value("Color", unreal.LinearColor(c[0], c[1], c[2], 1.0))
                self.water_mid.set_scalar_parameter_value("Roughness", 0.85 - 0.8 * wet)

    def destroy(self):
        super().destroy()
        for a in self.extras:
            _destroy(a)


class HismField:
    """Your field: per-instance custom data on a 400-instance (H)ISM."""

    def __init__(self, comp):
        self.comp = comp
        try:
            if comp.get_editor_property("num_custom_data_floats") < 6:
                comp.set_num_custom_data_floats(6)
        except Exception:
            comp.set_editor_property("num_custom_data_floats", 6)
        c00, c0n, cn0 = (self._center(0), self._center(GRID - 1), self._center((GRID - 1) * GRID))
        self.col_axis = (c0n - c00) * (1.0 / (GRID - 1))
        self.row_axis = (cn0 - c00) * (1.0 / (GRID - 1))
        self.corner = c00 - self.col_axis * 0.5 - self.row_axis * 0.5
        self.cm_per_m = self.col_axis.length() / 5.0     # the twin's cells are 5 m

    def _center(self, index):
        res = self.comp.get_instance_transform(index, True)
        if isinstance(res, tuple):
            res = next(x for x in res if isinstance(x, unreal.Transform))
        return res.translation

    def apply(self, ndvi, stress, fungus, std, moisture=None, anomalies=()):
        anomalies = set(anomalies)
        n = min(self.comp.get_instance_count(), len(ndvi))
        for i in range(n):
            vals = [ndvi[i], stress[i], std[i] if std else 0.0,
                    moisture[i] if moisture else 0.0, fungus[i], 1.0 if i in anomalies else 0.0]
            last = i == n - 1
            if hasattr(self.comp, "set_custom_data"):
                self.comp.set_custom_data(i, vals, last)
            else:
                for k, v in enumerate(vals):
                    self.comp.set_custom_data_value(i, k, v, last and k == 5)

    def size_cm(self):
        return self.col_axis.length() * GRID

    def world_to_grid(self, p):
        # Solve corner + col_axis * col + row_axis * row = p in the ground plane.
        a, b = self.col_axis, self.row_axis
        dx, dy = p.x - self.corner.x, p.y - self.corner.y
        det = a.x * b.y - a.y * b.x
        if abs(det) < 1e-9:
            return GRID / 2.0, GRID / 2.0
        col = (dx * b.y - dy * b.x) / det
        row = (a.x * dy - a.y * dx) / det
        return row, col

    def grid_to_world(self, row, col, alt_m):
        # Altitude scaled like the field, so drones keep their proportions on a resized field.
        return self.corner + self.col_axis * col + self.row_axis * row + \
            unreal.Vector(0.0, 0.0, alt_m * self.cm_per_m)

    def destroy(self):
        pass


def _find_hism_fields():
    twin = reality = forecast = None
    for actor in _actors():
        try:
            comps = actor.get_components_by_class(unreal.InstancedStaticMeshComponent)
        except Exception:
            continue
        for comp in comps:
            if comp.get_instance_count() != GRID * GRID:
                continue
            label = actor.get_actor_label()
            if REALITY_LABEL_HINT and REALITY_LABEL_HINT.lower() in label.lower():
                reality = reality or comp
            elif FORECAST_LABEL_HINT and FORECAST_LABEL_HINT.lower() in label.lower():
                forecast = forecast or comp
            elif twin is None and (TWIN_ACTOR_LABEL is None or label == TWIN_ACTOR_LABEL):
                twin = comp
    return twin, reality, forecast


# ===================================================================== client
class TwinLive:
    def __init__(self):
        self.frame, self.seq, self.lock, self.running = None, -1, threading.Lock(), True
        self.twin_field = self.truth_field = self.forecast_field = None
        self.drones, self.targets, self.caption = {}, {}, None
        self.errors = 0
        twin, reality, forecast = (None, None, None) if FIELD_MODE == "blocks" else _find_hism_fields()
        if twin is not None:
            self.twin_field = HismField(twin)
            self.truth_field = HismField(reality) if reality is not None else None
            self.forecast_field = HismField(forecast) if forecast is not None else None
            unreal.log(f"[twin_live] hism mode: field '{twin.get_owner().get_actor_label()}'"
                       + (f", reality '{reality.get_owner().get_actor_label()}'" if reality else "")
                       + (f", forecast '{forecast.get_owner().get_actor_label()}'" if forecast else ""))
        elif FIELD_MODE == "hism":
            raise RuntimeError("FIELD_MODE='hism' but no 400-instance ISM/HISM found in the level")
        else:
            # Your own CROP_MESH wins over the generated rice.
            assets = paddy_assets() if FIELD_LOOK == "paddy" and not CROP_MESH else None
            make = (lambda o, n: PaddyField(o, n, assets)) if assets else BlocksField
            self.twin_field = make(BLOCKS_ORIGIN, "Twin")
            gap = (GRID + BLOCKS_GAP_CELLS) * CELL_CM
            self.truth_field = make(BLOCKS_ORIGIN + unreal.Vector(gap, 0.0, 0.0), "Reality")
            if SHOW_FORECAST:
                self.forecast_field = make(BLOCKS_ORIGIN + unreal.Vector(2 * gap, 0.0, 0.0), "Forecast")
            unreal.log(f"[twin_live] {'paddy' if assets else 'blocks'} look: built 'Twin', 'Reality'"
                       + (" and 'Forecast'" if SHOW_FORECAST else "") + " fields")
        size = self.twin_field.size_cm()
        views = [(self.twin_field, "TWIN", (120, 200, 255)),
                 (self.truth_field, "REALITY", (255, 200, 120)),
                 (self.forecast_field, "FORECAST +7 DAYS", (200, 160, 255))]
        views = [v for v in views if v[0] is not None]
        centers = [f.grid_to_world(GRID / 2.0, GRID / 2.0, 0.0) for f, _, _ in views]
        mean = centers[0]
        for c in centers[1:]:
            mean = mean + c
        self.center = mean * (1.0 / len(centers))
        self.span = size + (centers[-1] - centers[0]).length()
        # Caption above the scene's north edge; labels above each field.
        north = self.twin_field.grid_to_world(-1.0, GRID / 2.0, 0.0) - centers[0]
        self.cm_per_m = size / (GRID * 5.0)        # the twin's cells are 5 m
        # Caption in front of the fields (towards the camera), like a subtitle.
        south = north * -1.0
        self.caption = _text_actor(self.center + south * 1.45 + unreal.Vector(0.0, 0.0, size * 0.08),
                                   "Digital twin: waiting for the API...", size * 0.05, "TwinCaption",
                                   (255, 255, 255))
        # Labels stand just behind each field's north edge, clear of the blocks.
        self.labels = [_text_actor(c + north * 1.5 + unreal.Vector(0.0, 0.0, size * 0.06), name,
                                   size * (0.08 if len(name) < 10 else 0.055),
                                   f"Label_{name.split()[0]}", color)
                       for c, (_, name, color) in zip(centers, views)]
        self.texts = [self.caption] + self.labels
        self.panel, self.what_if = None, None
        if SHOW_PANEL:
            # Results board up in the sky behind the fields, where the view is otherwise empty.
            self.panel = _text_actor(self.center + north * 2.4 + unreal.Vector(0.0, 0.0, size * 0.45),
                                     "Results: waiting for data...", size * 0.045, "TwinResults",
                                     (235, 245, 235))
            self.texts.append(self.panel)
        if SHOW_LEGEND:
            self.legend = _text_actor(self.center + south * 1.8 + unreal.Vector(0.0, 0.0, size * 0.03),
                                      LEGEND_TEXT, size * 0.026, "TwinLegend", (225, 225, 210))
            self.texts.append(self.legend)
        self.boards = [_board(t) for t in self.texts] if SHOW_BOARDS else []
        # Farmland under the fields (paddy look): earth instead of the level's grey floor.
        self.ground = None
        if isinstance(self.twin_field, PaddyField):
            self.ground, _ = _mesh_actor(_PLANE, self.center - unreal.Vector(0.0, 0.0, 3.0),
                                         unreal.Vector(self.span * 1.6 / 100.0, size * 3.0 / 100.0, 1.0),
                                         (0.16, 0.13, 0.08), "TwinGround")
        self.view, self.view_drone = "overview", None
        self.rain, self.rain_on, self.raining, self.rain_streaks = [], False, False, None
        if SHOW_RAIN:
            self._build_rain(size)
        if AUTO_FRAME_CAMERA:
            self.frame_camera()
        self.poller = threading.Thread(target=self._poll, daemon=True)
        self.poller.start()
        self.tick_handle = unreal.register_slate_post_tick_callback(self._tick)

    def _build_rain(self, size):
        """RAIN_NIAGARA over the scene, or light-blue streaks falling over every field."""
        top = size * 0.6
        fx = _niagara_actor(RAIN_NIAGARA, self.center + unreal.Vector(0.0, 0.0, top), "TwinRain")
        if fx is not None:
            fx.set_actor_scale3d(unreal.Vector(self.span / 1000.0, size / 1000.0, 1.0))
            self.rain, self.rain_streaks = [fx], None
        else:
            # A fixed pseudo-random layout so the scene looks the same every run.
            n, half_x, half_y = 90, self.span * 0.5, size * 0.5
            self.rain_streaks = []
            k = size / 2000.0
            for i in range(n):
                x = ((i * 0.618034) % 1.0 - 0.5) * 2 * half_x
                y = ((i * 0.381966 * 7) % 1.0 - 0.5) * 2 * half_y
                z = ((i * 0.7548777) % 1.0) * top
                a, _ = _mesh_actor(_CYLINDER, self.center + unreal.Vector(x, y, z),
                                   unreal.Vector(k * 0.06, k * 0.06, k * 2.0), (0.55, 0.7, 0.95), f"Rain_{i:02d}")
                self.rain.append(a)
                self.rain_streaks.append([x, y, z])
            self.rain_top = top
        for a in self.rain:
            _set_active(a, False)

    def _update_rain(self, dt):
        if self.raining != self.rain_on:
            for a in self.rain:
                _set_active(a, self.raining)
            self.rain_on = self.raining
        if not (self.rain_on and self.rain_streaks):
            return
        fall = self.rain_top * 1.5 * max(dt, 0.0)            # top to ground in ~0.7 s
        for a, p in zip(self.rain, self.rain_streaks):
            p[2] -= fall
            if p[2] < 0.0:
                p[2] += self.rain_top
            a.set_actor_location(self.center + unreal.Vector(p[0], p[1], p[2]), False, True)

    def set_view(self, view):
        """Camera presets: 'overview', 'closeup' (low over the twin's crop) or 'follow' (chase a drone)."""
        self.view = view
        if view == "overview":
            self.frame_camera()
        elif view == "closeup":
            size = self.twin_field.size_cm()
            look = self.twin_field.grid_to_world(GRID * 0.55, GRID * 0.5, 0.0)
            cam = self.twin_field.grid_to_world(GRID * 1.05, GRID * 0.35, 0.0) + unreal.Vector(0.0, 0.0, size * 0.09)
            self._move_camera(cam, look)
        elif view == "follow":
            self.view_drone, self.view_snap = None, True     # jump to the drone, then glide with it
            self._follow_drone()

    def _follow_drone(self):
        """Chase camera behind and above a flying drone (the one spraying, else the first)."""
        flying = [d for d in self.drones.values() if d["pos"] is not None]
        if not flying:
            return
        pick = next((d for d in flying if d["spraying"]), None) or self.view_drone or flying[0]
        self.view_drone = pick
        pos = pick["pos"]
        size = self.twin_field.size_cm()
        cam = pos + unreal.Vector(0.0, size * 0.16, size * 0.09)
        old = _get_camera()
        if old is not None and old[0] is not None and not getattr(self, "view_snap", False):
            cam = old[0] + (cam - old[0]) * 0.15          # glide, don't jump
        self.view_snap = False
        self._move_camera(cam, pos - unreal.Vector(0.0, 0.0, size * 0.02))

    def _move_camera(self, cam, look):
        try:
            _editor().set_level_viewport_camera_info(cam, _look_rotation(cam, look))
        except Exception as e:
            unreal.log_warning(f"[twin_live] could not move the camera ({e})")

    # background thread: HTTP only, never touches UObjects
    def frame_camera(self):
        """Raised three-quarter view from the south, both fields in shot."""
        cam = self.center + unreal.Vector(0.0, self.span * 0.62, self.span * 0.42)
        try:
            _editor().set_level_viewport_camera_info(cam, _look_rotation(cam, self.center))
        except Exception as e:
            unreal.log_warning(f"[twin_live] could not move the camera ({e}); fly there yourself")

    def _poll(self):
        while self.running:
            try:
                with urllib.request.urlopen(f"{API}/ue5/frame?since={self.seq}", timeout=2.0) as r:
                    frame = json.loads(r.read().decode("utf-8"))
                with self.lock:
                    # A field the tick has not applied yet is never lost: self.seq only
                    # advances in the tick, so the next poll returns the field again.
                    self.frame = frame
                self.errors = 0
            except Exception:
                self.errors += 1
            time.sleep(1.0 / POLL_HZ)

    # game thread: apply the latest frame
    def _tick(self, delta_seconds):
        with self.lock:
            frame, self.frame = self.frame, None
        if frame is not None:
            field = frame.get("field")
            if field is not None:
                self.seq = frame["seq"]
                self._apply_field(field)
            self._apply_status(frame.get("status"), field)
            if field is not None and field.get("what_if"):
                self.what_if = field["what_if"]
            if self.panel is not None and frame.get("metrics"):
                _set_text(self.panel, panel_text(frame["metrics"], self.what_if))
            if frame.get("weather"):
                self.raining = SHOW_RAIN and frame["weather"].get("precipitation_mm", 0.0) > RAIN_MM
            if frame.get("drones"):
                self._apply_drones(frame["drones"]["drones"])
        elif self.errors > 20 and self.caption is not None:
            _set_text(self.caption, f"Digital twin: cannot reach {API}")
        self._smooth_drones(delta_seconds)
        self._update_rain(delta_seconds)
        if getattr(self, "view", None) == "follow":     # subclasses (twin_paddy) run their own cameras
            self._follow_drone()
        cam = _get_camera()
        if cam is not None:
            for t in self.texts:
                _face_camera(t, cam[0])

    def _apply_field(self, f):
        if isinstance(self.twin_field, HismField):
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"),
                                  f.get("moisture"), f.get("anomalies", ()))
        elif isinstance(self.twin_field, PaddyField):
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"), f.get("moisture"))
        else:
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"))
        if self.forecast_field is not None and f.get("forecast_stress"):
            if isinstance(self.forecast_field, HismField):
                self.forecast_field.apply(f["forecast_ndvi"], f["forecast_stress"], f["forecast_fungus"],
                                          f.get("forecast_std"), None, ())
            else:
                self.forecast_field.apply(f["forecast_ndvi"], f["forecast_stress"], f["forecast_fungus"],
                                          f.get("forecast_std"))
        if self.truth_field is None or not f.get("truth_stress"):
            return
        no_uncertainty = [0.0] * len(f["truth_stress"])     # reality is known exactly
        if isinstance(self.truth_field, HismField):
            self.truth_field.apply(f["truth_ndvi"], f["truth_stress"], f["truth_fungus"], no_uncertainty,
                                   f.get("moisture"), ())
        elif isinstance(self.truth_field, PaddyField):
            self.truth_field.apply(f["truth_ndvi"], f["truth_stress"], f["truth_fungus"], no_uncertainty,
                                   f.get("truth_moisture") or f.get("moisture"))
        else:
            self.truth_field.apply(f["truth_ndvi"], f["truth_stress"], f["truth_fungus"], no_uncertainty)

    def _apply_status(self, status, field):
        if self.caption is None:
            return
        if status:
            err = status.get("error") or {}
            extra = f"   |   twin vs reality error: {err['fungus']:.3f}" if err.get("fungus") is not None else ""
            _set_text(self.caption, f"{status['text']}{extra}")
        elif field is not None:
            _set_text(self.caption, f"Day {field['sim_day']}  ({field['growth_stage']})")

    def _apply_drones(self, drones):
        for i, (d_id, p) in enumerate(sorted(drones.items())):
            row, col, alt, spraying, heading = p
            if d_id not in self.drones:
                self.drones[d_id] = self._build_drone(d_id, _DRONE_COLORS[i % len(_DRONE_COLORS)],
                                                      self.twin_field.grid_to_world(row, col, alt))
            target = self.twin_field.grid_to_world(row, col, alt)
            self.targets[d_id] = (target, bool(spraying), heading)

    def _build_drone(self, d_id, color, at):
        """Your DRONE_MESH, or a quadcopter of engine shapes (body, X arms, spinning rotors)."""
        size = DRONE_SIZE_M * self.cm_per_m                   # cm
        k = size / 100.0                                      # engine shapes are 100 cm
        parts = []                                            # (actor, local offset cm, local yaw, spins)
        if _asset(DRONE_MESH) is not None:
            kk = k * DRONE_MESH_SCALE
            body, _ = _mesh_actor(DRONE_MESH, at, unreal.Vector(kk, kk, kk), color, f"Drone_{d_id}",
                                  own_material=True)
        else:
            body, _ = _mesh_actor(_CUBE, at, unreal.Vector(k * 0.32, k * 0.32, k * 0.12), color,
                                  f"Drone_{d_id}")
            arm = size * 0.36                                 # hub to rotor
            for j, yaw in enumerate((45.0, 135.0)):
                a, _ = _mesh_actor(_CUBE, at, unreal.Vector(k * 1.02, k * 0.05, k * 0.04), (0.12, 0.12, 0.13),
                                   f"Drone_{d_id}_arm{j}")
                parts.append((a, (0.0, 0.0, 0.0), yaw, False))
            for j, yaw in enumerate((45.0, 135.0, 225.0, 315.0)):
                off = (arm * math.cos(math.radians(yaw)), arm * math.sin(math.radians(yaw)), size * 0.05)
                motor, _ = _mesh_actor(_CYLINDER, at, unreal.Vector(k * 0.07, k * 0.07, k * 0.07),
                                       (0.1, 0.1, 0.1), f"Drone_{d_id}_motor{j}")
                rotor, _ = _mesh_actor(_CUBE, at, unreal.Vector(k * 0.36, k * 0.035, k * 0.01),
                                       (0.85, 0.85, 0.85), f"Drone_{d_id}_rotor{j}")
                parts.append((motor, off, 0.0, False))
                parts.append((rotor, (off[0], off[1], size * 0.09), yaw, True))
            # Spray tank under the hub, in the drone's colour.
            tank, _ = _mesh_actor(_CYLINDER, at, unreal.Vector(k * 0.16, k * 0.16, k * 0.12), color,
                                  f"Drone_{d_id}_tank")
            parts.append((tank, (0.0, 0.0, -size * 0.1), 0.0, False))
        spray = _niagara_actor(SPRAY_NIAGARA, at, f"Spray_{d_id}")
        if spray is None:
            # Negative Z scale flips the cone so it points down from the drone.
            spray, _ = _mesh_actor(_CONE, at, unreal.Vector(k * 1.3, k * 1.3, -k * 1.2), (0.45, 0.75, 1.0),
                                   f"Spray_{d_id}")
        _set_active(spray, False)
        return {"body": body, "parts": parts, "cone": spray, "pos": None, "yaw": 0.0, "spin": 0.0,
                "spraying": False, "size": size}

    def _smooth_drones(self, dt=0.05):
        for d_id, (target, spraying, heading) in self.targets.items():
            d = self.drones[d_id]
            pos = target if d["pos"] is None else d["pos"] + (target - d["pos"]) * SMOOTHING
            d["pos"] = pos
            if heading is not None:
                d["yaw"] = heading - 90.0
            d["body"].set_actor_location(pos, False, True)
            d["body"].set_actor_rotation(unreal.Rotator(roll=0.0, pitch=0.0, yaw=d["yaw"]), True)
            d["spin"] = (d["spin"] + 1440.0 * max(dt, 0.0)) % 360.0     # 4 rev/s reads as "spinning"
            yr = math.radians(d["yaw"])
            cy, sy = math.cos(yr), math.sin(yr)
            for actor, (ox, oy, oz), yaw, spins in d["parts"]:
                actor.set_actor_location(pos + unreal.Vector(ox * cy - oy * sy, ox * sy + oy * cy, oz), False, True)
                actor.set_actor_rotation(
                    unreal.Rotator(roll=0.0, pitch=0.0, yaw=d["yaw"] + yaw + (d["spin"] if spins else 0.0)), True)
            drop = d["size"] * 0.8                             # spray hangs just under the drone
            d["cone"].set_actor_location(pos - unreal.Vector(0.0, 0.0, drop), False, True)
            if spraying != d["spraying"]:
                _set_active(d["cone"], spraying)
                d["spraying"] = spraying

    def stop(self):
        self.running = False
        poller = getattr(self, "poller", None)
        if poller is not None and poller.is_alive() and poller is not threading.current_thread():
            poller.join(timeout=3.0)      # let an in-flight request finish before tearing down
        try:
            unreal.unregister_slate_post_tick_callback(self.tick_handle)
        except Exception:
            pass
        for d in self.drones.values():
            _destroy(d["body"])
            _destroy(d["cone"])
            for part in d["parts"]:
                _destroy(part[0])
        for a in self.rain:
            _destroy(a)
        for t in getattr(self, "texts", [self.caption] if self.caption else []):
            _destroy(t)
        for b in getattr(self, "boards", []):
            _destroy(b)
        _TEXTS.clear()
        if getattr(self, "ground", None) is not None:
            _destroy(self.ground)
        for f in (self.twin_field, self.truth_field, self.forecast_field):
            if f is not None:
                f.destroy()


def twin_live_stop():
    import builtins
    live = getattr(builtins, "_twin_live", None)
    if live is not None:
        live.stop()
        builtins._twin_live = None
        unreal.log("[twin_live] stopped")


def panel_text(m, what_if=None):
    """Plain-ASCII results panel (UE's default text font)."""
    h, f, ms, e = m["harvest_t_ha"], m["fungicide"], m["missions"], m["twin_error"]
    zones = ", ".join(str(z) for z in h.get("spray_zones") or []) or "none"
    lines = [
        f"DAY {m['sim_day']}  |  {m['growth_stage']}",
        f"Expected harvest:  {h['now']:.2f} t/ha",
        f"  in 7 days, no spray:  {h['in_7d_no_spray']:.2f} t/ha",
        f"  in 7 days, sprayed:   {h['in_7d_if_sprayed']:.2f} t/ha  (zones {zones})",
    ]
    if what_if:
        lines.append(f"Worst disease in 7 days:  {what_if['no_spray']['fungus_peak'] * 100:.0f}%"
                     f"  ->  {what_if['if_sprayed']['fungus_peak'] * 100:.0f}% if sprayed")
    if f["blanket_kg"]:
        lines.append(f"Fungicide:  {f['used_kg']:.1f} kg  vs  {f['blanket_kg']:.0f} kg whole-field"
                     f"  ({f['saved_pct']:.0f}% saved)")
    else:
        lines.append("Fungicide:  none used yet")
    if e.get("first") is not None:
        lines.append(f"Twin vs reality error:  {e['first']:.3f}  ->  {e['now']:.3f}")
    lines.append(f"Missions:  {ms['scout']} scout  |  {ms['spray']} spray  |  {ms['flight_min']:.1f} min flown")
    return "\n".join(lines)


# ======================================================== editor menu + actions
def _post(path, body):
    req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5.0) as r:
        return json.loads(r.read().decode("utf-8"))


def _target_cell(live):
    """(row, col) for 'Inject outbreak': the selected field block, else the view centre on the field."""
    fields = [f for f in (live.twin_field, live.truth_field) if f is not None]
    try:
        selected = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
    except Exception:
        selected = []
    for actor in selected or []:
        label = actor.get_actor_label()
        parts = label.split("_")
        if len(parts) == 3 and parts[0] in ("Twin", "Reality") and parts[1].isdigit() and parts[2].isdigit():
            return int(parts[1]) + 0.5, int(parts[2]) + 0.5
        loc = actor.get_actor_location()
        for f in fields:
            r, c = f.world_to_grid(loc)
            if 0 <= r <= GRID and 0 <= c <= GRID:
                return r, c
    cam = _get_camera()
    if cam is not None and cam[1] is not None:
        loc, rot = cam
        fwd = rot.get_forward_vector()
        ground_z = live.center.z
        if fwd.z < -1e-3:
            hit = loc + fwd * ((ground_z - loc.z) / fwd.z)
            for f in fields:
                r, c = f.world_to_grid(hit)
                if 0 <= r <= GRID and 0 <= c <= GRID:
                    return r, c
    return GRID / 2.0, GRID / 2.0


def twin_live_action(action):
    """Run one menu action without blocking the editor."""
    import builtins
    live = getattr(builtins, "_twin_live", None)
    body, path = {"action": action}, "/live/action"
    if action == "demo":
        path, body = "/demo/start", {}
    elif action == "stop":
        path, body = "/demo/stop", {}
    elif action == "outbreak" and live is not None:
        row, col = _target_cell(live)
        body.update(row=round(row, 2), col=round(col, 2))

    def worker():
        try:
            _post(path, body)
        except urllib.error.HTTPError as e:
            msg = "busy: wait for the current step, or Stop" if e.code == 409 else f"HTTP {e.code}"
            unreal.log_warning(f"[twin_live] {action}: {msg}")
        except Exception as e:
            unreal.log_warning(f"[twin_live] {action}: cannot reach {API} ({e})")
    threading.Thread(target=worker, daemon=True).start()
    unreal.log(f"[twin_live] {action} {body if action == 'outbreak' else ''}".rstrip())


MENU_ITEMS = [
    ("Run", "TwinDemo", "Start demo (full story)", "twin_live_action('demo')"),
    ("Run", "TwinStop", "Stop", "twin_live_action('stop')"),
    ("Step", "TwinNextDay", "Next day", "twin_live_action('next_day')"),
    ("Step", "TwinScout", "Scout now", "twin_live_action('scout')"),
    ("Step", "TwinSpray", "Spray now", "twin_live_action('spray')"),
    ("Step", "TwinAuto", "Scout + spray", "twin_live_action('auto')"),
    ("What if", "TwinOutbreak", "Inject outbreak (selected block / view centre)", "twin_live_action('outbreak')"),
    ("What if", "TwinReset", "Reset field (new hidden outbreak)", "twin_live_action('reset')"),
    ("View", "TwinCamera", "Overview camera", "twin_live_view('overview')"),
    ("View", "TwinCloseup", "Close-up of the crop", "twin_live_view('closeup')"),
    ("View", "TwinFollow", "Follow a drone", "twin_live_view('follow')"),
]

# Toolbar above the viewport: (name, label, tooltip, command)
TOOLBAR_ITEMS = [
    ("TwinTbDemo", "Demo", "Run the full story: outbreak, scout, decide, spray, recovery", "twin_live_action('demo')"),
    ("TwinTbStop", "Stop", "Stop the story", "twin_live_action('stop')"),
    ("TwinTbDay", "Next day", "The crop grows and the disease spreads one day", "twin_live_action('next_day')"),
    ("TwinTbScout", "Scout", "Drones scout the field; the twin updates from their images", "twin_live_action('scout')"),
    ("TwinTbSpray", "Spray", "Drones spray the zones the twin flags", "twin_live_action('spray')"),
    ("TwinTbOutbreak", "Outbreak", "Plant a hidden disease patch in REALITY (selected patch or view centre)",
     "twin_live_action('outbreak')"),
    ("TwinTbReset", "Reset", "Start again with a new hidden outbreak", "twin_live_action('reset')"),
    ("TwinTbOverview", "Overview", "Camera: all three fields", "twin_live_view('overview')"),
    ("TwinTbCloseup", "Close-up", "Camera: low over the twin's crop", "twin_live_view('closeup')"),
    ("TwinTbFollow", "Follow drone", "Camera: chase a drone while it flies", "twin_live_view('follow')"),
]


def _register_menu():
    """Add the 'Digital Twin' menu to the editor's main menu bar (once per editor session)."""
    try:
        menus = unreal.ToolMenus.get()
        main = menus.find_menu("LevelEditor.MainMenu")
        menu = menus.find_menu("LevelEditor.MainMenu.DigitalTwin")
        if menu is None:
            menu = main.add_sub_menu("TwinLive", "", "DigitalTwin", "Digital Twin", "Digital twin demo controls")
        for section, name, label, command in MENU_ITEMS:
            entry = unreal.ToolMenuEntry(name=name, type=unreal.MultiBlockType.MENU_ENTRY)
            entry.set_label(label)
            entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", command)
            menu.add_menu_entry(section, entry)
        menus.refresh_all_widgets()
        return True
    except Exception as e:
        unreal.log_warning(f"[twin_live] could not add the Digital Twin menu ({e}); "
                           "use twin_live_action('scout') etc. in the Python console")
        return False


def _register_toolbar():
    """Digital Twin buttons on the level editor toolbar, above the viewport."""
    if not SHOW_TOOLBAR:
        return False
    try:
        menus = unreal.ToolMenus.get()
        bar = None
        for name in ("LevelEditor.LevelEditorToolBar.User", "LevelEditor.LevelEditorToolBar.PlayToolBar",
                     "LevelEditor.LevelEditorToolBar"):
            bar = menus.find_menu(name) or (menus.extend_menu(name) if hasattr(menus, "extend_menu") else None)
            if bar is not None:
                break
        if bar is None:
            raise RuntimeError("no level editor toolbar found")
        try:
            bar.add_section("DigitalTwin", "Digital Twin")
        except Exception:
            pass
        for name, label, tip, command in TOOLBAR_ITEMS:
            entry = unreal.ToolMenuEntry(name=name, type=unreal.MultiBlockType.TOOL_BAR_BUTTON)
            entry.set_label(label)
            try:
                entry.set_tool_tip(tip)
            except Exception:
                pass
            entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", command)
            bar.add_menu_entry("DigitalTwin", entry)
        menus.refresh_all_widgets()
        return True
    except Exception as e:
        unreal.log_warning(f"[twin_live] could not add the toolbar buttons ({e}); use the Digital Twin menu")
        return False


def twin_live_view(view="overview"):
    """Camera presets: 'overview', 'closeup' or 'follow'."""
    import builtins
    live = getattr(builtins, "_twin_live", None)
    if live is None:
        return
    if hasattr(live, "set_view") and not hasattr(live, "closeup_camera"):
        live.set_view(view)
    else:   # twin_paddy.TwinPaddy (PaddyTwin level) has its own camera methods
        {"overview": live.frame_camera, "closeup": live.closeup_camera,
         "follow": live.follow_camera}.get(view, live.frame_camera)()


def twin_live_camera():
    """Re-frame the editor camera over the fields (and stop following a drone)."""
    twin_live_view("overview")


def twin_live_start():
    import builtins
    twin_live_stop()                       # re-running the script restarts cleanly
    builtins._twin_live = TwinLive()
    builtins.twin_live_stop = twin_live_stop
    builtins.twin_live_camera = twin_live_camera
    builtins.twin_live_action = twin_live_action
    builtins.twin_live_view = twin_live_view
    _register_menu()
    _register_toolbar()
    unreal.log(f"[twin_live] running against {API}; twin_live_stop() to stop, "
               "twin_live_camera() to re-frame the view")
    return builtins._twin_live


if __name__ == "__main__":     # File > Execute Python Script; importing it (twin_paddy.py) only reuses the code
    twin_live_start()
