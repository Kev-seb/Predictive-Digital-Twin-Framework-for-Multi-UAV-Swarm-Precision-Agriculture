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
  "blocks" no setup at all: builds a 20 x 20 field of engine cubes (height =
           NDVI, colour = stress / fungus, faded where the twin is uncertain)
           plus a "Reality" field beside it.

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


def _mesh_actor(mesh_path, location, scale, color, label, own_material=False, color_param="Color"):
    """Static mesh actor tinted through a dynamic material instance.

    own_material=True keeps the mesh's own material (your crop or drone mesh)
    and only sets ``color_param`` on it; engine shapes get BasicShapeMaterial.
    """
    actor = _spawn(unreal.StaticMeshActor, location, label)
    comp = actor.static_mesh_component
    comp.set_static_mesh(_asset(mesh_path) if own_material else unreal.load_asset(mesh_path))
    actor.set_actor_scale3d(scale)
    try:
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
            self.twin_field = BlocksField(BLOCKS_ORIGIN, "Twin")
            gap = (GRID + BLOCKS_GAP_CELLS) * CELL_CM
            self.truth_field = BlocksField(BLOCKS_ORIGIN + unreal.Vector(gap, 0.0, 0.0), "Reality")
            if SHOW_FORECAST:
                self.forecast_field = BlocksField(BLOCKS_ORIGIN + unreal.Vector(2 * gap, 0.0, 0.0), "Forecast")
            unreal.log("[twin_live] blocks mode: built 'Twin', 'Reality'"
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
        self.boards = [_board(t) for t in self.texts] if SHOW_BOARDS else []
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
        cam = _get_camera()
        if cam is not None:
            for t in self.texts:
                _face_camera(t, cam[0])

    def _apply_field(self, f):
        if isinstance(self.twin_field, HismField):
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"),
                                  f.get("moisture"), f.get("anomalies", ()))
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
    ("View", "TwinCamera", "Re-frame camera", "twin_live_camera()"),
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


def twin_live_camera():
    """Re-frame the editor camera over the fields."""
    import builtins
    live = getattr(builtins, "_twin_live", None)
    if live is not None:
        live.frame_camera()


def twin_live_start():
    import builtins
    twin_live_stop()                       # re-running the script restarts cleanly
    builtins._twin_live = TwinLive()
    builtins.twin_live_stop = twin_live_stop
    builtins.twin_live_camera = twin_live_camera
    builtins.twin_live_action = twin_live_action
    _register_menu()
    unreal.log(f"[twin_live] running against {API}; twin_live_stop() to stop, "
               "twin_live_camera() to re-frame the view")
    return builtins._twin_live


twin_live_start()
