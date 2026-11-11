"""
twin_live.py - show the paddy-field digital twin live in the UE5 editor.

No C++ and no extra plugins: runs in UE5's built-in Python
(Edit > Plugins > "Python Editor Script Plugin", then restart the editor).

  1. Start the twin:            python digital_twin_api.py
  2. In UE5: File > Execute Python Script... > twin_live.py
     (or in the Output Log, Python mode:  exec(open(r"C:/path/to/twin_live.py").read()) )
  3. Start the story:           curl -X POST localhost:8008/demo/start -H 'content-type: application/json' -d '{}'
  4. Stop the script:           run it again, or in the Python console: twin_live_stop()

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
the demo's narration and the twin-vs-reality error.
"""

import json
import threading
import time
import urllib.request

import unreal

# ------------------------------------------------------------------ settings
API = "http://127.0.0.1:8008"
POLL_HZ = 10.0
FIELD_MODE = "auto"              # "auto" | "hism" | "blocks"
TWIN_ACTOR_LABEL = None          # hism mode: label of your field actor (None = first 400-instance ISM)
REALITY_LABEL_HINT = "Reality"   # hism mode: label substring of the optional reality field
BLOCKS_ORIGIN = unreal.Vector(0.0, 0.0, 0.0)   # blocks mode: north-west corner of the twin field (cm)
BLOCKS_GAP_CELLS = 4             # blocks mode: gap between twin and reality fields
CELL_CM = 500.0                  # blocks mode: 5 m cells, like the twin's 20 x 20 grid
GRID = 20
SMOOTHING = 0.35                 # drone motion smoothing per editor tick (0..1)

_SPHERE = "/Engine/BasicShapes/Sphere"
_CUBE = "/Engine/BasicShapes/Cube"
_CONE = "/Engine/BasicShapes/Cone"
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


def _mesh_actor(mesh_path, location, scale, color, label):
    actor = _spawn(unreal.StaticMeshActor, location, label)
    comp = actor.static_mesh_component
    comp.set_static_mesh(unreal.load_asset(mesh_path))
    actor.set_actor_scale3d(scale)
    mid = comp.create_dynamic_material_instance(0, unreal.load_asset(_SHAPE_MAT))
    if color is not None and mid is not None:
        mid.set_vector_parameter_value("Color", unreal.LinearColor(color[0], color[1], color[2], 1.0))
    return actor, mid


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def crop_color(stress, fungus, std=0.0):
    """Healthy green -> yellow (stress) -> brown, darkened by fungus, faded by uncertainty."""
    green, yellow, brown, grey = (0.10, 0.55, 0.12), (0.85, 0.78, 0.20), (0.42, 0.24, 0.10), (0.5, 0.5, 0.5)
    s = max(0.0, min(1.0, stress * 2.5))
    c = _lerp(green, yellow, min(1.0, s * 2.0)) if s < 0.5 else _lerp(yellow, brown, (s - 0.5) * 2.0)
    c = _lerp(c, (0.30, 0.05, 0.05), max(0.0, min(1.0, fungus * 1.6)))
    return _lerp(c, grey, max(0.0, min(0.8, std * 8.0)))


# ======================================================================= fields
class BlocksField:
    """A 20 x 20 field of cubes: height = NDVI, colour = stress / fungus / uncertainty."""

    def __init__(self, origin, label):
        self.origin, self.cells = origin, []
        for r in range(GRID):
            for c in range(GRID):
                loc = unreal.Vector(origin.x + (c + 0.5) * CELL_CM, origin.y + (r + 0.5) * CELL_CM, origin.z)
                actor, mid = _mesh_actor(_CUBE, loc, unreal.Vector(CELL_CM / 105.0, CELL_CM / 105.0, 0.3),
                                         (0.1, 0.55, 0.12), f"{label}_{r:02d}_{c:02d}")
                self.cells.append((actor, mid, loc))

    def apply(self, ndvi, stress, fungus, std):
        for i, (actor, mid, loc) in enumerate(self.cells):
            height = 0.15 + 1.6 * max(0.0, ndvi[i])
            actor.set_actor_scale3d(unreal.Vector(CELL_CM / 105.0, CELL_CM / 105.0, height))
            actor.set_actor_location(unreal.Vector(loc.x, loc.y, loc.z + height * 50.0), False, True)
            if mid is not None:
                r, g, b = crop_color(stress[i], fungus[i], std[i] if std else 0.0)
                mid.set_vector_parameter_value("Color", unreal.LinearColor(r, g, b, 1.0))

    def grid_to_world(self, row, col, alt_m):
        return unreal.Vector(self.origin.x + col * CELL_CM, self.origin.y + row * CELL_CM,
                             self.origin.z + alt_m * 100.0)

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

    def grid_to_world(self, row, col, alt_m):
        # Altitude scaled like the field, so drones keep their proportions on a resized field.
        return self.corner + self.col_axis * col + self.row_axis * row + \
            unreal.Vector(0.0, 0.0, alt_m * self.cm_per_m)

    def destroy(self):
        pass


def _find_hism_fields():
    twin = reality = None
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
            elif twin is None and (TWIN_ACTOR_LABEL is None or label == TWIN_ACTOR_LABEL):
                twin = comp
    return twin, reality


# ===================================================================== client
class TwinLive:
    def __init__(self):
        self.frame, self.seq, self.lock, self.running = None, -1, threading.Lock(), True
        self.twin_field = self.truth_field = None
        self.drones, self.targets, self.caption = {}, {}, None
        self.errors = 0
        twin, reality = (None, None) if FIELD_MODE == "blocks" else _find_hism_fields()
        if twin is not None:
            self.twin_field = HismField(twin)
            self.truth_field = HismField(reality) if reality is not None else None
            unreal.log(f"[twin_live] hism mode: field '{twin.get_owner().get_actor_label()}'"
                       + (f", reality '{reality.get_owner().get_actor_label()}'" if reality else ""))
        elif FIELD_MODE == "hism":
            raise RuntimeError("FIELD_MODE='hism' but no 400-instance ISM/HISM found in the level")
        else:
            self.twin_field = BlocksField(BLOCKS_ORIGIN, "Twin")
            gap = (GRID + BLOCKS_GAP_CELLS) * CELL_CM
            self.truth_field = BlocksField(BLOCKS_ORIGIN + unreal.Vector(gap, 0.0, 0.0), "Reality")
            unreal.log("[twin_live] blocks mode: built 'Twin' and 'Reality' fields")
        cap_loc = self.twin_field.grid_to_world(-2.0, GRID / 2.0, 12.0)
        self.caption = _spawn(unreal.TextRenderActor, cap_loc, "TwinCaption")
        try:   # (UE's enum really is spelled "HorizTextAligment")
            self.caption.text_render.set_horizontal_alignment(unreal.HorizTextAligment.EHTA_CENTER)
        except Exception:
            pass
        self.caption.text_render.set_world_size(120.0)
        self.caption.text_render.set_text("Digital twin: waiting for the API...")
        self.poller = threading.Thread(target=self._poll, daemon=True)
        self.poller.start()
        self.tick_handle = unreal.register_slate_post_tick_callback(self._tick)

    # background thread: HTTP only, never touches UObjects
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
            if frame.get("drones"):
                self._apply_drones(frame["drones"]["drones"])
        elif self.errors > 20 and self.caption is not None:
            self.caption.text_render.set_text(f"Digital twin: cannot reach {API}")
        self._smooth_drones()

    def _apply_field(self, f):
        if isinstance(self.twin_field, HismField):
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"),
                                  f.get("moisture"), f.get("anomalies", ()))
        else:
            self.twin_field.apply(f["ndvi"], f["stress"], f["fungus"], f.get("stress_std"))
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
            self.caption.text_render.set_text(f"{status['text']}{extra}")
        elif field is not None:
            self.caption.text_render.set_text(f"Day {field['sim_day']}  ({field['growth_stage']})")

    def _apply_drones(self, drones):
        for i, (d_id, p) in enumerate(sorted(drones.items())):
            row, col, alt, spraying, heading = p
            if d_id not in self.drones:
                color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
                body, _ = _mesh_actor(_SPHERE, self.twin_field.grid_to_world(row, col, alt),
                                      unreal.Vector(1.6, 1.6, 0.6), color, f"Drone_{d_id}")
                cone, _ = _mesh_actor(_CONE, self.twin_field.grid_to_world(row, col, max(alt - 1.5, 0.0)),
                                      unreal.Vector(2.5, 2.5, -1.5), (0.45, 0.75, 1.0), f"Spray_{d_id}")
                self.drones[d_id] = {"body": body, "cone": cone, "pos": None}
            target = self.twin_field.grid_to_world(row, col, alt)
            self.targets[d_id] = (target, bool(spraying), heading)

    def _smooth_drones(self):
        for d_id, (target, spraying, heading) in self.targets.items():
            d = self.drones[d_id]
            pos = target if d["pos"] is None else d["pos"] + (target - d["pos"]) * SMOOTHING
            d["pos"] = pos
            d["body"].set_actor_location(pos, False, True)
            if heading is not None:
                d["body"].set_actor_rotation(unreal.Rotator(roll=0.0, pitch=0.0, yaw=heading - 90.0), True)
            d["cone"].set_actor_location(pos - unreal.Vector(0.0, 0.0, 160.0), False, True)
            d["cone"].set_is_temporarily_hidden_in_editor(not spraying)

    def stop(self):
        self.running = False
        try:
            unreal.unregister_slate_post_tick_callback(self.tick_handle)
        except Exception:
            pass
        for d in self.drones.values():
            _destroy(d["body"])
            _destroy(d["cone"])
        if self.caption is not None:
            _destroy(self.caption)
        for f in (self.twin_field, self.truth_field):
            if f is not None:
                f.destroy()


def twin_live_stop():
    import builtins
    live = getattr(builtins, "_twin_live", None)
    if live is not None:
        live.stop()
        builtins._twin_live = None
        unreal.log("[twin_live] stopped")


def twin_live_start():
    import builtins
    twin_live_stop()                       # re-running the script restarts cleanly
    builtins._twin_live = TwinLive()
    builtins.twin_live_stop = twin_live_stop
    unreal.log(f"[twin_live] running against {API}; twin_live_stop() to stop")
    return builtins._twin_live


twin_live_start()
