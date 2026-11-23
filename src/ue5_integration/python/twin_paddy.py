"""
twin_paddy.py - live controller for the PaddyTwin level (/Game/TwinDemo/Maps/PaddyTwin).

Reuses twin_live.py (API polling, frame handling, actions, Digital Twin menu) and drives the level
that paddy_scene.py built:

  * Field_TWIN / Field_REALITY / Field_FORECAST actors, each with two HISM components:
      Rice   400 instances (index = row * 20 + col, row 0 = north). Custom data 0-5:
             ndvi, stress, stress_std, moisture, fungus, anomaly (M_RiceCrop colours from these).
             Instance Z scale = plant height from NDVI (0.3 m .. 1.0 m).
      Water  400 instances. Custom data 0-4: moisture, ndvi, stress, stress_std, fungus
             (M_PaddyWater: glassy water when flooded, matte cracked mud when dry; far canopy tint).
  * three spray quadcopters (SM_DroneBody + 4 spinning SM_Rotor), NS_SprayMist under each while spraying
  * RainStreaks (GPU-animated streak mesh) over the fields on days with precipitation_mm > RAIN_MM
  * the EUW_TwinControl editor panel: buttons, camera views, live readouts, status line, legend

Start:  in the editor's Python console    import twin_paddy; twin_paddy.start()
        (or File > Execute Python Script > twin_paddy.py)
Stop:   twin_paddy.stop()
"""
import builtins
import math
import os
import sys
import threading
import time

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import twin_live as tl  # noqa: E402  (reused: polling, frame handling, actions, menu)

ROOT = "/Game/TwinDemo"
GRID = tl.GRID
DRONE_SCALE = 1.5          # ~2.4 m real span, drawn 1.5x so drones read from the overview camera
DRONE_COLORS = [(0.85, 0.86, 0.88), (0.95, 0.55, 0.08), (0.2, 0.55, 0.95)]
HEIGHT_MIN_M, HEIGHT_MAX_M = 0.3, 1.0
MESH_HEIGHT_M = 0.95       # SM_RiceCell is ~0.95 m tall at scale 1
EPS = 0.004                # skip custom-data writes smaller than this (keeps updates cheap)
RAIN_MM = tl.RAIN_MM
SPRAY_FX = f"{ROOT}/FX/NS_SprayMist.NS_SprayMist"
WIDGET = f"{ROOT}/UI/EUW_TwinControl.EUW_TwinControl"


def _actor(label):
    for a in tl._actors():
        if a.get_actor_label() == label:
            return a
    return None


def _comp(actor, name):
    for c in actor.get_components_by_class(unreal.InstancedStaticMeshComponent):
        if c.get_name() == name:
            return c
    return None


def ndvi_to_height_m(ndvi):
    t = max(0.0, min(1.0, (ndvi - 0.2) / 0.65))
    return HEIGHT_MIN_M + (HEIGHT_MAX_M - HEIGHT_MIN_M) * t


# ======================================================================= fields
class PaddyField(tl.HismField):
    """A built paddy: rice HISM drives the grid geometry (twin_live.HismField), plus water and heights."""

    def __init__(self, actor):
        rice, water = _comp(actor, "Rice"), _comp(actor, "Water")
        if rice is None or water is None:
            raise RuntimeError(f"{actor.get_actor_label()}: expected Rice and Water HISM components")
        super().__init__(rice)
        self.actor, self.water = actor, water
        self.base = []
        for i in range(GRID * GRID):
            t = rice.get_instance_transform(i, False)
            if isinstance(t, tuple):
                t = next(x for x in t if isinstance(x, unreal.Transform))
            self.base.append((t.translation, t.rotation.rotator()))
        self.rice_vals = [[None] * 6 for _ in range(GRID * GRID)]
        self.water_vals = [[None] * 5 for _ in range(GRID * GRID)]
        self.heights = [None] * (GRID * GRID)

    @staticmethod
    def _write(comp, cache, i, vals):
        changed = [k for k, v in enumerate(vals) if v is not None and
                   (cache[i][k] is None or abs(cache[i][k] - v) > EPS)]
        for k in changed:
            comp.set_custom_data_value(i, k, float(vals[k]), False)
            cache[i][k] = vals[k]
        return bool(changed)

    def apply(self, ndvi, stress, fungus, std, moisture=None, anomalies=()):
        anomalies = set(anomalies or ())
        n = min(GRID * GRID, len(ndvi))
        dirty_r = dirty_w = False
        moved = []
        for i in range(n):
            m = moisture[i] if moisture else None
            s_ = std[i] if std else 0.0
            dirty_r |= self._write(self.comp, self.rice_vals, i,
                                   [ndvi[i], stress[i], s_, m, fungus[i], 1.0 if i in anomalies else 0.0])
            dirty_w |= self._write(self.water, self.water_vals, i, [m, ndvi[i], stress[i], s_, fungus[i]])
            h = ndvi_to_height_m(ndvi[i])
            if self.heights[i] is None or abs(self.heights[i] - h) > 0.01:
                self.heights[i] = h
                moved.append(i)
        if moved:
            lo, hi = min(moved), max(moved)
            ts = []
            for i in range(lo, hi + 1):
                loc, rot = self.base[i]
                z = self.heights[i] / MESH_HEIGHT_M
                ts.append(unreal.Transform(loc, rot, unreal.Vector(1.0, 1.0, z)))
            self.comp.batch_update_instances_transforms(lo, ts, False, True, True)
        # one render-state refresh per component (the last custom-data write marks it dirty)
        if dirty_r:
            self.comp.set_custom_data_value(n - 1, 5, float(self.rice_vals[n - 1][5] or 0.0), True)
        if dirty_w:
            self.water.set_custom_data_value(n - 1, 4, float(self.water_vals[n - 1][4] or 0.0), True)


# ======================================================================= controller
class TwinPaddy(tl.TwinLive):
    """twin_live.TwinLive for the PaddyTwin level (no blocks, no in-world caption: the panel shows it)."""

    def __init__(self):
        self.frame, self.seq, self.lock, self.running = None, -1, threading.Lock(), True
        self.drones, self.targets, self.caption, self.panel = {}, {}, None, None
        self.errors, self.texts, self.what_if = 0, [], None
        self.status_text, self.metrics, self.field = "Connecting to the twin API...", None, None
        self.cam_mode, self.follow_id, self.fps = "overview", None, 0.0
        self.force_rain = None     # True/False overrides the API weather (twin_paddy.rain(...)); None = API
        self._fps_acc, self._fps_n, self._ui_t = 0.0, 0, 0.0
        fields = {}
        for key in ("TWIN", "REALITY", "FORECAST"):
            a = _actor(f"Field_{key}")
            if a is None:
                raise RuntimeError(f"Field_{key} not found: open /Game/TwinDemo/Maps/PaddyTwin first")
            fields[key] = PaddyField(a)
        self.twin_field, self.truth_field, self.forecast_field = fields["TWIN"], fields["REALITY"], fields["FORECAST"]
        size = self.twin_field.size_cm()
        centers = [f.grid_to_world(GRID / 2.0, GRID / 2.0, 0.0) for f in fields.values()]
        self.center = (centers[0] + centers[2]) * 0.5
        self.span = size + (centers[-1] - centers[0]).length()
        self.cm_per_m = size / (GRID * 5.0)
        self.rain, self.rain_on, self.raining, self.rain_streaks = [], False, False, None
        self._build_rain(size)
        self.ui = PanelBinding(self)
        self.frame_camera()
        self.poller = threading.Thread(target=self._poll, daemon=True)
        self.poller.start()
        self.tick_handle = unreal.register_slate_post_tick_callback(self._tick)

    # ---------------------------------------------------------------- rain
    def _build_rain(self, size):
        """RainStreaks (paddy_scene.build_rain): GPU-animated streaks, shown on rainy days.

        UE 5.8 note: a Niagara rain authored through the toolset did not render reliably in the editor
        viewport, so the streaks fall in the material (WPO) instead - no per-frame Python cost."""
        a = _actor("RainStreaks")
        if a is None:
            unreal.log_warning("[twin_paddy] RainStreaks actor missing (paddy_scene.build_rain()); rain disabled")
            return
        self.rain = [a]
        a.set_is_temporarily_hidden_in_editor(True)

    def _update_rain(self, dt):
        if self.raining != self.rain_on:
            for a in self.rain:
                a.set_is_temporarily_hidden_in_editor(not self.raining)
            self.rain_on = self.raining
            self._weather_look(self.raining)

    @staticmethod
    def _weather_look(rainy):
        """Overcast look on rainy days: dimmer sun, thicker haze (restored when it stops)."""
        sun, fog = _actor("Sun"), _actor("HeightFog")
        if sun is not None:
            sun.light_component.set_intensity(2.5 if rainy else 9.0)
        if fog is not None:
            fog.component.set_fog_density(0.035 if rainy else 0.007)
    # ---------------------------------------------------------------- status / panel
    def _apply_status(self, status, field):
        if status:
            err = status.get("error") or {}
            extra = f"   |   twin vs reality error {err['fungus']:.3f}" if err.get("fungus") is not None else ""
            self.status_text = f"{status['text']}{extra}"
        elif field is not None:
            self.status_text = f"Day {field['sim_day']}  ({field['growth_stage']})"

    def _tick(self, dt):
        with self.lock:
            peek = self.frame
        if peek is not None:
            if peek.get("metrics"):
                self.metrics = peek["metrics"]
            if peek.get("field"):
                self.field = peek["field"]
        super()._tick(dt)
        if self.force_rain is not None and self.raining != self.force_rain:
            self.raining = self.force_rain
            self._update_rain(0.0)
        self._fps_acc += dt
        self._fps_n += 1
        if self._fps_acc >= 1.0:
            self.fps, self._fps_acc, self._fps_n = self._fps_n / self._fps_acc, 0.0, 0
        if self.cam_mode == "follow":
            self._follow_camera(dt)
        self._ui_t += dt
        if self._ui_t >= 0.25:
            self._ui_t = 0.0
            self.ui.refresh()

    # ---------------------------------------------------------------- drones
    def _build_drone(self, d_id, color, at):
        idx = len(self.drones)
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        body = tl._spawn(unreal.StaticMeshActor, at, f"Drone_{d_id}")
        mc = body.static_mesh_component
        mc.set_static_mesh(unreal.load_asset(f"{ROOT}/Meshes/SM_DroneBody"))
        mc.set_mobility(unreal.ComponentMobility.MOVABLE)
        body.set_actor_scale3d(unreal.Vector(DRONE_SCALE, DRONE_SCALE, DRONE_SCALE))
        mid = mc.create_dynamic_material_instance(0, None)
        if mid is not None:
            mid.set_vector_parameter_value("Color", unreal.LinearColor(color[0], color[1], color[2], 1.0))
        rotors = []
        import paddy_scene
        rotor_mesh = unreal.load_asset(f"{ROOT}/Meshes/SM_Rotor")
        for k in range(4):
            a = math.radians(45 + 90 * k)
            rc = paddy_scene._add_comp(body, unreal.StaticMeshComponent, f"Rotor{k}")
            rc.set_static_mesh(rotor_mesh)
            rc.set_mobility(unreal.ComponentMobility.MOVABLE)
            rc.set_relative_location(unreal.Vector(math.cos(a) * 120.0, math.sin(a) * 120.0, 18.0), False, True)
            if mid is not None:
                rc.set_material(0, mid)
            rotors.append([rc, k * 37.0, 1.0 if k % 2 else -1.0])
        # Spray mist: a separate Niagara actor that follows the drone (a Niagara component added to a
        # placed actor at runtime does not simulate in the editor world; a NiagaraActor does).
        spray = tl._niagara_actor(SPRAY_FX, at - unreal.Vector(0, 0, 80.0), f"Spray_{d_id}")
        if spray is not None:
            tl._set_active(spray, False)
        return {"body": body, "rotors": rotors, "spray": spray, "pos": None, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "spraying": False, "parts": [], "cone": None}

    def _smooth_drones(self, dt=0.016):
        dt = max(dt, 0.0)
        k = 1.0 - math.exp(-dt * 5.0)
        for d_id, (target, spraying, heading) in self.targets.items():
            d = self.drones[d_id]
            prev = d["pos"]
            pos = target if prev is None else prev + (target - prev) * k
            d["pos"] = pos
            if heading is not None:
                want = heading - 90.0
                diff = (want - d["yaw"] + 540.0) % 360.0 - 180.0
                d["yaw"] += diff * min(1.0, dt * 4.0)
            # lean into the direction of travel
            if prev is not None and dt > 0:
                v = (pos - prev) * (1.0 / dt)
                yr = math.radians(d["yaw"])
                fwd = v.x * math.cos(yr) + v.y * math.sin(yr)
                side = -v.x * math.sin(yr) + v.y * math.cos(yr)
                d["pitch"] += (max(-14.0, min(14.0, -fwd / 60.0)) - d["pitch"]) * min(1.0, dt * 3.0)
                d["roll"] += (max(-14.0, min(14.0, side / 60.0)) - d["roll"]) * min(1.0, dt * 3.0)
            d["body"].set_actor_location(pos, False, True)
            d["body"].set_actor_rotation(unreal.Rotator(roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"]), True)
            airborne = target.z - self.twin_field.grid_to_world(0, 0, 0).z > 150.0
            rate = 2600.0 if airborne else 900.0
            for r in d["rotors"]:
                r[1] = (r[1] + r[2] * rate * dt) % 360.0
                r[0].set_relative_rotation(unreal.Rotator(0.0, 0.0, r[1]), False, True)
            if d["spray"] is not None:
                d["spray"].set_actor_location(pos - unreal.Vector(0, 0, 70.0 * DRONE_SCALE), False, True)
                if spraying != d["spraying"]:
                    tl._set_active(d["spray"], spraying)
            d["spraying"] = spraying

    # ---------------------------------------------------------------- cameras
    def frame_camera(self):
        self.cam_mode = "overview"
        cam = self.center + unreal.Vector(0.0, 25000.0, 9500.0)
        look = self.center + unreal.Vector(0.0, -2500.0, 0.0)
        tl._editor().set_level_viewport_camera_info(cam, tl._look_rotation(cam, look))

    def closeup_camera(self):
        self.cam_mode = "closeup"
        f = self.twin_field
        cam = f.grid_to_world(23.0, 8.0, 7.0)
        look = f.grid_to_world(12.0, 11.0, 0.0)
        tl._editor().set_level_viewport_camera_info(cam, tl._look_rotation(cam, look))

    def follow_camera(self, d_id=None):
        if not self.drones:
            unreal.log_warning("[twin_paddy] no drone yet: start the demo or Scout first")
            return
        ids = sorted(self.drones)
        self.follow_id = d_id if d_id in self.drones else (
            ids[(ids.index(self.follow_id) + 1) % len(ids)] if self.follow_id in ids else ids[0])
        self.cam_mode = "follow"
        self._follow_pos = None

    def _follow_camera(self, dt):
        # prefer a drone that is spraying right now, so the mist is in shot
        cur = self.drones.get(self.follow_id)
        if cur is None or not cur["spraying"]:
            for k in sorted(self.drones):
                if self.drones[k]["spraying"]:
                    self.follow_id, cur = k, self.drones[k]
                    self._follow_pos = None          # cut to the sprayer instead of a long glide
                    break
        d = cur
        if d is None or d["pos"] is None:
            return
        yr = math.radians(d["yaw"])
        back = unreal.Vector(-math.cos(yr), -math.sin(yr), 0.0)
        side = unreal.Vector(-math.sin(yr), math.cos(yr), 0.0)
        # behind and to the side, a little above: drone, spray and crop all in frame
        want = d["pos"] + back * 750.0 + side * 600.0 + unreal.Vector(0, 0, 160.0)
        p = getattr(self, "_follow_pos", None)
        p = want if p is None else p + (want - p) * min(1.0, dt * 4.0)
        self._follow_pos = p
        tl._editor().set_level_viewport_camera_info(p, tl._look_rotation(p, d["pos"] - unreal.Vector(0, 0, 220.0)))

    def stop(self):
        self.running = False
        if self.poller.is_alive() and self.poller is not threading.current_thread():
            self.poller.join(timeout=3.0)
        try:
            unreal.unregister_slate_post_tick_callback(self.tick_handle)
        except Exception:
            pass
        for d in self.drones.values():
            tl._destroy(d["body"])
            if d["spray"] is not None:
                tl._destroy(d["spray"])
        for a in self.rain:          # level actor: hide, never destroy
            a.set_is_temporarily_hidden_in_editor(True)
        self._weather_look(False)
        self.ui.unbind()


def _target_cell(live):
    """Inject-outbreak cell: the selected CellCursor (or any selected non-field actor) if it sits on a field,
    else where the centre of the view meets the ground."""
    fields = [live.twin_field, live.truth_field]
    try:
        selected = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
    except Exception:
        selected = []
    for actor in selected or []:
        if actor.get_actor_label().startswith("Field_"):
            continue
        loc = actor.get_actor_location()
        for f in fields:
            r, c = f.world_to_grid(loc)
            if 0 <= r < GRID and 0 <= c < GRID:
                return r, c
    cam = tl._get_camera()
    if cam is not None and cam[1] is not None:
        loc, rot = cam
        fwd = rot.get_forward_vector()
        if fwd.z < -1e-3:
            hit = loc + fwd * ((0.0 - loc.z) / fwd.z)
            for f in fields:
                r, c = f.world_to_grid(hit)
                if 0 <= r < GRID and 0 <= c < GRID:
                    return r, c
    return GRID / 2.0, GRID / 2.0


tl._target_cell = _target_cell     # twin_live_action('outbreak') uses this


# ======================================================================= editor panel
LEGEND = [("Healthy", (0.045, 0.2, 0.018)), ("Water stress", (0.55, 0.45, 0.06)), ("Severe stress", (0.24, 0.12, 0.035)),
          ("Fungus", (0.20, 0.055, 0.025)), ("Uncertain (twin)", (0.42, 0.42, 0.42))]


class PanelBinding:
    """Binds the EUW_TwinControl tab (built by twin_ui.py) to the controller: buttons + live readouts."""

    BUTTONS = {"Btn_Start": "demo", "Btn_Stop": "stop", "Btn_NextDay": "next_day", "Btn_Scout": "scout",
               "Btn_Spray": "spray", "Btn_Auto": "auto", "Btn_Outbreak": "outbreak", "Btn_Reset": "reset"}
    CAMS = {"Btn_CamOverview": "overview", "Btn_CamClose": "closeup", "Btn_CamFollow": "follow"}

    def __init__(self, live):
        self.live, self.widget, self._bound = live, None, set()
        self.open()

    def open(self):
        try:
            bp = unreal.load_asset(WIDGET)
        except Exception:
            bp = None
        if bp is None:
            unreal.log_warning("[twin_paddy] EUW_TwinControl not found; use the Digital Twin menu instead")
            return
        sub = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
        w = sub.find_utility_widget_from_blueprint(bp)
        if w is None:
            w = sub.spawn_and_register_tab(bp)
        self._bind(w)

    def _bind(self, w):
        if w is None or w == self.widget:
            return
        self.widget, self._bound = w, set()
        for name, action in self.BUTTONS.items():
            self._on(name, self._action_cb(action))
        for name, mode in self.CAMS.items():
            self._on(name, self._camera_cb(mode))

    # OnClicked callables must take no parameters at all (UE counts defaulted ones too)
    @staticmethod
    def _action_cb(action):
        def cb():
            tl.twin_live_action(action)
        return cb

    def _camera_cb(self, mode):
        def cb():
            self._camera(mode)
        return cb

    def _on(self, name, fn):
        try:
            btn = self.widget.get_editor_property(name)
        except Exception:
            btn = None
        if btn is None:
            return
        btn.on_clicked.add_callable(fn)
        self._bound.add(name)

    def _camera(self, mode):
        if mode == "overview":
            self.live.frame_camera()
        elif mode == "closeup":
            self.live.closeup_camera()
        else:
            self.live.follow_camera()

    def _set(self, name, text):
        try:
            tb = self.widget.get_editor_property(name)
            if tb is not None:
                tb.set_text(unreal.Text(text))
        except Exception:
            pass

    def refresh(self):
        if self.widget is None or not unreal.SystemLibrary.is_valid(self.widget):
            self.widget = None
            bp = unreal.load_asset(WIDGET) if unreal.EditorAssetLibrary.does_asset_exist(WIDGET.split(".")[0]) else None
            if bp is not None:
                self._bind(unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem).find_utility_widget_from_blueprint(bp))
            if self.widget is None:
                return
        live, m = self.live, self.live.metrics
        self._set("Txt_Status", live.status_text)
        if m:
            h, f, ms, e = m["harvest_t_ha"], m["fungicide"], m["missions"], m["twin_error"]
            self._set("Txt_Day", f"Day {m['sim_day']}   |   {m['growth_stage']}")
            self._set("Txt_HarvestNow", f"{h['now']:.2f} t/ha")
            self._set("Txt_Harvest7NoSpray", f"{h['in_7d_no_spray']:.2f} t/ha")
            zones = ", ".join(str(z) for z in h.get("spray_zones") or []) or "none"
            self._set("Txt_Harvest7Spray", f"{h['in_7d_if_sprayed']:.2f} t/ha   (zones {zones})")
            if f["blanket_kg"]:
                self._set("Txt_Fungicide", f"{f['used_kg']:.1f} kg  vs  {f['blanket_kg']:.0f} kg whole-field   "
                                           f"({f['saved_pct']:.0f}% saved)")
            else:
                self._set("Txt_Fungicide", "none used yet")
            if e.get("first") is not None:
                self._set("Txt_Error", f"{e['first']:.3f}  ->  {e['now']:.3f}")
            self._set("Txt_Missions", f"{ms['scout']} scout   {ms['spray']} spray   {ms['flight_min']:.1f} min flown")
        w = live.what_if
        if w:
            self._set("Txt_Disease", f"{w['no_spray']['fungus_peak'] * 100:.0f}%  ->  "
                                     f"{w['if_sprayed']['fungus_peak'] * 100:.0f}% if sprayed")
        cam = {"overview": "Overview", "closeup": "Close-up (TWIN)", "follow": f"Follow {live.follow_id or ''}"}[live.cam_mode]
        self._set("Txt_View", f"View: {cam}     {live.fps:.0f} fps")

    def unbind(self):
        self.widget = None


# ======================================================================= entry points
def start():
    stop()
    live = TwinPaddy()
    builtins._twin_live = live
    builtins.twin_live_action = tl.twin_live_action
    builtins.twin_live_camera = live.frame_camera
    builtins.twin_live_view = tl.twin_live_view       # the menu's camera entries
    builtins.twin_paddy = sys.modules[__name__]
    tl._register_menu()
    unreal.log(f"[twin_paddy] running against {tl.API}; twin_paddy.stop() to stop")
    return live


def stop():
    live = getattr(builtins, "_twin_live", None)
    if live is not None:
        try:
            live.stop()
        except Exception as e:
            unreal.log_warning(f"[twin_paddy] stop: {e}")
        builtins._twin_live = None
    # remove leftovers from an earlier session (e.g. after a Python reload)
    for a in tl._actors():
        lab = a.get_actor_label()
        if lab.startswith(("Drone_drone", "Spray_drone", "TwinRain_", "FXTest_")):
            tl._destroy(a)


def live():
    return getattr(builtins, "_twin_live", None)


def overview():
    live().frame_camera()


def closeup():
    live().closeup_camera()


def follow(d_id=None):
    live().follow_camera(d_id)


def rain(on=True):
    """Force rain on/off for a shot (None = follow the API weather, precipitation_mm > RAIN_MM)."""
    live().force_rain = on


if __name__ == "__main__":
    start()
