"""
Minimal stand-in for UE5's `unreal` Python module, covering exactly the API
src/ue5_integration/python/twin_live.py uses, so the script can be tested
without the editor. It records state (locations, colours, custom data,
visibility, captions) for assertions.
"""

import math

_TICKS = {}
_LEVEL = []
LOG = []


def log(msg):
    LOG.append(str(msg))


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k):
        return Vector(self.x * k, self.y * k, self.z * k)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return f"Vector({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


class Rotator:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def get_forward_vector(self):
        p, y = math.radians(self.pitch), math.radians(self.yaw)
        return Vector(math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))


class LinearColor:
    def __init__(self, r, g, b, a=1.0):
        self.r, self.g, self.b, self.a = r, g, b, a


class Transform:
    def __init__(self, translation):
        self.translation = translation


class HorizTextAligment:
    EHTA_CENTER = "center"


class _MID:
    def __init__(self):
        self.params = {}

    def set_vector_parameter_value(self, name, value):
        self.params[name] = value


class _MeshComponent:
    def __init__(self):
        self.mesh = None
        self.mid = None

    def set_static_mesh(self, mesh):
        self.mesh = mesh

    def create_dynamic_material_instance(self, index, source=None, name=None):
        self.mid = _MID()
        return self.mid


class _TextRender:
    def __init__(self):
        self.text, self.size, self.align = "", 0.0, None

    def set_text(self, text):
        self.text = str(text)

    def set_world_size(self, size):
        self.size = size

    def set_horizontal_alignment(self, value):
        self.align = value

    def set_text_render_color(self, color):
        self.color = color


class Actor:
    def __init__(self, location=None):
        self.location = location or Vector()
        self.label = ""
        self.scale = Vector(1, 1, 1)
        self.rotation = Rotator()
        self.hidden = False
        self.components = []

    def get_actor_location(self):
        return self.location

    def set_actor_label(self, label):
        self.label = label

    def get_actor_label(self):
        return self.label

    def set_actor_scale3d(self, scale):
        self.scale = scale

    def set_actor_location(self, loc, sweep, teleport):
        self.location = loc

    def set_actor_rotation(self, rot, teleport):
        self.rotation = rot

    def set_is_temporarily_hidden_in_editor(self, hidden):
        self.hidden = hidden

    def get_components_by_class(self, cls):
        return [c for c in self.components if isinstance(c, cls)]


class StaticMeshActor(Actor):
    def __init__(self, location=None):
        super().__init__(location)
        self.static_mesh_component = _MeshComponent()


class _NiagaraComponent:
    def __init__(self):
        self.asset, self.active = None, False

    def set_asset(self, asset):
        self.asset = asset

    def activate(self, reset=False):
        self.active = True

    def deactivate(self):
        self.active = False


class NiagaraActor(Actor):
    def __init__(self, location=None):
        super().__init__(location)
        self.niagara_component = _NiagaraComponent()


class TextRenderActor(Actor):
    def __init__(self, location=None):
        super().__init__(location)
        self.text_render = _TextRender()


class InstancedStaticMeshComponent:
    def __init__(self, owner, centers):
        self.owner, self.centers = owner, centers
        self.num_custom_data_floats = 0
        self.custom = {}
        self.dirty_marks = 0
        owner.components.append(self)

    def get_instance_count(self):
        return len(self.centers)

    def get_instance_transform(self, index, world_space=False):
        return True, Transform(self.centers[index])   # UE returns (success, transform)

    def get_editor_property(self, name):
        return getattr(self, name)

    def set_num_custom_data_floats(self, n):
        self.num_custom_data_floats = n

    def set_custom_data(self, index, values, mark_dirty):
        assert len(values) <= self.num_custom_data_floats
        self.custom[index] = list(values)
        self.dirty_marks += int(bool(mark_dirty))

    def get_owner(self):
        return self.owner


class HierarchicalInstancedStaticMeshComponent(InstancedStaticMeshComponent):
    pass


class Color:
    def __init__(self, r, g, b, a=255):
        self.r, self.g, self.b, self.a = r, g, b, a


def log_warning(msg):
    LOG.append("WARNING: " + str(msg))


class UnrealEditorSubsystem:
    camera = None

    def get_level_viewport_camera_info(self):
        return UnrealEditorSubsystem.camera          # UE returns (location, rotation) or None

    def set_level_viewport_camera_info(self, location, rotation):
        UnrealEditorSubsystem.camera = (location, rotation)


class EditorActorSubsystem:
    selected = []

    def get_all_level_actors(self):
        return list(_LEVEL)

    def get_selected_level_actors(self):
        return list(EditorActorSubsystem.selected)

    def spawn_actor_from_class(self, cls, location, rotation):
        actor = cls(location)
        _LEVEL.append(actor)
        return actor

    def destroy_actor(self, actor):
        if actor in _LEVEL:
            _LEVEL.remove(actor)


_SUBSYSTEM = EditorActorSubsystem()


_EDITOR = UnrealEditorSubsystem()


class MultiBlockType:
    MENU_ENTRY = "menu_entry"


class ToolMenuStringCommandType:
    PYTHON = "python"


class ToolMenuEntry:
    def __init__(self, name="", type=None):
        self.name, self.type, self.label, self.command = name, type, "", None

    def set_label(self, label):
        self.label = label

    def set_string_command(self, kind, custom_type, command):
        self.command = (kind, command)


class ToolMenu:
    def __init__(self, name, label=""):
        self.name, self.label, self.entries = name, label, []

    def add_sub_menu(self, owner, section, name, label, tooltip=""):
        sub = ToolMenu(f"{self.name}.{name}", label)
        ToolMenus._menus[sub.name] = sub
        return sub

    def add_menu_entry(self, section, entry):
        self.entries = [e for e in self.entries if e[1].name != entry.name] + [(section, entry)]


class ToolMenus:
    _menus = {}
    _instance = None
    refreshed = 0

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ToolMenus()
            cls._menus["LevelEditor.MainMenu"] = ToolMenu("LevelEditor.MainMenu")
        return cls._instance

    def find_menu(self, name):
        return ToolMenus._menus.get(name)

    def refresh_all_widgets(self):
        ToolMenus.refreshed += 1


def get_editor_subsystem(cls):
    return _EDITOR if cls is UnrealEditorSubsystem else _SUBSYSTEM


MISSING_ASSETS = set()


def load_asset(path):
    return None if path in MISSING_ASSETS else path


def register_slate_post_tick_callback(fn):
    handle = len(_TICKS) + 1
    _TICKS[handle] = fn
    return handle


def unregister_slate_post_tick_callback(handle):
    _TICKS.pop(handle, None)


# --- test helpers (not part of the real API) -------------------------------
def _reset():
    UnrealEditorSubsystem.camera = None
    EditorActorSubsystem.selected = []
    ToolMenus._menus.clear()
    ToolMenus._instance = None
    MISSING_ASSETS.clear()
    _TICKS.clear()
    _LEVEL.clear()
    LOG.clear()


def _tick_all(dt=0.05):
    for fn in list(_TICKS.values()):
        fn(dt)
