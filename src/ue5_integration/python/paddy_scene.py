"""
paddy_scene.py - build the paddy-field digital twin level inside the UE5 editor (UE 5.8).

Run in the editor's Python (File > Execute Python Script, or remote execution):
    import paddy_scene; paddy_scene.build_all()
or step by step: step_level(), step_assets(), step_environment(), step_fields(), save().

Everything is saved under /Game/TwinDemo. Engine content is only read, never modified.
twin_paddy.py (the live controller) drives what this builds.

Layout (cm, X = east = grid col, Y = south = grid row, Z = up), matching twin_live.py:
    TWIN      north-west corner (0, 0)
    REALITY   (12000, 0)
    FORECAST  (24000, 0)
Each field is 20 x 20 cells of 5 m (10000 cm square), 20 m apart.
"""
import math
import os
import random

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.normpath(os.path.join(HERE, "..", "generated_meshes"))

ROOT = "/Game/TwinDemo"
MAP = f"{ROOT}/Maps/PaddyTwin"
GRID, CELL = 20, 500.0
FIELD = GRID * CELL
GAP = 2000.0
FIELDS = [("TWIN", 0.0), ("REALITY", FIELD + GAP), ("FORECAST", 2 * (FIELD + GAP))]
WATER_Z = 6.0
BUND_H, BUND_W = 45.0, 140.0
CANAL_Y = -2600.0          # canal centre line (north of the fields)
ROAD_Y = 13200.0           # farm road centre line (south, beyond the drone pads)

mel = unreal.MaterialEditingLibrary
atools = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary


def _actors_sub():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def log(msg):
    unreal.log(f"[paddy_scene] {msg}")


# ------------------------------------------------------------------ level
def step_level():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if eal.does_asset_exist(MAP):
        les.load_level(MAP)
        log("loaded existing level")
    else:
        for d in ("Maps", "Meshes", "Materials", "FX", "UI", "Blueprints"):
            eal.make_directory(f"{ROOT}/{d}")
        les.new_level(MAP)
        log("created level")
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world().get_name()


def save():
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    eal.save_directory(ROOT, only_if_is_dirty=True, recursive=True)
    log("saved")


# ------------------------------------------------------------------ assets: meshes
def set_nanite(mesh, shape="PRESERVE_AREA"):
    ns = mesh.get_editor_property("nanite_settings")
    ns.enabled = True
    ns.shape_preservation = getattr(unreal.NaniteShapePreservation, shape)  # keep thin leaves at a distance
    mesh.set_editor_property("nanite_settings", ns)


def import_obj(name, nanite=False, collision=False):
    dst = f"{ROOT}/Meshes"
    task = unreal.AssetImportTask()
    task.filename = os.path.join(MESH_DIR, name + ".obj")
    task.destination_path = dst
    task.destination_name = name
    task.automated = True
    task.replace_existing = True
    task.save = False
    atools.import_asset_tasks([task])
    mesh = unreal.load_asset(f"{dst}/{name}.{name}")
    if mesh is None:
        # Interchange may name the asset after the file's object; find it
        for p in eal.list_assets(dst, recursive=False):
            if name in p:
                mesh = unreal.load_asset(p)
    if mesh is None:
        raise RuntimeError(f"import failed: {name}")
    if nanite:
        set_nanite(mesh, "VOXELIZE" if name.startswith("SM_Rice") else "PRESERVE_AREA")
    if not collision:
        try:
            unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem).remove_collisions(mesh)
        except Exception:
            pass
    b = mesh.get_bounding_box()
    log(f"{name}: bounds min {b.min} max {b.max}")
    return mesh


# ------------------------------------------------------------------ assets: materials
def _new_material(name):
    path = f"{ROOT}/Materials/{name}"
    if eal.does_asset_exist(path):          # rebuild in place so meshes keep their reference
        m = unreal.load_asset(path)
        mel.delete_all_material_expressions(m)
        return m
    return atools.create_asset(name, f"{ROOT}/Materials", unreal.Material, unreal.MaterialFactoryNew())


def _custom(m, code, inputs, out_type=unreal.CustomMaterialOutputType.CMOT_FLOAT3, x=-400, y=0, desc="Custom"):
    node = mel.create_material_expression(m, unreal.MaterialExpressionCustom, x, y)
    node.set_editor_property("code", code)
    node.set_editor_property("output_type", out_type)
    node.set_editor_property("description", desc)
    ins = []
    for n in inputs:
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", n)
        ins.append(ci)
    node.set_editor_property("inputs", ins)
    return node


def _picd(m, index, x, y, default=0.0):
    n = mel.create_material_expression(m, unreal.MaterialExpressionPerInstanceCustomData, x, y)
    n.set_editor_property("data_index", index)
    n.set_editor_property("const_default_value", default)
    return n


def _scalar(m, v, x, y):
    n = mel.create_material_expression(m, unreal.MaterialExpressionConstant, x, y)
    n.set_editor_property("r", v)
    return n


def _param(m, name, v, x, y):
    n = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("default_value", v)
    return n


def _vparam(m, name, rgb, x, y):
    n = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("default_value", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    return n


def _texcoord(m, x, y):
    return mel.create_material_expression(m, unreal.MaterialExpressionTextureCoordinate, x, y)


def _worldpos(m, x, y):
    return mel.create_material_expression(m, unreal.MaterialExpressionWorldPosition, x, y)


def _link(a, b, inp, out=""):
    if not mel.connect_material_expressions(a, out, b, inp):
        raise RuntimeError(f"could not connect {a.get_name()}.{out} -> {b.get_name()}.{inp}")


def _prop(a, prop, out=""):
    if not mel.connect_material_property(a, out, prop):
        raise RuntimeError(f"could not connect {a.get_name()} -> {prop}")


def _finish(m):
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


RICE_COLOR = r"""
float s = saturate(S * 2.5);
float3 green  = float3(0.045, 0.2, 0.018);
float3 yellow = float3(0.55, 0.45, 0.06);
float3 brown  = float3(0.24, 0.12, 0.035);
float3 c = lerp(lerp(green, yellow, saturate(s * 2.0)), brown, saturate(s * 2.0 - 1.0));
c = lerp(c, float3(0.20, 0.055, 0.025), saturate(F * 1.6));
float lum = dot(c, float3(0.3, 0.59, 0.11));
c = lerp(c, lum.xxx * 0.9 + 0.06, saturate(U * 8.0) * 0.85);
float h = frac(sin(dot(floor(W.xy / 37.0), float2(12.9898, 78.233))) * 43758.5453);
c *= lerp(0.45, 1.15, saturate(V)) * lerp(0.88, 1.08, h);
c = lerp(c, c * float3(1.15, 1.08, 0.7), smoothstep(0.8, 1.0, V) * 0.4);
return c;
"""


def mat_rice():
    m = _new_material("M_RiceCrop")
    m.set_editor_property("two_sided", True)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    m.set_editor_property("used_with_instanced_static_meshes", True)
    m.set_editor_property("used_with_nanite", True)
    tc = _texcoord(m, -1100, 300)
    mask_v = mel.create_material_expression(m, unreal.MaterialExpressionComponentMask, -900, 300)
    mask_v.set_editor_property("g", True); mask_v.set_editor_property("r", False)
    _link(tc, mask_v, "")
    wp = _worldpos(m, -1100, 450)
    col = _custom(m, RICE_COLOR, ["S", "F", "U", "V", "W"], desc="RiceHealthColor")
    _link(_picd(m, 1, -900, -200), col, "S")
    _link(_picd(m, 4, -900, -80), col, "F")
    _link(_picd(m, 2, -900, 40), col, "U")
    _link(mask_v, col, "V")
    _link(wp, col, "W")
    _prop(col, unreal.MaterialProperty.MP_BASE_COLOR)
    sss = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -150, 120)
    _link(col, sss, "A"); _link(_scalar(m, 0.7, -300, 200), sss, "B")
    _prop(sss, unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
    _prop(_scalar(m, 0.55, -300, 280), unreal.MaterialProperty.MP_ROUGHNESS)
    _prop(_scalar(m, 0.35, -300, 340), unreal.MaterialProperty.MP_SPECULAR)
    # wind: engine SimpleGrassWind, weighted by height along the leaf (UV.y)
    fn = unreal.load_asset("/Engine/Functions/Engine_MaterialFunctions01/WorldPositionOffset/SimpleGrassWind")
    call = mel.create_material_expression(m, unreal.MaterialExpressionMaterialFunctionCall, -400, 520)
    call.set_material_function(fn)
    _link(_param(m, "WindIntensity", 0.18, -700, 480), call, "WindIntensity")
    _link(mask_v, call, "WindWeight")
    _link(_param(m, "WindSpeed", 0.5, -700, 600), call, "WindSpeed")
    _link(mel.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -700, 700), call, "AdditionalWPO")
    _prop(call, unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
    return _finish(m)


CANOPY_FN = r"""
float s = saturate(S * 2.5);
float3 green  = float3(0.045, 0.2, 0.018);
float3 yellow = float3(0.55, 0.45, 0.06);
float3 brown  = float3(0.24, 0.12, 0.035);
float3 c = lerp(lerp(green, yellow, saturate(s * 2.0)), brown, saturate(s * 2.0 - 1.0));
c = lerp(c, float3(0.20, 0.055, 0.025), saturate(F * 1.6));
float lum = dot(c, float3(0.3, 0.59, 0.11));
c = lerp(c, lum.xxx * 0.9 + 0.06, saturate(U * 8.0) * 0.85);
"""
# Far canopy: thin leaves are sub-pixel from the overview cameras, so the cell's water/mud tile blends
# toward the crop colour with distance (and NDVI) - same per-cell data as the rice, so it stays live.
WATER_COLOR = r"""
float wet = smoothstep(0.55, 0.88, M);
float h = frac(sin(dot(floor(W.xy / 23.0), float2(12.9898, 78.233))) * 43758.5453);
float h2 = frac(sin(dot(floor(W.xy / 160.0), float2(39.346, 11.135))) * 24634.6345);
float3 mud = float3(0.115, 0.075, 0.042) * lerp(0.75, 1.15, h) * lerp(0.85, 1.1, h2);
float3 crack = float3(0.20, 0.15, 0.10);
mud = lerp(mud, crack, saturate(1.0 - M * 1.6) * 0.6 * h);
float3 water = float3(0.018, 0.028, 0.022);
float3 ground = lerp(mud, water, wet);
""" + CANOPY_FN + r"""
float d = length(W - Cam);
float cover = saturate(N * 1.1) * smoothstep(1200.0, 7000.0, d) * 0.82;
float leafnoise = frac(sin(dot(floor(W.xy / 41.0), float2(7.13, 157.1))) * 1731.7);
return lerp(ground, c * lerp(0.62, 0.9, leafnoise), cover);
"""
WATER_ROUGH = r"""
float wet = smoothstep(0.55, 0.88, M);
float cover = saturate(N * 1.15) * smoothstep(1200.0, 7000.0, length(W - Cam)) * 0.92;
return lerp(lerp(0.9, 0.03, wet), 0.75, cover);
"""
WATER_NORMAL = r"""
float wet = smoothstep(0.55, 0.88, M);
float t = Time * 0.6;
float2 p = W.xy / 90.0;
float nx = sin(p.x * 3.1 + t) * 0.5 + sin(p.y * 2.3 + p.x * 1.7 - t * 1.3) * 0.5;
float ny = cos(p.y * 2.9 - t) * 0.5 + cos(p.x * 2.1 - p.y * 1.4 + t * 1.1) * 0.5;
return normalize(float3(nx * 0.04 * wet, ny * 0.04 * wet, 1.0));
"""


def mat_water():
    """Per-cell water/mud tile. Custom data: 0 moisture, 1 ndvi, 2 stress, 3 stress_std, 4 fungus."""
    m = _new_material("M_PaddyWater")
    m.set_editor_property("used_with_instanced_static_meshes", True)
    moist = _picd(m, 0, -1100, -200, default=0.9)
    ndvi = _picd(m, 1, -1100, -100, default=0.0)
    st = _picd(m, 2, -1100, 0)
    sd = _picd(m, 3, -1100, 100)
    fu = _picd(m, 4, -1100, 200)
    wp = _worldpos(m, -1100, 300)
    cam = mel.create_material_expression(m, unreal.MaterialExpressionCameraPositionWS, -1100, 400)
    tm = mel.create_material_expression(m, unreal.MaterialExpressionTime, -1100, 500)
    c = _custom(m, WATER_COLOR, ["M", "N", "S", "U", "F", "W", "Cam"], desc="WaterMudCanopy", y=-100)
    for node, pin in ((moist, "M"), (ndvi, "N"), (st, "S"), (sd, "U"), (fu, "F"), (wp, "W"), (cam, "Cam")):
        _link(node, c, pin)
    _prop(c, unreal.MaterialProperty.MP_BASE_COLOR)
    r = _custom(m, WATER_ROUGH, ["M", "N", "W", "Cam"], out_type=unreal.CustomMaterialOutputType.CMOT_FLOAT1,
                desc="Roughness", y=100)
    for node, pin in ((moist, "M"), (ndvi, "N"), (wp, "W"), (cam, "Cam")):
        _link(node, r, pin)
    _prop(r, unreal.MaterialProperty.MP_ROUGHNESS)
    n = _custom(m, WATER_NORMAL, ["M", "W", "Time"], desc="Ripples", y=300)
    _link(moist, n, "M"); _link(wp, n, "W"); _link(tm, n, "Time")
    _prop(n, unreal.MaterialProperty.MP_NORMAL)
    _prop(_scalar(m, 0.6, -300, 450), unreal.MaterialProperty.MP_SPECULAR)
    return _finish(m)


SURFACE = r"""
float2 p = W.xy / Cell;
float acc = 0.0, amp = 0.5;
for (int o = 0; o < 4; o++) {
    float2 i = floor(p), f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = frac(sin(dot(i, float2(12.9898, 78.233))) * 43758.5453);
    float b = frac(sin(dot(i + float2(1, 0), float2(12.9898, 78.233))) * 43758.5453);
    float c = frac(sin(dot(i + float2(0, 1), float2(12.9898, 78.233))) * 43758.5453);
    float d = frac(sin(dot(i + float2(1, 1), float2(12.9898, 78.233))) * 43758.5453);
    acc += amp * lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
    p *= 2.13; amp *= 0.5;
}
float n = acc / 0.9375;
float3 col = C;
if (Patch > 0.0) {
    float2 cell = floor((W.xy + float2(0.0, 1000.0)) / float2(9000.0, 6000.0));
    float k = frac(sin(dot(cell, float2(27.17, 91.31))) * 15731.743);
    float3 tints[5] = { float3(1.0, 1.0, 1.0), float3(1.35, 1.2, 0.55), float3(0.75, 0.95, 0.7),
                        float3(1.9, 1.15, 0.75), float3(0.85, 1.15, 0.85) };
    col = lerp(C, C * tints[(int)(k * 4.999)], Patch);
}
return col * lerp(1.0 - Var, 1.0 + Var, n);
"""


def mat_surface():
    """Generic surface: Color * procedural variation. Instances: ground, bund mud, road, boards, trunk."""
    m = _new_material("M_Surface")
    m.set_editor_property("two_sided", True)
    m.set_editor_property("used_with_instanced_static_meshes", True)
    c = _custom(m, SURFACE, ["C", "W", "Cell", "Var", "Patch"], desc="Variation")
    _link(_param(m, "Patchwork", 0.0, -900, 350), c, "Patch")
    _link(_vparam(m, "Color", (0.1, 0.2, 0.05), -900, -100), c, "C")
    _link(_worldpos(m, -900, 50), c, "W")
    _link(_param(m, "NoiseCell", 60.0, -900, 150), c, "Cell")
    _link(_param(m, "Variation", 0.15, -900, 250), c, "Var")
    _prop(c, unreal.MaterialProperty.MP_BASE_COLOR)
    _prop(_param(m, "Roughness", 0.85, -400, 250), unreal.MaterialProperty.MP_ROUGHNESS)
    _prop(_param(m, "Specular", 0.3, -400, 350), unreal.MaterialProperty.MP_SPECULAR)
    _prop(_param(m, "Metallic", 0.0, -400, 450), unreal.MaterialProperty.MP_METALLIC)
    em = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -200, 550)
    _link(_vparam(m, "Emissive", (0, 0, 0), -400, 550), em, "A")
    _link(_param(m, "EmissiveStrength", 0.0, -400, 650), em, "B")
    _prop(em, unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    return _finish(m)


DRONE = r"""
float u = UV.x;
float3 body = Body;
float3 dark = float3(0.025, 0.026, 0.03);
float3 tank = float3(0.75, 0.78, 0.8);
float3 accent = float3(0.9, 0.35, 0.03);
return u < 0.2 ? body : (u < 0.45 ? dark : (u < 0.7 ? tank : accent));
"""
FOLIAGE = r"""
float u = UV.x;
float h = frac(sin(dot(floor(W.xy / 300.0), float2(12.9898, 78.233))) * 43758.5453);
float3 leaf = lerp(float3(0.05, 0.16, 0.025), float3(0.12, 0.2, 0.03), h) * lerp(0.6, 1.1, UV.y);
float3 trunk = float3(0.16, 0.12, 0.08);
return u > 0.4 ? trunk : leaf;
"""


def mat_banded(name, code, extra=None, foliage=False):
    m = _new_material(name)
    m.set_editor_property("two_sided", True)
    m.set_editor_property("used_with_instanced_static_meshes", True)
    if foliage:
        m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    ins = ["UV", "W"] + ([extra[0]] if extra else [])
    c = _custom(m, code, ins, desc=name)
    _link(_texcoord(m, -900, 0), c, "UV")
    _link(_worldpos(m, -900, 100), c, "W")
    if extra:
        _link(_vparam(m, extra[1], extra[2], -900, 200), c, extra[0])
    _prop(c, unreal.MaterialProperty.MP_BASE_COLOR)
    if foliage:
        sss = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -150, 120)
        _link(c, sss, "A"); _link(_scalar(m, 0.6, -300, 200), sss, "B")
        _prop(sss, unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
        fn = unreal.load_asset("/Engine/Functions/Engine_MaterialFunctions01/WorldPositionOffset/SimpleGrassWind")
        call = mel.create_material_expression(m, unreal.MaterialExpressionMaterialFunctionCall, -400, 520)
        call.set_material_function(fn)
        _link(_scalar(m, 0.06, -700, 480), call, "WindIntensity")
        mk = mel.create_material_expression(m, unreal.MaterialExpressionComponentMask, -700, 560)
        mk.set_editor_property("g", True); mk.set_editor_property("r", False)
        _link(_texcoord(m, -900, 560), mk, "")
        _link(mk, call, "WindWeight")
        _link(_scalar(m, 0.3, -700, 620), call, "WindSpeed")
        _link(mel.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -700, 700), call, "AdditionalWPO")
        _prop(call, unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
        _prop(_scalar(m, 0.6, -300, 300), unreal.MaterialProperty.MP_ROUGHNESS)
    else:
        _prop(_scalar(m, 0.35, -300, 300), unreal.MaterialProperty.MP_ROUGHNESS)
        _prop(_scalar(m, 0.5, -300, 360), unreal.MaterialProperty.MP_SPECULAR)
    return _finish(m)


def mi(name, parent, vectors=None, scalars=None):
    path = f"{ROOT}/Materials/{name}"
    if eal.does_asset_exist(path):
        inst = unreal.load_asset(path)
    else:
        inst = atools.create_asset(name, f"{ROOT}/Materials", unreal.MaterialInstanceConstant,
                                   unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, parent)
    for k, v in (vectors or {}).items():
        mel.set_material_instance_vector_parameter_value(inst, k, unreal.LinearColor(v[0], v[1], v[2], 1.0))
    for k, v in (scalars or {}).items():
        mel.set_material_instance_scalar_parameter_value(inst, k, v)
    mel.update_material_instance(inst)
    eal.save_loaded_asset(inst)
    return inst


MESHES = [("SM_RiceCell", True), ("SM_RiceCellLite", True), ("SM_WaterTile", False), ("SM_DroneBody", False), ("SM_Rotor", False),
          ("SM_RainStreak", False),
          ("SM_Palm", True), ("SM_Shrub", True)]


def step_assets():
    for name, nanite in MESHES:
        import_obj(name, nanite=nanite)
    return step_materials()


def step_materials():
    meshes = {name: unreal.load_asset(f"{ROOT}/Meshes/{name}.{name}") for name, _ in MESHES if name != "SM_RainStreak"}
    rice = mat_rice()
    water = mat_water()
    surf = mat_surface()
    drone = mat_banded("M_Drone", DRONE, extra=("Body", "Color", (0.8, 0.82, 0.85)))
    foliage = mat_banded("M_Foliage", FOLIAGE, foliage=True)
    mi("MI_Ground", surf, {"Color": (0.06, 0.13, 0.025)}, {"NoiseCell": 80.0, "Variation": 0.25, "Roughness": 0.9,
                                                            "Patchwork": 0.85})
    mi("MI_BundMud", surf, {"Color": (0.12, 0.08, 0.045)}, {"NoiseCell": 25.0, "Variation": 0.2, "Roughness": 0.95})
    mi("MI_BundGrass", surf, {"Color": (0.07, 0.12, 0.03)}, {"NoiseCell": 30.0, "Variation": 0.3, "Roughness": 0.9})
    mi("MI_Road", surf, {"Color": (0.22, 0.18, 0.13)}, {"NoiseCell": 15.0, "Variation": 0.2, "Roughness": 0.95})
    mi("MI_CanalWater", surf, {"Color": (0.02, 0.035, 0.03)}, {"NoiseCell": 200.0, "Variation": 0.05,
                                                                 "Roughness": 0.05, "Specular": 0.7})
    mi("MI_Concrete", surf, {"Color": (0.32, 0.31, 0.29)}, {"NoiseCell": 20.0, "Variation": 0.1, "Roughness": 0.8})
    mi("MI_Board", surf, {"Color": (0.012, 0.014, 0.018)}, {"Variation": 0.0, "Roughness": 0.6})
    mi("MI_Cursor", surf, {"Color": (0.9, 0.35, 0.02), "Emissive": (1.0, 0.4, 0.02)},
       {"Variation": 0.0, "EmissiveStrength": 4.0})
    mi("MI_Pad", surf, {"Color": (0.35, 0.33, 0.3), "Emissive": (1.0, 0.55, 0.05)},
       {"Variation": 0.05, "EmissiveStrength": 0.0})
    for name, mat in (("SM_RiceCell", rice), ("SM_RiceCellLite", rice), ("SM_WaterTile", water), ("SM_DroneBody", drone),
                      ("SM_Rotor", drone), ("SM_Palm", foliage), ("SM_Shrub", foliage)):
        meshes[name].set_material(0, mat)
        eal.save_loaded_asset(meshes[name])
    log("assets ready")
    return {k: str(v.get_path_name()) for k, v in meshes.items()}


# ------------------------------------------------------------------ actors
def _tag(actor, folder):
    actor.set_folder_path(folder)
    actor.tags = list(actor.tags) + [unreal.Name("PaddyTwin")]


def _spawn(cls, loc, label, folder, rot=None):
    a = _actors_sub().spawn_actor_from_class(cls, loc, rot or unreal.Rotator(0, 0, 0))
    a.set_actor_label(label)
    _tag(a, folder)
    return a


def _mesh(mesh_path, loc, scale, label, folder, material=None, rot=None, shadows=True):
    a = _spawn(unreal.StaticMeshActor, loc, label, folder, rot)
    c = a.static_mesh_component
    c.set_static_mesh(unreal.load_asset(mesh_path))
    a.set_actor_scale3d(scale)
    if material:
        c.set_material(0, unreal.load_asset(material))
    c.set_cast_shadow(shadows)
    return a


def clear(prefix_folder=None):
    """Remove actors built by this script (tag PaddyTwin), optionally only in one outliner folder."""
    n = 0
    for a in _actors_sub().get_all_level_actors():
        if unreal.Name("PaddyTwin") in a.tags:
            if prefix_folder and not str(a.get_folder_path()).startswith(prefix_folder):
                continue
            a.destroy_actor(); n += 1
    log(f"cleared {n} actors")


CUBE = "/Engine/BasicShapes/Cube.Cube"
PLANE = "/Engine/BasicShapes/Plane.Plane"
MAT = f"{ROOT}/Materials"


def step_environment():
    clear("Environment")
    cx = FIELDS[-1][1] / 2 + FIELD / 2
    # ---- sky and light: bright tropical morning, sun low in the east
    sun = _spawn(unreal.DirectionalLight, unreal.Vector(cx, 0, 3000), "Sun", "Environment/Lighting",
                 unreal.Rotator(roll=0, pitch=-32.0, yaw=200.0))
    lc = sun.light_component
    lc.set_editor_property("intensity", 9.0)
    lc.set_editor_property("light_color", unreal.Color(r=255, g=244, b=228, a=255))
    lc.set_editor_property("atmosphere_sun_light", True)
    lc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    try:
        lc.set_editor_property("dynamic_shadow_distance_movable_light", 40000.0)
    except Exception:
        pass
    _spawn(unreal.SkyAtmosphere, unreal.Vector(cx, 0, 0), "SkyAtmosphere", "Environment/Lighting")
    sky = _spawn(unreal.SkyLight, unreal.Vector(cx, 0, 2000), "SkyLight", "Environment/Lighting")
    sc = sky.light_component
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)
    sc.set_editor_property("intensity", 1.0)
    cl = _spawn(unreal.VolumetricCloud, unreal.Vector(cx, 0, 0), "VolumetricCloud", "Environment/Lighting")
    try:
        cc = cl.get_component_by_class(unreal.VolumetricCloudComponent)
        cc.set_editor_property("layer_bottom_altitude", 2.5)
        cc.set_editor_property("layer_height", 6.0)
    except Exception as e:
        log(f"cloud settings skipped: {e}")
    fog = _spawn(unreal.ExponentialHeightFog, unreal.Vector(cx, 0, -200), "HeightFog", "Environment/Lighting")
    fc = fog.component
    fc.set_editor_property("fog_density", 0.007)
    fc.set_editor_property("fog_height_falloff", 0.25)
    fc.set_editor_property("start_distance", 8000.0)
    fc.set_editor_property("fog_inscattering_luminance" if hasattr(fc, "fog_inscattering_luminance") else "fog_inscattering_color",
                           unreal.LinearColor(0.45, 0.6, 0.8, 1.0))
    ppv = _spawn(unreal.PostProcessVolume, unreal.Vector(cx, 0, 0), "PostProcess", "Environment/Lighting")
    ppv.set_editor_property("unbound", True)
    s = ppv.settings
    s.set_editor_property("override_auto_exposure_method", True)
    s.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    s.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    s.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    s.set_editor_property("override_auto_exposure_bias", True)
    s.set_editor_property("auto_exposure_bias", -0.6)
    s.set_editor_property("override_bloom_intensity", True)
    s.set_editor_property("bloom_intensity", 0.25)
    s.set_editor_property("override_vignette_intensity", True)
    s.set_editor_property("vignette_intensity", 0.25)
    ppv.set_editor_property("settings", s)

    # ---- ground: farmland around the fields
    g = _mesh(PLANE, unreal.Vector(cx, 5000, -2.0), unreal.Vector(6000, 6000, 1), "Ground", "Environment/Ground",
              f"{MAT}/MI_Ground")
    # ---- farm road south of the drone pads, and the irrigation canal north of the fields
    span_x = cx * 2 + 30000
    _mesh(CUBE, unreal.Vector(cx, ROAD_Y, 4), unreal.Vector(span_x / 100, 5.0, 0.1), "FarmRoad",
          "Environment/Ground", f"{MAT}/MI_Road")
    for s_ in (-1, 1):
        _mesh(CUBE, unreal.Vector(cx, ROAD_Y + s_ * 300, 2), unreal.Vector(span_x / 100, 1.0, 0.06),
              f"RoadVerge{s_}", "Environment/Ground", f"{MAT}/MI_BundMud")
    _mesh(CUBE, unreal.Vector(cx, CANAL_Y, -60), unreal.Vector(span_x / 100, 6.0, 1.0), "CanalBed",
          "Environment/Canal", f"{MAT}/MI_BundMud")
    _mesh(CUBE, unreal.Vector(cx, CANAL_Y, -18), unreal.Vector(span_x / 100, 5.6, 0.02), "CanalWater",
          "Environment/Canal", f"{MAT}/MI_CanalWater")
    for s_ in (-1, 1):
        _mesh(CUBE, unreal.Vector(cx, CANAL_Y + s_ * 330, 0), unreal.Vector(span_x / 100, 0.8, 0.5),
              f"CanalWall{s_}", "Environment/Canal", f"{MAT}/MI_Concrete")
    # feeder channels from the canal to each field (water inlets)
    for name, ox in FIELDS:
        _mesh(CUBE, unreal.Vector(ox + FIELD / 2, (CANAL_Y + 330) / 2, 0), unreal.Vector(0.8, abs(CANAL_Y + 330) / 100, 0.3),
              f"Inlet_{name}", "Environment/Canal", f"{MAT}/MI_Concrete")

    # ---- palms and shrubs along the canal and road, plus a tree line far north
    rnd = random.Random(42)
    trees = []
    x = -6000.0
    while x < span_x - 8000:
        trees.append(("SM_Palm", x + rnd.uniform(-500, 500), CANAL_Y - 900 - rnd.uniform(0, 1400), rnd.uniform(0.85, 1.25)))
        x += rnd.uniform(1400, 2600)
    x = -5000.0
    while x < span_x - 8000:
        if rnd.random() < 0.6:
            trees.append(("SM_Palm", x, ROAD_Y + 900 + rnd.uniform(0, 1200), rnd.uniform(0.8, 1.15)))
        trees.append(("SM_Shrub", x + 700, ROAD_Y + 600 + rnd.uniform(0, 600), rnd.uniform(0.8, 1.6)))
        x += rnd.uniform(1800, 3200)
    for k in range(40):     # tree line far behind the canal
        trees.append(("SM_Shrub", -8000 + k * 1600 + rnd.uniform(-300, 300), CANAL_Y - 4500 - rnd.uniform(0, 2500),
                      rnd.uniform(2.5, 4.5)))
    for side in (-1, 1):    # west and east edges
        for k in range(8):
            xx = (-4500 if side < 0 else FIELDS[-1][1] + FIELD + 4500) + rnd.uniform(-1500, 1500)
            trees.append(("SM_Palm", xx, -1500 + k * 1900 + rnd.uniform(-400, 400), rnd.uniform(0.85, 1.2)))
    for i, (mesh, tx, ty, s_) in enumerate(trees):
        _mesh(f"{ROOT}/Meshes/{mesh}.{mesh}", unreal.Vector(tx, ty, 0), unreal.Vector(s_, s_, s_), f"{mesh[3:]}_{i:03d}",
              "Environment/Trees", rot=unreal.Rotator(0, 0, rnd.uniform(0, 360)))
    log(f"environment: {len(trees)} trees")


def overview_camera():
    cx = FIELDS[-1][1] / 2 + FIELD / 2
    target = unreal.Vector(cx, FIELD / 2 - 2500, 0)
    cam = unreal.Vector(cx, FIELD / 2 + 25000, 9500)
    d = target - cam
    rot = unreal.Rotator(roll=0, pitch=math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))),
                         yaw=math.degrees(math.atan2(d.y, d.x)))
    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(cam, rot)


def set_camera(loc, look_at):
    d = look_at - loc
    rot = unreal.Rotator(roll=0, pitch=math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))),
                         yaw=math.degrees(math.atan2(d.y, d.x)))
    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(loc, rot)


# ------------------------------------------------------------------ fields
_SDS = None


def _add_comp(actor, cls, name, parent=None):
    """Add a component to a placed actor (saved with the level)."""
    global _SDS
    _SDS = _SDS or unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    lib = unreal.SubobjectDataBlueprintFunctionLibrary
    handles = _SDS.k2_gather_subobject_data_for_instance(actor)
    parent_h = handles[0]
    if parent is not None:
        for h in handles:
            if lib.get_associated_object(lib.get_data(h)) == parent:
                parent_h = h
    h, fail = _SDS.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=parent_h, new_class=cls,
                                                                  blueprint_context=None))
    if str(fail):
        raise RuntimeError(f"add component {name}: {fail}")
    _SDS.rename_subobject(h, unreal.Text(name))
    return lib.get_associated_object(lib.get_data(h))


def _hash(r, c, k=0):
    return (math.sin(r * 127.1 + c * 311.7 + k * 74.7) * 43758.5453) % 1.0


def _hism(actor, name, mesh, material, floats, shadows=True):
    comp = _add_comp(actor, unreal.HierarchicalInstancedStaticMeshComponent, name)
    comp.set_static_mesh(unreal.load_asset(mesh))
    if material:
        comp.set_material(0, unreal.load_asset(material))
    comp.set_editor_property("num_custom_data_floats", floats)
    comp.set_cast_shadow(shadows)
    comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    return comp


def build_field(label, ox, oy=0.0, live=True, health=None):
    mesh = "SM_RiceCell" if live else "SM_RiceCellLite"
    """One 100 m paddy: bunds, per-cell water tiles, 20 x 20 rice cells (instance = row * 20 + col).

    The actor sits at the field's north-west corner. Components:
      Rice  HISM, 400 instances, custom data 0-5 = ndvi, stress, stress_std, moisture, fungus, anomaly
      Water HISM, 400 instances, custom data 0 = moisture
    """
    a = _spawn(unreal.Actor, unreal.Vector(ox, oy, 0), f"Field_{label}", "Fields")
    scene = _add_comp(a, unreal.SceneComponent, "Root")
    rice = _hism(a, "Rice", f"{ROOT}/Meshes/{mesh}.{mesh}", None, 6)
    water = _hism(a, "Water", f"{ROOT}/Meshes/SM_WaterTile.SM_WaterTile", None, 5, shadows=False)
    rice_t, water_t = [], []
    for r in range(GRID):
        for c in range(GRID):
            x, y = (c + 0.5) * CELL, (r + 0.5) * CELL
            yaw = 90.0 * int(_hash(r, c) * 4)
            h = 0.75 if health is None else health(r, c)[0]
            rice_t.append(unreal.Transform(unreal.Vector(x, y, 0.0), unreal.Rotator(0, 0, yaw), unreal.Vector(1, 1, h)))
            water_t.append(unreal.Transform(unreal.Vector(x, y, WATER_Z), unreal.Rotator(0, 0, 0), unreal.Vector(1, 1, 1)))
    rice.add_instances(rice_t, False, False)
    water.add_instances(water_t, False, False)
    for i in range(GRID * GRID):
        r, c = divmod(i, GRID)
        if health is None:
            vals = [0.75, 0.05, 0.0, 0.9, 0.0, 0.0]
        else:
            vals = health(r, c)
        last = i == GRID * GRID - 1
        for k in range(5):
            rice.set_custom_data_value(i, k, vals[k], last and k == 4)
        for k, v in enumerate((vals[3], vals[0], vals[1], vals[2], vals[4])):
            water.set_custom_data_value(i, k, v, last and k == 4)
    # bunds (levees) around the field
    t = BUND_W
    for nm, cx_, cy_, sx, sy in (("N", FIELD / 2, -t / 2, FIELD + 2 * t, t), ("S", FIELD / 2, FIELD + t / 2, FIELD + 2 * t, t),
                                 ("W", -t / 2, FIELD / 2, t, FIELD), ("E", FIELD + t / 2, FIELD / 2, t, FIELD)):
        b = _mesh(CUBE, unreal.Vector(ox + cx_, oy + cy_, BUND_H / 2 - 5), unreal.Vector(sx / 100, sy / 100, BUND_H / 100),
                  f"Bund_{label}_{nm}", "Fields/Bunds", f"{MAT}/MI_BundMud")
        g = _mesh(CUBE, unreal.Vector(ox + cx_, oy + cy_, BUND_H - 4), unreal.Vector(sx / 100 * 0.98, sy / 100 * 0.6, 0.03),
                  f"BundGrass_{label}_{nm}", "Fields/Bunds", f"{MAT}/MI_BundGrass")
    return a


LABEL_COLORS = {"TWIN": (120, 200, 255), "REALITY": (255, 196, 110), "FORECAST": (205, 165, 255)}
LABEL_TITLE = {"TWIN": "TWIN", "REALITY": "REALITY", "FORECAST": "FORECAST +7 DAYS"}
LABEL_SUB = {"TWIN": "what the twin believes now", "REALITY": "hidden ground truth",
             "FORECAST": "predicted, if nothing is done"}


def build_label(label, ox):
    """Dark board on posts behind the field's north bund, text facing south (towards the cameras)."""
    w, h = 7600.0, 1250.0
    cx, y, z = ox + FIELD / 2, -1100.0, 1500.0
    board = _mesh(CUBE, unreal.Vector(cx, y, z), unreal.Vector(w / 100, 0.3, h / 100), f"LabelBoard_{label}",
                  "Fields/Labels", f"{MAT}/MI_Board")
    for px in (-w * 0.4, w * 0.4):
        _mesh(CUBE, unreal.Vector(cx + px, y + 10, (z - h / 2) / 2), unreal.Vector(0.25, 0.25, (z - h / 2) / 100),
              f"LabelPost_{label}_{int(px)}", "Fields/Labels", f"{MAT}/MI_Concrete")
    t = _spawn(unreal.TextRenderActor, unreal.Vector(cx, y + 20, z + 160), f"Label_{label}", "Fields/Labels",
               unreal.Rotator(0, 0, 90))
    tr = t.text_render
    tr.set_horizontal_alignment(unreal.HorizTextAligment.EHTA_CENTER)
    tr.set_vertical_alignment(unreal.VerticalTextAligment.EVRTA_TEXT_CENTER)
    tr.set_world_size(720.0)
    tr.set_text_render_color(unreal.Color(r=LABEL_COLORS[label][0], g=LABEL_COLORS[label][1], b=LABEL_COLORS[label][2], a=255))
    tr.set_text(LABEL_TITLE[label])
    sub = _spawn(unreal.TextRenderActor, unreal.Vector(cx, y + 20, z - 430), f"LabelSub_{label}", "Fields/Labels",
                 unreal.Rotator(0, 0, 90))
    st = sub.text_render
    st.set_horizontal_alignment(unreal.HorizTextAligment.EHTA_CENTER)
    st.set_vertical_alignment(unreal.VerticalTextAligment.EVRTA_TEXT_CENTER)
    st.set_world_size(300.0)
    st.set_text_render_color(unreal.Color(r=225, g=228, b=232, a=255))
    st.set_text(LABEL_SUB[label])
    return t


def build_pads():
    """Three drone launch pads just south of the TWIN field (row ~21, as in the twin's mission planner)."""
    for i, col in enumerate((8.4, 10.0, 11.6)):
        _mesh(CUBE, unreal.Vector(col * CELL, 21.2 * CELL, 5), unreal.Vector(3.2, 3.2, 0.12), f"DronePad_{i + 1}",
              "Fields/Pads", f"{MAT}/MI_Pad")
    _mesh(CUBE, unreal.Vector(10 * CELL, 21.2 * CELL, 1), unreal.Vector(16, 6, 0.05), "PadApron", "Fields/Pads",
          f"{MAT}/MI_Concrete")
    # track from the road to the apron
    _mesh(CUBE, unreal.Vector(10 * CELL, (21.2 * CELL + ROAD_Y) / 2, 2), unreal.Vector(3.5, (ROAD_Y - 21.2 * CELL) / 100, 0.05),
          "PadTrack", "Fields/Pads", f"{MAT}/MI_Road")


def _neighbour_health(seed):
    rnd = random.Random(seed)
    stage = rnd.uniform(0.55, 0.95)
    def f(r, c):
        n = _hash(r + seed, c - seed, 3)
        return [stage * (0.95 + 0.06 * n), 0.01 + 0.02 * n, 0.0, 0.9, 0.0, 0.0]
    return f


_EAST = FIELDS[-1][1] + FIELD + 6000.0
NEIGHBOURS = [(-FIELD - 6000.0, 0.0), (_EAST, 0.0),
              (-FIELD - 6000.0, ROAD_Y + 2600), (-FIELD / 2 + 1000, ROAD_Y + 2600), (FIELD / 2 + GAP + 1000, ROAD_Y + 2600),
              (1.5 * FIELD + 2 * GAP + 1000, ROAD_Y + 2600), (_EAST, ROAD_Y + 2600)]


def step_fields(neighbours=True):
    clear("Fields")
    for label, ox in FIELDS:
        build_field(label, ox)
        build_label(label, ox)
    build_pads()
    if neighbours:     # ordinary farmers' paddies around the twin, for context (not driven by the API)
        for k, (ox, oy) in enumerate(NEIGHBOURS):
            build_field(f"Neighbour{k + 1}", ox, oy, live=False, health=_neighbour_health(k + 5))
    log("fields built")


# ------------------------------------------------------------------ editor panel asset
EUW = f"{ROOT}/UI/EUW_TwinControl"


def create_euw():
    """Empty Editor Utility Widget; twin_ui.py (run outside UE, via the MCP UMG toolset) fills it."""
    try:
        sub = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
        if eal.does_asset_exist(EUW):
            bp = unreal.load_asset(EUW)
            w = sub.find_utility_widget_from_blueprint(bp)
            if w is not None:
                sub.close_tab_by_id(sub.find_utility_widget_from_blueprint(bp).get_name())
    except Exception:
        pass
    if eal.does_asset_exist(EUW):
        eal.delete_asset(EUW)
    f = unreal.EditorUtilityWidgetBlueprintFactory()
    f.set_editor_property("parent_class", unreal.EditorUtilityWidget)
    a = atools.create_asset("EUW_TwinControl", f"{ROOT}/UI", unreal.EditorUtilityWidgetBlueprint, f)
    log(f"created {a.get_path_name()}")
    return a.get_path_name()


def build_all():
    """Everything that runs inside UE. Then, outside UE: paddy_fx.py and twin_ui.py (MCP), and save()."""
    step_level()
    step_assets()
    step_environment()
    step_fields()
    build_cursor()
    build_rain()
    create_euw()
    save()
    overview_camera()


def build_cursor():
    """CellCursor: drag it onto a cell of TWIN or REALITY, then 'Inject outbreak' hits that cell."""
    for a in _actors_sub().get_all_level_actors():
        if a.get_actor_label() == "CellCursor":
            a.destroy_actor()
    c = _mesh("/Engine/BasicShapes/Cylinder.Cylinder", unreal.Vector(FIELD + GAP + 4.5 * CELL, 14.5 * CELL, 60.0),
              unreal.Vector(4.6, 4.6, 0.04), "CellCursor", "Fields", f"{MAT}/MI_Cursor", shadows=False)
    return c


def mat_cursor():
    mi("MI_Cursor", unreal.load_asset(f"{ROOT}/Materials/M_Surface"), {"Color": (0.9, 0.35, 0.02), "Emissive": (1.0, 0.4, 0.02)},
       {"Variation": 0.0, "EmissiveStrength": 4.0})


# ------------------------------------------------------------------ rain (GPU-animated streaks)
RAIN_TOP = 3600.0
RAIN_FALL = r"""
float t = frac(Time * 0.75 + R * 7.31);
return float3(t * 260.0, t * 90.0, -t * Top);
"""


def mat_rain():
    """Falling streaks: every instance loops from RAIN_TOP to the ground, phase from PerInstanceRandom."""
    m = _new_material("M_RainStreak")
    m.set_editor_property("two_sided", True)
    m.set_editor_property("used_with_instanced_static_meshes", True)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    _prop(_vparam(m, "Color", (0.7, 0.76, 0.85), -400, 0), unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    f = _custom(m, RAIN_FALL, ["Time", "R", "Top"], desc="Fall", y=300)
    _link(mel.create_material_expression(m, unreal.MaterialExpressionTime, -900, 250), f, "Time")
    _link(mel.create_material_expression(m, unreal.MaterialExpressionPerInstanceRandom, -900, 350), f, "R")
    _link(_scalar(m, RAIN_TOP, -900, 450), f, "Top")
    _prop(f, unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
    return _finish(m)


def build_rain(per_field=9000):
    """Actor 'RainStreaks' (hidden until twin_paddy turns rain on) over the three fields."""
    for a in _actors_sub().get_all_level_actors():
        if a.get_actor_label() == "RainStreaks":
            a.destroy_actor()
    mesh = unreal.load_asset(f"{ROOT}/Meshes/SM_RainStreak")
    mesh.set_material(0, mat_rain())
    a = _spawn(unreal.Actor, unreal.Vector(0, 0, 0), "RainStreaks", "Environment")
    _add_comp(a, unreal.SceneComponent, "Root")
    h = _hism(a, "Streaks", f"{ROOT}/Meshes/SM_RainStreak.SM_RainStreak", None, 0, shadows=False)
    h.set_editor_property("bounds_scale", 40.0)      # WPO moves streaks far from their rest position
    rnd = random.Random(17)
    ts = []
    for _, ox in FIELDS:
        for _ in range(per_field):
            ts.append(unreal.Transform(unreal.Vector(ox - 600 + rnd.random() * (FIELD + 1200),
                                                     -600 + rnd.random() * (FIELD + 1200), RAIN_TOP),
                                       unreal.Rotator(0, 0, 0), unreal.Vector(1, 1, rnd.uniform(0.8, 1.3))))
    h.add_instances(ts, False, False)
    a.set_is_temporarily_hidden_in_editor(True)
    log(f"rain streaks: {len(ts)}")
    return a
