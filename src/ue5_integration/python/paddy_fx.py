"""
paddy_fx.py - author the Niagara effects for the paddy twin through the Unreal MCP server
(UE 5.8 NiagaraToolsets; Python cannot edit Niagara module inputs directly).

Run from a normal terminal while the editor is open with the MCP server on :8000:
    python paddy_fx.py

  NS_SprayMist  fine downward mist under a spraying drone (fountain emitter, cone pointed down)
  NS_Rain       rain streaks falling over one 100 m field (box spawn, velocity-aligned streaks)
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/mcp"
NS = "NiagaraToolsets.NiagaraToolset_System"
FX = "/Game/TwinDemo/FX"
_sid = None


def rpc(method, params=None, notify=False):
    global _sid
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        body["id"] = 1
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if _sid:
        h.update({"Mcp-Session-Id": _sid, "Mcp-Protocol-Version": "2025-06-18"})
    req = urllib.request.Request(URL, json.dumps(body).encode(), h, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        _sid = _sid or r.headers.get("Mcp-Session-Id")
        txt = r.read().decode("utf-8", "replace")
    if notify or not txt.strip():
        return None
    if txt.lstrip().startswith(("event:", "data:")):
        txt = [l[5:].strip() for l in txt.splitlines() if l.startswith("data:")][-1]
    return json.loads(txt)


def tool(name, args, toolset=NS):
    r = rpc("tools/call", {"name": "call_tool", "arguments": {"toolset_name": toolset, "tool_name": name,
                                                             "arguments": args}})
    res = r.get("result", {})
    text = "".join(c.get("text", "") for c in res.get("content", []))
    if res.get("isError"):
        raise RuntimeError(f"{name}: {text[:600]}")
    return json.loads(text) if text.startswith("{") else text


def ref(system, module="", inp=None, script="ParticleSpawnScript", emitter="Fountain", renderer=-1):
    return {"system": {"refPath": system}, "emitterName": emitter, "scriptName": script if module else "",
            "moduleName": module, "rendererIndex": renderer, "inputNameStack": [inp] if inp else []}


def F(v):
    return {"struct": {"refPath": "/Script/Niagara.NiagaraFloat"}, "value": {"value": v}}


def B(v):
    return {"struct": {"refPath": "/Script/Niagara.NiagaraBool"}, "value": {"value": -1 if v else 0}}


def V3(x, y, z):
    return {"struct": {"refPath": "/Script/CoreUObject.Vector3f"}, "value": {"x": x, "y": y, "z": z}}


def V2(x, y):
    return {"struct": {"refPath": "/Script/CoreUObject.Vector2f"}, "value": {"x": x, "y": y}}


def C(r, g, b, a):
    return {"struct": {"refPath": "/Script/CoreUObject.LinearColor"}, "value": {"r": r, "g": g, "b": b, "a": a}}


def E(enum_path, name):
    return {"struct": {"refPath": "/Script/NiagaraEditor.NiagaraExt_StackInputData_Enum"},
            "value": {"enum": {"refPath": enum_path}, "enumName": f"{enum_path.split('.')[-1]}::{name}"}}


SHAPES = "/Niagara/Enums/Location/ENiagara_LocationShapes.ENiagara_LocationShapes"
SIZE_MODE = "/Niagara/Enums/ENiagara_SizeScaleMode.ENiagara_SizeScaleMode"


def setin(system, module, inp, value, script="ParticleSpawnScript"):
    try:
        tool("SetStackInputData", {"stackInputRef": ref(system, module, inp, script), "inputData": value})
        print(f"  {module}.{inp} ok")
    except RuntimeError as e:
        print(f"  {module}.{inp} FAILED: {str(e)[:300]}")


def new_system(name):
    path = f"{FX}/{name}.{name}"
    try:
        tool("CreateNiagaraSystem", {"assetName": name, "assetPath": FX,
                                     "templateSystem": {"refPath": "/Niagara/DefaultAssets/DefaultSystem.DefaultSystem"}})
    except RuntimeError as e:
        print(f"  (create {name}: {str(e)[:120]}; editing the existing asset)")
    summary = tool("GetSystemSummary", {"system": {"refPath": path}})["returnValue"]
    for em in summary["emitters"]:
        if em["emitterName"] != "Fountain":
            tool("RemoveEmitter", {"emitterToRemove": ref(path, emitter=em["emitterName"])})
    return path


def spray():
    s = new_system("NS_SprayMist")
    print(s)
    setin(s, "SpawnRate", "SpawnRate", F(1100.0), "EmitterUpdateScript")
    setin(s, "InitializeParticle", "Lifetime Min", F(0.7))
    setin(s, "InitializeParticle", "Lifetime Max", F(1.15))
    setin(s, "InitializeParticle", "Color", C(0.85, 0.92, 0.98, 0.3))
    setin(s, "InitializeParticle", "Uniform Sprite Size Min", F(12.0))
    setin(s, "InitializeParticle", "Uniform Sprite Size Max", F(30.0))
    setin(s, "ShapeLocation", "Sphere Radius", F(95.0))
    setin(s, "AddVelocity", "Cone Axis", V3(0.0, 0.0, -1.0))
    setin(s, "AddVelocity", "Cone Angle", F(14.0))
    setin(s, "AddVelocity", "Velocity Speed", F(330.0))
    setin(s, "Drag", "Drag", F(1.6), "ParticleUpdateScript")
    setin(s, "GravityForce", "Gravity", V3(0.0, 0.0, -420.0), "ParticleUpdateScript")
    renderer(s, {"bCastShadows": False})


def rain():
    s = new_system("NS_Rain")
    print(s)
    # CPU sim: in 5.8 this template's velocity/gravity did not move GPU particles (they hung at spawn height).
    # Fixed bounds keep the field-sized emitter from being culled.
    emitter(s, {"SimTarget": "CPUSim", "CalculateBoundsMode": "Fixed",
                "FixedBounds": {"min": {"x": -5600, "y": -5600, "z": -3800},
                                "max": {"x": 5600, "y": 5600, "z": 400}, "isValid": True}})
    setin(s, "SpawnRate", "SpawnRate", F(6000.0), "EmitterUpdateScript")
    setin(s, "InitializeParticle", "Lifetime Min", F(1.3))
    setin(s, "InitializeParticle", "Lifetime Max", F(1.5))
    setin(s, "InitializeParticle", "Color", C(0.7, 0.78, 0.88, 0.45))
    setin(s, "InitializeParticle", "Sprite Size Mode", E(SIZE_MODE, "NewEnumerator4"))       # Non-Uniform
    setin(s, "InitializeParticle", "Sprite Size", V2(6.0, 160.0))
    setin(s, "ShapeLocation", "Shape Primitive", E(SHAPES, "NewEnumerator2"))               # Box / Plane
    setin(s, "ShapeLocation", "Box Size", V3(10000.0, 10000.0, 200.0))
    setin(s, "AddVelocity", "Cone Axis", V3(0.08, 0.03, -1.0))
    setin(s, "AddVelocity", "Cone Angle", F(2.0))
    setin(s, "AddVelocity", "Velocity Speed", F(2400.0))
    setin(s, "Drag", "Drag", F(0.0), "ParticleUpdateScript")
    renderer(s, {"Alignment": "VelocityAligned", "bCastShadows": False})   # long, thin streaks


def emitter(system, changes):
    """Edit emitter properties (sim target, bounds); GetEmitterData returns them as a JSON string."""
    try:
        ed = tool("GetEmitterData", {"emitterRef": ref(system)})["returnValue"]
        props = json.loads(ed["propertyValues"])
        props.update(changes)
        ed["propertyValues"] = json.dumps(props)
        tool("SetEmitterData", {"emitter": ref(system), "emitterData": ed})
        print(f"  emitter {list(changes)} ok")
    except RuntimeError as e:
        print(f"  emitter FAILED: {str(e)[:400]}")


def renderer(system, changes):
    """Edit sprite-renderer properties (GetRendererData returns them as a JSON string)."""
    try:
        rd = tool("GetRendererData", {"rendererRef": ref(system, renderer=0)})["returnValue"]
        props = json.loads(rd["propertyValues"])
        props.update(changes)
        rd["propertyValues"] = json.dumps(props)
        tool("SetRendererData", {"renderer": ref(system, renderer=0), "rendererData": rd})
        print(f"  renderer {changes} ok")
    except RuntimeError as e:
        print(f"  renderer FAILED: {str(e)[:400]}")


def main():
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "paddy_fx", "version": "1"}})
    rpc("notifications/initialized", notify=True)
    which = sys.argv[1:] or ["spray"]
    if "spray" in which:
        spray()
    if "rain" in which:
        rain()
    for n in ("NS_SprayMist",):
        try:
            st = tool("GetSystemCompileState", {"system": {"refPath": f"{FX}/{n}.{n}"}})
            print(n, "compile:", json.dumps(st)[:300])
        except RuntimeError as e:
            print(n, "compile state:", str(e)[:200])


if __name__ == "__main__":
    main()
