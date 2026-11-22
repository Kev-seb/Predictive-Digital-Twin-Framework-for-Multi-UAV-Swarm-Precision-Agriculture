"""
twin_ui.py - build the EUW_TwinControl editor panel through the Unreal MCP server (UE 5.8 UMGToolSet).

Run from a normal terminal while the editor is open (MCP server on :8000), after creating the empty
asset /Game/TwinDemo/UI/EUW_TwinControl (paddy_scene.create_euw() does that):
    python twin_ui.py

The panel has no Blueprint logic: twin_paddy.py binds every button's OnClicked to Python and writes the
readouts every 0.25 s. Widget names (Btn_*, Txt_*) are the contract between the two files.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/mcp"
BP_PATH = "/Game/TwinDemo/UI/EUW_TwinControl.EUW_TwinControl"
UMG = "UMGToolSet.UMGToolSet"
OBJ = "editor_toolset.toolsets.object.ObjectTools"
_sid = None

BG = (0.012, 0.014, 0.018, 1.0)
CARD = (0.03, 0.035, 0.045, 1.0)
TEXT = (0.86, 0.88, 0.9, 1.0)
DIM = (0.5, 0.54, 0.58, 1.0)
ACCENT = (0.35, 0.72, 1.0, 1.0)
GREEN = (0.32, 0.85, 0.42, 1.0)


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
    with urllib.request.urlopen(req, timeout=120) as r:
        _sid = _sid or r.headers.get("Mcp-Session-Id")
        txt = r.read().decode("utf-8", "replace")
    if notify or not txt.strip():
        return None
    if txt.lstrip().startswith(("event:", "data:")):
        txt = [l[5:].strip() for l in txt.splitlines() if l.startswith("data:")][-1]
    return json.loads(txt)


def tool(toolset, name, args):
    r = rpc("tools/call", {"name": "call_tool", "arguments": {"toolset_name": toolset, "tool_name": name,
                                                             "arguments": args}})
    res = r.get("result", {})
    text = "".join(c.get("text", "") for c in res.get("content", []))
    if res.get("isError"):
        raise RuntimeError(f"{name}: {text[:500]}")
    return json.loads(text)["returnValue"] if text.startswith("{") else text


BP = {"refPath": BP_PATH}


def add(cls, name, parent=None, var=False):
    info = tool(UMG, "AddWidget", {"widgetBlueprint": BP, "widgetClass": {"refPath": f"/Script/UMG.{cls}"},
                                   "widgetDisplayName": name, "parentWidget": parent, "childIndex": -1})
    if var:
        tool(UMG, "ToggleWidgetAsVariable", {"widgetBlueprint": BP, "widget": info["widget"], "bIsVariable": True})
    return info


def setp(obj, values):
    if obj in (None, "None"):
        return
    try:
        ok = tool(OBJ, "set_properties", {"instance": obj, "values": json.dumps(values)})
        if ok is False:
            print(f"  ! set {list(values)} on {obj['refPath'].split('.')[-1]} returned False")
    except RuntimeError as e:
        print(f"  ! set {list(values)} on {obj['refPath'].split('.')[-1]}: {str(e)[:160]}")


def color(c):
    return {"r": c[0], "g": c[1], "b": c[2], "a": c[3]}


def slate(c):
    return {"specifiedColor": color(c), "colorUseRule": "UseColor_Specified"}


def pad(l, t, r, b):
    return {"left": l, "top": t, "right": r, "bottom": b}


def text(parent, name, value, size=10, col=TEXT, bold=False, var=False, wrap=False, slot_pad=(0, 2, 0, 2)):
    w = add("TextBlock", name, parent, var)
    v = {"Text": value, "ColorAndOpacity": slate(col),
         "Font": {"fontObject": {"refPath": "/Engine/EngineFonts/Roboto.Roboto"},
                  "typefaceFontName": "Bold" if bold else "Regular", "size": size}}
    if wrap:
        v["AutoWrapText"] = True
    setp(w["widget"], v)
    setp(w["slot"], {"Padding": pad(*slot_pad)})
    return w


def button(parent, name, label, tint=(0.09, 0.11, 0.14, 1.0), fill=False):
    b = add("Button", name, parent, var=True)
    setp(b["widget"], {"BackgroundColor": color(tint)})
    sp = {"Padding": pad(2, 2, 2, 2)}
    if fill:
        sp["Size"] = {"sizeRule": "Fill", "value": 1.0}
    setp(b["slot"], sp)
    t = text(b["widget"], f"{name}_Label", label, size=10, bold=True, slot_pad=(6, 3, 6, 3))
    return b


def header(parent, name, label):
    text(parent, name, label, size=9, col=DIM, bold=True, slot_pad=(0, 12, 0, 4))


def readout(parent, name, label, var_name):
    row = add("HorizontalBox", f"Row_{name}", parent)
    lab = add("SizeBox", f"LabBox_{name}", row["widget"])
    setp(lab["widget"], {"bOverride_WidthOverride": True, "WidthOverride": 175.0})
    text(lab["widget"], f"Lab_{name}", label, size=10, col=DIM)
    v = text(row["widget"], var_name, "-", size=10, col=TEXT, bold=True, var=True)
    setp(v["slot"], {"Size": {"sizeRule": "Fill", "value": 1.0}, "Padding": pad(0, 2, 0, 2)})


def legend_row(parent, i, label, c):
    row = add("HorizontalBox", f"Legend_{i}", parent)
    box = add("SizeBox", f"LegendBox_{i}", row["widget"])
    setp(box["widget"], {"bOverride_WidthOverride": True, "WidthOverride": 16.0,
                         "bOverride_HeightOverride": True, "HeightOverride": 16.0})
    setp(box["slot"], {"Padding": pad(0, 2, 8, 2), "VerticalAlignment": "VAlign_Center"})
    sw = add("Border", f"LegendSwatch_{i}", box["widget"])
    setp(sw["widget"], {"BrushColor": color(c)})
    text(row["widget"], f"LegendText_{i}", label, size=10)


def build():
    root = add("Border", "Root")
    setp(root["widget"], {"BrushColor": color(BG), "Padding": pad(12, 10, 12, 10)})
    scroll = add("ScrollBox", "Scroll", root["widget"])
    main = add("VerticalBox", "Main", scroll["widget"])
    m = main["widget"]
    text(m, "Title", "PADDY DIGITAL TWIN", size=15, col=ACCENT, bold=True, slot_pad=(0, 0, 0, 2))
    text(m, "Txt_Day", "Day -  |  -", size=12, bold=True, var=True)
    status_card = add("Border", "StatusCard", m)
    setp(status_card["widget"], {"BrushColor": color(CARD), "Padding": pad(8, 6, 8, 6)})
    setp(status_card["slot"], {"Padding": pad(0, 6, 0, 0)})
    text(status_card["widget"], "Txt_Status", "Connecting to the twin API...", size=10, wrap=True, var=True)

    header(m, "H_Sim", "SIMULATION")
    g1 = add("HorizontalBox", "RowRun", m)
    button(g1["widget"], "Btn_Start", "Start demo", (0.05, 0.25, 0.1, 1.0), fill=True)
    button(g1["widget"], "Btn_Stop", "Stop", (0.28, 0.07, 0.06, 1.0), fill=True)
    button(g1["widget"], "Btn_Reset", "Reset", fill=True)
    g2 = add("HorizontalBox", "RowStep", m)
    button(g2["widget"], "Btn_NextDay", "Next day", fill=True)
    button(g2["widget"], "Btn_Scout", "Scout", fill=True)
    button(g2["widget"], "Btn_Spray", "Spray", fill=True)
    g3 = add("HorizontalBox", "RowMore", m)
    button(g3["widget"], "Btn_Auto", "Scout + Spray", (0.05, 0.16, 0.3, 1.0), fill=True)
    button(g3["widget"], "Btn_Outbreak", "Inject outbreak", (0.3, 0.16, 0.03, 1.0), fill=True)
    text(m, "Hint_Outbreak", "Outbreak cell: select a field cell marker or CellCursor, else the centre of the view.",
         size=8, col=DIM, wrap=True)

    header(m, "H_Cam", "CAMERA")
    g4 = add("HorizontalBox", "RowCam", m)
    button(g4["widget"], "Btn_CamOverview", "Overview", fill=True)
    button(g4["widget"], "Btn_CamClose", "Close-up", fill=True)
    button(g4["widget"], "Btn_CamFollow", "Follow drone", fill=True)
    text(m, "Txt_View", "View: Overview", size=9, col=DIM, var=True)

    header(m, "H_Res", "RESULTS")
    res = add("Border", "ResultsCard", m)
    setp(res["widget"], {"BrushColor": color(CARD), "Padding": pad(8, 6, 8, 6)})
    rv = add("VerticalBox", "Results", res["widget"])
    r = rv["widget"]
    readout(r, "HarvestNow", "Expected harvest now", "Txt_HarvestNow")
    readout(r, "H7No", "In 7 days, no spray", "Txt_Harvest7NoSpray")
    readout(r, "H7Sp", "In 7 days, if sprayed", "Txt_Harvest7Spray")
    readout(r, "Disease", "Worst disease in 7 days", "Txt_Disease")
    readout(r, "Fung", "Fungicide used", "Txt_Fungicide")
    readout(r, "Err", "Twin vs reality error", "Txt_Error")
    readout(r, "Miss", "Missions", "Txt_Missions")

    header(m, "H_Leg", "LEGEND  (leaf colour per 5 m cell; height = NDVI)")
    for i, (label, c) in enumerate([("Healthy", (0.16, 0.5, 0.1, 1)), ("Stressed (water / nutrients)", (0.78, 0.7, 0.2, 1)),
                                    ("Severe stress", (0.52, 0.33, 0.12, 1)), ("Fungal disease", (0.47, 0.15, 0.07, 1)),
                                    ("Twin uncertain (greyed)", (0.55, 0.55, 0.55, 1)),
                                    ("Water: glassy = flooded, matte mud = dry", (0.1, 0.14, 0.13, 1))]):
        legend_row(m, i, label, c)
    text(m, "Footer", "Fields, left to right: TWIN (what the twin believes) | REALITY (hidden truth) | "
                      "FORECAST +7 DAYS (if nothing is done).", size=8, col=DIM, wrap=True, slot_pad=(0, 10, 0, 0))
    ok = tool(UMG, "CompileWidgetBlueprint", {"widgetBlueprint": BP})
    print("compile:", ok)


def main():
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "twin_ui", "version": "1"}})
    rpc("notifications/initialized", notify=True)
    tree = tool(UMG, "GetWidgets", {"widgetBlueprint": BP})
    if json.dumps(tree).count("refPath") > 2 and "--force" not in sys.argv:
        print("EUW_TwinControl already has widgets; recreate the empty asset first or pass --force")
    build()


if __name__ == "__main__":
    main()
