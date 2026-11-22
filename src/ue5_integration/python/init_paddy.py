"""
init_paddy.py - editor startup script (Project Settings > Plugins > Python > Startup Scripts).

Adds this folder to sys.path and a "Digital Twin > Start paddy twin" menu entry. Nothing heavy runs
at startup: the controller starts when you click the entry (it needs the PaddyTwin level loaded).
"""
import os
import sys

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _register():
    menus = unreal.ToolMenus.get()
    main = menus.find_menu("LevelEditor.MainMenu")
    menu = menus.find_menu("LevelEditor.MainMenu.DigitalTwin")
    if menu is None:
        menu = main.add_sub_menu("TwinLive", "", "DigitalTwin", "Digital Twin", "Digital twin demo controls")
    for name, label, cmd in (
            ("PaddyStart", "Start paddy twin (opens the control panel)", "import twin_paddy; twin_paddy.start()"),
            ("PaddyStop", "Stop paddy twin", "import twin_paddy; twin_paddy.stop()"),
            ("PaddyOpenLevel", "Open the PaddyTwin level",
             "import unreal; unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level("
             "'/Game/TwinDemo/Maps/PaddyTwin')")):
        entry = unreal.ToolMenuEntry(name=name, type=unreal.MultiBlockType.MENU_ENTRY)
        entry.set_label(label)
        entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", cmd)
        menu.add_menu_entry("Paddy", entry)
    menus.refresh_all_widgets()


try:
    _register()
except Exception as e:      # never block editor startup
    unreal.log_warning(f"[init_paddy] could not add the Digital Twin menu: {e}")
