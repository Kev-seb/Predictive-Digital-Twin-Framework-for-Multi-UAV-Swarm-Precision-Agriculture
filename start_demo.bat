@echo off
rem ===== One-click start for the UE5 digital twin demo =====
rem Set UE_PROJECT once to your .uproject to also open UE5 with the twin script running.
rem Leave it empty to start only the twin (then run twin_live.py from UE5's File menu).
set UE_PROJECT=
rem Optional: set UE_EDITOR if UE5 is not in C:\Program Files\Epic Games
set UE_EDITOR=

cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% scripts\start_demo.py %*
if errorlevel 1 pause
