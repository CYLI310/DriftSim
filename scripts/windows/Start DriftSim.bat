@echo off
rem Start the DriftSim dataset GUI from the source checkout (no build needed).
rem Double-click this file. The first run creates .venv and installs the package (a few minutes).
rem Close this window (or press Ctrl+C) to stop the server; datasets go to the repository's exports folder.
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv with Python 3.11 ...
    py -3.11 -m venv .venv || (echo Python 3.11 is required: install it from python.org & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e . || (pause & exit /b 1)
)
".venv\Scripts\python.exe" -m rc_drift_sim.app %*
if errorlevel 1 pause
