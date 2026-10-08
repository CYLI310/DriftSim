@echo off
rem Start the DriftSim GUI from the Python environment made by setup_windows.ps1 (training, export, BeamNG tests).
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\driftsim-gui.exe" (
  echo Run scripts\windows\setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\driftsim-gui.exe" %*
pause
