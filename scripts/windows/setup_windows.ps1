# Set up DriftSim on a Windows PC for training (NVIDIA GPU if present) and BeamNG tests.
#   Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
# Creates .venv in the DriftSim folder, installs PyTorch (CUDA build if an NVIDIA GPU is found),
# DriftSim with ONNX export and beamngpy, then checks that BeamNG can be found.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

function Find-Python {
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try { $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $exe) { return $exe } } catch {}
    }
    try { $exe = & python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12); print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0) { return $exe } } catch {}
    return $null
}
$py = Find-Python
if (-not $py) { Write-Host "Python 3.10, 3.11 or 3.12 is needed: install it from python.org (tick 'Add to PATH') and run this again." -ForegroundColor Red; exit 1 }
Write-Host "Python: $py"

if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py -m venv .venv }
$vpy = Join-Path $root ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "NVIDIA GPU found: installing the CUDA build of PyTorch"
    & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "No NVIDIA GPU found: installing the CPU build of PyTorch (training on the CPU is slow; train on the Jetson instead)"
    & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $vpy -m pip install -e ".[export,beamng]"

& $vpy -c @"
import torch, beamngpy
from rc_drift_sim.deploy.beamng import find_beamng_home
print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('beamngpy', beamngpy.__version__)
home = find_beamng_home()
print('BeamNG found at:', home or 'not found - give its folder in the GUI or with --beamng-home')
"@

Write-Host ""
Write-Host "Done. Start the GUI with 'scripts\windows\DriftSim GUI (Python).bat' (RL runs > your run > Export final model > Test in BeamNG)." -ForegroundColor Green
Write-Host "beamngpy must match your BeamNG version (BeamNG 0.39 -> beamngpy 1.36, 0.38 -> 1.35.1, ...):  .venv\Scripts\pip install beamngpy==<version>"
Write-Host "Command line:  .venv\Scripts\driftsim-drive --model path\to\final_model.zip --car beamng --seconds 20"
