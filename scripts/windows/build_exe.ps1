# Build the DriftSim executable on Windows: dist\DriftSim\DriftSim.exe (plus dist\DriftSim.zip).
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_exe.ps1             CPU only (about 60 MB)
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_exe.ps1 -WithTorch  bundle PyTorch (GPU)
#
# Needs Python 3.11 (python.org installer, "py" launcher). The script makes a build virtualenv in
# .venv-build, installs the package and PyInstaller, draws the icon, builds, and runs the self-test.
# For -WithTorch on an NVIDIA PC, install the CUDA build of torch into .venv-build first
# (https://pytorch.org, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128);
# otherwise the CPU build of torch is bundled.
param([switch]$WithTorch)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$Venv = Join-Path $Root ".venv-build"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "creating $Venv"
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { $global:LASTEXITCODE = 1 }
    if ($LASTEXITCODE -ne 0) { python -m venv $Venv }     # e.g. a CI runner with python 3.11 on PATH
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 not found: install it from python.org (with the py launcher)" }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install -e . pyinstaller pillow
if ($WithTorch) {
    & $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    if ($LASTEXITCODE -ne 0) { & $Py -m pip install torch }
    $env:DRIFTSIM_WITH_TORCH = "1"
} else {
    $env:DRIFTSIM_WITH_TORCH = "0"
}

& $Py scripts\windows\make_ico.py build\DriftSim.ico
& $Py -m PyInstaller --noconfirm --distpath dist --workpath build\pyinstaller scripts\windows\DriftSim.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

& dist\DriftSim\DriftSim.exe --self-test
if ($LASTEXITCODE -ne 0) { throw "self-test of the built executable failed" }

Compress-Archive -Path dist\DriftSim -DestinationPath dist\DriftSim.zip -Force
Write-Host ""
Write-Host "built dist\DriftSim\DriftSim.exe (zip: dist\DriftSim.zip)"
