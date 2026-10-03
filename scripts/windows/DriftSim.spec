# PyInstaller spec for the DriftSim executable (one folder: dist/DriftSim/DriftSim.exe + _internal/).
#
#     pyinstaller --noconfirm scripts/windows/DriftSim.spec            (run from the repository root)
#
# Set DRIFTSIM_WITH_TORCH=1 to bundle PyTorch for GPU data generation (adds about 300 MB for the
# CPU build and 2-3 GB for a CUDA build of torch; whatever torch is installed gets bundled).
# The same spec also builds a macOS / Linux binary, which is how the bundle is tested off Windows.
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
WITH_TORCH = os.environ.get("DRIFTSIM_WITH_TORCH", "0") == "1"
ICON = os.path.join(ROOT, "build", "DriftSim.ico")

datas = [
    (os.path.join(ROOT, "rc_drift_sim", "app", "static"), os.path.join("rc_drift_sim", "app", "static")),
    (os.path.join(ROOT, "rc_drift_sim", "configs"), os.path.join("rc_drift_sim", "configs")),
    (os.path.join(ROOT, "examples", "specs"), os.path.join("examples", "specs")),
]
# not used by the GUI or the exporter (plotting, the RL environment, development tools)
excludes = ["matplotlib", "jax", "jaxlib", "gymnasium", "pygame", "tkinter", "IPython", "pytest",
            "rc_drift_sim.viz", "rc_drift_sim.rl"]
if not WITH_TORCH:
    excludes.append("torch")

a = Analysis(
    [os.path.join(SPECPATH, "driftsim_app.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=["rc_drift_sim.app.server", "rc_drift_sim.datagen.runner", "rc_drift_sim.sim.xp", "yaml"],
    excludes=excludes,
    noarchive=False,
    # keep the .py sources too: the variable catalog reads the units and descriptions from the
    # comments in sim/params.py
    module_collection_mode={"rc_drift_sim": "pyz+py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DriftSim",
    console=True,              # the window shows the server log; closing it stops the server
    icon=ICON if os.path.exists(ICON) else None,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DriftSim")
