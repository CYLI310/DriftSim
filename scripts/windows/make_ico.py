"""Write the Windows icon (.ico with 16-256 px images) from the macOS app icon drawing.

    python scripts/windows/make_ico.py build/DriftSim.ico      (needs Pillow)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/DriftSim.ico")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "icon.png"
        subprocess.run([sys.executable, str(HERE.parent / "mac" / "make_icon.py"), str(png)], check=True)
        Image.open(png).convert("RGBA").save(out, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
