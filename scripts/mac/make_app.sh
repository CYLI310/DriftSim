#!/bin/bash
# Build DriftSim.app: double-click it to start the dataset GUI server in the background and open it
# in the browser; double-click it again to open or stop the server.
#
#   scripts/mac/make_app.sh                  builds DriftSim.app in the repository folder
#   scripts/mac/make_app.sh ~/Applications   builds it there instead
#
# The app runs scripts/driftsim-gui.sh from this checkout, so you can drag it to the Dock or copy it
# anywhere; rebuild it if you move the repository.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEST="${1:-$REPO}"
APP="$DEST/DriftSim.app"
RES="$APP/Contents/Resources"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"
if [ -e "$APP" ]; then
    # only replace an app this script built
    [ -f "$RES/repo_path" ] || { echo "$APP exists and was not built by this script; not replacing it." >&2; exit 1; }
    rm -rf "$APP"
fi

osacompile -o "$APP" "$HERE/DriftSim.applescript"
printf '%s\n' "$REPO" > "$RES/repo_path"

# icon (skipped if no Python with Pillow is around; the app then has the default script icon)
PY=""
for p in "$REPO/.venv/bin/python" "$REPO/../.venv/bin/python" "$(command -v python3 2>/dev/null)"; do
    if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import PIL, numpy" >/dev/null 2>&1; then PY="$p"; break; fi
done
if [ -n "$PY" ] && "$PY" "$HERE/make_icon.py" "$TMP/icon.png"; then
    mkdir "$TMP/DriftSim.iconset"
    for size in 16 32 128 256 512; do
        sips -z $size $size "$TMP/icon.png" --out "$TMP/DriftSim.iconset/icon_${size}x${size}.png" >/dev/null
        sips -z $((size * 2)) $((size * 2)) "$TMP/icon.png" --out "$TMP/DriftSim.iconset/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns -o "$RES/applet.icns" "$TMP/DriftSim.iconset"
    rm -f "$RES/Assets.car"                                   # it would take precedence over applet.icns
    /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$APP/Contents/Info.plist" 2>/dev/null || true
else
    echo "note: no Python with Pillow and NumPy found, keeping the default icon"
fi

plist() { /usr/libexec/PlistBuddy -c "Set :$1 $2" "$APP/Contents/Info.plist" 2>/dev/null ||
          /usr/libexec/PlistBuddy -c "Add :$1 string $2" "$APP/Contents/Info.plist"; }
plist CFBundleIdentifier local.driftsim.gui
plist CFBundleName DriftSim

codesign --force --sign - "$APP" 2>/dev/null     # re-seal after the edits (ad-hoc, local use)
touch "$APP"                                      # make Finder pick up the new icon
echo "built $APP"
