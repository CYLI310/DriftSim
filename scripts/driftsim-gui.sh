#!/bin/bash
# Start, stop or find the DriftSim dataset GUI (the web server behind driftsim-gui).
#
#   scripts/driftsim-gui.sh               run here in the terminal; Ctrl+C stops it
#   scripts/driftsim-gui.sh --background  start in the background, open the browser, return
#   scripts/driftsim-gui.sh --stop        stop the server
#   scripts/driftsim-gui.sh --status      print the address if it is running (exit 1 if not)
#
# If a DriftSim server is already running, starting just opens it in the browser again.
# Background logs: ~/Library/Logs/DriftSim/server.log. Set DRIFTSIM_NO_OPEN=1 to skip the browser.
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"   # apps and Finder start with a bare PATH

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$HOME/Library/Application Support/DriftSim"
LOG_DIR="$HOME/Library/Logs/DriftSim"
PIDFILE="$STATE_DIR/server.pid"
URLFILE="$STATE_DIR/server.url"
LOG="$LOG_DIR/server.log"
PORTS="$(seq 8765 8784)"            # the server takes the first free port from 8765 up

# A Python that can import the package: the project's virtualenv first.
find_python() {
    local p
    for p in "$REPO/.venv/bin/python" "$REPO/../.venv/bin/python" \
             "$(command -v python3.11 2>/dev/null)" "$(command -v python3 2>/dev/null)"; do
        if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import rc_drift_sim.app" >/dev/null 2>&1; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

is_driftsim() {   # $1 = base URL; true if a DriftSim GUI answers there
    curl -fsS -m 1 "${1}api/catalog" 2>/dev/null | grep -q '"default_spec"'
}

running_url() {   # print the URL of a live DriftSim server
    local url port
    if [ -f "$URLFILE" ]; then
        url="$(cat "$URLFILE")"
        if is_driftsim "$url"; then echo "$url"; return 0; fi
    fi
    for port in $PORTS; do
        url="http://127.0.0.1:$port/"
        if is_driftsim "$url"; then echo "$url"; return 0; fi
    done
    return 1
}

server_pid() {    # PID of the server listening behind $1 (our pidfile first, then lsof)
    local pid port
    if [ -f "$PIDFILE" ]; then
        pid="$(cat "$PIDFILE")"
        if kill -0 "$pid" 2>/dev/null && ps -o command= -p "$pid" | grep -q "rc_drift_sim"; then echo "$pid"; return 0; fi
    fi
    port="$(echo "$1" | sed -E 's#.*:([0-9]+)/?$#\1#')"
    pid="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)"
    if [ -n "$pid" ] && ps -o command= -p "$pid" | grep -q "rc_drift_sim"; then echo "$pid"; return 0; fi
    return 1
}

open_url() { [ -n "${DRIFTSIM_NO_OPEN:-}" ] || open "$1" 2>/dev/null || true; }

cmd="${1:-}"
case "$cmd" in
  --status)
    running_url
    exit $?
    ;;

  --stop)
    url="$(running_url)" || { echo "DriftSim is not running."; rm -f "$PIDFILE" "$URLFILE"; exit 0; }
    pid="$(server_pid "$url")" || { echo "DriftSim is running at $url but its process was not found." >&2; exit 1; }
    kill -INT "$pid" 2>/dev/null          # same as Ctrl+C: cancels running jobs, keeps written shards
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    rm -f "$PIDFILE" "$URLFILE"
    echo "DriftSim stopped."
    ;;

  --background)
    if url="$(running_url)"; then open_url "$url"; echo "$url"; exit 0; fi
    PY="$(find_python)" || { echo "No Python with rc_drift_sim installed. Run: pip install -e \"$REPO[dev]\"" >&2; exit 2; }
    mkdir -p "$STATE_DIR" "$LOG_DIR"
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') starting with $PY" >> "$LOG"
    start_line=$(wc -l < "$LOG")
    cd "$REPO" || exit 2
    nohup "$PY" -u -m rc_drift_sim.app --no-browser >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PIDFILE"
    url=""
    for _ in $(seq 1 80); do              # wait up to 20 s for "DriftSim dataset GUI: <url>"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "DriftSim stopped while starting. Last log lines:" >&2
            tail -n 15 "$LOG" >&2
            rm -f "$PIDFILE"
            exit 3
        fi
        url="$(tail -n +"$((start_line + 1))" "$LOG" | grep -o 'http://127.0.0.1:[0-9]*/' | head -1)"
        if [ -n "$url" ] && is_driftsim "$url"; then break; fi
        url=""
        sleep 0.25
    done
    if [ -z "$url" ]; then echo "DriftSim did not answer within 20 s; see $LOG" >&2; exit 4; fi
    echo "$url" > "$URLFILE"
    open_url "$url"
    echo "$url"
    ;;

  "")
    if url="$(running_url)"; then
        echo "DriftSim is already running at $url (opening it)."
        open_url "$url"
        exit 0
    fi
    PY="$(find_python)" || { echo "No Python with rc_drift_sim installed. Run: pip install -e \"$REPO[dev]\"" >&2; exit 2; }
    cd "$REPO" || exit 2
    exec "$PY" -m rc_drift_sim.app ${DRIFTSIM_NO_OPEN:+--no-browser}
    ;;

  *)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
    ;;
esac
