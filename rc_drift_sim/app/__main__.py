"""Start the dataset GUI: ``python -m rc_drift_sim.app [--port 8765] [--out DIR] [--no-browser]``.

Also installed as the ``driftsim-gui`` command. Stop it with Ctrl+C (running jobs are cancelled;
shards already written are kept).
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser

from .server import make_server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="driftsim-gui", description="Web GUI for generating RC drift datasets.")
    ap.add_argument("--port", type=int, default=8765, help="port (the next free one is used if taken)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to bind (default 127.0.0.1: only this computer can connect)")
    ap.add_argument("--out", default=None, help="export root folder (default: <repo>/exports)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    # Background launches inherit "ignore SIGINT"; restore Ctrl+C and treat SIGTERM and a closed
    # terminal window (SIGHUP, unless nohup ignores it) the same way, so every way of stopping goes
    # through the clean shutdown below.
    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, _stop)
    httpd = make_server(args.host, args.port, args.out, verbose=args.verbose)
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"warning: listening on {args.host}; anyone who can reach this address can start jobs")
    print(f"DriftSim dataset GUI: {httpd.url}  (exports go to {httpd.root})")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(httpd.url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        httpd.jobs.cancel_all()
        if not httpd.jobs.wait_idle(timeout=15.0):
            print("a job did not stop within 15 s; exiting anyway")
        httpd.server_close()
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
