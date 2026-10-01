"""Local web GUI for generating datasets: ``python -m rc_drift_sim.app`` (or ``driftsim-gui``).

Standard library only. The server (``server.py``) serves the single-page app in ``static/`` and a
JSON API around ``rc_drift_sim.datagen``.
"""
from .server import GuiServer, make_server

__all__ = ["GuiServer", "make_server"]
