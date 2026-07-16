"""Desktop user interface for the lamination peel comparator."""

from __future__ import annotations

__all__ = ["MainWindow", "run_app"]


def __getattr__(name: str):
    if name in __all__:
        from .main_window import MainWindow, run_app

        return {"MainWindow": MainWindow, "run_app": run_app}[name]
    raise AttributeError(name)

