"""Background worker for the CPU-bound comparison run."""

from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .core_bridge import run_project


class SimulationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str)

    def __init__(self, project: Any) -> None:
        super().__init__()
        self.project = project

    @Slot()
    def run(self) -> None:
        self.progress.emit("명목 조건과 민감도 시나리오를 계산하고 있습니다…")
        try:
            bundle = run_project(self.project)
        except Exception as exc:  # Worker must return errors to the UI thread.
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.finished.emit(bundle)

