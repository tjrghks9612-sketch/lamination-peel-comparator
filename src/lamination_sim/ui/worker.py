"""Background worker for the CPU-bound comparison run."""

from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .core_bridge import get_value, run_project, sequence


class SimulationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str)

    def __init__(self, project: Any) -> None:
        super().__init__()
        self.project = project

    @Slot()
    def run(self) -> None:
        sweep = get_value(self.project, "tension_sweep", default={})
        preload_count = max(1, len(sequence(get_value(sweep, "preload_levels", default=[]))))
        # Tape stiffness is a fixed PET estimate; only preload is swept.
        combinations = preload_count
        material_enabled = bool(get_value(self.project, "run_uncertainty", default=False))
        nested = bool(get_value(sweep, "nest_material_uncertainty", default=False))
        material_count = int(
            get_value(
                get_value(self.project, "assumptions", default={}),
                "uncertainty_samples",
                default=1,
            )
        )
        runs = combinations * 2
        if material_enabled and nested:
            runs = combinations * max(material_count, 1) * 2
        elif material_enabled:
            runs += max(material_count - 1, 0) * 2
        self.progress.emit(
            f"장력 {combinations}조합 × A/B를 계산하고 있습니다… 예상 {runs}회 해석"
        )
        try:
            bundle = run_project(self.project)
        except Exception as exc:  # Worker must return errors to the UI thread.
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.finished.emit(bundle)
