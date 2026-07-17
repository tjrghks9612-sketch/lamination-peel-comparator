from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from lamination_sim.models import AssumptionSet
from lamination_sim.presets import measured_project
from lamination_sim.simulation import simulate
from lamination_sim.ui.visualization import (
    PeelView,
    _interpolate_at_time,
    _time_bracket,
)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_time_interpolation_uses_physical_time_not_normalized_index() -> None:
    times = [0.0, 1.0, 4.0]

    assert _time_bracket(times, 2.5) == pytest.approx((1, 2, 0.5))
    assert _interpolate_at_time(times, [0.0, 10.0, 40.0], 2.5) == pytest.approx(25.0)


def test_peel_view_uses_solver_position_history(qt_app: QApplication) -> None:
    condition = measured_project().condition_b
    result = simulate(condition, AssumptionSet(), "coarse")
    view = PeelView("B", "#ff9955")
    view.set_data(condition, result)
    index = len(result.time_s) // 2

    view.set_time(result.time_s[index], 0.25)

    assert view._result_position()[:3] == pytest.approx(result.position_xyz_mm[index])


def test_off_panel_grip_is_not_clamped_to_panel_edge(qt_app: QApplication) -> None:
    condition = measured_project().condition_b
    result = simulate(condition, AssumptionSet(), "coarse")
    view = PeelView("B", "#ff9955")
    view.set_data(condition, result)
    panel_rect = QRectF(40.0, 40.0, 100.0, 200.0)

    point = view._map_extended_xy(
        result.position_xyz_mm[-1][0],
        result.position_xyz_mm[-1][1],
        panel_rect,
        condition.panel.width_mm,
        condition.panel.height_mm,
    )

    assert point.y() < panel_rect.top()
    assert point.y() >= panel_rect.top() - 28.0 - 1.0e-12
