from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication
from pydantic import ValidationError

from lamination_sim.models import PanelConfig
from lamination_sim.presets import measured_project
from lamination_sim.ui.condition_editor import ConditionEditor
from lamination_sim.ui.visualization import _trim_display_geometry


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_legacy_panel_payload_gets_pretrim_geometry_defaults() -> None:
    panel = PanelConfig.model_validate(
        {
            "preset": "pro",
            "width_mm": 71.5,
            "height_mm": 149.6,
            "thickness_mm": 0.056,
            "corner_radius_mm": 0.0,
        }
    )

    assert panel.trim_geometry.pretrim_margin_mm == pytest.approx(1.5)
    assert panel.trim_geometry.cell_corner_radius_mm == pytest.approx(6.0)
    assert panel.trim_geometry.pad_height_mm == pytest.approx(3.0)


def test_trim_geometry_must_fit_inside_pretrim_rectangle() -> None:
    with pytest.raises(ValidationError):
        PanelConfig.model_validate(
            {
                "width_mm": 20.0,
                "height_mm": 20.0,
                "trim_geometry": {
                    "pretrim_margin_mm": 9.0,
                    "cell_corner_radius_mm": 1.0,
                    "pad_height_mm": 3.0,
                    "island_width_mm": 1.0,
                    "island_height_mm": 1.0,
                },
            }
        )


def test_small_custom_panel_scales_internal_island_instead_of_rejecting_input() -> None:
    panel = PanelConfig(width_mm=20.0, height_mm=20.0)
    geometry = _trim_display_geometry(QRectF(0.0, 0.0, 200.0, 200.0), panel)

    assert geometry.cell_rect.contains(geometry.island_rect)


def test_approved_layout_has_uniform_tight_outer_margin_and_top_pad() -> None:
    panel = measured_project().condition_a.panel
    rect = QRectF(10.0, 20.0, 286.0, 598.4)

    geometry = _trim_display_geometry(rect, panel)
    margin_x = panel.trim_geometry.pretrim_margin_mm * rect.width() / panel.width_mm
    margin_y = panel.trim_geometry.pretrim_margin_mm * rect.height() / panel.height_mm

    assert geometry.cell_rect.left() - rect.left() == pytest.approx(margin_x)
    assert rect.right() - geometry.cell_rect.right() == pytest.approx(margin_x)
    assert rect.bottom() - geometry.cell_rect.bottom() == pytest.approx(margin_y)
    assert geometry.pad_rect.top() - rect.top() == pytest.approx(margin_y)
    assert geometry.pad_rect.bottom() == pytest.approx(geometry.cell_rect.top())
    assert geometry.pad_rect.width() == pytest.approx(
        geometry.cell_rect.width() - 2.0 * geometry.cell_radius_x
    )
    assert geometry.pad_rect.center().y() < geometry.cell_rect.center().y()
    assert geometry.island_rect.center().y() > geometry.cell_rect.center().y()


def test_internal_holes_are_hidden_inside_dynamic_island_cover() -> None:
    panel = measured_project().condition_a.panel
    geometry = _trim_display_geometry(QRectF(0.0, 0.0, 286.0, 598.4), panel)

    assert geometry.island_rect.contains(geometry.hole_pill_rect)
    assert geometry.island_rect.contains(geometry.hole_circle_rect)


def test_condition_editor_round_trips_trim_geometry(qt_app: QApplication) -> None:
    condition = measured_project().condition_a.model_copy(deep=True)
    condition.panel.trim_geometry.pretrim_margin_mm = 1.2
    condition.panel.trim_geometry.cell_corner_radius_mm = 5.5
    condition.panel.trim_geometry.pad_height_mm = 2.4
    condition.panel.trim_geometry.island_width_mm = 19.0
    condition.panel.trim_geometry.island_height_mm = 5.0
    editor = ConditionEditor("A", "#58A6FF")

    editor.set_condition(condition)
    payload = editor.condition_payload()
    restored = PanelConfig.model_validate(payload["panel"])

    assert restored.trim_geometry == condition.panel.trim_geometry
