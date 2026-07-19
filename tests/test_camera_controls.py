from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lamination_sim.ui.results_view import ResultsView
from lamination_sim.ui.visualization import (
    CAMERA_PRESETS,
    PeelView,
    _camera_plane_transform,
    _orbit_project,
)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_orbit_projection_has_true_top_and_depth_sensitive_quarter_views() -> None:
    top_low = _orbit_project(10.0, 20.0, 0.0, *CAMERA_PRESETS["top"])
    top_high = _orbit_project(10.0, 20.0, 30.0, *CAMERA_PRESETS["top"])
    quarter_low = _orbit_project(10.0, 20.0, 0.0, *CAMERA_PRESETS["quarter"])
    quarter_high = _orbit_project(10.0, 20.0, 30.0, *CAMERA_PRESETS["quarter"])

    assert top_low.x() == pytest.approx(10.0)
    assert top_low.y() == pytest.approx(-20.0)
    assert top_high == top_low
    assert quarter_high.y() < quarter_low.y()


def test_camera_presets_project_the_panel_to_distinct_quadrilaterals() -> None:
    panel = QRectF(80.0, 80.0, 100.0, 200.0)
    area = QRectF(30.0, 50.0, 320.0, 420.0)
    top = _camera_plane_transform(panel, area, *CAMERA_PRESETS["top"])
    quarter = _camera_plane_transform(panel, area, *CAMERA_PRESETS["quarter"])

    top_corner = top.map(panel.topRight())
    quarter_corner = quarter.map(panel.topRight())

    assert top_corner != quarter_corner
    for transform in (top, quarter):
        for corner in (
            panel.topLeft(),
            panel.topRight(),
            panel.bottomLeft(),
            panel.bottomRight(),
        ):
            projected = transform.map(corner)
            assert area.adjusted(-1.0, -1.0, 1.0, 20.0).contains(projected)


def test_camera_projection_is_affine_without_view_dependent_warp() -> None:
    """Orbiting must rotate/foreshorten the sheet, never projectively warp it."""

    panel = QRectF(80.0, 80.0, 100.0, 200.0)
    area = QRectF(30.0, 50.0, 320.0, 420.0)
    for yaw, elevation in CAMERA_PRESETS.values():
        transform = _camera_plane_transform(panel, area, yaw, elevation)

        # QTransform's third row is the projective term.  The old
        # quadToQuad fitting populated m13/m23 and changed the local scale
        # whenever a camera angle changed; a true orthographic camera keeps
        # those terms at zero and m33 at one.
        assert transform.m13() == pytest.approx(0.0)
        assert transform.m23() == pytest.approx(0.0)
        assert transform.m33() == pytest.approx(1.0)


def test_mouse_drag_orbits_camera_continuously(qt_app: QApplication) -> None:
    view = PeelView("A", "#5FA8FF")
    view.resize(420, 560)
    view.show()
    qt_app.processEvents()
    initial = (view.camera_yaw_deg, view.camera_elevation_deg)
    emitted: list[tuple[float, float]] = []
    view.camera_changed.connect(lambda yaw, elevation: emitted.append((yaw, elevation)))

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(180, 240))
    QTest.mouseMove(view, QPoint(220, 210), delay=1)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(220, 210))

    assert emitted
    assert view.camera_yaw_deg > initial[0]
    assert view.camera_elevation_deg > initial[1]
    assert view._camera_drag_position is None


def test_results_view_keeps_a_and_b_camera_angles_paired(qt_app: QApplication) -> None:
    results = ResultsView()

    results._set_camera_preset("side")
    assert (results.view_a.camera_yaw_deg, results.view_a.camera_elevation_deg) == pytest.approx(
        CAMERA_PRESETS["side"]
    )
    assert (results.view_b.camera_yaw_deg, results.view_b.camera_elevation_deg) == pytest.approx(
        CAMERA_PRESETS["side"]
    )

    results.view_a.set_camera_angles(24.0, 51.0, notify=True)
    qt_app.processEvents()

    assert results.view_b.camera_yaw_deg == pytest.approx(24.0)
    assert results.view_b.camera_elevation_deg == pytest.approx(51.0)
    assert not any(button.isChecked() for button in results.camera_buttons.values())


def test_results_view_keeps_visual_layers_paired(qt_app: QApplication) -> None:
    results = ResultsView()
    results.layer_buttons["top_film"].setChecked(False)
    results._update_visual_layers()

    assert results.view_a.show_top_film is False
    assert results.view_b.show_top_film is False
    assert results.view_a.show_equipment is True
    assert results.view_b.show_force_vectors is True
