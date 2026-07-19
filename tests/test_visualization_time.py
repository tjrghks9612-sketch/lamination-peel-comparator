from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QApplication

from lamination_sim.models import AssumptionSet
from lamination_sim.presets import measured_project
from lamination_sim.simulation import simulate
from lamination_sim.ui.visualization import (
    PeelView,
    _damage_contour_segments,
    _film_peel_fields,
    _film_surface_boundary,
    _film_surface_cells,
    _film_surface_grid,
    _interpolate_at_time,
    _limited_pull_offset,
    _projected_z_offset,
    _time_bracket,
    _top_film_surface_grid,
    _vector_arrow_points,
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

    point = view._map_extended_xy_base(
        result.position_xyz_mm[-1][0],
        result.position_xyz_mm[-1][1],
        panel_rect,
        condition.panel.width_mm,
        condition.panel.height_mm,
    )

    assert point.y() < panel_rect.top()
    assert point.y() >= panel_rect.top() - 28.0 - 1.0e-12


def test_film_surface_uses_shared_vertices_without_tearing() -> None:
    panel = QRectF(20.0, 30.0, 100.0, 200.0)
    damage = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]

    surface = _film_surface_grid(
        panel,
        damage,
        damage,
        QPointF(12.0, -8.0),
        QPointF(6.0, -14.0),
        0.0,
    )
    cells = _film_surface_cells(surface)

    assert len(cells) == 2
    first, second = cells
    assert first[0][1] == second[0][0]
    assert first[0][2] == second[0][3]


def test_fully_detached_surface_remains_one_full_size_sheet() -> None:
    panel = QRectF(20.0, 30.0, 100.0, 200.0)
    pull = QPointF(12.0, -8.0)
    depth = QPointF(6.0, -14.0)

    surface = _film_surface_grid(
        panel,
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        pull,
        depth,
        0.0,
    )
    boundary = _film_surface_boundary(surface)

    assert max(point.x() for point in boundary) - min(point.x() for point in boundary) == pytest.approx(
        panel.width()
    )
    assert max(point.y() for point in boundary) - min(point.y() for point in boundary) == pytest.approx(
        panel.height()
    )
    assert min(point.x() for point in boundary) == pytest.approx(
        panel.left() + pull.x() + depth.x()
    )
    assert min(point.y() for point in boundary) == pytest.approx(
        panel.top() + pull.y() + depth.y()
    )


def test_visual_film_bridges_intact_islands_instead_of_drawing_holes() -> None:
    damage = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
    ]

    visual_damage, _ = _film_peel_fields(damage, "bottom_left", 0.0)

    assert visual_damage[2][2] == pytest.approx(1.0)
    assert visual_damage[4][4] == pytest.approx(0.0)
    assert sum(map(sum, visual_damage)) == pytest.approx(sum(map(sum, damage)))


def test_visual_film_curves_up_from_front_toward_gripped_corner() -> None:
    damage = [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    _, lift = _film_peel_fields(damage, "bottom_left", 0.0)

    assert lift[0][0] > lift[0][2]
    assert lift[0][0] > lift[1][1]
    assert lift[3][3] == pytest.approx(0.0)


def test_z_height_responds_to_orbit_camera_elevation() -> None:
    low = _projected_z_offset(4.0, 56.0)
    high = _projected_z_offset(14.0, 56.0)
    top = _projected_z_offset(14.0, 56.0, 90.0)
    front = _projected_z_offset(14.0, 56.0, 12.0)

    assert low.x() == pytest.approx(0.0)
    assert low.y() < 0.0
    assert high.y() < low.y()
    assert top.y() == pytest.approx(0.0, abs=1.0e-10)
    assert front.y() < high.y()


def test_in_plane_film_motion_preserves_measured_diagonal_pull_direction() -> None:
    anchor = QPointF(10.0, 90.0)
    grip = QPointF(70.0, 20.0)

    offset = _limited_pull_offset(anchor, grip)

    assert offset.x() > 0.0
    assert offset.y() < 0.0
    assert math.hypot(offset.x(), offset.y()) <= 25.0 + 1.0e-12


def test_damage_contour_spans_the_actual_full_height_boundary() -> None:
    damage = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]

    segments = _damage_contour_segments(damage)
    points = [point for segment in segments for point in segment]

    assert segments
    assert min(point[1] for point in points) == pytest.approx(0.0)
    assert max(point[1] for point in points) == pytest.approx(1.0)
    assert {round(point[0], 8) for point in points} == {0.5}


def test_damage_contour_follows_wavy_damage_instead_of_a_pca_line() -> None:
    boundaries = [1, 2, 1, 3, 2, 1]
    damage = [
        [1.0 if column <= boundary else 0.0 for column in range(5)]
        for boundary in boundaries
    ]

    segments = _damage_contour_segments(damage)
    points = [point for segment in segments for point in segment]

    assert min(point[1] for point in points) == pytest.approx(0.0)
    assert max(point[1] for point in points) == pytest.approx(1.0)
    assert len({round(point[0], 4) for point in points}) >= 3


def test_uniform_damage_has_no_active_peel_front() -> None:
    assert _damage_contour_segments([[0.0, 0.0], [0.0, 0.0]]) == []
    assert _damage_contour_segments([[1.0, 1.0], [1.0, 1.0]]) == []


def test_top_film_surface_keeps_full_domain_and_separates_damaged_region() -> None:
    panel = QRectF(20.0, 30.0, 100.0, 200.0)
    transform = QTransform()
    surface = _top_film_surface_grid(
        panel,
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0]],
        transform,
        38.0,
        1.0,
    )

    assert surface[0][0][0].x() == pytest.approx(panel.left())
    assert surface[-1][-1][0].x() == pytest.approx(panel.right())
    # Fully damaged film stays at its reference plane while bonded film follows
    # the exaggerated panel lift, making the separation visible in 3-D views.
    assert surface[0][0][0].y() == pytest.approx(panel.bottom())
    assert surface[0][1][0].y() < panel.bottom()


def test_projected_force_arrow_has_stable_length_and_head() -> None:
    arrow = _vector_arrow_points(QPointF(10.0, 10.0), QPointF(3.0, -4.0), 25.0)

    assert arrow is not None
    start, end, wing_a, wing_b = arrow
    assert math.hypot(end.x() - start.x(), end.y() - start.y()) == pytest.approx(25.0)
    assert wing_a != wing_b
