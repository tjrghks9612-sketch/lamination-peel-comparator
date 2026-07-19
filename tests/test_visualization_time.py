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
    _camera_plane_transform,
    _damage_contour_segments,
    _film_fold_world_grid,
    _film_peel_fields,
    _project_film_fold_grid,
    _film_surface_boundary,
    _film_surface_cells,
    _interpolate_at_time,
    _pull_tape_attachment_world,
    _pull_tape_world_grid,
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


def test_180_degree_fold_is_continuous_at_front_then_reverses_direction() -> None:
    damage = [[1.0] * 3 for _ in range(10)]
    damage += [[0.5] * 3]
    damage += [[0.0] * 3 for _ in range(10)]
    radius = 0.025 * math.hypot(100.0, 20.0)
    grip = (50.0, 25.0, 2.0 * radius)

    world = _film_fold_world_grid(
        100.0,
        20.0,
        damage,
        "bottom_left",
        grip,
        0.0,
    )
    front = world[10][1][0]
    rising = world[9][1][0]
    tail = world[1][1][0]
    anchor = world[0][1][0]

    assert front == pytest.approx((50.0, 10.0, 0.0))
    assert rising[1] < front[1]
    assert 0.0 < rising[2] < 2.0 * radius
    assert tail[1] > front[1]
    assert tail[2] == pytest.approx(2.0 * radius)
    assert anchor[1] > tail[1]
    assert anchor[2] == pytest.approx(2.0 * radius)


def test_film_body_never_narrows_to_pull_tape_or_grip() -> None:
    damage = [[1.0] * 21 for _ in range(5)] + [[0.0] * 21 for _ in range(4)]
    narrow = _film_fold_world_grid(
        100.0,
        40.0,
        damage,
        "bottom_left",
        (120.0, 20.0, 18.0),
        0.0,
        pull_tape_width_mm=2.0,
        pull_tape_length_mm=5.0,
    )
    wide = _film_fold_world_grid(
        100.0,
        40.0,
        damage,
        "bottom_left",
        (120.0, 20.0, 18.0),
        0.0,
        pull_tape_width_mm=30.0,
        pull_tape_length_mm=80.0,
    )

    assert narrow == wide
    first = narrow[2][0][0]
    last = narrow[2][-1][0]
    assert first[0] == pytest.approx(0.0)
    assert last[0] == pytest.approx(100.0)
    assert first[1:] == pytest.approx(last[1:])


def test_pull_tape_uses_only_its_actual_width_on_film_edge() -> None:
    damage = [[0.0] * 21 for _ in range(9)]
    world = _film_fold_world_grid(
        100.0,
        40.0,
        damage,
        "bottom_left",
        (0.0, 0.0, 12.0),
        0.0,
        mesh_subdivisions=2,
    )

    first, second = _pull_tape_attachment_world(world, 100.0, "bottom_left", 10.0)

    assert first == pytest.approx((0.0, 0.0, 0.0))
    assert second == pytest.approx((10.0, 0.0, 0.0))
    assert len(world) > len(damage)
    assert len(world[0]) > len(damage[0])


def test_pull_tape_is_a_separate_constant_width_mesh_that_stops_before_head() -> None:
    attachment = ((5.0, 80.0, 8.0), (15.0, 80.0, 8.0))
    grip = (13.0, 120.0, 24.0)

    tape = _pull_tape_world_grid(attachment, grip, segments=12, head_clearance_mm=3.0)

    assert len(tape) == 13
    assert tape[0][0] == pytest.approx(attachment[0])
    assert tape[0][1] == pytest.approx(attachment[1])
    for edge in tape:
        assert math.dist(edge[0], edge[1]) == pytest.approx(10.0)
    end_center = tuple((tape[-1][0][axis] + tape[-1][1][axis]) * 0.5 for axis in range(3))
    assert math.dist(end_center, grip) == pytest.approx(3.0)
    assert all(math.dist(point, grip) > 0.0 for edge in tape for point in edge)


def test_diagonal_fold_has_no_collapsed_cells_or_crossed_width_strips() -> None:
    damage = [
        [1.0 if row <= 2.0 + column * 0.5 else 0.0 for column in range(11)]
        for row in range(11)
    ]
    world = _film_fold_world_grid(
        100.0,
        100.0,
        damage,
        "bottom_left",
        (50.0, 120.0, 20.0),
        0.0,
        mesh_subdivisions=2,
    )

    for row in world:
        assert all(
            row[column + 1][0][0] > row[column][0][0]
            for column in range(len(row) - 1)
        )
    for row in range(len(world) - 1):
        for column in range(len(world[row]) - 1):
            first = world[row][column][0]
            across = world[row][column + 1][0]
            forward = world[row + 1][column][0]
            u = tuple(across[axis] - first[axis] for axis in range(3))
            v = tuple(forward[axis] - first[axis] for axis in range(3))
            cross = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            assert math.sqrt(sum(value * value for value in cross)) > 1.0e-8


def test_fully_detached_full_width_sheet_follows_head_as_one_rigid_mesh() -> None:
    active_front = [[1.0] * 6 + [0.0] * 5 for _ in range(5)]
    fully_detached = [[1.0] * 11 for _ in range(5)]
    delta = (7.0, -3.0, 4.0)
    reference = _film_fold_world_grid(
        100.0,
        40.0,
        fully_detached,
        "bottom_left",
        (88.0, 30.0, 18.0),
        1.0,
        front_damage_grid=active_front,
    )
    followed = _film_fold_world_grid(
        100.0,
        40.0,
        fully_detached,
        "bottom_left",
        (88.0, 30.0, 18.0),
        1.0,
        front_damage_grid=active_front,
        follow_delta_xyz_mm=delta,
    )

    for reference_row, followed_row in zip(reference, followed):
        for reference_vertex, followed_vertex in zip(reference_row, followed_row):
            assert followed_vertex[0] == pytest.approx(
                tuple(reference_vertex[0][axis] + delta[axis] for axis in range(3))
            )

    assert math.dist(reference[0][0][0], reference[0][-1][0]) == pytest.approx(100.0)
    assert math.dist(reference[0][0][0], reference[-1][0][0]) == pytest.approx(40.0)


def test_full_release_follow_reference_is_first_detached_frame(
    qt_app: QApplication,
) -> None:
    condition = measured_project().condition_a
    result = simulate(condition, AssumptionSet(), "coarse")
    first_detached_frame = next(
        frame_number
        for frame_number, frame in enumerate(result.bottom_damage_frames)
        if frame and all(value >= 0.5 for value in frame)
    )
    solver_index = result.frame_indices[first_detached_frame]
    view = PeelView("A", "#60A5FA")
    view.set_data(condition, result)
    view.set_time(result.time_s[solver_index], solver_index / (len(result.time_s) - 1))

    _current, _front, reference_pose = view._bottom_damage_display_fields()

    assert reference_pose == pytest.approx(result.position_xyz_mm[solver_index])


def test_tail_cross_sections_keep_full_width_instead_of_fanning() -> None:
    damage = [[1.0] * 5 + [0.5] + [0.0] * 5 for _ in range(11)]
    world = _film_fold_world_grid(
        100.0,
        100.0,
        damage,
        "bottom_left",
        (100.0, 100.0, 20.0),
        0.0,
        pull_tape_length_mm=0.0,
    )
    first = world[1][1][0]
    second = world[1][2][0]

    assert first[0] - second[0] == pytest.approx(-10.0)
    assert first[1] == pytest.approx(second[1])
    assert first[2] == pytest.approx(second[2])


def test_diagonal_and_fully_detached_fold_remain_one_shared_mesh() -> None:
    diagonal = [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    panel = QRectF(20.0, 30.0, 100.0, 200.0)
    transform = _camera_plane_transform(panel, QRectF(0.0, 0.0, 400.0, 400.0), 38.0, 32.0)

    for damage in (diagonal, [[1.0] * 4 for _ in range(4)]):
        world = _film_fold_world_grid(
            71.5,
            149.6,
            damage,
            "bottom_left",
            (84.0, 165.0, 28.0),
            0.0,
        )
        surface = _project_film_fold_grid(
            panel, world, 71.5, 149.6, transform, 56.0, 32.0
        )
        cells = _film_surface_cells(surface)

        assert len(cells) == 9
        assert len(_film_surface_boundary(surface)) == 12
        assert cells[0][0][1] == cells[1][0][0]
        assert cells[0][0][2] == cells[1][0][3]


def test_fold_projects_as_3d_geometry_when_camera_rotates() -> None:
    damage = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
    world = _film_fold_world_grid(
        60.0, 120.0, damage, "bottom_left", (72.0, 132.0, 24.0), 0.0
    )
    panel = QRectF(20.0, 30.0, 100.0, 200.0)
    area = QRectF(0.0, 0.0, 400.0, 400.0)
    quarter = _project_film_fold_grid(
        panel,
        world,
        60.0,
        120.0,
        _camera_plane_transform(panel, area, 38.0, 32.0),
        56.0,
        32.0,
    )
    side = _project_film_fold_grid(
        panel,
        world,
        60.0,
        120.0,
        _camera_plane_transform(panel, area, 90.0, 20.0),
        56.0,
        20.0,
    )

    assert world[0][0][0][2] > 0.0
    assert quarter[0][0][0] != side[0][0][0]
    assert quarter[0][1][0] != side[0][1][0]


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
