"""Dependency-light visualizations rendered with QPainter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .core_bridge import get_value, result_series, sequence
from .theme import COLORS, color


CAMERA_PRESETS: dict[str, tuple[float, float]] = {
    "top": (-90.0, 90.0),
    "quarter": (-135.0, 38.0),
    "front": (-90.0, 12.0),
    "side": (0.0, 12.0),
}


@dataclass(frozen=True)
class TrimDisplayGeometry:
    """Projected-plane process outlines inside the rectangular laminate."""

    outer_rect: QRectF
    cell_rect: QRectF
    pad_rect: QRectF
    island_rect: QRectF
    hole_pill_rect: QRectF
    hole_circle_rect: QRectF
    cell_radius_x: float
    cell_radius_y: float


def _trim_display_geometry(
    panel_rect: QRectF,
    panel: Any,
) -> TrimDisplayGeometry:
    """Resolve the approved 180-degree pre-trim layout in panel coordinates.

    The laminate stays a sharp rectangle.  The finished cell, top pad and
    bottom island/hole outlines are future trim references and never mask the
    numerical damage domain.
    """

    width_mm = max(1.0e-9, _float(get_value(panel, "width_mm", default=71.5), 71.5))
    height_mm = max(
        1.0e-9,
        _float(get_value(panel, "height_mm", default=149.6), 149.6),
    )
    trim = get_value(panel, "trim_geometry", default={})
    margin_mm = max(
        0.05,
        _float(get_value(trim, "pretrim_margin_mm", default=1.5), 1.5),
    )
    pad_height_mm = max(
        0.05,
        _float(get_value(trim, "pad_height_mm", default=3.0), 3.0),
    )
    maximum_margin = max(0.05, min(width_mm, height_mm - pad_height_mm) * 0.24)
    margin_mm = min(margin_mm, maximum_margin)
    cell_width_mm = max(width_mm - 2.0 * margin_mm, width_mm * 0.1)
    cell_height_mm = max(
        height_mm - 2.0 * margin_mm - pad_height_mm,
        height_mm * 0.1,
    )
    radius_mm = min(
        max(
            0.0,
            _float(get_value(trim, "cell_corner_radius_mm", default=6.0), 6.0),
        ),
        0.5 * min(cell_width_mm, cell_height_mm),
    )

    scale_x = panel_rect.width() / width_mm
    scale_y = panel_rect.height() / height_mm
    margin_x = margin_mm * scale_x
    margin_y = margin_mm * scale_y
    pad_height = min(pad_height_mm * scale_y, panel_rect.height() - 2.0 * margin_y)
    cell_rect = QRectF(
        panel_rect.left() + margin_x,
        panel_rect.top() + margin_y + pad_height,
        panel_rect.width() - 2.0 * margin_x,
        panel_rect.height() - 2.0 * margin_y - pad_height,
    )
    radius_x = min(radius_mm * scale_x, cell_rect.width() * 0.5)
    radius_y = min(radius_mm * scale_y, cell_rect.height() * 0.5)
    pad_rect = QRectF(
        cell_rect.left() + radius_x,
        panel_rect.top() + margin_y,
        max(0.0, cell_rect.width() - 2.0 * radius_x),
        pad_height,
    )

    island_width = min(
        _float(get_value(trim, "island_width_mm", default=22.0), 22.0) * scale_x,
        cell_rect.width() * 0.72,
    )
    island_height = min(
        _float(get_value(trim, "island_height_mm", default=6.0), 6.0) * scale_y,
        cell_rect.height() * 0.12,
    )
    island_bottom_gap = max(0.65 * margin_y, 1.0)
    island_rect = QRectF(
        cell_rect.center().x() - island_width * 0.5,
        cell_rect.bottom() - island_bottom_gap - island_height,
        island_width,
        island_height,
    )
    hole_height = island_height * 0.52
    hole_circle_size = hole_height
    hole_pill_width = island_width * 0.49
    hole_gap = island_width * 0.06
    group_width = hole_circle_size + hole_gap + hole_pill_width
    group_left = island_rect.center().x() - group_width * 0.5
    hole_y = island_rect.center().y() - hole_height * 0.5
    hole_circle_rect = QRectF(group_left, hole_y, hole_circle_size, hole_circle_size)
    hole_pill_rect = QRectF(
        group_left + hole_circle_size + hole_gap,
        hole_y,
        hole_pill_width,
        hole_height,
    )
    return TrimDisplayGeometry(
        outer_rect=QRectF(panel_rect),
        cell_rect=cell_rect,
        pad_rect=pad_rect,
        island_rect=island_rect,
        hole_pill_rect=hole_pill_rect,
        hole_circle_rect=hole_circle_rect,
        cell_radius_x=radius_x,
        cell_radius_y=radius_y,
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _compact(value: float) -> str:
    if value != 0 and (abs(value) < 0.01 or abs(value) >= 10000):
        return f"{value:.2e}"
    return f"{value:.2f}"


def _orbit_project(
    x: float,
    y: float,
    z: float,
    yaw_degrees: float,
    elevation_degrees: float,
) -> QPointF:
    """Orthographically project a world point through an orbit camera."""

    yaw = math.radians(yaw_degrees)
    elevation = math.radians(max(0.0, min(90.0, elevation_degrees)))
    screen_x = -math.sin(yaw) * x + math.cos(yaw) * y
    screen_y = (
        math.sin(elevation) * (math.cos(yaw) * x + math.sin(yaw) * y)
        - math.cos(elevation) * z
    )
    return QPointF(screen_x, screen_y)


def _camera_plane_transform(
    panel_rect: QRectF,
    camera_area: QRectF,
    yaw_degrees: float,
    elevation_degrees: float,
) -> QTransform:
    """Map the panel through one fixed-scale orthographic affine camera.

    The previous implementation fit each projected quadrilateral independently.
    That silently changed zoom and made a drag look like the specimen was
    stretching.  A fixed scale based on the panel diagonal preserves metric
    consistency across yaw changes; elevation only contributes real
    foreshortening of the panel-normal projection.
    """

    width = max(panel_rect.width(), 1.0e-9)
    height = max(panel_rect.height(), 1.0e-9)
    diagonal = math.hypot(width, height)
    # Use one scale for every yaw/elevation.  The camera area's width is the
    # guaranteed limiting dimension because any orthographic projection of the
    # panel has a bounding span no larger than its diagonal.  Keeping a small
    # fixed margin lets the specimen remain fully visible without per-view
    # re-fitting (which was the source of the apparent stretching).
    scale = 0.96 * camera_area.width() / diagonal
    yaw = math.radians(yaw_degrees)
    elevation = math.radians(max(0.0, min(90.0, elevation_degrees)))

    # QTransform maps x' = m11*x + m21*y + dx and
    # y' = m12*x + m22*y + dy.  The base panel plane uses screen y-down,
    # while the orbit basis uses world y-up, hence the negative y coefficients.
    m11 = scale * (-math.sin(yaw))
    m21 = scale * (-math.cos(yaw))
    m12 = scale * math.sin(elevation) * math.cos(yaw)
    m22 = scale * (-math.sin(elevation) * math.sin(yaw))

    elevation_fraction = max(0.0, min(1.0, elevation_degrees / 90.0))
    target_center = camera_area.center() + QPointF(
        0.0, (1.0 - elevation_fraction) * 18.0
    )
    panel_center = panel_rect.center()
    dx = target_center.x() - m11 * panel_center.x() - m21 * panel_center.y()
    dy = target_center.y() - m12 * panel_center.x() - m22 * panel_center.y()
    return QTransform(m11, m12, m21, m22, dx, dy)


def _time_bracket(times: list[float], selected_time: float) -> tuple[int, int, float]:
    """Return causal interpolation indices and fraction for a physical time."""

    if len(times) <= 1 or selected_time <= times[0]:
        return (0, 0, 0.0)
    if selected_time >= times[-1]:
        last = len(times) - 1
        return (last, last, 0.0)
    low = 0
    high = len(times) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if times[middle] <= selected_time:
            low = middle
        else:
            high = middle
    span = times[high] - times[low]
    fraction = 0.0 if span <= 0.0 else (selected_time - times[low]) / span
    return (low, high, max(0.0, min(1.0, fraction)))


def _interpolate_at_time(
    times: list[float], values: list[float], selected_time: float, fallback: float = 0.0
) -> float:
    if not values:
        return fallback
    if not times or len(times) != len(values):
        index = min(len(values) - 1, round(max(0.0, min(1.0, selected_time)) * (len(values) - 1)))
        return values[index]
    first, second, fraction = _time_bracket(times, selected_time)
    return _lerp(values[first], values[second], fraction)


def _coarsen_damage_grid(
    grid: list[list[float]], max_rows: int = 30, max_columns: int = 18
) -> list[list[float]]:
    """Average a dense solver grid without changing its full-sheet extent."""

    rows = len(grid)
    columns = max((len(row) for row in grid), default=0)
    if rows == 0 or columns == 0:
        return []
    out_rows = min(rows, max_rows)
    out_columns = min(columns, max_columns)
    result: list[list[float]] = []
    for out_row in range(out_rows):
        row_start = out_row * rows // out_rows
        row_end = max(row_start + 1, (out_row + 1) * rows // out_rows)
        values: list[float] = []
        for out_column in range(out_columns):
            column_start = out_column * columns // out_columns
            column_end = max(
                column_start + 1,
                (out_column + 1) * columns // out_columns,
            )
            samples = [
                _float(grid[row][column])
                for row in range(row_start, min(row_end, rows))
                for column in range(column_start, min(column_end, len(grid[row])))
            ]
            values.append(sum(samples) / len(samples) if samples else 0.0)
        result.append(values)
    return result


def _smooth_lift_fraction(damage: float) -> float:
    """Map cohesive damage to a continuous visual lift without discontinuities."""

    value = max(0.0, min(1.0, damage))
    return value * value * (3.0 - 2.0 * value)


def _film_peel_fields(
    damage_grid: list[list[float]],
    corner: str,
    fallback_damage: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Create a topology-safe visual damage field and a curved lift field.

    The numerical cohesive field remains untouched. For visualization, intact
    components not connected to the far/opposite corner are bridged instead of
    being rendered as literal holes in one continuous film. Lift then rises
    gradually from the active front toward the gripped start corner.
    """

    grid = _coarsen_damage_grid(damage_grid)
    if not grid:
        damage = max(0.0, min(1.0, fallback_damage))
        grid = [[damage, damage], [damage, damage]]
    elif len(grid) == 1:
        grid = [list(grid[0]), list(grid[0])]
    if len(grid[0]) == 1:
        grid = [[row[0], row[0]] for row in grid]
    grid = [
        [max(0.0, min(1.0, _float(value))) for value in row]
        for row in grid
    ]
    target_damage_sum = sum(sum(row) for row in grid)

    rows, columns = len(grid), len(grid[0])
    intact = {(row, column) for row in range(rows) for column in range(columns) if grid[row][column] < 0.5}
    components: list[set[tuple[int, int]]] = []
    remaining = set(intact)
    while remaining:
        component = {remaining.pop()}
        pending = list(component)
        while pending:
            row, column = pending.pop()
            for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbour = (row + row_delta, column + column_delta)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    pending.append(neighbour)
        components.append(component)

    far_corner = (
        0 if "top" in corner else rows - 1,
        0 if "right" in corner else columns - 1,
    )
    kept_intact: set[tuple[int, int]] = set()
    if components:
        kept_intact = next(
            (component for component in components if far_corner in component),
            max(components, key=len),
        )
        for row, column in intact - kept_intact:
            grid[row][column] = 1.0

        # Bridging a visual hole must not make the displayed peel fraction jump.
        # Give the same area back to the far-corner adhered component by moving
        # its front outward until the original damage sum is restored.
        excess = sum(sum(row) for row in grid) - target_damage_sum
        frontier = {
            neighbour
            for row, column in kept_intact
            for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0))
            if (
                0 <= (neighbour := (row + row_delta, column + column_delta))[0] < rows
                and 0 <= neighbour[1] < columns
                and neighbour not in kept_intact
            )
        }
        while excess > 1.0e-12 and frontier:
            node = min(
                frontier,
                key=lambda item: (
                    math.hypot(item[0] - far_corner[0], item[1] - far_corner[1]),
                    item[0],
                    item[1],
                ),
            )
            frontier.remove(node)
            row, column = node
            reduction = min(grid[row][column], excess)
            grid[row][column] -= reduction
            excess -= reduction
            if grid[row][column] < 0.5:
                kept_intact.add(node)
                for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    neighbour = (row + row_delta, column + column_delta)
                    if (
                        0 <= neighbour[0] < rows
                        and 0 <= neighbour[1] < columns
                        and neighbour not in kept_intact
                    ):
                        frontier.add(neighbour)

    detached = {(row, column) for row in range(rows) for column in range(columns) if grid[row][column] >= 0.5}
    if not detached:
        return grid, [[0.0 for _ in range(columns)] for _ in range(rows)]
    if not kept_intact:
        return grid, [[1.0 for _ in range(columns)] for _ in range(rows)]

    front_nodes = {
        (row, column)
        for row, column in detached
        if any(
            (row + row_delta, column + column_delta) in kept_intact
            for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0))
        )
    }
    distance = {node: 0 for node in front_nodes}
    pending = list(front_nodes)
    cursor = 0
    while cursor < len(pending):
        row, column = pending[cursor]
        cursor += 1
        for row_delta, column_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbour = (row + row_delta, column + column_delta)
            if neighbour in detached and neighbour not in distance:
                distance[neighbour] = distance[(row, column)] + 1
                pending.append(neighbour)
    maximum_distance = max(distance.values(), default=0)
    start_row = rows - 1 if "top" in corner else 0
    start_column = columns - 1 if "right" in corner else 0
    diagonal = max(math.hypot(rows - 1, columns - 1), 1.0)
    lift_grid: list[list[float]] = []
    for row in range(rows):
        lift_row: list[float] = []
        for column in range(columns):
            if (row, column) not in detached:
                lift_row.append(0.0)
                continue
            distance_fraction = (
                distance.get((row, column), maximum_distance) / maximum_distance
                if maximum_distance > 0
                else 1.0
            )
            corner_fraction = 1.0 - min(
                math.hypot(row - start_row, column - start_column) / diagonal,
                1.0,
            )
            profile = max(0.0, min(1.0, 0.82 * distance_fraction + 0.18 * corner_fraction))
            lift_row.append(_smooth_lift_fraction(grid[row][column]) * profile)
        lift_grid.append(lift_row)
    return grid, lift_grid


def _film_fold_world_grid(
    width_mm: float,
    height_mm: float,
    damage_grid: list[list[float]],
    corner: str,
    grip_xyz_mm: tuple[float, float, float],
    fallback_damage: float,
    *,
    pull_tape_width_mm: float = 10.0,
    pull_tape_length_mm: float = 10.0,
    front_damage_grid: list[list[float]] | None = None,
    follow_delta_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[tuple[tuple[float, float, float], float, float]]]:
    """Build a visualization-only, head-driven laid-down U peel shape.

    The damage=0.5 contour remains the physical display front, but every strip
    folds in one common front-to-head direction.  This prevents a diagonal
    front from producing the fan/tent shape caused by per-vertex local normals.
    The final fifth of the tail narrows to the pull-tape width and joins the
    tape one configured tape length before the grip.  These coordinates never
    participate in cohesive damage, force, or actuator-work calculations.
    """

    grid = [list(row) for row in damage_grid if row]
    if not grid:
        damage = max(0.0, min(1.0, fallback_damage))
        grid = [[damage, damage], [damage, damage]]
    elif len(grid) == 1:
        grid = [list(grid[0]), list(grid[0])]
    columns = min((len(row) for row in grid), default=0)
    if columns == 1:
        grid = [[row[0], row[0]] for row in grid]
        columns = 2
    elif columns > 1:
        grid = [row[:columns] for row in grid]
    rows = len(grid)

    width = max(float(width_mm), 1.0e-9)
    height = max(float(height_mm), 1.0e-9)
    contour_source = front_damage_grid if front_damage_grid else grid
    contour = _damage_contour_segments(contour_source)
    all_detached = all(_float(value) >= 0.5 for row in grid for value in row)
    opposite = (
        width if "right" not in corner else 0.0,
        height if "top" not in corner else 0.0,
    )
    # Once the last bonded node releases there is no current threshold contour.
    # Its limiting position is the opposite corner, which keeps the final frame
    # a single folded sheet instead of snapping back to a rigid translation.
    front_segments_mm = [
        (
            (start[0] * width, start[1] * height),
            (end[0] * width, end[1] * height),
        )
        for start, end in contour
    ]
    if not front_segments_mm and all_detached:
        front_segments_mm = [(opposite, opposite)]

    start_corner = (
        width if "right" in corner else 0.0,
        height if "top" in corner else 0.0,
    )
    fold_radius = max(2.5, min(7.0, 0.04 * math.hypot(width, height)))
    arc_length = math.pi * fold_radius
    front_points = [point for segment in front_segments_mm for point in segment]
    front_center = (
        sum(point[0] for point in front_points) / max(len(front_points), 1),
        sum(point[1] for point in front_points) / max(len(front_points), 1),
    )
    head_direction = (
        grip_xyz_mm[0] - front_center[0],
        grip_xyz_mm[1] - front_center[1],
    )
    head_distance = math.hypot(*head_direction)
    if head_distance <= 1.0e-9:
        head_direction = (
            opposite[0] - start_corner[0],
            opposite[1] - start_corner[1],
        )
        head_distance = max(math.hypot(*head_direction), 1.0e-9)
    tangent = (head_direction[0] / head_distance, head_direction[1] / head_distance)
    lateral_axis = (-tangent[1], tangent[0])

    def coordinates(point: tuple[float, float]) -> tuple[float, float]:
        return (
            point[0] * tangent[0] + point[1] * tangent[1],
            point[0] * lateral_axis[0] + point[1] * lateral_axis[1],
        )

    front_coordinates = [
        (coordinates(first), coordinates(second))
        for first, second in front_segments_mm
    ]

    def front_station(lateral: float, material_station: float) -> float:
        crossings: list[float] = []
        endpoints: list[tuple[float, float]] = []
        for (station_a, lateral_a), (station_b, lateral_b) in front_coordinates:
            endpoints.extend(((station_a, lateral_a), (station_b, lateral_b)))
            lateral_delta = lateral_b - lateral_a
            if abs(lateral_delta) <= 1.0e-12:
                if abs(lateral - lateral_a) <= 1.0e-9:
                    crossings.extend((station_a, station_b))
                continue
            fraction = (lateral - lateral_a) / lateral_delta
            if -1.0e-9 <= fraction <= 1.0 + 1.0e-9:
                crossings.append(_lerp(station_a, station_b, max(0.0, min(1.0, fraction))))
        if crossings:
            ahead = [value for value in crossings if value >= material_station - 1.0e-9]
            return min(ahead) if ahead else min(crossings, key=lambda value: abs(value - material_station))
        if endpoints:
            return min(endpoints, key=lambda value: abs(value[1] - lateral))[0]
        return coordinates(opposite)[0]

    def folded_point(
        x: float, y: float
    ) -> tuple[tuple[float, float, float], float, float]:
        material_station, lateral = coordinates((x, y))
        station = front_station(lateral, material_station)
        distance = max(station - material_station, 0.0)
        fx = station * tangent[0] + lateral * lateral_axis[0]
        fy = station * tangent[1] + lateral * lateral_axis[1]
        if distance <= arc_length:
            angle = distance / fold_radius
            along = fold_radius * math.sin(angle)
            return (
                (
                    fx - tangent[0] * along,
                    fy - tangent[1] * along,
                    fold_radius * (1.0 - math.cos(angle)),
                ),
                distance,
                lateral,
            )
        tail = distance - arc_length
        return (
            (
                fx + tangent[0] * tail,
                fy + tangent[1] * tail,
                2.0 * fold_radius,
            ),
            distance,
            lateral,
        )

    anchor_point, anchor_distance, anchor_lateral = folded_point(*start_corner)
    grip_x, grip_y, grip_z = grip_xyz_mm
    tape_attachment = (
        grip_x - tangent[0] * max(0.0, pull_tape_length_mm),
        grip_y - tangent[1] * max(0.0, pull_tape_length_mm),
        max(0.0, grip_z),
    )
    anchor_delta = (
        tape_attachment[0] - anchor_point[0],
        tape_attachment[1] - anchor_point[1],
        tape_attachment[2] - anchor_point[2],
    )
    longitudinal_correction = anchor_delta[0] * tangent[0] + anchor_delta[1] * tangent[1]
    lateral_correction = anchor_delta[0] * lateral_axis[0] + anchor_delta[1] * lateral_axis[1]
    panel_laterals = [
        coordinates(point)[1]
        for point in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    ]
    panel_lateral_span = max(max(panel_laterals) - min(panel_laterals), 1.0e-9)
    tape_width_scale = min(1.0, max(0.0, pull_tape_width_mm) / panel_lateral_span)

    world: list[list[tuple[tuple[float, float, float], float, float]]] = []
    for row, values in enumerate(grid):
        output_row: list[tuple[tuple[float, float, float], float, float]] = []
        y = row / (rows - 1) * height
        for column, value in enumerate(values):
            x = column / (columns - 1) * width
            damage = max(0.0, min(1.0, _float(value)))
            if damage < 0.5 or not front_segments_mm:
                output_row.append(((x, y, 0.0), damage, 0.0))
                continue
            point, distance, lateral = folded_point(x, y)
            tail_progress = distance / max(anchor_distance, fold_radius)
            weight = max(0.0, min(1.0, (tail_progress - 0.8) / 0.2))
            weight = _smooth_lift_fraction(weight)
            lateral_narrowing = (
                lateral_correction
                + (lateral - anchor_lateral) * (tape_width_scale - 1.0)
            )
            point = (
                point[0]
                + (tangent[0] * longitudinal_correction + lateral_axis[0] * lateral_narrowing)
                * weight
                + follow_delta_xyz_mm[0],
                point[1]
                + (tangent[1] * longitudinal_correction + lateral_axis[1] * lateral_narrowing)
                * weight
                + follow_delta_xyz_mm[1],
                max(0.0, point[2] + anchor_delta[2] * weight + follow_delta_xyz_mm[2]),
            )
            fold_fraction = max(0.0, min(1.0, distance / max(arc_length, 1.0e-9)))
            output_row.append((point, damage, fold_fraction))
        world.append(output_row)
    return world


def _project_film_fold_grid(
    panel_rect: QRectF,
    world_grid: list[list[tuple[tuple[float, float, float], float, float]]],
    width_mm: float,
    height_mm: float,
    plane_transform: QTransform,
    z_reference_mm: float,
    elevation_degrees: float,
) -> list[list[tuple[QPointF, float, float]]]:
    """Project the visualization-only peel mesh through the shared orbit camera."""

    width = max(width_mm, 1.0e-9)
    height = max(height_mm, 1.0e-9)
    surface: list[list[tuple[QPointF, float, float]]] = []
    for row in world_grid:
        output_row: list[tuple[QPointF, float, float]] = []
        for (x, y, z), damage, fold_fraction in row:
            on_plane = QPointF(
                panel_rect.left() + x / width * panel_rect.width(),
                panel_rect.bottom() - y / height * panel_rect.height(),
            )
            projected = plane_transform.map(on_plane) + _projected_z_offset(
                z, z_reference_mm, elevation_degrees
            )
            output_row.append((projected, damage, fold_fraction))
        surface.append(output_row)
    return surface


def _film_surface_cells(
    surface: list[list[tuple[QPointF, float, float]]],
) -> list[tuple[tuple[QPointF, QPointF, QPointF, QPointF], float, float]]:
    """Return connected quadrilateral cells, ordered back-to-front."""

    if len(surface) < 2 or len(surface[0]) < 2:
        return []
    cells: list[tuple[tuple[QPointF, QPointF, QPointF, QPointF], float, float]] = []
    # The physical top edge is farther from the fixed camera, so paint it first.
    for row in range(len(surface) - 2, -1, -1):
        for column in range(len(surface[row]) - 1):
            vertices = (
                surface[row][column],
                surface[row][column + 1],
                surface[row + 1][column + 1],
                surface[row + 1][column],
            )
            cells.append(
                (
                    tuple(vertex[0] for vertex in vertices),  # type: ignore[arg-type]
                    sum(vertex[1] for vertex in vertices) / 4.0,
                    sum(vertex[2] for vertex in vertices) / 4.0,
                )
            )
    return cells


def _film_surface_boundary(
    surface: list[list[tuple[QPointF, float, float]]],
) -> list[QPointF]:
    """Trace the single closed material boundary of the projected sheet."""

    if not surface or not surface[0]:
        return []
    if len(surface) == 1:
        return [vertex[0] for vertex in surface[0]]
    boundary = [vertex[0] for vertex in surface[0]]
    boundary.extend(row[-1][0] for row in surface[1:])
    boundary.extend(vertex[0] for vertex in reversed(surface[-1][:-1]))
    boundary.extend(row[0][0] for row in reversed(surface[1:-1]))
    return boundary


def _top_film_surface_grid(
    panel_rect: QRectF,
    panel_z_grid: list[list[float]],
    damage_grid: list[list[float]],
    risk_grid: list[list[float]],
    reaction_grid: list[list[float]],
    plane_transform: QTransform,
    elevation_degrees: float,
    maximum_lift_mm: float,
) -> list[list[tuple[QPointF, float, float, float]]]:
    """Build the full top-film sheet from solver fields.

    The plate displacement is transferred to bonded film by ``1-damage``.
    Fully damaged film is left near its undeformed reference plane because the
    reduced-order solver does not contain a separate membrane model for its
    post-release motion.  The display uses one fixed result-wide magnification
    so animation frames do not breathe or rescale as the camera moves.
    """

    source = next(
        (grid for grid in (damage_grid, risk_grid, panel_z_grid, reaction_grid) if grid and grid[0]),
        [],
    )
    source_rows = len(source) if source else 2
    source_columns = len(source[0]) if source else 2
    rows = max(2, min(source_rows, 28))
    columns = max(2, min(source_columns, 18))
    lift_reference = max(maximum_lift_mm, 1.0e-12)
    normal_pixels_per_mm = 22.0 / lift_reference
    normal_projection = math.cos(
        math.radians(max(0.0, min(90.0, elevation_degrees)))
    )

    surface: list[list[tuple[QPointF, float, float, float]]] = []
    for row in range(rows):
        y_fraction = row / (rows - 1)
        output_row: list[tuple[QPointF, float, float, float]] = []
        for column in range(columns):
            x_fraction = column / (columns - 1)
            damage = max(
                0.0,
                min(1.0, _sample_normalized_grid(damage_grid, x_fraction, y_fraction)),
            )
            risk = max(0.0, _sample_normalized_grid(risk_grid, x_fraction, y_fraction))
            reaction = max(
                0.0,
                _sample_normalized_grid(reaction_grid, x_fraction, y_fraction),
            )
            panel_lift = max(
                0.0,
                _sample_normalized_grid(panel_z_grid, x_fraction, y_fraction),
            )
            film_lift = panel_lift * (1.0 - damage)
            base = plane_transform.map(
                QPointF(
                    panel_rect.left() + x_fraction * panel_rect.width(),
                    panel_rect.bottom() - y_fraction * panel_rect.height(),
                )
            )
            offset = QPointF(
                0.0,
                -film_lift * normal_pixels_per_mm * normal_projection,
            )
            output_row.append((base + offset, damage, risk, reaction))
        surface.append(output_row)
    return surface


def _vector_arrow_points(
    start: QPointF,
    vector: QPointF,
    length_pixels: float,
    head_pixels: float = 6.0,
) -> tuple[QPointF, QPointF, QPointF, QPointF] | None:
    """Return a stable 2-D arrow glyph for a projected 3-D vector."""

    magnitude = math.hypot(vector.x(), vector.y())
    if magnitude <= 1.0e-12 or length_pixels <= 0.0:
        return None
    unit = QPointF(vector.x() / magnitude, vector.y() / magnitude)
    end = start + unit * length_pixels
    normal = QPointF(-unit.y(), unit.x())
    neck = end - unit * head_pixels
    return start, end, neck + normal * (head_pixels * 0.48), neck - normal * (head_pixels * 0.48)


def _sample_normalized_grid(
    grid: list[list[float]], x_fraction: float, y_fraction: float
) -> float:
    """Bilinearly sample a physical-bottom-origin normalized grid."""

    if not grid or not grid[0]:
        return 0.0
    rows, columns = len(grid), len(grid[0])
    x = max(0.0, min(1.0, x_fraction)) * (columns - 1)
    y = max(0.0, min(1.0, y_fraction)) * (rows - 1)
    column0, row0 = int(math.floor(x)), int(math.floor(y))
    column1, row1 = min(column0 + 1, columns - 1), min(row0 + 1, rows - 1)
    x_local, y_local = x - column0, y - row0
    bottom = _lerp(grid[row0][column0], grid[row0][column1], x_local)
    top = _lerp(grid[row1][column0], grid[row1][column1], x_local)
    return _lerp(bottom, top, y_local)


def _projected_z_offset(
    absolute_z_mm: float,
    z_reference_mm: float,
    elevation_degrees: float = CAMERA_PRESETS["quarter"][1],
    maximum_pixels: float = 82.0,
) -> QPointF:
    """Project panel-normal height for the current orbit elevation."""

    fraction = min(max(absolute_z_mm, 0.0) / max(z_reference_mm, 1.0e-9), 1.0)
    elevation = math.radians(max(0.0, min(90.0, elevation_degrees)))
    depth = maximum_pixels * fraction * math.cos(elevation)
    return QPointF(0.0, -depth)


def _damage_contour_segments(
    damage_grid: list[list[float]], threshold: float = 0.5
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract the actual damage/intact boundary with marching squares.

    Coordinates are normalized to the full panel: x and y both span 0..1,
    with y=0 at the physical bottom edge.  A fully intact or fully detached
    sheet has no active peel front and therefore returns no segments.
    """

    rows = len(damage_grid)
    columns = min((len(row) for row in damage_grid), default=0)
    if rows < 2 or columns < 2:
        return []

    def crossing(
        first: tuple[float, float],
        second: tuple[float, float],
        first_value: float,
        second_value: float,
    ) -> tuple[float, float] | None:
        first_above = first_value >= threshold
        second_above = second_value >= threshold
        if first_above == second_above:
            return None
        delta = second_value - first_value
        fraction = 0.5 if abs(delta) <= 1.0e-12 else (threshold - first_value) / delta
        fraction = max(0.0, min(1.0, fraction))
        return (
            _lerp(first[0], second[0], fraction),
            _lerp(first[1], second[1], fraction),
        )

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            x0 = column / (columns - 1)
            x1 = (column + 1) / (columns - 1)
            y0 = row / (rows - 1)
            y1 = (row + 1) / (rows - 1)
            values = (
                _float(damage_grid[row][column]),
                _float(damage_grid[row][column + 1]),
                _float(damage_grid[row + 1][column + 1]),
                _float(damage_grid[row + 1][column]),
            )
            points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            edges = (
                crossing(points[0], points[1], values[0], values[1]),
                crossing(points[1], points[2], values[1], values[2]),
                crossing(points[2], points[3], values[2], values[3]),
                crossing(points[3], points[0], values[3], values[0]),
            )
            active = [(edge, point) for edge, point in enumerate(edges) if point is not None]
            if len(active) == 2:
                segments.append((active[0][1], active[1][1]))
            elif len(active) == 4:
                case = sum((1 << index) for index, value in enumerate(values) if value >= threshold)
                center_above = sum(values) / 4.0 >= threshold
                if (case == 5 and center_above) or (case == 10 and not center_above):
                    pairs = ((0, 1), (2, 3))
                else:
                    pairs = ((3, 0), (1, 2))
                for first_edge, second_edge in pairs:
                    first = edges[first_edge]
                    second = edges[second_edge]
                    if first is not None and second is not None:
                        segments.append((first, second))
    return segments


class PeelView(QWidget):
    """Engineering view with a continuous, orbit-camera 3-D film projection."""

    progress_requested = Signal(float)
    camera_changed = Signal(float, float)

    def __init__(self, label: str, accent: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.accent = QColor(accent)
        self.condition: Any = None
        self.result: Any = None
        self.progress = 0.0
        self.selected_time_s = 0.0
        self.z_reference_mm = 1.0
        self.camera_yaw_deg, self.camera_elevation_deg = CAMERA_PRESETS["quarter"]
        self.show_top_film = True
        self.show_bottom_film = True
        self.show_equipment = True
        self.show_force_vectors = True
        self._camera_drag_position: QPointF | None = None
        self.setMinimumSize(330, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("좌클릭 드래그: 3D 시점 회전 · 휠: 시간 이동 · 더블클릭: 쿼터뷰")

    def set_data(self, condition: Any, result: Any) -> None:
        self.condition = condition
        self.result = result
        self.progress = 0.0
        times = result_series(result, "time")
        self.selected_time_s = times[0] if times else 0.0
        self.update()

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, float(progress)))
        times = result_series(self.result, "time")
        if times:
            self.selected_time_s = _lerp(times[0], times[-1], self.progress)
        self.update()

    def set_time(self, selected_time_s: float, global_progress: float | None = None) -> None:
        self.selected_time_s = float(selected_time_s)
        if global_progress is not None:
            self.progress = max(0.0, min(1.0, float(global_progress)))
        self.update()

    def set_z_reference(self, maximum_absolute_z_mm: float) -> None:
        self.z_reference_mm = max(float(maximum_absolute_z_mm), 1.0e-9)
        self.update()

    def set_layer_visibility(
        self,
        *,
        top_film: bool | None = None,
        bottom_film: bool | None = None,
        equipment: bool | None = None,
        force_vectors: bool | None = None,
    ) -> None:
        """Toggle engineering layers without changing simulation state."""

        if top_film is not None:
            self.show_top_film = bool(top_film)
        if bottom_film is not None:
            self.show_bottom_film = bool(bottom_film)
        if equipment is not None:
            self.show_equipment = bool(equipment)
        if force_vectors is not None:
            self.show_force_vectors = bool(force_vectors)
        self.update()

    def set_camera_angles(
        self,
        yaw_degrees: float,
        elevation_degrees: float,
        *,
        notify: bool = False,
    ) -> None:
        """Update the orbit camera without introducing a gesture discontinuity."""

        self.camera_yaw_deg = ((float(yaw_degrees) + 180.0) % 360.0) - 180.0
        self.camera_elevation_deg = max(8.0, min(90.0, float(elevation_degrees)))
        self.update()
        if notify:
            self.camera_changed.emit(self.camera_yaw_deg, self.camera_elevation_deg)

    def set_camera_preset(self, name: str, *, notify: bool = False) -> None:
        if name not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset: {name!r}")
        self.set_camera_angles(*CAMERA_PRESETS[name], notify=notify)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._camera_drag_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if (
            self._camera_drag_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta = event.position() - self._camera_drag_position
            self._camera_drag_position = event.position()
            self.set_camera_angles(
                self.camera_yaw_deg + delta.x() * 0.55,
                self.camera_elevation_deg - delta.y() * 0.42,
                notify=True,
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._camera_drag_position is not None:
            self._camera_drag_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._camera_drag_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.set_camera_preset("quarter", notify=True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        delta = 0.025 if event.angleDelta().y() > 0 else -0.025
        self.progress_requested.emit(max(0.0, min(1.0, self.progress + delta)))
        event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), color("surface"))

        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(color("border_soft"), 1))
        painter.drawRoundedRect(outer, 10, 10)

        self._draw_header(painter)
        if self.condition is None:
            self._draw_empty(painter)
            return

        panel = get_value(self.condition, "panel", default={})
        width_mm = max(1.0, _float(get_value(panel, "width_mm", default=71.5), 71.5))
        height_mm = max(1.0, _float(get_value(panel, "height_mm", default=149.6), 149.6))
        panel_rect = self._panel_rect(width_mm, height_mm)
        plane_transform = self._camera_transform(panel_rect)

        painter.save()
        painter.setTransform(plane_transform, combine=True)
        self._draw_panel_shadow(painter, panel_rect)
        self._draw_panel(painter, panel_rect)
        self._draw_mesh(painter, panel_rect)
        self._draw_trim_features(painter, panel_rect)
        painter.restore()
        if self.show_top_film:
            self._draw_top_film(painter, panel_rect, width_mm, height_mm)
        if self.show_bottom_film:
            self._draw_front_and_film(painter, panel_rect, width_mm, height_mm)
        elif self.show_equipment or self.show_force_vectors:
            self._draw_equipment_without_film(painter, panel_rect, width_mm, height_mm)
        self._draw_trajectory(painter, panel_rect, width_mm, height_mm)
        self._draw_footer(painter)

    def _draw_header(self, painter: QPainter) -> None:
        badge = QRectF(14, 13, 28, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.accent)
        painter.drawRoundedRect(badge, 6, 6)
        painter.setPen(QColor("#071015"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, self.label)
        painter.setPen(color("text"))
        painter.drawText(QPointF(52, 30), f"조건 {self.label} · 필름/장비 3D")

        time_values = result_series(self.result, "time")
        if time_values:
            shown_time = max(time_values[0], min(time_values[-1], self.selected_time_s))
            text = f"{shown_time:.3f} s"
        else:
            text = f"{self.progress * 100:.0f}%"
        painter.setPen(color("text_muted"))
        painter.drawText(
            QRectF(self.width() - 100, 12, 84, 26),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_empty(self, painter: QPainter) -> None:
        painter.setPen(color("text_dim"))
        painter.drawText(
            QRectF(20, 55, self.width() - 40, self.height() - 85),
            Qt.AlignmentFlag.AlignCenter,
            "조건을 입력하고\n비교 실행을 눌러주세요",
        )

    def _panel_rect(self, width_mm: float, height_mm: float) -> QRectF:
        # Reserve depth above the panel so positive Z can be seen instead of
        # being clipped into the header like a flat top-down trajectory.
        available = QRectF(42, 78, self.width() - 84, self.height() - 133)
        ratio = width_mm / height_mm
        target_width = available.height() * ratio
        if target_width > available.width():
            target_width = available.width()
            target_height = target_width / ratio
        else:
            target_height = available.height()
        return QRectF(
            available.center().x() - target_width / 2,
            available.center().y() - target_height / 2,
            target_width,
            target_height,
        )

    def _camera_area(self) -> QRectF:
        return QRectF(34.0, 66.0, self.width() - 68.0, self.height() - 116.0)

    def _camera_transform(self, panel_rect: QRectF) -> QTransform:
        return _camera_plane_transform(
            panel_rect,
            self._camera_area(),
            self.camera_yaw_deg,
            self.camera_elevation_deg,
        )

    def _draw_panel_shadow(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 85))
        painter.drawRect(rect.translated(5, 7))

    def _draw_panel(self, painter: QPainter, rect: QRectF) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#222B37"))
        gradient.setColorAt(0.55, QColor("#151C25"))
        gradient.setColorAt(1, QColor("#10161D"))
        painter.setBrush(gradient)
        outer_pen = QPen(QColor("#4EA2F5"), 1.25)
        outer_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(outer_pen)
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255, 16), 0.8))
        painter.drawRect(rect.adjusted(2.5, 2.5, -2.5, -2.5))

    def _draw_trim_features(self, painter: QPainter, rect: QRectF) -> None:
        panel = get_value(self.condition, "panel", default={})
        geometry = _trim_display_geometry(rect, panel)

        cell_pen = QPen(QColor("#F59A45"), 1.45)
        cell_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(cell_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            geometry.cell_rect,
            geometry.cell_radius_x,
            geometry.cell_radius_y,
        )

        pad_fill = QColor("#78C75A")
        pad_fill.setAlpha(50)
        painter.setBrush(pad_fill)
        painter.setPen(QPen(QColor("#82D766"), 1.25))
        painter.drawRect(geometry.pad_rect)

        island_fill = QColor("#05070A")
        island_fill.setAlpha(205)
        painter.setBrush(island_fill)
        painter.setPen(QPen(QColor(225, 230, 236, 150), 0.8))
        island_radius = geometry.island_rect.height() * 0.5
        painter.drawRoundedRect(geometry.island_rect, island_radius, island_radius)

        hole_pen = QPen(QColor("#F05DB2"), 1.15, Qt.PenStyle.DashLine)
        hole_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(hole_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        hole_radius = geometry.hole_pill_rect.height() * 0.5
        painter.drawRoundedRect(geometry.hole_pill_rect, hole_radius, hole_radius)
        painter.drawEllipse(geometry.hole_circle_rect)

    def _draw_mesh(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setClipRect(rect.adjusted(1, 1, -1, -1))
        painter.setPen(QPen(color("grid", 95), 0.7))
        for index in range(1, 8):
            x = rect.left() + rect.width() * index / 8
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(1, 14):
            y = rect.top() + rect.height() * index / 14
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.restore()

    def _draw_top_film(
        self,
        painter: QPainter,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        """Render the complete top film, its damage front and PSA reaction."""

        panel_z = self._frame_grid("panel_z_frames_mm")
        damage = self._frame_grid("top_damage_frames")
        risk = self._frame_grid("top_risk_frames", "risk_frames")
        reaction = self._frame_grid("top_interface_reaction_frames_n")
        maximum_lift = max(
            result_series(self.result, "panel_lift") or [0.0],
        )
        surface = _top_film_surface_grid(
            rect,
            panel_z,
            damage,
            risk,
            reaction,
            self._camera_transform(rect),
            self.camera_elevation_deg,
            maximum_lift,
        )
        if not surface or not surface[0]:
            return

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(len(surface) - 2, -1, -1):
            for column in range(len(surface[row]) - 1):
                vertices = (
                    surface[row][column],
                    surface[row][column + 1],
                    surface[row + 1][column + 1],
                    surface[row + 1][column],
                )
                points = QPolygonF([vertex[0] for vertex in vertices])
                local_damage = sum(vertex[1] for vertex in vertices) / 4.0
                local_risk = sum(vertex[2] for vertex in vertices) / 4.0
                film_color = QColor("#57C7DB")
                film_color.setAlpha(24 + round(34 * (1.0 - local_damage)))
                painter.setBrush(film_color)
                painter.drawPolygon(points)
                if local_risk > 0.01:
                    risk_level = math.sqrt(min(1.0, local_risk))
                    risk_color = self._risk_color(risk_level)
                    risk_color.setAlpha(round(24 + 112 * risk_level))
                    painter.setBrush(risk_color)
                    painter.drawPolygon(points)
                if local_damage > 0.01:
                    damage_color = QColor("#F05DB2")
                    damage_color.setAlpha(round(24 + 132 * min(1.0, local_damage)))
                    painter.setBrush(damage_color)
                    painter.drawPolygon(points)

        # A continuous cyan boundary makes the entire top-film domain visible
        # even in top view, where panel-normal separation projects to zero.
        boundary = [vertex[0] for vertex in surface[0]]
        boundary.extend(row[-1][0] for row in surface[1:])
        boundary.extend(vertex[0] for vertex in reversed(surface[-1][:-1]))
        boundary.extend(row[0][0] for row in reversed(surface[1:-1]))
        outline_pen = QPen(QColor("#70DCEB"), 1.65)
        outline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(QPolygonF([*boundary, boundary[0]]))

        mesh_pen = QPen(QColor(132, 226, 238, 38), 0.65)
        painter.setPen(mesh_pen)
        for row in range(0, len(surface), max(1, len(surface) // 10)):
            painter.drawPolyline(QPolygonF([vertex[0] for vertex in surface[row]]))
        for column in range(0, len(surface[0]), max(1, len(surface[0]) // 7)):
            painter.drawPolyline(QPolygonF([row[column][0] for row in surface]))

        # The magenta line is the actual top cohesive damage boundary, not a
        # fitted diagonal proxy.
        contour_pen = QPen(QColor("#FF8CCC"), 2.0)
        contour_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(contour_pen)
        for start, end in _damage_contour_segments(damage):
            painter.drawLine(
                self._top_surface_point(surface, start[0], start[1]),
                self._top_surface_point(surface, end[0], end[1]),
            )

        if self.show_force_vectors:
            self._draw_top_reaction_vectors(painter, surface, reaction, rect, width_mm, height_mm)
        painter.restore()

    @staticmethod
    def _top_surface_point(
        surface: list[list[tuple[QPointF, float, float, float]]],
        x_fraction: float,
        y_fraction: float,
    ) -> QPointF:
        rows, columns = len(surface), len(surface[0])
        row = min(rows - 1, max(0, round(y_fraction * (rows - 1))))
        column = min(columns - 1, max(0, round(x_fraction * (columns - 1))))
        return surface[row][column][0]

    def _draw_top_reaction_vectors(
        self,
        painter: QPainter,
        surface: list[list[tuple[QPointF, float, float, float]]],
        reaction_grid: list[list[float]],
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        normal = _orbit_project(
            0.0, 0.0, 1.0, self.camera_yaw_deg, self.camera_elevation_deg
        ) - _orbit_project(
            0.0, 0.0, 0.0, self.camera_yaw_deg, self.camera_elevation_deg
        )
        if math.hypot(normal.x(), normal.y()) <= 0.03:
            return
        sampled = [
            (vertex[3], row_index, column_index)
            for row_index, row in enumerate(surface)
            for column_index, vertex in enumerate(row)
            if vertex[3] > 0.0
        ]
        sampled.sort(reverse=True)
        local_max = sampled[0][0] if sampled else 0.0
        for value, row, column in sampled[:: max(1, len(sampled) // 7)][:7]:
            length = 5.0 + 8.0 * math.sqrt(value / max(local_max, 1.0e-15))
            self._draw_vector_arrow(
                painter,
                surface[row][column][0],
                normal,
                length,
                QColor(104, 225, 239, 125),
                0.9,
                3.5,
            )

        normal_force = self._result_series_value(
            result_series(self.result, "top_interface_force")
        )
        if normal_force <= 1.0e-10:
            return
        centroid_x = self._result_series_value(
            result_series(self.result, "top_reaction_x"), width_mm * 0.5
        )
        centroid_y = self._result_series_value(
            result_series(self.result, "top_reaction_y"), height_mm * 0.5
        )
        start = self._camera_transform(rect).map(
            self._map_xy(centroid_x, centroid_y, rect, width_mm, height_mm)
        )
        force_history = result_series(self.result, "top_interface_force")
        length = 18.0 + 22.0 * math.sqrt(
            normal_force / max(max(force_history or [normal_force]), 1.0e-12)
        )
        end = self._draw_vector_arrow(
            painter,
            start,
            normal,
            length,
            QColor("#68E1EF"),
            1.8,
            6.0,
        )
        if end is not None:
            self._draw_callout(
                painter,
                end + QPointF(6.0, -15.0),
                f"상면 Fn {_compact(normal_force)} N",
                QColor("#B9F5FA"),
            )

    @staticmethod
    def _draw_vector_arrow(
        painter: QPainter,
        start: QPointF,
        vector: QPointF,
        length: float,
        arrow_color: QColor,
        width: float,
        head: float,
    ) -> QPointF | None:
        points = _vector_arrow_points(start, vector, length, head)
        if points is None:
            return None
        origin, end, wing_a, wing_b = points
        pen = QPen(arrow_color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(origin, end)
        painter.drawPolyline(QPolygonF([wing_a, end, wing_b]))
        return end

    def _draw_callout(
        self,
        painter: QPainter,
        anchor: QPointF,
        text: str,
        text_color: QColor,
    ) -> None:
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 12.0
        height = metrics.height() + 7.0
        left = max(8.0, min(anchor.x(), self.width() - width - 8.0))
        top = max(45.0, min(anchor.y(), self.height() - height - 38.0))
        box = QRectF(left, top, width, height)
        painter.setPen(QPen(QColor(255, 255, 255, 34), 0.8))
        painter.setBrush(QColor(8, 15, 22, 205))
        painter.drawRoundedRect(box, 4.0, 4.0)
        painter.setPen(text_color)
        painter.drawText(
            box.adjusted(6.0, 0.0, -6.0, 0.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_risk_field(self, painter: QPainter, rect: QRectF) -> None:
        grid = self._frame_grid("top_risk_frames", "risk_frames", "top_damage_frames")
        if grid:
            rows = len(grid)
            columns = max((len(row) for row in grid), default=0)
            if rows and columns:
                painter.save()
                painter.setClipRect(rect)
                cell_w = rect.width() / max(1, columns)
                cell_h = rect.height() / max(1, rows)
                for row_index, row in enumerate(grid):
                    for column_index, value in enumerate(row):
                        # Rtop=1 is the physical threshold. A square-root visual
                        # transfer keeps subcritical fields visible without
                        # normalizing every frame to an artificial red maximum.
                        level = math.sqrt(max(0.0, min(1.0, _float(value))))
                        if level <= 0.01:
                            continue
                        risk_color = self._risk_color(level)
                        risk_color.setAlpha(int(35 + level * 155))
                        painter.fillRect(
                            QRectF(
                                rect.left() + column_index * cell_w,
                                rect.bottom() - (row_index + 1) * cell_h,
                                cell_w + 0.6,
                                cell_h + 0.6,
                            ),
                            risk_color,
                        )
                painter.restore()
                return

        risk_values = result_series(self.result, "top_risk")
        risk = self._result_series_value(risk_values)
        if risk <= 0:
            return
        current = self._result_position()
        panel = get_value(self.condition, "panel", default={})
        width_mm = max(1.0, _float(get_value(panel, "width_mm", default=71.5), 71.5))
        height_mm = max(1.0, _float(get_value(panel, "height_mm", default=149.6), 149.6))
        center = self._map_extended_xy_base(
            current[0], current[1], rect, width_mm, height_mm
        )
        radius = max(22.0, min(rect.width(), rect.height()) * 0.23)
        radial = QLinearGradient(center - QPointF(radius, radius), center + QPointF(radius, radius))
        c = self._risk_color(min(1.0, risk))
        c.setAlpha(min(145, int(35 + risk * 85)))
        radial.setColorAt(0, c)
        transparent = QColor(c)
        transparent.setAlpha(0)
        radial.setColorAt(1, transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(radial)
        painter.drawEllipse(center, radius, radius)

    def _draw_front_and_film(
        self,
        painter: QPainter,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        peel_values = result_series(self.result, "bottom_peel")
        peel = self._result_series_value(peel_values, self.progress)
        peel = max(0.0, min(1.0, peel))
        corner = get_value(
            get_value(self.condition, "pull_tape", default={}),
            "start_corner",
            default="bottom_left",
        )
        plane_transform = self._camera_transform(rect)
        p1, p2 = self._result_front_line(rect, width_mm, height_mm, peel, str(corner))
        p1, p2 = plane_transform.map(p1), plane_transform.map(p2)
        current_xyz = self._result_position()
        grip_base = self._map_extended_xy_base(
            current_xyz[0], current_xyz[1], rect, width_mm, height_mm
        )
        grip_plane = plane_transform.map(grip_base)
        depth_offset = _projected_z_offset(
            current_xyz[2],
            self.z_reference_mm,
            self.camera_elevation_deg,
        )
        grip_lifted = grip_plane + depth_offset
        midpoint = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
        (
            raw_damage_grid,
            front_damage_grid,
            detached_reference_xyz,
        ) = self._bottom_damage_display_fields()
        damage_grid, _lift_grid = _film_peel_fields(
            raw_damage_grid,
            str(corner),
            peel,
        )
        # Display-only geometry: use the solver's current damage front as the
        # root of a 180-degree fold.  This mesh never feeds back into mechanics.
        visual_grip_xyz = (
            (grip_base.x() - rect.left()) / max(rect.width(), 1.0e-9) * width_mm,
            (rect.bottom() - grip_base.y()) / max(rect.height(), 1.0e-9) * height_mm,
            current_xyz[2],
        )
        shape_grip_xyz = visual_grip_xyz
        follow_delta_xyz = (0.0, 0.0, 0.0)
        if detached_reference_xyz is not None:
            reference_base = self._map_extended_xy_base(
                detached_reference_xyz[0],
                detached_reference_xyz[1],
                rect,
                width_mm,
                height_mm,
            )
            reference_visual_xyz = (
                (reference_base.x() - rect.left()) / max(rect.width(), 1.0e-9) * width_mm,
                (rect.bottom() - reference_base.y()) / max(rect.height(), 1.0e-9) * height_mm,
                detached_reference_xyz[2],
            )
            shape_grip_xyz = reference_visual_xyz
            follow_delta_xyz = tuple(
                visual_grip_xyz[index] - reference_visual_xyz[index]
                for index in range(3)
            )
        tape = get_value(self.condition, "pull_tape", default={})
        fold_world = _film_fold_world_grid(
            width_mm,
            height_mm,
            raw_damage_grid or damage_grid,
            str(corner),
            shape_grip_xyz,
            peel,
            pull_tape_width_mm=_float(get_value(tape, "width_mm", default=10.0), 10.0),
            pull_tape_length_mm=_float(get_value(tape, "length_mm", default=10.0), 10.0),
            front_damage_grid=front_damage_grid,
            follow_delta_xyz_mm=follow_delta_xyz,
        )
        surface = _project_film_fold_grid(
            rect,
            fold_world,
            width_mm,
            height_mm,
            plane_transform,
            self.z_reference_mm,
            self.camera_elevation_deg,
        )

        # Paint a single connected sheet. Every quadrilateral shares vertices
        # with its neighbours, so no damage pattern can tear the visual film
        # into independently translated tiles.
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for points, damage, lift_fraction in _film_surface_cells(surface):
            film_color = QColor(COLORS["b"])
            film_color.setAlpha(round(12 + 108 * lift_fraction + 12 * damage))
            painter.setBrush(film_color)
            painter.drawPolygon(QPolygonF(points))

        # Sparse material grid lines make surface curvature and diagonal lift
        # legible without implying that the film consists of separate pieces.
        mesh_pen = QPen(QColor(255, 210, 178, 34), 0.65)
        mesh_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mesh_pen)
        row_stride = max(1, len(surface) // 12)
        column_stride = max(1, len(surface[0]) // 8)
        for row in range(0, len(surface), row_stride):
            painter.drawPolyline(QPolygonF([vertex[0] for vertex in surface[row]]))
        for column in range(0, len(surface[0]), column_stride):
            painter.drawPolyline(QPolygonF([row[column][0] for row in surface]))

        material_boundary = _film_surface_boundary(surface)
        depth_edge_pen = QPen(QColor(236, 157, 105, 92), 1.0, Qt.PenStyle.DotLine)
        depth_edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(depth_edge_pen)
        base_and_surface_corners = (
            (plane_transform.map(rect.bottomLeft()), surface[0][0][0]),
            (plane_transform.map(rect.bottomRight()), surface[0][-1][0]),
            (plane_transform.map(rect.topRight()), surface[-1][-1][0]),
            (plane_transform.map(rect.topLeft()), surface[-1][0][0]),
        )
        for base_corner, surface_corner in base_and_surface_corners:
            if math.hypot(
                surface_corner.x() - base_corner.x(),
                surface_corner.y() - base_corner.y(),
            ) > 1.0:
                painter.drawLine(base_corner, surface_corner)

        # The film is the full sharp-cornered pre-trim sheet.  Keep its
        # material boundary visually distinct from the orange future trim line.
        outline = QColor("#69B5FF")
        outline.setAlpha(205)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        outline_pen = QPen(outline, 1.45, Qt.PenStyle.DashLine)
        outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        outline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline_pen)
        if material_boundary:
            painter.drawPolyline(QPolygonF([*material_boundary, material_boundary[0]]))
        painter.restore()

        anchor_row = -1 if "top" in str(corner) else 0
        anchor_column = -1 if "right" in str(corner) else 0
        material_anchor = surface[anchor_row][anchor_column][0]
        # The small riser is the projected Z component. The full pull-tape
        # ribbon and gripper head are rendered in screen space so their width
        # remains legible without changing the physical trajectory geometry.
        if math.hypot(depth_offset.x(), depth_offset.y()) > 0.5:
            riser_pen = QPen(QColor(176, 190, 205, 105), 1.0, Qt.PenStyle.DotLine)
            riser_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(riser_pen)
            painter.drawLine(grip_plane, grip_lifted)
        if self.show_equipment:
            self._draw_pull_tape_and_head(painter, material_anchor, grip_lifted)
        if self.show_force_vectors:
            self._draw_applied_force_vector(painter, material_anchor)

        # The mechanics solver keeps a PCA-reduced straight segment as a load
        # reference.  It is not the visible peel wave.  Draw the display front
        # from the exact same damage frame as the material tiles so the overlay
        # cannot diverge from the propagated interface state.
        contour_segments = _damage_contour_segments(raw_damage_grid or damage_grid)
        if contour_segments:
            shadow_pen = QPen(QColor(35, 20, 14, 150), 4.2)
            shadow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            shadow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            front_pen = QPen(QColor("#F8D2BD"), 2.1)
            front_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            front_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            for normalized_start, normalized_end in contour_segments:
                base_start = QPointF(
                    rect.left() + normalized_start[0] * rect.width(),
                    rect.bottom() - normalized_start[1] * rect.height(),
                )
                base_end = QPointF(
                    rect.left() + normalized_end[0] * rect.width(),
                    rect.bottom() - normalized_end[1] * rect.height(),
                )
                base_start = plane_transform.map(base_start)
                base_end = plane_transform.map(base_end)
                painter.setPen(shadow_pen)
                painter.drawLine(base_start, base_end)
                painter.setPen(front_pen)
                painter.drawLine(base_start, base_end)
        elif not raw_damage_grid:
            # Compatibility only: legacy results do not contain damage frames.
            front_length = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            if front_length >= 1.0 and peel < 1.0 - 1.0e-6:
                legacy_pen = QPen(QColor("#F8D2BD"), 2.0)
                legacy_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(legacy_pen)
                painter.drawLine(p1, p2)
            elif peel < 1.0 - 1.0e-6:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#F8D2BD"))
                painter.drawEllipse(midpoint, 2.5, 2.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(205, 217, 230, 150))
        painter.drawEllipse(grip_plane, 2.2, 2.2)
        if not self.show_equipment:
            painter.setBrush(QColor(COLORS["b"]))
            painter.drawEllipse(grip_lifted, 4.5, 4.5)

    def _draw_equipment_without_film(
        self,
        painter: QPainter,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        corner = str(
            get_value(
                get_value(self.condition, "pull_tape", default={}),
                "start_corner",
                default="bottom_left",
            )
        )
        transform = self._camera_transform(rect)
        anchor = transform.map(self._panel_corner(rect, corner))
        x, y, z, _speed = self._result_position()
        grip = self._project_grip_xyz(x, y, z, rect, width_mm, height_mm)
        if self.show_equipment:
            self._draw_pull_tape_and_head(painter, anchor, grip)
        if self.show_force_vectors:
            self._draw_applied_force_vector(painter, anchor)

    def _draw_pull_tape_and_head(
        self,
        painter: QPainter,
        material_anchor: QPointF,
        grip: QPointF,
    ) -> None:
        vector = grip - material_anchor
        distance = math.hypot(vector.x(), vector.y())
        if distance <= 1.0e-9:
            vector = QPointF(0.0, -1.0)
            distance = 1.0
        unit = QPointF(vector.x() / distance, vector.y() / distance)
        normal = QPointF(-unit.y(), unit.x())
        tape_width_mm = _float(
            get_value(get_value(self.condition, "pull_tape", default={}), "width_mm", default=8.0),
            8.0,
        )
        shown_width = max(4.0, min(9.0, 3.0 + tape_width_mm * 0.32))
        half = normal * (shown_width * 0.5)
        jaw_center = grip - unit * 7.0
        tape_end = jaw_center - unit * 2.0
        ribbon = QPolygonF(
            [material_anchor + half, tape_end + half, tape_end - half, material_anchor - half]
        )
        tape_gradient = QLinearGradient(material_anchor, tape_end)
        tape_gradient.setColorAt(0.0, QColor(101, 181, 255, 82))
        tape_gradient.setColorAt(0.6, QColor(151, 208, 255, 132))
        tape_gradient.setColorAt(1.0, QColor(213, 235, 250, 176))
        painter.setPen(QPen(QColor(120, 195, 255, 190), 0.9))
        painter.setBrush(tape_gradient)
        painter.drawPolygon(ribbon)
        painter.setPen(QPen(QColor(224, 241, 255, 125), 0.75, Qt.PenStyle.DashLine))
        painter.drawLine(material_anchor, tape_end)

        angle = math.degrees(math.atan2(unit.y(), unit.x())) + 90.0
        painter.save()
        painter.translate(grip)
        painter.rotate(angle)
        body_gradient = QLinearGradient(QPointF(-12, -30), QPointF(12, 8))
        body_gradient.setColorAt(0.0, QColor("#7891A7"))
        body_gradient.setColorAt(0.52, QColor("#334454"))
        body_gradient.setColorAt(1.0, QColor("#18222C"))
        painter.setPen(QPen(QColor("#9DB4C7"), 1.0))
        painter.setBrush(body_gradient)
        painter.drawRoundedRect(QRectF(-11.0, -31.0, 22.0, 27.0), 4.0, 4.0)
        painter.setBrush(QColor("#1C2732"))
        painter.drawRoundedRect(QRectF(-15.0, -6.0, 30.0, 9.0), 3.0, 3.0)
        painter.setBrush(QColor("#B8C8D5"))
        painter.drawRoundedRect(QRectF(-9.0, 1.0, 18.0, 6.0), 2.0, 2.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.accent)
        painter.drawEllipse(QPointF(0.0, -21.0), 3.0, 3.0)
        painter.restore()

        tension = self._result_series_value(result_series(self.result, "tension"))
        label_point = material_anchor + vector * 0.58 + normal * 8.0
        self._draw_callout(
            painter,
            label_point,
            f"풀테이프  T={_compact(tension)} N",
            QColor("#B7DFFF"),
        )

    def _draw_applied_force_vector(self, painter: QPainter, start: QPointF) -> None:
        fx = self._result_series_value(result_series(self.result, "force_x"))
        fy = self._result_series_value(result_series(self.result, "force_y"))
        fz = self._result_series_value(result_series(self.result, "force_z"))
        force = math.sqrt(fx * fx + fy * fy + fz * fz)
        if force <= 1.0e-10:
            return
        projected = _orbit_project(
            fx, fy, fz, self.camera_yaw_deg, self.camera_elevation_deg
        ) - _orbit_project(
            0.0, 0.0, 0.0, self.camera_yaw_deg, self.camera_elevation_deg
        )
        force_history = result_series(self.result, "force")
        length = 24.0 + 26.0 * math.sqrt(
            force / max(max(force_history or [force]), 1.0e-12)
        )
        end = self._draw_vector_arrow(
            painter,
            start,
            projected,
            length,
            QColor("#FFD15A"),
            2.0,
            7.0,
        )
        if end is not None:
            self._draw_callout(
                painter,
                end + QPointF(5.0, 4.0),
                f"F ({_compact(fx)}, {_compact(fy)}, {_compact(fz)}) N",
                QColor("#FFE8A0"),
            )

    def _draw_trajectory(
        self,
        painter: QPainter,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        x_values = result_series(self.result, "position_x")
        y_values = result_series(self.result, "position_y")
        z_values = result_series(self.result, "position_z")
        if (
            len(x_values) < 2
            or len(x_values) != len(y_values)
            or len(x_values) != len(z_values)
        ):
            trajectory = self._trajectory()
            x_values = [point[0] for point in trajectory]
            y_values = [point[1] for point in trajectory]
            z_values = [point[2] for point in trajectory]
        if len(x_values) < 2:
            return
        stride = max(1, len(x_values) // 180)
        sample_indices = list(range(0, len(x_values), stride))
        if sample_indices[-1] != len(x_values) - 1:
            sample_indices.append(len(x_values) - 1)
        points = [
            self._project_grip_xyz(
                x_values[i], y_values[i], z_values[i], rect, width_mm, height_mm
            )
            for i in sample_indices
        ]
        painter.setPen(QPen(self.accent, 1.2, Qt.PenStyle.DashLine))
        painter.drawPolyline(QPolygonF(points))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#F4F7FB"))
        painter.drawEllipse(points[0], 2.6, 2.6)
        painter.setBrush(self.accent)
        painter.drawEllipse(points[-1], 2.6, 2.6)

    def _draw_footer(self, painter: QPainter) -> None:
        risk_values = result_series(self.result, "top_risk")
        peel_values = result_series(self.result, "bottom_peel")
        risk = self._result_series_value(risk_values)
        peel = self._result_series_value(peel_values, self.progress)
        risk_text = f"Rtop {_compact(risk)}" if risk_values else "Rtop —"
        peel_text = f"하면 박리 {peel * 100:.0f}%"
        top_force = self._result_series_value(
            result_series(self.result, "top_interface_force")
        )
        damage_values = result_series(self.result, "top_damage")
        top_damage = self._result_series_value(damage_values)
        y = self.height() - 27
        painter.setPen(color("text_muted"))
        painter.drawText(
            QPointF(15, y),
            f"{risk_text} · 상면 손상 {top_damage:.1f} mm² · Fn {_compact(top_force)} N",
        )
        painter.drawText(
            QRectF(self.width() - 145, y - 15, 130, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            peel_text,
        )
        if risk_values:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._risk_color(min(1.0, risk)))
            painter.drawEllipse(QPointF(93, y - 4), 3.5, 3.5)

        legend_y = 55.0
        entries = (
            (QColor("#70DCEB"), "상면 필름"),
            (QColor(COLORS["b"]), "하면 필름"),
            (QColor("#F05DB2"), "상면 damage"),
            (QColor("#FFD15A"), "Fx/Fy/Fz"),
        )
        x = 15.0
        for marker, label in entries:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(marker)
            painter.drawRoundedRect(QRectF(x, legend_y - 7.0, 9.0, 4.0), 2.0, 2.0)
            painter.setPen(color("text_dim"))
            painter.drawText(QPointF(x + 13.0, legend_y), label)
            x += 13.0 + painter.fontMetrics().horizontalAdvance(label) + 12.0

    def _trajectory(self) -> list[tuple[float, float, float, float]]:
        points = sequence(get_value(self.condition, "trajectory", "points", default=[]))
        result = []
        for point in points:
            result.append(
                (
                    _float(get_value(point, "x_mm", "x", default=0)),
                    _float(get_value(point, "y_mm", "y", default=0)),
                    _float(get_value(point, "z_mm", "z", default=0)),
                    _float(get_value(point, "speed_mm_s", "speed", default=1), 1),
                )
            )
        return result

    def _result_position(self) -> tuple[float, float, float, float]:
        times = result_series(self.result, "time")
        components = [
            result_series(self.result, "position_x"),
            result_series(self.result, "position_y"),
            result_series(self.result, "position_z"),
            result_series(self.result, "speed"),
        ]
        if times and all(len(values) == len(times) for values in components):
            return tuple(
                _interpolate_at_time(times, values, self.selected_time_s)
                for values in components
            )  # type: ignore[return-value]
        return self._trajectory_position(self._trajectory(), self.progress)

    @staticmethod
    def _trajectory_position(
        trajectory: list[tuple[float, float, float, float]], progress: float
    ) -> tuple[float, float, float, float]:
        if not trajectory:
            return (0, 0, 0, 1)
        if len(trajectory) == 1:
            return trajectory[0]
        scaled = max(0.0, min(1.0, progress)) * (len(trajectory) - 1)
        index = min(len(trajectory) - 2, int(scaled))
        local = scaled - index
        first, second = trajectory[index], trajectory[index + 1]
        return tuple(_lerp(first[i], second[i], local) for i in range(4))  # type: ignore[return-value]

    @staticmethod
    def _panel_corner(rect: QRectF, corner: str) -> QPointF:
        return QPointF(
            rect.right() if "right" in corner else rect.left(),
            rect.top() if "top" in corner else rect.bottom(),
        )

    @staticmethod
    def _map_xy(
        x: float, y: float, rect: QRectF, width_mm: float, height_mm: float
    ) -> QPointF:
        px = rect.left() + max(0.0, min(1.0, x / width_mm)) * rect.width()
        py = rect.bottom() - max(0.0, min(1.0, y / height_mm)) * rect.height()
        return QPointF(px, py)

    def _map_extended_xy_base(
        self,
        x: float,
        y: float,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> QPointF:
        """Map off-panel gripper motion onto the unprojected panel plane."""

        x_values = result_series(self.result, "position_x") or [x]
        y_values = result_series(self.result, "position_y") or [y]
        x_min = min(0.0, min(x_values))
        x_max = max(width_mm, max(x_values))
        y_min = min(0.0, min(y_values))
        y_max = max(height_mm, max(y_values))
        margin = min(28.0, max(10.0, rect.left() - 4.0))

        def extended(
            value: float,
            lower: float,
            upper: float,
            pixel_low: float,
            pixel_high: float,
            high_overflow: float,
        ) -> float:
            direction = 1.0 if pixel_high >= pixel_low else -1.0
            if 0.0 <= value <= upper:
                return pixel_low + value / upper * (pixel_high - pixel_low)
            if value < 0.0:
                span = max(-lower, 1.0e-12)
                return pixel_low - direction * margin * min(1.0, -value / span)
            span = max(high_overflow, 1.0e-12)
            return pixel_high + direction * margin * min(1.0, (value - upper) / span)

        px = extended(
            x, x_min, width_mm, rect.left(), rect.right(), x_max - width_mm
        )
        # Y pixels run downward, so compute the physical-axis map then invert.
        py_up = extended(
            y, y_min, height_mm, rect.bottom(), rect.top(), y_max - height_mm
        )
        return QPointF(px, py_up)

    def _map_extended_xy(
        self,
        x: float,
        y: float,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> QPointF:
        """Map an XY point through the current orbit camera."""

        return self._camera_transform(rect).map(
            self._map_extended_xy_base(x, y, rect, width_mm, height_mm)
        )

    def _bounded_view_point(self, point: QPointF) -> QPointF:
        """Keep projected 3-D cues visible without pinning them to the panel."""

        return QPointF(
            max(7.0, min(self.width() - 7.0, point.x())),
            max(43.0, min(self.height() - 40.0, point.y())),
        )

    def _project_grip_xyz(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> QPointF:
        """Project the measured grip XYZ through the same camera as the film."""

        on_plane = self._map_extended_xy(x_mm, y_mm, rect, width_mm, height_mm)
        return self._bounded_view_point(
            on_plane
            + _projected_z_offset(
                z_mm,
                self.z_reference_mm,
                self.camera_elevation_deg,
            )
        )

    @staticmethod
    def _front_line(rect: QRectF, progress: float, corner: str) -> tuple[QPointF, QPointF]:
        # Compatibility fallback for result files created by the v1 solver.
        s = max(0.001, min(1.999, progress * 2.0))
        coordinates: list[tuple[float, float]] = []
        if s <= 1:
            coordinates = [(0, s), (s, 0)]
        else:
            coordinates = [(s - 1, 1), (1, s - 1)]
        transformed = []
        for x, y in coordinates:
            if "right" in corner:
                x = 1 - x
            if "top" in corner:
                y = 1 - y
            transformed.append(QPointF(rect.left() + x * rect.width(), rect.bottom() - y * rect.height()))
        return transformed[0], transformed[1]

    def _result_front_line(
        self,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
        peel: float,
        corner: str,
    ) -> tuple[QPointF, QPointF]:
        segments = sequence(
            get_value(
                self.result,
                "front_segments_mm",
                "peel_front_segments",
                default=[],
            )
        )
        if segments:
            times = result_series(self.result, "time")
            if len(times) == len(segments):
                first, second, fraction = _time_bracket(times, self.selected_time_s)
                left = sequence(segments[first])
                right = sequence(segments[second])
                segment = [
                    _lerp(_float(left[i]), _float(right[i]), fraction)
                    for i in range(min(len(left), len(right)))
                ]
            else:
                index = min(
                    len(segments) - 1,
                    round(self.progress * (len(segments) - 1)),
                )
                segment = sequence(segments[index])
            if len(segment) >= 4:
                return (
                    self._map_xy(
                        _float(segment[0]),
                        _float(segment[1]),
                        rect,
                        width_mm,
                        height_mm,
                    ),
                    self._map_xy(
                        _float(segment[2]),
                        _float(segment[3]),
                        rect,
                        width_mm,
                        height_mm,
                    ),
                )
        return self._front_line(rect, peel, corner)

    def _frame_grid(self, *names: str) -> list[list[float]]:
        frames = sequence(get_value(self.result, *names, default=[]))
        if not frames:
            return []
        frame_indices = [int(value) for value in sequence(get_value(self.result, "frame_indices", default=[]))]
        times = result_series(self.result, "time")
        if len(frame_indices) == len(frames) and times:
            frame_times = [times[min(max(index, 0), len(times) - 1)] for index in frame_indices]
            index = min(
                range(len(frame_times)),
                key=lambda item: abs(frame_times[item] - self.selected_time_s),
            )
        else:
            index = min(len(frames) - 1, round(self.progress * (len(frames) - 1)))
        frame = frames[index]
        if hasattr(frame, "tolist"):
            try:
                frame = frame.tolist()
            except Exception:
                return []
        if not isinstance(frame, (list, tuple)):
            return []
        # The public solver stores frame data as a compact flattened node array.
        # Reconstruct its (ny, nx) display grid when mesh_shape is available.
        if frame and not isinstance(frame[0], (list, tuple)):
            shape = sequence(get_value(self.result, "mesh_shape", default=[]))
            if len(shape) == 2:
                rows_count, columns_count = int(shape[0]), int(shape[1])
                if rows_count > 0 and columns_count > 0 and len(frame) >= rows_count * columns_count:
                    return [
                        [_float(value) for value in frame[row * columns_count : (row + 1) * columns_count]]
                        for row in range(rows_count)
                    ]
        rows = []
        for row in frame:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (list, tuple)):
                rows.append([_float(value) for value in row])
        return rows

    def _bottom_damage_display_fields(
        self,
    ) -> tuple[
        list[list[float]],
        list[list[float]],
        tuple[float, float, float] | None,
    ]:
        """Return current damage and the last physical front for full release.

        A uniformly detached frame has no active contour.  The last preceding
        non-uniform solver frame supplies the U-fold shape, and its grip pose is
        returned so the whole detached mesh can follow the current head pose.
        This is display history only; solver state is never modified.
        """

        current = self._frame_grid("bottom_damage_frames")
        if (
            not current
            or _damage_contour_segments(current)
            or not all(value >= 0.5 for row in current for value in row)
        ):
            return current, current, None

        frames = sequence(get_value(self.result, "bottom_damage_frames", default=[]))
        if not frames:
            return current, current, None
        frame_indices = [
            int(value)
            for value in sequence(get_value(self.result, "frame_indices", default=[]))
        ]
        times = result_series(self.result, "time")
        if len(frame_indices) == len(frames) and times:
            frame_times = [
                times[min(max(index, 0), len(times) - 1)] for index in frame_indices
            ]
            current_frame = min(
                range(len(frame_times)),
                key=lambda item: abs(frame_times[item] - self.selected_time_s),
            )
        else:
            current_frame = min(
                len(frames) - 1,
                round(self.progress * (len(frames) - 1)),
            )

        shape = sequence(get_value(self.result, "mesh_shape", default=[]))
        for frame_number in range(current_frame - 1, -1, -1):
            frame = frames[frame_number]
            if hasattr(frame, "tolist"):
                frame = frame.tolist()
            candidate: list[list[float]] = []
            if isinstance(frame, (list, tuple)) and frame:
                if isinstance(frame[0], (list, tuple)):
                    candidate = [
                        [_float(value) for value in row]
                        for row in frame
                        if isinstance(row, (list, tuple))
                    ]
                elif len(shape) == 2:
                    rows_count, columns_count = int(shape[0]), int(shape[1])
                    if len(frame) >= rows_count * columns_count:
                        candidate = [
                            [
                                _float(value)
                                for value in frame[
                                    row * columns_count : (row + 1) * columns_count
                                ]
                            ]
                            for row in range(rows_count)
                        ]
            if not candidate or not _damage_contour_segments(candidate):
                continue

            solver_index = (
                frame_indices[frame_number]
                if len(frame_indices) == len(frames)
                else round(frame_number / max(len(frames) - 1, 1) * max(len(times) - 1, 0))
            )
            positions = sequence(get_value(self.result, "position_xyz_mm", default=[]))
            if positions:
                pose = positions[min(max(solver_index, 0), len(positions) - 1)]
                if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                    return (
                        current,
                        candidate,
                        (_float(pose[0]), _float(pose[1]), _float(pose[2])),
                    )
            return current, candidate, None
        return current, current, None

    @staticmethod
    def _series_value(values: list[float], progress: float, fallback: float = 0.0) -> float:
        if not values:
            return fallback
        index = min(len(values) - 1, round(progress * (len(values) - 1)))
        return values[index]

    def _result_series_value(self, values: list[float], fallback: float = 0.0) -> float:
        return _interpolate_at_time(
            result_series(self.result, "time"),
            values,
            self.selected_time_s,
            fallback,
        )

    @staticmethod
    def _risk_color(level: float) -> QColor:
        level = max(0.0, min(1.0, level))
        if level < 0.55:
            local = level / 0.55
            return QColor(
                round(_lerp(85, 243, local)),
                round(_lerp(214, 201, local)),
                round(_lerp(190, 105, local)),
            )
        local = (level - 0.55) / 0.45
        return QColor(
            round(_lerp(243, 240, local)),
            round(_lerp(201, 113, local)),
            round(_lerp(105, 120, local)),
        )


class LineChart(QWidget):
    """Small synchronized A/B line chart without a QtCharts dependency."""

    def __init__(
        self,
        title: str,
        series_key: str,
        unit: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.series_key = series_key
        self.unit = unit
        self.result_a: Any = None
        self.result_b: Any = None
        self.progress = 0.0
        self.selected_time_s = 0.0
        self.time_start_s = 0.0
        self.time_end_s = 1.0
        self.setMinimumHeight(190)

    def set_results(self, result_a: Any, result_b: Any) -> None:
        self.result_a = result_a
        self.result_b = result_b
        self.update()

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, progress))
        self.selected_time_s = _lerp(
            self.time_start_s, self.time_end_s, self.progress
        )
        self.update()

    def set_time_range(self, start_s: float, end_s: float) -> None:
        self.time_start_s = float(start_s)
        self.time_end_s = max(float(end_s), self.time_start_s + 1.0e-12)
        self.update()

    def set_time(self, selected_time_s: float, global_progress: float | None = None) -> None:
        self.selected_time_s = float(selected_time_s)
        if global_progress is not None:
            self.progress = max(0.0, min(1.0, float(global_progress)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), color("surface"))
        painter.setPen(QPen(color("border_soft"), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
        painter.setPen(color("text"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QPointF(14, 24), self.title)

        data_a = result_series(self.result_a, self.series_key)
        data_b = result_series(self.result_b, self.series_key)
        plot = QRectF(59, 39, self.width() - 75, self.height() - 69)
        if not data_a and not data_b:
            painter.setPen(color("text_dim"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "실행 후 그래프가 표시됩니다")
            return

        combined = data_a + data_b
        minimum = min(combined)
        maximum = max(combined)
        if math.isclose(minimum, maximum):
            pad = max(0.1, abs(maximum) * 0.1)
            minimum -= pad
            maximum += pad
        else:
            pad = (maximum - minimum) * 0.12
            minimum = min(0.0, minimum - pad)
            maximum += pad

        self._draw_grid(painter, plot, minimum, maximum)
        self._draw_series(
            painter,
            plot,
            result_series(self.result_a, "time"),
            data_a,
            minimum,
            maximum,
            QColor(COLORS["a"]),
        )
        self._draw_series(
            painter,
            plot,
            result_series(self.result_b, "time"),
            data_b,
            minimum,
            maximum,
            QColor(COLORS["b"]),
        )
        cursor_fraction = (self.selected_time_s - self.time_start_s) / (
            self.time_end_s - self.time_start_s
        )
        cursor_x = plot.left() + max(0.0, min(1.0, cursor_fraction)) * plot.width()
        painter.setPen(QPen(QColor(255, 255, 255, 65), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(cursor_x, plot.top()), QPointF(cursor_x, plot.bottom()))
        self._draw_legend(painter)

    def _draw_grid(self, painter: QPainter, plot: QRectF, minimum: float, maximum: float) -> None:
        painter.setPen(QPen(color("grid", 110), 0.8))
        for index in range(4):
            y = plot.top() + index * plot.height() / 3
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            value = maximum - index * (maximum - minimum) / 3
            painter.setPen(color("text_dim"))
            label = f"{_compact(value)}{self.unit}"
            painter.drawText(
                QRectF(3, y - 9, 52, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(QPen(color("grid", 110), 0.8))

    def _draw_series(
        self,
        painter: QPainter,
        plot: QRectF,
        times: list[float],
        values: list[float],
        minimum: float,
        maximum: float,
        series_color: QColor,
    ) -> None:
        if not values:
            return
        path = QPainterPath()
        for index, value in enumerate(values):
            if len(times) == len(values):
                time_fraction = (times[index] - self.time_start_s) / (
                    self.time_end_s - self.time_start_s
                )
            else:
                time_fraction = index / max(1, len(values) - 1)
            x = plot.left() + max(0.0, min(1.0, time_fraction)) * plot.width()
            y = plot.bottom() - ((value - minimum) / (maximum - minimum)) * plot.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(series_color, 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _draw_legend(self, painter: QPainter) -> None:
        y = self.height() - 18
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["a"]))
        painter.drawEllipse(QPointF(15, y - 3), 3, 3)
        painter.setPen(color("text_muted"))
        painter.drawText(QPointF(23, y), "A")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["b"]))
        painter.drawEllipse(QPointF(47, y - 3), 3, 3)
        painter.setPen(color("text_muted"))
        painter.drawText(QPointF(55, y), "B")
