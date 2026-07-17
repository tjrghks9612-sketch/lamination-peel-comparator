"""Dependency-light visualizations rendered with QPainter."""

from __future__ import annotations

import math
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
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .core_bridge import get_value, result_series, sequence
from .theme import COLORS, color


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


class PeelView(QWidget):
    """Top-down engineering view with a compact pseudo-3D peeled-film cue."""

    progress_requested = Signal(float)

    def __init__(self, label: str, accent: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.accent = QColor(accent)
        self.condition: Any = None
        self.result: Any = None
        self.progress = 0.0
        self.selected_time_s = 0.0
        self.z_reference_mm = 1.0
        self.setMinimumSize(330, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("마우스 휠 또는 하단 타임라인으로 프레임을 이동합니다.")

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

        self._draw_panel_shadow(painter, panel_rect)
        self._draw_panel(painter, panel_rect)
        self._draw_risk_field(painter, panel_rect)
        self._draw_mesh(painter, panel_rect)
        self._draw_front_and_film(painter, panel_rect, width_mm, height_mm)
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
        painter.drawText(QPointF(52, 30), f"조건 {self.label} · 상면 위험도")

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
        available = QRectF(34, 54, self.width() - 68, self.height() - 105)
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

    def _draw_panel_shadow(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 85))
        painter.drawRoundedRect(rect.translated(5, 7), 16, 16)

    def _draw_panel(self, painter: QPainter, rect: QRectF) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#222B37"))
        gradient.setColorAt(0.55, QColor("#151C25"))
        gradient.setColorAt(1, QColor("#10161D"))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#435064"), 1.2))
        painter.drawRoundedRect(rect, 15, 15)
        inset = rect.adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor(255, 255, 255, 16), 1))
        painter.drawRoundedRect(inset, 12, 12)

    def _draw_mesh(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(1, 1, -1, -1), 14, 14)
        painter.setClipPath(path)
        painter.setPen(QPen(color("grid", 95), 0.7))
        for index in range(1, 8):
            x = rect.left() + rect.width() * index / 8
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(1, 14):
            y = rect.top() + rect.height() * index / 14
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.restore()

    def _draw_risk_field(self, painter: QPainter, rect: QRectF) -> None:
        grid = self._frame_grid("top_risk_frames", "risk_frames", "top_damage_frames")
        if grid:
            rows = len(grid)
            columns = max((len(row) for row in grid), default=0)
            if rows and columns:
                painter.save()
                clip = QPainterPath()
                clip.addRoundedRect(rect, 15, 15)
                painter.setClipPath(clip)
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
        center = self._map_extended_xy(current[0], current[1], rect, width_mm, height_mm)
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
        p1, p2 = self._result_front_line(
            rect, width_mm, height_mm, peel, str(corner)
        )
        current_xyz = self._result_position()
        grip = self._map_extended_xy(current_xyz[0], current_xyz[1], rect, width_mm, height_mm)
        z_scale = 34.0 * min(1.0, abs(current_xyz[2]) / self.z_reference_mm)
        vertical_room = max(grip.y() - 46.0, 0.0)
        shown_vertical_lift = min(z_scale, vertical_room)
        compressed_lift = z_scale - shown_vertical_lift
        grip_lifted = grip + QPointF(
            z_scale * 0.35 + compressed_lift * 0.65,
            -shown_vertical_lift,
        )
        midpoint = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)

        if math.hypot(p2.x() - p1.x(), p2.y() - p1.y()) < 1.0:
            # At an unresolved point front (including completed peel), a
            # closed pseudo-sheet degenerates into a misleading triangle.
            painter.setPen(QPen(QColor(COLORS["b"]), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(p1, grip_lifted)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["b"]))
            painter.drawEllipse(grip_lifted, 4.5, 4.5)
            return

        film = QPainterPath()
        film.moveTo(p1)
        film.quadTo(midpoint + QPointF(0, -z_scale * 0.35), grip_lifted)
        film.quadTo(midpoint + QPointF(10, -z_scale * 0.15), p2)
        film.closeSubpath()
        gradient = QLinearGradient(midpoint, grip_lifted)
        gradient.setColorAt(0, QColor(245, 158, 106, 55))
        gradient.setColorAt(1, QColor(245, 158, 106, 170))
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(COLORS["b"]), 1.2))
        painter.drawPath(film)

        painter.setPen(QPen(QColor("#F8D2BD"), 2.0))
        painter.drawLine(p1, p2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["b"]))
        painter.drawEllipse(grip_lifted, 4.5, 4.5)

    def _draw_trajectory(
        self,
        painter: QPainter,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> None:
        x_values = result_series(self.result, "position_x")
        y_values = result_series(self.result, "position_y")
        if len(x_values) < 2 or len(x_values) != len(y_values):
            trajectory = self._trajectory()
            x_values = [point[0] for point in trajectory]
            y_values = [point[1] for point in trajectory]
        if len(x_values) < 2:
            return
        stride = max(1, len(x_values) // 180)
        sample_indices = list(range(0, len(x_values), stride))
        if sample_indices[-1] != len(x_values) - 1:
            sample_indices.append(len(x_values) - 1)
        points = [
            self._map_extended_xy(x_values[i], y_values[i], rect, width_mm, height_mm)
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
        y = self.height() - 27
        painter.setPen(color("text_muted"))
        painter.drawText(QPointF(15, y), risk_text)
        painter.drawText(
            QRectF(self.width() - 145, y - 15, 130, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            peel_text,
        )
        if risk_values:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._risk_color(min(1.0, risk)))
            painter.drawEllipse(QPointF(93, y - 4), 3.5, 3.5)

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
    def _map_xy(
        x: float, y: float, rect: QRectF, width_mm: float, height_mm: float
    ) -> QPointF:
        px = rect.left() + max(0.0, min(1.0, x / width_mm)) * rect.width()
        py = rect.bottom() - max(0.0, min(1.0, y / height_mm)) * rect.height()
        return QPointF(px, py)

    def _map_extended_xy(
        self,
        x: float,
        y: float,
        rect: QRectF,
        width_mm: float,
        height_mm: float,
    ) -> QPointF:
        """Map off-panel gripper motion into the view without edge pinning."""

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
