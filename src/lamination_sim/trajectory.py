"""Piecewise-linear six-point trajectory interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import TrajectoryPoint


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrajectorySeries:
    time_s: FloatArray
    xyz_mm: FloatArray
    speed_mm_s: FloatArray
    waypoint_times_s: FloatArray

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])


def waypoint_times(points: list[TrajectoryPoint]) -> FloatArray:
    if len(points) != 6:
        raise ValueError("trajectory must contain exactly 6 points")
    xyz = np.asarray([(p.x_mm, p.y_mm, p.z_mm) for p in points], dtype=float)
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speeds = np.asarray([p.speed_mm_s for p in points[:-1]], dtype=float)
    if np.any(distances <= 0.0) or np.any(speeds <= 0.0):
        raise ValueError("trajectory segments and speeds must be positive")
    return np.concatenate(([0.0], np.cumsum(distances / speeds)))


def interpolate_trajectory(
    points: list[TrajectoryPoint], samples: int = 101
) -> TrajectorySeries:
    """Interpolate the exact polyline on a uniform time grid.

    A point's speed applies to the segment starting at that point.  P6 speed is
    retained by the input schema but is not used by this v1 interpretation.
    """

    if samples < 2:
        raise ValueError("samples must be at least 2")
    xyz = np.asarray([(p.x_mm, p.y_mm, p.z_mm) for p in points], dtype=float)
    speeds = np.asarray([p.speed_mm_s for p in points], dtype=float)
    way_times = waypoint_times(points)
    time = np.linspace(0.0, way_times[-1], samples, dtype=float)
    segments = np.searchsorted(way_times[1:], time, side="right")
    segments = np.minimum(segments, len(points) - 2)
    t0 = way_times[segments]
    t1 = way_times[segments + 1]
    fraction = np.divide(time - t0, t1 - t0, out=np.zeros_like(time), where=t1 > t0)
    positions = xyz[segments] + fraction[:, None] * (
        xyz[segments + 1] - xyz[segments]
    )
    positions[0] = xyz[0]
    positions[-1] = xyz[-1]
    segment_speeds = speeds[segments]
    segment_speeds[-1] = speeds[-2]
    return TrajectorySeries(time, positions, segment_speeds, way_times)


def projected_progress(xyz_mm: FloatArray) -> FloatArray:
    """Monotone diagonal progress from the XY projection of P1 toward P6."""

    xy = np.asarray(xyz_mm[:, :2], dtype=float)
    direction = xy[-1] - xy[0]
    denominator = float(direction @ direction)
    if denominator <= 1.0e-15:
        increments = np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        raw = cumulative / max(float(cumulative[-1]), 1.0e-15)
    else:
        raw = ((xy - xy[0]) @ direction) / denominator
    raw = np.clip(raw, 0.0, 1.0)
    progress = np.maximum.accumulate(raw)
    progress[0] = 0.0
    progress[-1] = 1.0
    return progress

