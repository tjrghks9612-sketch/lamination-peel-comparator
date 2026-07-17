"""Causal interpolation of waypoint-targeted 3-D grip trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import TrajectoryPoint


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrajectorySeries:
    time_s: FloatArray
    xyz_mm: FloatArray
    speed_mm_s: FloatArray
    waypoint_times_s: FloatArray
    waypoint_indices: IntArray
    path_parameter: FloatArray

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])


def _point_arrays(points: list[TrajectoryPoint]) -> tuple[FloatArray, FloatArray]:
    if len(points) < 2:
        raise ValueError("trajectory must contain at least 2 points")
    xyz = np.asarray([(p.x_mm, p.y_mm, p.z_mm) for p in points], dtype=float)
    speeds = np.asarray([p.speed_mm_s for p in points], dtype=float)
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(speeds)):
        raise ValueError("trajectory coordinates and speeds must be finite")
    if np.any(distances <= 0.0):
        raise ValueError("trajectory segments must have positive length")
    if np.any(speeds < 0.0):
        raise ValueError("waypoint target speeds must be non-negative")
    if np.any((speeds[:-1] + speeds[1:]) <= 0.0):
        raise ValueError(
            "a moving segment cannot have zero target speed at both endpoints"
        )
    return xyz, speeds


def waypoint_times(points: list[TrajectoryPoint]) -> FloatArray:
    """Return waypoint times for linear target-speed interpolation.

    For a segment of length ``L`` with endpoint target speeds ``v0`` and
    ``v1``, the acceleration is constant and the duration is
    ``2 L / (v0 + v1)``.  A zero-speed P1 is therefore a valid stationary
    start as long as P2 has a positive target speed.
    """

    xyz, speeds = _point_arrays(points)
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    durations = 2.0 * distances / (speeds[:-1] + speeds[1:])
    return np.concatenate(([0.0], np.cumsum(durations)))


def _interval_counts(samples: int, segment_count: int) -> IntArray:
    if samples < segment_count + 1:
        raise ValueError(
            "samples must be large enough to include every trajectory waypoint"
        )
    quotient, remainder = divmod(samples - 1, segment_count)
    counts = np.full(segment_count, quotient, dtype=np.int64)
    counts[:remainder] += 1
    return counts


def interpolate_trajectory(
    points: list[TrajectoryPoint],
    samples: int = 101,
    *,
    segment_intervals: NDArray[np.int64] | list[int] | None = None,
) -> TrajectorySeries:
    """Interpolate a polyline using each waypoint's target speed.

    Samples are allocated by segment ordinal, never by total path duration or
    the P1-to-P6 vector.  Consequently, changing a future waypoint cannot
    resample or otherwise change an already completed segment.  Within each
    segment speed varies linearly in time; position follows the corresponding
    constant-acceleration kinematics along the local straight segment.
    """

    xyz, speeds = _point_arrays(points)
    way_times = waypoint_times(points)
    segment_count = len(points) - 1
    if segment_intervals is None:
        counts = _interval_counts(samples, segment_count)
    else:
        counts = np.asarray(segment_intervals, dtype=np.int64)
        if counts.shape != (segment_count,) or np.any(counts < 1):
            raise ValueError(
                "segment_intervals must provide one positive count per segment"
            )
    waypoint_indices = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )

    time_parts: list[FloatArray] = []
    position_parts: list[FloatArray] = []
    speed_parts: list[FloatArray] = []
    parameter_parts: list[FloatArray] = []
    for segment, count_value in enumerate(counts):
        count = int(count_value)
        duration = float(way_times[segment + 1] - way_times[segment])
        local_time = np.linspace(0.0, duration, count + 1, dtype=float)
        acceleration = (speeds[segment + 1] - speeds[segment]) / duration
        travelled = (
            speeds[segment] * local_time
            + 0.5 * acceleration * local_time**2
        )
        segment_length = float(np.linalg.norm(xyz[segment + 1] - xyz[segment]))
        fraction = np.clip(travelled / segment_length, 0.0, 1.0)
        position = xyz[segment] + fraction[:, None] * (
            xyz[segment + 1] - xyz[segment]
        )
        speed = speeds[segment] + acceleration * local_time
        parameter = segment + fraction
        if segment:
            local_time = local_time[1:]
            position = position[1:]
            speed = speed[1:]
            parameter = parameter[1:]
        time_parts.append(way_times[segment] + local_time)
        position_parts.append(position)
        speed_parts.append(speed)
        parameter_parts.append(parameter)

    time = np.concatenate(time_parts)
    positions = np.concatenate(position_parts, axis=0)
    interpolated_speeds = np.concatenate(speed_parts)
    path_parameter = np.concatenate(parameter_parts)
    positions[waypoint_indices] = xyz
    interpolated_speeds[waypoint_indices] = speeds
    time[waypoint_indices] = way_times
    path_parameter[waypoint_indices] = np.arange(len(points), dtype=float)
    return TrajectorySeries(
        time_s=time,
        xyz_mm=positions,
        speed_mm_s=interpolated_speeds,
        waypoint_times_s=way_times,
        waypoint_indices=waypoint_indices,
        path_parameter=path_parameter,
    )
