from __future__ import annotations

import numpy as np

from lamination_sim.models import TrajectoryPoint
from lamination_sim.trajectory import (
    interpolate_trajectory,
    projected_progress,
    waypoint_times,
)


def points():
    return [
        TrajectoryPoint(x_mm=float(i), y_mm=float(2 * i), z_mm=float(i) / 2, speed_mm_s=5 + i)
        for i in range(6)
    ]


def test_waypoint_time_uses_outbound_speed():
    values = points()
    times = waypoint_times(values)
    distance = np.sqrt(1.0**2 + 2.0**2 + 0.5**2)
    assert np.isclose(times[1], distance / values[0].speed_mm_s)
    assert np.isclose(times[-1] - times[-2], distance / values[4].speed_mm_s)


def test_interpolation_keeps_endpoints_and_progress_monotone():
    values = points()
    series = interpolate_trajectory(values, samples=31)
    assert np.allclose(series.xyz_mm[0], [0.0, 0.0, 0.0])
    assert np.allclose(series.xyz_mm[-1], [5.0, 10.0, 2.5])
    progress = projected_progress(series.xyz_mm)
    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert np.all(np.diff(progress) >= 0.0)

