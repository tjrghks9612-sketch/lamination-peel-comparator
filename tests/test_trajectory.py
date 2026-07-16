from __future__ import annotations

import numpy as np

from lamination_sim.models import TrajectoryPoint
from lamination_sim.trajectory import interpolate_trajectory, waypoint_times


def points() -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(
            x_mm=float(i),
            y_mm=float(2 * i),
            z_mm=float(i) / 2,
            speed_mm_s=5 + i,
        )
        for i in range(6)
    ]


def test_waypoint_time_uses_both_endpoint_target_speeds() -> None:
    values = points()
    times = waypoint_times(values)
    distance = np.sqrt(1.0**2 + 2.0**2 + 0.5**2)
    assert np.isclose(
        times[1],
        2.0 * distance / (values[0].speed_mm_s + values[1].speed_mm_s),
    )
    assert np.isclose(
        times[-1] - times[-2],
        2.0 * distance / (values[4].speed_mm_s + values[5].speed_mm_s),
    )


def test_zero_speed_p1_accelerates_to_p2_target() -> None:
    values = [
        TrajectoryPoint(
            x_mm=float(10 * index),
            y_mm=0.0,
            z_mm=4.0,
            speed_mm_s=float(10 * index),
        )
        for index in range(6)
    ]
    series = interpolate_trajectory(values, samples=51)
    p2_index = int(series.waypoint_indices[1])
    halfway_index = p2_index // 2

    assert series.speed_mm_s[0] == 0.0
    assert series.speed_mm_s[p2_index] == 10.0
    assert series.speed_mm_s[halfway_index] == 5.0
    # Constant acceleration from rest covers one quarter of the distance at
    # half the segment time.
    assert series.xyz_mm[halfway_index, 0] == 2.5
    assert series.waypoint_times_s[1] == 2.0


def test_interpolation_keeps_waypoints_and_causal_path_parameter() -> None:
    values = points()
    series = interpolate_trajectory(values, samples=31)
    expected = np.asarray(
        [(point.x_mm, point.y_mm, point.z_mm) for point in values], dtype=float
    )

    np.testing.assert_allclose(series.xyz_mm[series.waypoint_indices], expected)
    np.testing.assert_allclose(
        series.path_parameter[series.waypoint_indices], np.arange(6, dtype=float)
    )
    assert np.all(np.diff(series.path_parameter) >= 0.0)


def test_changing_p6_does_not_resample_p1_through_p4() -> None:
    first = points()
    second = [point.model_copy(deep=True) for point in first]
    second[-1].x_mm = 500.0
    second[-1].y_mm = -200.0
    second[-1].z_mm = 80.0
    second[-1].speed_mm_s = 500.0

    first_series = interpolate_trajectory(first, samples=31)
    second_series = interpolate_trajectory(second, samples=31)
    end = int(first_series.waypoint_indices[3]) + 1

    np.testing.assert_array_equal(first_series.time_s[:end], second_series.time_s[:end])
    np.testing.assert_array_equal(first_series.xyz_mm[:end], second_series.xyz_mm[:end])
    np.testing.assert_array_equal(
        first_series.speed_mm_s[:end], second_series.speed_mm_s[:end]
    )
    np.testing.assert_array_equal(
        first_series.path_parameter[:end], second_series.path_parameter[:end]
    )
