from __future__ import annotations

import numpy as np
import pytest

from lamination_sim.models import AssumptionSet, TrajectoryPoint
from lamination_sim.presets import (
    MEASURED_TRAJECTORY_A,
    MEASURED_TRAJECTORY_B,
    default_condition,
    measured_project,
)
from lamination_sim.simulation import simulate


def _trajectory_values(condition) -> list[tuple[float, float, float, float]]:
    return [
        (point.x_mm, point.y_mm, point.z_mm, point.speed_mm_s)
        for point in condition.trajectory
    ]


def _assert_same_prefix(first, second, end: int) -> None:
    fields = (
        "time_s",
        "position_xyz_mm",
        "speed_mm_s",
        "trajectory_progress",
        "peel_angle_deg",
        "force_xyz_n",
        "moment_xyz_n_mm",
        "bottom_peel_ratio",
        "top_peak_risk",
        "panel_max_lift_mm",
        "panel_twist_mm",
    )
    for field in fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(first, field))[:end],
            np.asarray(getattr(second, field))[:end],
            err_msg=field,
        )


def test_supplied_ab_data_are_accepted_without_z_normalization() -> None:
    project = measured_project()

    assert _trajectory_values(project.condition_a) == list(MEASURED_TRAJECTORY_A)
    assert _trajectory_values(project.condition_b) == list(MEASURED_TRAJECTORY_B)
    assert project.condition_a.trajectory[0].speed_mm_s == 0.0
    assert project.condition_b.trajectory[0].speed_mm_s == 0.0
    np.testing.assert_allclose(
        [point.z_mm for point in project.condition_b.trajectory[:5]],
        np.asarray([point.z_mm for point in project.condition_a.trajectory[:5]])
        + 10.0,
    )
    np.testing.assert_allclose(
        np.diff([point.z_mm for point in project.condition_a.trajectory[:5]]),
        np.diff([point.z_mm for point in project.condition_b.trajectory[:5]]),
    )


def test_p6_only_change_cannot_change_p1_through_p4_physics() -> None:
    baseline = measured_project().condition_a
    changed = baseline.model_copy(deep=True)
    changed.trajectory[-1].x_mm += 100.0
    changed.trajectory[-1].y_mm += 50.0
    changed.trajectory[-1].z_mm += 25.0
    changed.trajectory[-1].speed_mm_s = 100.0

    first = simulate(baseline, AssumptionSet(), "coarse")
    second = simulate(changed, AssumptionSet(), "coarse")
    end = first.trajectory_waypoint_indices[3] + 1

    _assert_same_prefix(first, second, end)


def test_identical_p1_p4_history_is_independent_of_later_path() -> None:
    baseline = measured_project().condition_a
    changed = baseline.model_copy(deep=True)
    changed.trajectory[4].x_mm = 60.0
    changed.trajectory[4].y_mm = 40.0
    changed.trajectory[4].z_mm = 30.0
    changed.trajectory[4].speed_mm_s = 12.0
    changed.trajectory[5].x_mm = 200.0
    changed.trajectory[5].y_mm = 100.0
    changed.trajectory[5].z_mm = 80.0
    changed.trajectory[5].speed_mm_s = 75.0

    first = simulate(baseline, AssumptionSet(), "coarse")
    second = simulate(changed, AssumptionSet(), "coarse")
    end = first.trajectory_waypoint_indices[3] + 1

    _assert_same_prefix(first, second, end)


def test_ten_mm_absolute_z_offset_changes_initial_angle_and_vertical_force() -> None:
    low = measured_project().condition_a
    high = low.model_copy(deep=True)
    for point in high.trajectory:
        point.z_mm += 10.0

    low_result = simulate(low, AssumptionSet(), "coarse")
    high_result = simulate(high, AssumptionSet(), "coarse")

    assert low_result.position_xyz_mm[0][2] == pytest.approx(4.0)
    assert high_result.position_xyz_mm[0][2] == pytest.approx(14.0)
    assert high_result.peel_angle_deg[0] != pytest.approx(
        low_result.peel_angle_deg[0]
    )
    assert high_result.force_xyz_n[0][2] != pytest.approx(
        low_result.force_xyz_n[0][2]
    )


def test_optional_initial_approach_is_explicit_p1_pre_history() -> None:
    condition = default_condition()
    p1 = condition.trajectory[0]
    condition.initial_approach = [
        TrajectoryPoint(
            x_mm=p1.x_mm - 2.0,
            y_mm=p1.y_mm,
            z_mm=p1.z_mm,
            speed_mm_s=2.0,
        ),
        p1.model_copy(deep=True),
    ]

    result = simulate(condition, AssumptionSet(), "coarse")

    assert result.initial_state_mode == "specified_approach"
    assert result.main_trajectory_start_index > 0
    assert min(result.time_s) < 0.0
    assert result.time_s[result.main_trajectory_start_index] == pytest.approx(0.0)
    assert result.trajectory_waypoint_indices[0] == result.main_trajectory_start_index
