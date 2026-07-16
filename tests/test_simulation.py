from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from lamination_sim.comparison import compare
from lamination_sim.models import AssumptionSet
from lamination_sim.presets import default_condition, default_project
from lamination_sim.simulation import simulate
from lamination_sim.trajectory import interpolate_trajectory, waypoint_times


def test_condition_requires_exactly_six_distinct_points() -> None:
    condition = default_condition()
    short_data = condition.model_dump()
    short_data["trajectory"] = short_data["trajectory"][:-1]
    with pytest.raises(ValidationError, match="exactly 6"):
        type(condition).model_validate(short_data)
    data = condition.model_dump()
    data["trajectory"][1] = data["trajectory"][0]
    with pytest.raises(ValidationError, match="must be distinct"):
        type(condition).model_validate(data)


def test_piecewise_linear_trajectory_uses_start_point_segment_speed() -> None:
    condition = default_condition()
    expected_times = waypoint_times(condition.trajectory)
    series = interpolate_trajectory(condition.trajectory, samples=51)
    first_distance = np.linalg.norm(
        np.asarray(
            [
                condition.trajectory[1].x_mm - condition.trajectory[0].x_mm,
                condition.trajectory[1].y_mm - condition.trajectory[0].y_mm,
                condition.trajectory[1].z_mm - condition.trajectory[0].z_mm,
            ]
        )
    )
    assert expected_times[1] == pytest.approx(
        first_distance / condition.trajectory[0].speed_mm_s
    )
    assert series.xyz_mm[0].tolist() == pytest.approx(
        [
            condition.trajectory[0].x_mm,
            condition.trajectory[0].y_mm,
            condition.trajectory[0].z_mm,
        ]
    )
    assert series.xyz_mm[-1].tolist() == pytest.approx(
        [
            condition.trajectory[-1].x_mm,
            condition.trajectory[-1].y_mm,
            condition.trajectory[-1].z_mm,
        ]
    )


def test_simulation_is_deterministic_and_damage_is_irreversible() -> None:
    condition = default_condition()
    condition.top_film.adhesion_gf = 0.001
    for index, point in enumerate(condition.trajectory[1:-1], start=1):
        point.x_mm = condition.panel.width_mm * min(1.0, index * 0.25)
        point.y_mm = condition.panel.height_mm * min(1.0, index * 0.05)
    assumptions = AssumptionSet()
    first = simulate(condition, assumptions, "coarse")
    second = simulate(condition, assumptions, "coarse")
    assert first.input_hash == second.input_hash
    assert first.top_peak_risk == second.top_peak_risk
    assert first.final_bottom_peel_ratio == pytest.approx(1.0)
    assert max(first.top_damage_area_mm2) > 0.0
    assert np.all(np.diff(first.top_damage_area_mm2) >= 0.0)
    assert len(first.panel_z_frames_mm) == len(first.frame_indices)
    assert len(first.panel_z_frames_mm[0]) == first.mesh_shape[0] * first.mesh_shape[1]


def test_initial_vertical_lift_is_not_suppressed_by_zero_xy_progress() -> None:
    low = default_condition()
    high = low.model_copy(deep=True)
    for condition, z_mm in ((low, 0.1), (high, 10.0)):
        condition.trajectory[1].x_mm = condition.trajectory[0].x_mm
        condition.trajectory[1].y_mm = condition.trajectory[0].y_mm
        condition.trajectory[1].z_mm = z_mm
    low_result = simulate(low, AssumptionSet(), "normal")
    high_result = simulate(high, AssumptionSet(), "normal")
    high_zero_progress_risk = max(
        risk
        for risk, progress in zip(
            high_result.top_peak_risk, high_result.peel_progress
        )
        if abs(progress) <= 1.0e-12
    )
    assert high_zero_progress_risk > 0.0
    assert high_result.peak_top_risk != low_result.peak_top_risk


def test_identical_conditions_compare_as_tie() -> None:
    project = default_project()
    project.run_uncertainty = False
    result = compare(project)
    assert result.winner == "tie"
    assert result.classification == "tie"
    assert result.result_a.peak_top_risk == pytest.approx(
        result.result_b.peak_top_risk, rel=0.0, abs=0.0
    )


def test_swapping_conditions_swaps_the_winner() -> None:
    project = default_project()
    project.run_uncertainty = False
    altered = project.condition_b
    for index, point in enumerate(altered.trajectory[1:-1], start=1):
        point.x_mm = altered.panel.width_mm * min(1.0, index * 0.25)
        point.y_mm = altered.panel.height_mm * min(1.0, index * 0.05)
    forward = compare(project)
    assert forward.winner == "a"
    project.condition_a, project.condition_b = project.condition_b, project.condition_a
    reverse = compare(project)
    assert reverse.winner == "b"
    assert forward.result_a.peak_top_risk == pytest.approx(
        reverse.result_b.peak_top_risk
    )
    assert forward.result_b.peak_top_risk == pytest.approx(
        reverse.result_a.peak_top_risk
    )


def test_paired_uncertainty_is_seed_reproducible() -> None:
    project = default_project()
    project.assumptions.uncertainty_samples = 4
    project.assumptions.time_steps_coarse = 21
    first = compare(project)
    second = compare(project)
    assert first.uncertainty_enabled
    assert [item.model_dump() for item in first.scenario_results] == [
        item.model_dump() for item in second.scenario_results
    ]
    assert first.risk_quantiles_a == second.risk_quantiles_a
    assert first.risk_quantiles_b == second.risk_quantiles_b
