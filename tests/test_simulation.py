from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from lamination_sim.comparison import compare
from lamination_sim.__main__ import _self_test
from lamination_sim.models import AssumptionSet, SweepLevel
from lamination_sim.presets import default_condition, default_project
from lamination_sim.simulation import simulate
from lamination_sim.trajectory import interpolate_trajectory, waypoint_times


def _use_single_high_tension(project) -> None:
    project.tension_sweep.enabled = False
    project.tension_sweep.preload_levels = [SweepLevel(label="test", value=20.0)]
    project.tension_sweep.stiffness_levels = [SweepLevel(label="test", value=1.0)]


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


def test_condition_rejects_p1_below_attachment_surface() -> None:
    data = default_condition().model_dump()
    data["trajectory"][0]["z_mm"] = -0.1

    with pytest.raises(ValidationError, match="P1 Z must be at or above"):
        type(default_condition()).model_validate(data)


def test_packaged_self_test_accepts_default_tension_sweep() -> None:
    assert _self_test() == 0


def test_piecewise_linear_trajectory_uses_waypoint_target_speeds() -> None:
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
        2.0
        * first_distance
        / (
            condition.trajectory[0].speed_mm_s
            + condition.trajectory[1].speed_mm_s
        )
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


def test_absolute_p1_height_changes_initial_equilibrium() -> None:
    low = default_condition()
    high = low.model_copy(deep=True)
    for point in low.trajectory:
        point.z_mm += 4.0
    for point in high.trajectory:
        point.z_mm += 14.0
    low_result = simulate(low, AssumptionSet(), "coarse")
    high_result = simulate(high, AssumptionSet(), "coarse")

    low_p1 = low_result.main_trajectory_start_index
    high_p1 = high_result.main_trajectory_start_index
    assert low_result.position_xyz_mm[0][2] == pytest.approx(0.0)
    assert high_result.position_xyz_mm[0][2] == pytest.approx(0.0)
    assert low_result.position_xyz_mm[low_p1][2] == pytest.approx(4.0)
    assert high_result.position_xyz_mm[high_p1][2] == pytest.approx(14.0)
    assert high_result.peel_angle_deg[high_p1] != pytest.approx(
        low_result.peel_angle_deg[low_p1]
    )
    assert high_result.force_xyz_n[high_p1][2] != pytest.approx(
        low_result.force_xyz_n[low_p1][2]
    )


def test_identical_conditions_compare_as_tie() -> None:
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)
    result = compare(project)
    assert result.winner == "tie"
    assert result.classification == "tie"
    assert result.result_a.peak_top_risk == pytest.approx(
        result.result_b.peak_top_risk, rel=0.0, abs=0.0
    )


def test_swapping_conditions_swaps_the_winner() -> None:
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)
    project.condition_b.bottom_film.adhesion_gf *= 10.0
    forward = compare(project)
    assert forward.winner == "a"
    assert forward.bottom_gate_pass_a
    assert not forward.bottom_gate_pass_b
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
    project.run_uncertainty = True
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
