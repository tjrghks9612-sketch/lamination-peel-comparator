from __future__ import annotations

import numpy as np
import pytest

from lamination_sim.comparison import compare
from lamination_sim.models import AssumptionSet, SweepLevel, TensionCase
from lamination_sim.presets import default_condition, default_project, measured_project
from lamination_sim.simulation import (
    _actuator_work_n_mm,
    _film_reach_mask,
    simulate,
)


HIGH_TENSION = TensionCase(initial_preload_n=20.0, tape_stiffness_n_per_mm=1.0)


def _use_single_high_tension(project) -> None:
    project.tension_sweep.enabled = False
    project.tension_sweep.preload_levels = [SweepLevel(label="test", value=20.0)]
    project.tension_sweep.stiffness_levels = [SweepLevel(label="test", value=1.0)]


def test_actuator_work_uses_force_direction_not_scalar_path_length() -> None:
    force = np.asarray([3.0, 4.0, 0.0])

    assert _actuator_work_n_mm(force, np.asarray([0.6, 0.8, 0.0])) == pytest.approx(5.0)
    assert _actuator_work_n_mm(force, np.asarray([-0.8, 0.6, 0.0])) == pytest.approx(0.0)
    assert _actuator_work_n_mm(force, np.asarray([-0.6, -0.8, 0.0])) == pytest.approx(0.0)


def test_actuator_work_uses_previous_and_current_force_average() -> None:
    previous = np.asarray([1.0, 0.0, 0.0])
    current = np.asarray([3.0, 0.0, 0.0])
    increment = np.asarray([2.0, 0.0, 0.0])
    assert _actuator_work_n_mm(current, increment, previous) == pytest.approx(4.0)


def test_film_payout_reach_is_isotropic_and_vertical_independent() -> None:
    local_x = np.asarray([8.0, 0.0, 6.0, 9.0])
    local_y = np.asarray([0.0, 8.0, 6.0, 0.0])

    reached = _film_reach_mask(local_x, local_y, 3.0, 5.0)

    np.testing.assert_array_equal(reached, [True, True, False, False])


def test_pure_vertical_motion_does_not_imply_full_bottom_peel() -> None:
    condition = default_condition()
    x = condition.trajectory[0].x_mm
    y = condition.trajectory[0].y_mm
    for index, point in enumerate(condition.trajectory):
        point.x_mm = x
        point.y_mm = y
        point.z_mm = 3.0 * index

    result = simulate(condition, AssumptionSet(), "normal")

    # The diagnostic path parameter still reaches its endpoint, but physical
    # peel is now a separate damage state and remains local to the corner.
    assert result.trajectory_progress[-1] == pytest.approx(1.0)
    assert result.final_bottom_peel_ratio < 0.10
    assert np.all(np.diff(result.bottom_peel_ratio) >= -1.0e-12)


def test_deprecated_vertical_reach_factor_cannot_change_peel() -> None:
    condition = default_condition()

    low = simulate(
        condition,
        AssumptionSet(vertical_front_reach_factor=0.0),
        "coarse",
    )
    high = simulate(
        condition,
        AssumptionSet(vertical_front_reach_factor=5.0),
        "coarse",
    )

    np.testing.assert_array_equal(low.bottom_peel_ratio, high.bottom_peel_ratio)


def test_insufficient_pull_force_stalls_bottom_front() -> None:
    assumptions = AssumptionSet(max_pull_force_n=0.1)

    result = simulate(default_condition(), assumptions, "coarse")

    assert result.final_bottom_peel_ratio < 0.01


def test_stronger_bottom_adhesion_reduces_actual_peel() -> None:
    baseline = default_condition()
    stronger = baseline.model_copy(deep=True)
    stronger.bottom_film.adhesion_gf *= 10.0
    assumptions = AssumptionSet()

    baseline_result = simulate(baseline, assumptions, "coarse", HIGH_TENSION)
    stronger_result = simulate(stronger, assumptions, "coarse", HIGH_TENSION)

    assert baseline_result.final_bottom_peel_ratio >= assumptions.bottom_completion_ratio
    assert stronger_result.final_bottom_peel_ratio < assumptions.bottom_completion_ratio
    assert (
        stronger_result.final_bottom_peel_ratio
        < baseline_result.final_bottom_peel_ratio
    )


def test_top_damage_reduces_local_foundation_and_converges() -> None:
    condition = default_condition()
    condition.top_film.adhesion_gf = 0.002
    assumptions = AssumptionSet(time_steps_coarse=21, damage_max_iterations=200)

    result = simulate(condition, assumptions, "coarse")

    assert result.final_top_damage_area_mm2 > 0.0
    assert min(result.top_min_foundation_retention) < 1.0
    assert max(result.top_damage_iterations) > 1
    assert all(result.top_damage_converged)
    assert len(result.top_risk_frames) == len(result.frame_indices)
    assert max(max(frame) for frame in result.top_risk_frames) > 0.0


def test_top_interface_reaction_is_damage_softened_and_exportable_per_frame() -> None:
    condition = default_condition()
    condition.top_film.adhesion_gf = 0.002
    result = simulate(
        condition,
        AssumptionSet(time_steps_coarse=21, damage_max_iterations=200),
        "coarse",
    )

    assert len(result.top_interface_normal_force_n) == len(result.time_s)
    assert len(result.top_interface_reaction_centroid_xy_mm) == len(result.time_s)
    assert len(result.top_interface_reaction_frames_n) == len(result.frame_indices)
    assert all(value >= 0.0 for value in result.top_interface_normal_force_n)
    assert all(
        value >= 0.0
        for frame in result.top_interface_reaction_frames_n
        for value in frame
    )
    for frame_index, frame in zip(
        result.frame_indices, result.top_interface_reaction_frames_n
    ):
        assert sum(frame) == pytest.approx(
            result.top_interface_normal_force_n[frame_index], rel=1.0e-10, abs=1.0e-12
        )


def test_thicker_panel_reduces_lift() -> None:
    thin = default_condition()
    thick = thin.model_copy(deep=True)
    thin.panel.thickness_mm = 0.4
    thick.panel.thickness_mm = 1.0

    thin_result = simulate(thin, AssumptionSet(), "coarse")
    thick_result = simulate(thick, AssumptionSet(), "coarse")

    assert thick_result.max_panel_lift_mm < thin_result.max_panel_lift_mm


def test_p1_surface_attachment_lifts_vertically_before_main_trajectory() -> None:
    condition = measured_project().condition_b
    result = simulate(condition, AssumptionSet(), "coarse")
    p1_index = result.main_trajectory_start_index

    assert result.initial_state_mode == "p1_attach_lift"
    assert result.time_s[0] == pytest.approx(0.0)
    assert p1_index > 0
    assert result.position_xyz_mm[0] == pytest.approx(
        [condition.trajectory[0].x_mm, condition.trajectory[0].y_mm, 0.0]
    )
    assert result.position_xyz_mm[p1_index] == pytest.approx(
        [
            condition.trajectory[0].x_mm,
            condition.trajectory[0].y_mm,
            condition.trajectory[0].z_mm,
        ]
    )
    assert all(
        point[:2] == pytest.approx(result.position_xyz_mm[0][:2])
        for point in result.position_xyz_mm[: p1_index + 1]
    )
    assert all(
        left <= right + 1.0e-12
        for left, right in zip(
            (point[2] for point in result.position_xyz_mm[:p1_index]),
            (point[2] for point in result.position_xyz_mm[1 : p1_index + 1]),
        )
    )
    assert result.peel_work_n_mm[p1_index] > 0.0
    assert result.tension_n[0] == pytest.approx(0.0)
    assert result.tension_n[p1_index] > result.initial_preload_n


def test_left_right_mirror_has_symmetric_scalar_results() -> None:
    left = default_condition()
    right = left.model_copy(deep=True)
    right.pull_tape.start_corner = "bottom_right"
    for point in right.trajectory:
        point.x_mm = right.panel.width_mm - point.x_mm

    left_result = simulate(left, AssumptionSet(), "normal")
    right_result = simulate(right, AssumptionSet(), "normal")

    assert right_result.final_bottom_peel_ratio == pytest.approx(
        left_result.final_bottom_peel_ratio, abs=1.0e-12
    )
    assert right_result.peak_top_risk == pytest.approx(
        left_result.peak_top_risk, rel=1.0e-10
    )
    assert right_result.max_panel_lift_mm == pytest.approx(
        left_result.max_panel_lift_mm, rel=1.0e-10
    )
    assert right_result.max_panel_twist_mm == pytest.approx(
        left_result.max_panel_twist_mm, rel=1.0e-10
    )


def test_mesh_refinement_converges_from_four_to_two_to_one_mm() -> None:
    condition = default_condition()
    assumptions = AssumptionSet()

    coarse = simulate(condition, assumptions, "coarse", HIGH_TENSION)
    normal = simulate(condition, assumptions, "normal", HIGH_TENSION)
    fine = simulate(condition, assumptions, "fine", HIGH_TENSION)

    assert coarse.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert normal.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert fine.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    coarse_to_normal = abs(coarse.peak_top_risk - normal.peak_top_risk)
    normal_to_fine = abs(normal.peak_top_risk - fine.peak_top_risk)
    assert normal_to_fine < coarse_to_normal
    # Endpoint-average work changes the damage-front timing slightly across
    # mesh resolutions; retain a broad mesh-robustness guard while the new
    # time-convergence test checks the tighter 51/101/201 acceptance.
    assert fine.peak_top_risk == pytest.approx(normal.peak_top_risk, rel=0.30)
    assert fine.max_panel_lift_mm == pytest.approx(
        normal.max_panel_lift_mm, rel=0.15
    )


def test_time_work_converges_at_51_101_201_steps() -> None:
    records = []
    for steps in (51, 101, 201):
        project = default_project()
        project.assumptions.time_steps_normal = steps
        project.run_uncertainty = False
        project.condition_b.bottom_film.adhesion_gf *= 10.0
        records.append(compare(project))

    def metric(result, name: str) -> float:
        return float(getattr(result, name))

    # Both sides' physical outputs must be time-converged, not only the
    # integrated actuator work.
    for side in ("result_a", "result_b"):
        for name in (
            "final_bottom_peel_ratio",
            "peak_top_risk",
            "max_panel_lift_mm",
            "max_tension_n",
            "total_peel_work_n_mm",
        ):
            middle = metric(getattr(records[1], side), name)
            fine = metric(getattr(records[2], side), name)
            assert abs(fine - middle) / max(abs(fine), 1.0e-12) < 0.05

    assert [item.winner for item in records] == ["a", "a", "a"]
    assert [
        (item.bottom_gate_pass_a, item.bottom_gate_pass_b) for item in records
    ] == [(True, False)] * 3
    assert all(
        0.0 <= item.result_a.final_bottom_peel_ratio <= 1.0
        and 0.0 <= item.result_b.final_bottom_peel_ratio <= 1.0
        for item in records
    )


def test_long_force_cap_saturation_adds_warning() -> None:
    result = simulate(
        default_condition(),
        AssumptionSet(max_pull_force_n=0.05),
        "normal",
        TensionCase(initial_preload_n=1.5, tape_stiffness_n_per_mm=2.25),
    )
    assert result.pull_force_saturation_fraction >= 0.10
    assert any("max_pull_force_n" in item for item in result.warnings)


def test_xy_path_order_changes_damage_front_and_mechanics() -> None:
    x_first = default_condition()
    y_first = x_first.model_copy(deep=True)
    width = x_first.panel.width_mm
    height = x_first.panel.height_mm
    x_path = ((0, 0), (0.4, 0.02), (0.8, 0.05), (1, 0.2), (1, 0.6), (1, 1))
    y_path = ((0, 0), (0.02, 0.4), (0.05, 0.8), (0.2, 1), (0.6, 1), (1, 1))
    for condition, path in ((x_first, x_path), (y_first, y_path)):
        for point, (x_fraction, y_fraction) in zip(condition.trajectory, path):
            point.x_mm = width * x_fraction
            point.y_mm = height * y_fraction

    x_result = simulate(x_first, AssumptionSet(), "coarse", HIGH_TENSION)
    y_result = simulate(y_first, AssumptionSet(), "coarse", HIGH_TENSION)

    assert x_result.final_bottom_peel_ratio == pytest.approx(1.0)
    assert y_result.final_bottom_peel_ratio == pytest.approx(1.0)
    assert not np.allclose(
        x_result.bottom_peel_ratio, y_result.bottom_peel_ratio
    )
    assert not np.allclose(
        x_result.bottom_damage_frames, y_result.bottom_damage_frames
    )
    assert x_result.peak_top_risk != pytest.approx(y_result.peak_top_risk)


def test_three_axis_force_and_moment_are_applied() -> None:
    result = simulate(default_condition(), AssumptionSet(), "coarse")
    force = np.asarray(result.force_xyz_n)
    moment = np.asarray(result.moment_xyz_n_mm)

    assert np.all(np.max(np.abs(force), axis=0) > 0.0)
    assert np.all(np.max(np.abs(moment), axis=0) > 0.0)
    assert result.max_moment_resultant_n_mm > 0.0


def test_bottom_completion_gate_precedes_top_risk_ranking() -> None:
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)
    project.condition_b.bottom_film.adhesion_gf *= 10.0

    result = compare(project)

    assert result.bottom_gate_pass_a
    assert not result.bottom_gate_pass_b
    assert result.winner == "a"
    assert "gate" in result.decision_basis


def test_crossing_pareto_metrics_produce_inconclusive_result() -> None:
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)
    project.condition_a.top_film.adhesion_gf = 10.0
    project.condition_a.panel.thickness_mm = 0.4
    project.condition_b.top_film.adhesion_gf = 2.0
    project.condition_b.panel.thickness_mm = 1.0

    result = compare(project)

    assert result.bottom_gate_pass_a and result.bottom_gate_pass_b
    assert result.result_a.peak_top_risk < result.result_b.peak_top_risk
    assert result.result_a.max_panel_lift_mm > result.result_b.max_panel_lift_mm
    assert result.winner == "inconclusive"
    assert "Pareto" in result.decision_basis


def test_too_few_uncertainty_samples_cannot_be_robust() -> None:
    project = default_project()
    project.run_uncertainty = True
    project.assumptions.uncertainty_samples = 4
    project.assumptions.time_steps_coarse = 21

    result = compare(project)

    assert result.classification not in {"robust_a", "robust_b"}
    assert any("robust" in warning for warning in result.warnings)
