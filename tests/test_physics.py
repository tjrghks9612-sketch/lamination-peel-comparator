from __future__ import annotations

import numpy as np
import pytest

from lamination_sim.comparison import compare
from lamination_sim.models import AssumptionSet
from lamination_sim.presets import default_condition, default_project
from lamination_sim.simulation import simulate


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


def test_insufficient_pull_force_stalls_bottom_front() -> None:
    assumptions = AssumptionSet(max_pull_force_n=0.1)

    result = simulate(default_condition(), assumptions, "coarse")

    assert result.final_bottom_peel_ratio < 0.01


def test_stronger_bottom_adhesion_reduces_actual_peel() -> None:
    baseline = default_condition()
    stronger = baseline.model_copy(deep=True)
    stronger.bottom_film.adhesion_gf *= 10.0
    assumptions = AssumptionSet()

    baseline_result = simulate(baseline, assumptions, "coarse")
    stronger_result = simulate(stronger, assumptions, "coarse")

    assert baseline_result.final_bottom_peel_ratio >= assumptions.bottom_completion_ratio
    assert stronger_result.final_bottom_peel_ratio < assumptions.bottom_completion_ratio
    assert (
        stronger_result.final_bottom_peel_ratio
        < baseline_result.final_bottom_peel_ratio
    )


def test_top_damage_reduces_local_foundation_and_converges() -> None:
    condition = default_condition()
    condition.top_film.adhesion_gf = 0.001
    assumptions = AssumptionSet(time_steps_coarse=21, damage_max_iterations=100)

    result = simulate(condition, assumptions, "coarse")

    assert result.final_top_damage_area_mm2 > 0.0
    assert min(result.top_min_foundation_retention) < 1.0
    assert max(result.top_damage_iterations) > 1
    assert all(result.top_damage_converged)


def test_thicker_panel_reduces_lift() -> None:
    thin = default_condition()
    thick = thin.model_copy(deep=True)
    thin.panel.thickness_mm = 0.4
    thick.panel.thickness_mm = 1.0

    thin_result = simulate(thin, AssumptionSet(), "coarse")
    thick_result = simulate(thick, AssumptionSet(), "coarse")

    assert thick_result.max_panel_lift_mm < thin_result.max_panel_lift_mm


def test_p1_static_equilibrium_loads_panel_without_advancing_bottom_damage() -> None:
    condition = default_condition()
    for point in condition.trajectory:
        point.z_mm += 4.0

    result = simulate(condition, AssumptionSet(), "coarse")

    assert result.initial_state_mode == "p1_equilibrium"
    assert result.time_s[0] == pytest.approx(0.0)
    assert result.speed_mm_s[0] == pytest.approx(condition.trajectory[0].speed_mm_s)
    assert result.force_resultant_n[0] > 0.0
    assert result.peel_angle_deg[0] > 0.0
    assert result.bottom_peel_ratio[0] == pytest.approx(0.0)


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

    coarse = simulate(condition, assumptions, "coarse")
    normal = simulate(condition, assumptions, "normal")
    fine = simulate(condition, assumptions, "fine")

    assert coarse.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert normal.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert fine.final_bottom_peel_ratio == pytest.approx(1.0, abs=1.0e-12)
    coarse_to_normal = abs(coarse.peak_top_risk - normal.peak_top_risk)
    normal_to_fine = abs(normal.peak_top_risk - fine.peak_top_risk)
    assert normal_to_fine < coarse_to_normal
    assert fine.peak_top_risk == pytest.approx(normal.peak_top_risk, rel=0.10)
    assert fine.max_panel_lift_mm == pytest.approx(
        normal.max_panel_lift_mm, rel=0.10
    )


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

    x_result = simulate(x_first, AssumptionSet(), "coarse")
    y_result = simulate(y_first, AssumptionSet(), "coarse")

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
    project.condition_b.bottom_film.adhesion_gf *= 10.0

    result = compare(project)

    assert result.bottom_gate_pass_a
    assert not result.bottom_gate_pass_b
    assert result.winner == "a"
    assert "gate" in result.decision_basis


def test_crossing_pareto_metrics_produce_inconclusive_result() -> None:
    project = default_project()
    project.run_uncertainty = False
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
    project.assumptions.uncertainty_samples = 4
    project.assumptions.time_steps_coarse = 21

    result = compare(project)

    assert result.classification not in {"robust_a", "robust_b"}
    assert any("robust" in warning for warning in result.warnings)
