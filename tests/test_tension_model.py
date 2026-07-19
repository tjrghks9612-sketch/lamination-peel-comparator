from __future__ import annotations

import pytest

from lamination_sim.comparison import (
    TensionScenarioResult,
    _classify_tension_sweep,
    compare,
)
from lamination_sim.models import (
    PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
    AssumptionSet,
    SweepLevel,
    TensionCase,
)
from lamination_sim.presets import default_condition, measured_project
from lamination_sim.simulation import simulate, tension_from_span


def test_zero_preload_and_stiffness_produce_zero_p1_load() -> None:
    result = simulate(
        default_condition(),
        AssumptionSet(),
        "coarse",
        TensionCase(initial_preload_n=0.0, tape_stiffness_n_per_mm=0.0),
    )

    assert result.tension_n[0] == pytest.approx(0.0)
    assert result.force_resultant_n[0] == pytest.approx(0.0)
    assert result.panel_max_lift_mm[0] == pytest.approx(0.0)


def test_higher_initial_preload_increases_p1_tension() -> None:
    condition = default_condition()
    low = simulate(
        condition,
        AssumptionSet(),
        "coarse",
        TensionCase(initial_preload_n=0.2, tape_stiffness_n_per_mm=0.2),
    )
    high = simulate(
        condition,
        AssumptionSet(),
        "coarse",
        TensionCase(initial_preload_n=1.2, tape_stiffness_n_per_mm=0.2),
    )

    assert low.tension_n[0] == pytest.approx(0.2)
    assert high.tension_n[0] == pytest.approx(1.2)
    assert high.tension_n[0] > low.tension_n[0]


def test_stiffness_controls_tension_change_for_same_span_change() -> None:
    low = tension_from_span(15.0, 10.0, 0.5, 0.05, 20.0)
    high = tension_from_span(15.0, 10.0, 0.5, 1.0, 20.0)

    assert low - 0.5 == pytest.approx(0.25)
    assert high - 0.5 == pytest.approx(5.0)
    assert high > low


def test_tension_is_unilateral_and_capped() -> None:
    assert tension_from_span(0.0, 10.0, 0.0, 1.0, 20.0) == 0.0
    assert tension_from_span(100.0, 0.0, 1.0, 5.0, 7.0) == 7.0


def test_equal_preload_pairs_identical_p1_tension_for_a_and_b() -> None:
    project = measured_project()
    project.tension_sweep.mode = "equal_preload"

    result = compare(project)

    assert len(result.tension_scenario_results) == 3
    assert (
        result.tension_wins_a
        + result.tension_wins_b
        + result.tension_ties
        + result.tension_inconclusive
        == 3
    )
    for scenario in result.tension_scenario_results:
        assert scenario.p1_tension_a_n == pytest.approx(scenario.initial_preload_n)
        assert scenario.p1_tension_b_n == pytest.approx(scenario.initial_preload_n)
        assert scenario.p1_tension_a_n == pytest.approx(scenario.p1_tension_b_n)


def test_fixed_pet_stiffness_is_not_a_sweep_axis() -> None:
    project = measured_project()
    project.tension_sweep.stiffness_levels = [
        SweepLevel(label="legacy-low", value=0.01),
        SweepLevel(label="legacy-high", value=50.0),
    ]

    result = compare(project)

    assert len(result.tension_scenario_results) == 3
    assert all(
        scenario.tape_stiffness_n_per_mm
        == pytest.approx(PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM)
        for scenario in result.tension_scenario_results
    )


def test_shared_rest_length_reflects_p1_geometry_difference() -> None:
    project = measured_project()
    project.tension_sweep.mode = "shared_rest_length"
    project.tension_sweep.rest_length_reference = "condition_a"

    result = compare(project)
    middle = result.tension_scenario_results[1]

    assert middle.p1_span_length_b_mm > middle.p1_span_length_a_mm
    assert middle.p1_tension_b_n > middle.p1_tension_a_n
    assert result.tension_advisory_only
    assert result.classification == "inconclusive"


def test_tension_sweep_rates_include_ties_and_inconclusive_in_denominator() -> None:
    winners = ["a"] * 7 + ["b", "inconclusive"]
    scenarios = [
        TensionScenarioResult.model_construct(winner=winner)
        for winner in winners
    ]

    classified = _classify_tension_sweep(scenarios, advisory_only=False)

    assert classified[3:7] == (7, 1, 0, 1)
    assert classified[7] == pytest.approx(7 / 9)
    assert classified[8] == pytest.approx(1 / 9)
    assert classified[1] == "weak_a"


def test_rank_flips_across_tension_cases_are_inconclusive() -> None:
    scenarios = [
        TensionScenarioResult.model_construct(winner=winner)
        for winner in (["a"] * 4 + ["b"] * 4 + ["tie"])
    ]

    winner, classification, *_rest = _classify_tension_sweep(
        scenarios, advisory_only=False
    )

    assert winner == "inconclusive"
    assert classification == "inconclusive"


def test_nested_material_uncertainty_keeps_tension_case_paired() -> None:
    project = measured_project()
    project.tension_sweep.preload_levels = [SweepLevel(label="Only", value=0.5)]
    project.tension_sweep.stiffness_levels = [SweepLevel(label="Only", value=0.2)]
    project.tension_sweep.nest_material_uncertainty = True
    project.run_uncertainty = True
    project.assumptions.uncertainty_samples = 2
    project.assumptions.time_steps_coarse = 11

    result = compare(project)

    assert result.material_uncertainty_enabled
    assert result.material_uncertainty_nested
    assert result.material_uncertainty_scenario_count == 2
    assert result.estimated_simulation_count == 4
    assert {item.tension_scenario_index for item in result.scenario_results} == {0}
    assert {item.material_index for item in result.scenario_results} == {0, 1}
