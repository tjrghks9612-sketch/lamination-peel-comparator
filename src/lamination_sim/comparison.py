"""Paired tension-sweep and optional material-uncertainty A/B comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import (
    PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
    AssumptionSet,
    ProjectV1,
    TensionCase,
)
from .simulation import SimulationResult, p1_span_length_mm, simulate


Winner = Literal["a", "b", "tie", "inconclusive"]
ScenarioWinner = Literal["a", "b", "tie", "inconclusive"]
PARETO_METRICS = (
    "peak_top_risk",
    "max_top_risk_area_mm2",
    "top_risk_exceedance_duration_s",
    "final_top_damage_area_mm2",
    "max_panel_lift_mm",
    "max_panel_twist_mm",
)


class UncertaintyScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    material_index: int
    tension_scenario_index: int | None = None
    assumptions: AssumptionSet
    peak_risk_a: float
    peak_risk_b: float
    bottom_gate_pass_a: bool
    bottom_gate_pass_b: bool
    winner: ScenarioWinner
    decision_basis: str
    a_vs_b_peak_risk_change_percent: float


class TensionScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    preload_label: str
    initial_preload_n: float
    stiffness_label: str
    tape_stiffness_n_per_mm: float
    mode: Literal["equal_preload", "shared_rest_length"]
    shared_rest_length_mm: float | None = None
    p1_span_length_a_mm: float
    p1_span_length_b_mm: float
    p1_tension_a_n: float
    p1_tension_b_n: float
    final_bottom_peel_ratio_a: float
    final_bottom_peel_ratio_b: float
    bottom_gate_pass_a: bool
    bottom_gate_pass_b: bool
    peak_top_risk_a: float
    peak_top_risk_b: float
    max_top_risk_area_mm2_a: float
    max_top_risk_area_mm2_b: float
    top_risk_exceedance_duration_s_a: float
    top_risk_exceedance_duration_s_b: float
    final_top_damage_area_mm2_a: float
    final_top_damage_area_mm2_b: float
    max_panel_lift_mm_a: float
    max_panel_lift_mm_b: float
    max_panel_twist_mm_a: float
    max_panel_twist_mm_b: float
    max_abs_force_xyz_n_a: list[float]
    max_abs_force_xyz_n_b: list[float]
    max_tension_n_a: float
    max_tension_n_b: float
    winner: ScenarioWinner
    decision_basis: str


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_schema_version: str
    result_a: SimulationResult
    result_b: SimulationResult
    winner: Winner
    verdict: str
    classification: Literal[
        "robust_a", "robust_b", "weak_a", "weak_b", "tie", "inconclusive"
    ]
    decision_basis: str
    bottom_completion_threshold: float
    bottom_gate_pass_a: bool
    bottom_gate_pass_b: bool
    pareto_metrics: list[str]
    a_vs_b_peak_risk_change_percent: float

    tension_sweep_enabled: bool = True
    tension_mode: Literal["equal_preload", "shared_rest_length"] = "equal_preload"
    tension_advisory_only: bool = False
    tension_wins_a: int = 0
    tension_wins_b: int = 0
    tension_ties: int = 0
    tension_inconclusive: int = 0
    tension_a_win_rate: float = 0.0
    tension_b_win_rate: float = 0.0
    tension_classification: Literal[
        "robust_a", "robust_b", "weak_a", "weak_b", "tie", "inconclusive"
    ] = "inconclusive"
    tension_verdict: str = ""
    tension_scenario_results: list[TensionScenarioResult] = Field(default_factory=list)
    estimated_simulation_count: int = 0
    material_uncertainty_nested: bool = False
    material_uncertainty_enabled: bool = False
    material_uncertainty_scenario_count: int = 0

    uncertainty_enabled: bool
    uncertainty_wins_a: int = 0
    uncertainty_wins_b: int = 0
    uncertainty_ties: int = 0
    uncertainty_inconclusive: int = 0
    uncertainty_a_win_rate: float = 0.0
    uncertainty_b_win_rate: float = 0.0
    median_a_vs_b_peak_risk_change_percent: float = 0.0
    risk_quantiles_a: dict[str, float] = Field(default_factory=dict)
    risk_quantiles_b: dict[str, float] = Field(default_factory=dict)
    scenario_results: list[UncertaintyScenarioResult] = Field(default_factory=list)
    rank_flip_scenarios: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class _ResolvedTensionCase:
    index: int
    preload_label: str
    preload_n: float
    stiffness_label: str
    stiffness_n_per_mm: float
    case_a: TensionCase
    case_b: TensionCase
    shared_rest_length_mm: float | None


def _a_improvement(a: float, b: float) -> float:
    """Positive means A has lower peak risk; negative means B is lower."""

    denominator = max(abs(b), 1.0e-15)
    return float((b - a) / denominator * 100.0)


def _metric_values(result: SimulationResult) -> dict[str, float]:
    return {name: float(getattr(result, name)) for name in PARETO_METRICS}


def _compare_results(
    result_a: SimulationResult,
    result_b: SimulationResult,
    completion_threshold: float,
    tolerance_percent: float,
) -> tuple[ScenarioWinner, str, bool, bool]:
    """Apply the bottom-completion gate, then Pareto dominance."""

    pass_a = result_a.final_bottom_peel_ratio >= completion_threshold
    pass_b = result_b.final_bottom_peel_ratio >= completion_threshold
    if pass_a and not pass_b:
        return "a", "하면 박리 완료 gate: A만 통과", pass_a, pass_b
    if pass_b and not pass_a:
        return "b", "하면 박리 완료 gate: B만 통과", pass_a, pass_b
    if not pass_a and not pass_b:
        return (
            "inconclusive",
            "하면 박리 완료 gate: 두 조건 모두 미통과",
            pass_a,
            pass_b,
        )

    metrics_a = _metric_values(result_a)
    metrics_b = _metric_values(result_b)
    better_a: list[str] = []
    better_b: list[str] = []
    for name in PARETO_METRICS:
        value_a = metrics_a[name]
        value_b = metrics_b[name]
        scale = max(abs(value_a), abs(value_b), 1.0e-15)
        difference_percent = abs(value_a - value_b) / scale * 100.0
        if difference_percent <= tolerance_percent:
            continue
        if value_a < value_b:
            better_a.append(name)
        else:
            better_b.append(name)

    if not better_a and not better_b:
        return "tie", "하면 gate 통과 후 모든 Pareto 지표가 동률 범위", pass_a, pass_b
    if better_a and not better_b:
        return (
            "a",
            "하면 gate 통과 후 A가 Pareto 지배: " + ", ".join(better_a),
            pass_a,
            pass_b,
        )
    if better_b and not better_a:
        return (
            "b",
            "하면 gate 통과 후 B가 Pareto 지배: " + ", ".join(better_b),
            pass_a,
            pass_b,
        )
    return (
        "inconclusive",
        "하면 gate 통과 후 지표 간 우열이 교차하여 Pareto 비지배",
        pass_a,
        pass_b,
    )


def _latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((samples, dimensions), dtype=float)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return values


def _uncertainty_assumptions(base: AssumptionSet) -> list[AssumptionSet]:
    count = base.uncertainty_samples
    if count == 1:
        return [base.model_copy(deep=True)]
    sample_count = count - 1
    lhs = _latin_hypercube(sample_count, 6, base.random_seed)
    rng = np.random.default_rng(base.random_seed + 1)
    angle_values = np.asarray(
        ([90.0, 180.0] * ((sample_count + 1) // 2))[:sample_count]
    )
    rng.shuffle(angle_values)
    scenarios: list[AssumptionSet] = [base.model_copy(deep=True)]
    for index, row in enumerate(lhs):
        values = base.model_dump()
        values.update(
            test_width_mm=12.5 + row[0] * (50.0 - 12.5),
            test_angle_deg=float(angle_values[index]),
            panel_young_modulus_gpa=35.0 + row[1] * (105.0 - 35.0),
            pet_young_modulus_gpa=2.0 + row[2] * (5.0 - 2.0),
            psa_modulus_mpa=float(10.0 ** (-1.30103 + row[3] * 2.0)),
            speed_exponent=row[4] * 0.25,
            grip_scale=0.5 + row[5],
        )
        scenarios.append(AssumptionSet.model_validate(values))
    return scenarios


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _nominal_verdict(winner: ScenarioWinner, basis: str) -> tuple[str, str]:
    if winner == "a":
        return "weak_a", f"명목 조건에서는 조건 A가 우세합니다. {basis}"
    if winner == "b":
        return "weak_b", f"명목 조건에서는 조건 B가 우세합니다. {basis}"
    if winner == "tie":
        return "tie", f"명목 조건에서는 두 조건이 동률입니다. {basis}"
    return "inconclusive", f"명목 조건의 판정을 보류합니다. {basis}"


def _resolved_tension_cases(project: ProjectV1) -> list[_ResolvedTensionCase]:
    config = project.tension_sweep
    span_a = p1_span_length_mm(project.condition_a)
    span_b = p1_span_length_mm(project.condition_b)
    fixed_stiffness = float(config.tape_stiffness_n_per_mm)
    cases: list[_ResolvedTensionCase] = []
    for preload in config.preload_levels:
        rest_length: float | None = None
        effective_a = preload.value
        effective_b = preload.value
        if config.mode == "shared_rest_length":
            if config.rest_length_reference == "custom":
                rest_length = float(config.custom_rest_length_mm or 0.0)
            elif config.rest_length_reference == "condition_b":
                rest_length = span_b
            else:
                rest_length = span_a
            effective_a = max(
                0.0,
                preload.value + fixed_stiffness * (span_a - rest_length),
            )
            effective_b = max(
                0.0,
                preload.value + fixed_stiffness * (span_b - rest_length),
            )
        cases.append(
            _ResolvedTensionCase(
                index=len(cases),
                preload_label=preload.label,
                preload_n=preload.value,
                stiffness_label=PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
                stiffness_n_per_mm=fixed_stiffness,
                case_a=TensionCase(
                    initial_preload_n=effective_a,
                    tape_stiffness_n_per_mm=fixed_stiffness,
                ),
                case_b=TensionCase(
                    initial_preload_n=effective_b,
                    tape_stiffness_n_per_mm=fixed_stiffness,
                ),
                shared_rest_length_mm=rest_length,
            )
        )
    return cases


def _selected_tension_index(project: ProjectV1) -> int:
    return len(project.tension_sweep.preload_levels) // 2


def _max_abs_force(result: SimulationResult) -> list[float]:
    force = np.asarray(result.force_xyz_n, dtype=float)
    if force.size == 0:
        return [0.0, 0.0, 0.0]
    return np.max(np.abs(force), axis=0).astype(float).tolist()


def _tension_scenario(
    case: _ResolvedTensionCase,
    result_a: SimulationResult,
    result_b: SimulationResult,
    assumptions: AssumptionSet,
    mode: Literal["equal_preload", "shared_rest_length"],
) -> TensionScenarioResult:
    winner, basis, pass_a, pass_b = _compare_results(
        result_a,
        result_b,
        assumptions.bottom_completion_ratio,
        assumptions.tie_tolerance_percent,
    )
    return TensionScenarioResult(
        index=case.index,
        preload_label=case.preload_label,
        initial_preload_n=case.preload_n,
        stiffness_label=case.stiffness_label,
        tape_stiffness_n_per_mm=case.stiffness_n_per_mm,
        mode=mode,
        shared_rest_length_mm=case.shared_rest_length_mm,
        p1_span_length_a_mm=result_a.p1_span_length_mm,
        p1_span_length_b_mm=result_b.p1_span_length_mm,
        p1_tension_a_n=result_a.tension_n[0],
        p1_tension_b_n=result_b.tension_n[0],
        final_bottom_peel_ratio_a=result_a.final_bottom_peel_ratio,
        final_bottom_peel_ratio_b=result_b.final_bottom_peel_ratio,
        bottom_gate_pass_a=pass_a,
        bottom_gate_pass_b=pass_b,
        peak_top_risk_a=result_a.peak_top_risk,
        peak_top_risk_b=result_b.peak_top_risk,
        max_top_risk_area_mm2_a=result_a.max_top_risk_area_mm2,
        max_top_risk_area_mm2_b=result_b.max_top_risk_area_mm2,
        top_risk_exceedance_duration_s_a=result_a.top_risk_exceedance_duration_s,
        top_risk_exceedance_duration_s_b=result_b.top_risk_exceedance_duration_s,
        final_top_damage_area_mm2_a=result_a.final_top_damage_area_mm2,
        final_top_damage_area_mm2_b=result_b.final_top_damage_area_mm2,
        max_panel_lift_mm_a=result_a.max_panel_lift_mm,
        max_panel_lift_mm_b=result_b.max_panel_lift_mm,
        max_panel_twist_mm_a=result_a.max_panel_twist_mm,
        max_panel_twist_mm_b=result_b.max_panel_twist_mm,
        max_abs_force_xyz_n_a=_max_abs_force(result_a),
        max_abs_force_xyz_n_b=_max_abs_force(result_b),
        max_tension_n_a=result_a.max_tension_n,
        max_tension_n_b=result_b.max_tension_n,
        winner=winner,
        decision_basis=basis,
    )


def _classify_tension_sweep(
    scenarios: list[TensionScenarioResult], advisory_only: bool
) -> tuple[Winner, str, str, int, int, int, int, float, float]:
    count = len(scenarios)
    wins_a = sum(item.winner == "a" for item in scenarios)
    wins_b = sum(item.winner == "b" for item in scenarios)
    ties = sum(item.winner == "tie" for item in scenarios)
    inconclusive = sum(item.winner == "inconclusive" for item in scenarios)
    a_rate = wins_a / max(count, 1)
    b_rate = wins_b / max(count, 1)
    if advisory_only:
        winner: Winner = "inconclusive"
        classification = "inconclusive"
        verdict = (
            "Shared rest length의 자연 길이가 계측되지 않아 참고 민감도로만 표시합니다. "
            f"A {wins_a}개, B {wins_b}개, 동률 {ties}개, 보류 {inconclusive}개입니다."
        )
    elif a_rate >= 0.80 and b_rate <= 0.10:
        winner = "a"
        classification = "robust_a"
        verdict = "고정 강성 2.25 N/mm에서 시험한 초기장력 조건 모두 우세: 조건 A."
    elif b_rate >= 0.80 and a_rate <= 0.10:
        winner = "b"
        classification = "robust_b"
        verdict = "고정 강성 2.25 N/mm에서 시험한 초기장력 조건 모두 우세: 조건 B."
    elif a_rate >= 0.60:
        winner = "a"
        classification = "weak_a"
        verdict = f"조건 A가 {count}개 중 {wins_a}개에서 우세하지만 장력 가정 민감성이 남습니다."
    elif b_rate >= 0.60:
        winner = "b"
        classification = "weak_b"
        verdict = f"조건 B가 {count}개 중 {wins_b}개에서 우세하지만 장력 가정 민감성이 남습니다."
    elif ties == count and count:
        winner = "tie"
        classification = "tie"
        verdict = "모든 장력 조합에서 두 조건이 동률 범위입니다."
    else:
        winner = "inconclusive"
        classification = "inconclusive"
        verdict = (
            "초기장력과 등가 인장강성에 따라 우열이 바뀌거나 Pareto 지표가 교차합니다. "
            "현재 정보만으로 두 궤적의 우열을 확정할 수 없습니다."
        )
    return (
        winner,
        classification,
        verdict,
        wins_a,
        wins_b,
        ties,
        inconclusive,
        a_rate,
        b_rate,
    )


def _compare_legacy(project: ProjectV1) -> ComparisonResult:
    """Compare A and B with an actual peel-completion gate and Pareto metrics."""

    nominal_a = simulate(project.condition_a, project.assumptions, "normal")
    nominal_b = simulate(project.condition_b, project.assumptions, "normal")
    threshold = project.assumptions.bottom_completion_ratio
    nominal_winner, nominal_basis, pass_a, pass_b = _compare_results(
        nominal_a,
        nominal_b,
        threshold,
        project.assumptions.tie_tolerance_percent,
    )
    nominal_change = _a_improvement(
        nominal_a.peak_top_risk, nominal_b.peak_top_risk
    )
    classification, verdict = _nominal_verdict(nominal_winner, nominal_basis)
    warnings = [
        "판정은 하면 완료 gate와 다중 지표의 상대 우열이며 실제 불량률 예측이 아닙니다."
    ]

    if not project.run_uncertainty:
        return ComparisonResult(
            project_schema_version=project.schema_version,
            result_a=nominal_a,
            result_b=nominal_b,
            winner=nominal_winner,
            verdict=verdict,
            classification=classification,
            decision_basis=nominal_basis,
            bottom_completion_threshold=threshold,
            bottom_gate_pass_a=pass_a,
            bottom_gate_pass_b=pass_b,
            pareto_metrics=list(PARETO_METRICS),
            a_vs_b_peak_risk_change_percent=nominal_change,
            uncertainty_enabled=False,
            warnings=warnings,
        )

    scenario_results: list[UncertaintyScenarioResult] = []
    risks_a: list[float] = []
    risks_b: list[float] = []
    improvements: list[float] = []
    wins_a = wins_b = ties = inconclusive = 0
    for index, assumptions in enumerate(
        _uncertainty_assumptions(project.assumptions)
    ):
        result_a = simulate(project.condition_a, assumptions, "coarse")
        result_b = simulate(project.condition_b, assumptions, "coarse")
        scenario_winner, basis, scenario_pass_a, scenario_pass_b = _compare_results(
            result_a,
            result_b,
            assumptions.bottom_completion_ratio,
            assumptions.tie_tolerance_percent,
        )
        improvement = _a_improvement(
            result_a.peak_top_risk, result_b.peak_top_risk
        )
        risks_a.append(result_a.peak_top_risk)
        risks_b.append(result_b.peak_top_risk)
        improvements.append(improvement)
        wins_a += int(scenario_winner == "a")
        wins_b += int(scenario_winner == "b")
        ties += int(scenario_winner == "tie")
        inconclusive += int(scenario_winner == "inconclusive")
        scenario_results.append(
            UncertaintyScenarioResult(
                index=index,
                assumptions=assumptions,
                peak_risk_a=result_a.peak_top_risk,
                peak_risk_b=result_b.peak_top_risk,
                bottom_gate_pass_a=scenario_pass_a,
                bottom_gate_pass_b=scenario_pass_b,
                winner=scenario_winner,
                decision_basis=basis,
                a_vs_b_peak_risk_change_percent=improvement,
            )
        )

    sample_count = len(scenario_results)
    a_rate = wins_a / sample_count
    b_rate = wins_b / sample_count
    median_improvement = float(np.median(improvements))
    robust_allowed = sample_count >= project.assumptions.minimum_robust_samples
    robust_a = robust_allowed and a_rate >= 0.80
    robust_b = robust_allowed and b_rate >= 0.80
    if not robust_allowed:
        warnings.append(
            f"불확실성 표본 {sample_count}개는 robust 판정 최소값 "
            f"{project.assumptions.minimum_robust_samples}개보다 적습니다."
        )

    if robust_a:
        winner: Winner = "a"
        classification = "robust_a"
        verdict = "조건 A가 paired 불확실성 표본의 80% 이상에서 gate/Pareto 우세를 유지합니다."
    elif robust_b:
        winner = "b"
        classification = "robust_b"
        verdict = "조건 B가 paired 불확실성 표본의 80% 이상에서 gate/Pareto 우세를 유지합니다."
    elif a_rate >= 0.60:
        winner = "a"
        classification = "weak_a"
        verdict = "조건 A가 더 자주 우세하지만 가정 민감성이 있어 약한 우세로 판정합니다."
    elif b_rate >= 0.60:
        winner = "b"
        classification = "weak_b"
        verdict = "조건 B가 더 자주 우세하지만 가정 민감성이 있어 약한 우세로 판정합니다."
    elif ties == sample_count:
        winner = "tie"
        classification = "tie"
        verdict = "모든 불확실성 표본에서 두 조건이 동률 범위입니다."
    else:
        winner = "inconclusive"
        classification = "inconclusive"
        verdict = "가정에 따라 gate 또는 Pareto 우열이 바뀌어 판정을 보류합니다."

    rank_flips = [
        scenario.index
        for scenario in scenario_results
        if nominal_winner in ("a", "b")
        and scenario.winner in ("a", "b")
        and scenario.winner != nominal_winner
    ]
    return ComparisonResult(
        project_schema_version=project.schema_version,
        result_a=nominal_a,
        result_b=nominal_b,
        winner=winner,
        verdict=verdict,
        classification=classification,
        decision_basis=nominal_basis,
        bottom_completion_threshold=threshold,
        bottom_gate_pass_a=pass_a,
        bottom_gate_pass_b=pass_b,
        pareto_metrics=list(PARETO_METRICS),
        a_vs_b_peak_risk_change_percent=nominal_change,
        uncertainty_enabled=True,
        uncertainty_wins_a=wins_a,
        uncertainty_wins_b=wins_b,
        uncertainty_ties=ties,
        uncertainty_inconclusive=inconclusive,
        uncertainty_a_win_rate=a_rate,
        uncertainty_b_win_rate=b_rate,
        median_a_vs_b_peak_risk_change_percent=median_improvement,
        risk_quantiles_a=_quantiles(risks_a),
        risk_quantiles_b=_quantiles(risks_b),
        scenario_results=scenario_results,
        rank_flip_scenarios=rank_flips,
        warnings=warnings,
    )


def compare(project: ProjectV1) -> ComparisonResult:
    """Run paired tension sensitivity, optionally nested material uncertainty."""

    config = project.tension_sweep
    cases = _resolved_tension_cases(project)
    if not cases:
        raise ValueError("tension sweep has no combinations")
    selected_index = min(_selected_tension_index(project), len(cases) - 1)
    if not config.enabled:
        cases = [cases[selected_index]]
        selected_index = 0
    material_sets = (
        _uncertainty_assumptions(project.assumptions)
        if project.run_uncertainty
        else [project.assumptions]
    )
    nested = bool(project.run_uncertainty and config.nest_material_uncertainty)

    sweep_results: list[TensionScenarioResult] = []
    base_pairs: dict[int, tuple[SimulationResult, SimulationResult]] = {}
    material_records: list[UncertaintyScenarioResult] = []
    uncertainty_pairs: list[
        tuple[int, int, AssumptionSet, SimulationResult, SimulationResult]
    ] = []

    for case in cases:
        result_a = simulate(
            project.condition_a, project.assumptions, "normal", case.case_a
        )
        result_b = simulate(
            project.condition_b, project.assumptions, "normal", case.case_b
        )
        base_pairs[case.index] = (result_a, result_b)
        sweep_results.append(
            _tension_scenario(
                case, result_a, result_b, project.assumptions, config.mode
            )
        )
        if nested:
            uncertainty_pairs.append(
                (case.index, 0, material_sets[0], result_a, result_b)
            )
            for material_index, assumptions in enumerate(material_sets[1:], start=1):
                uncertainty_pairs.append(
                    (
                        case.index,
                        material_index,
                        assumptions,
                        simulate(project.condition_a, assumptions, "coarse", case.case_a),
                        simulate(project.condition_b, assumptions, "coarse", case.case_b),
                    )
                )

    if project.run_uncertainty and not nested:
        selected_case = cases[selected_index]
        selected_a, selected_b = base_pairs[selected_index]
        uncertainty_pairs.append(
            (selected_index, 0, material_sets[0], selected_a, selected_b)
        )
        for material_index, assumptions in enumerate(material_sets[1:], start=1):
            uncertainty_pairs.append(
                (
                    selected_index,
                    material_index,
                    assumptions,
                    simulate(
                        project.condition_a,
                        assumptions,
                        "coarse",
                        selected_case.case_a,
                    ),
                    simulate(
                        project.condition_b,
                        assumptions,
                        "coarse",
                        selected_case.case_b,
                    ),
                )
            )

    advisory_only = bool(
        config.mode == "shared_rest_length"
        and config.rest_length_reference != "custom"
    )
    (
        tension_winner,
        tension_classification,
        tension_verdict,
        tension_wins_a,
        tension_wins_b,
        tension_ties,
        tension_inconclusive,
        tension_a_rate,
        tension_b_rate,
    ) = _classify_tension_sweep(sweep_results, advisory_only)

    nominal_a, nominal_b = base_pairs[selected_index]
    nominal_scenario = sweep_results[selected_index]
    nominal_change = _a_improvement(
        nominal_a.peak_top_risk, nominal_b.peak_top_risk
    )
    warnings = [
        "장력 민감도 판정은 계측 전 가정 범위의 상대 우열이며 실제 불량률 예측이 아닙니다."
    ]
    if advisory_only:
        warnings.append(
            "Shared rest length의 자연 길이가 계측되지 않아 이 모드의 결과는 참고 민감도입니다."
        )
    insufficient_material_samples = bool(
        project.run_uncertainty
        and project.assumptions.uncertainty_samples
        < project.assumptions.minimum_robust_samples
    )
    if insufficient_material_samples:
        warnings.append(
            "재료 불확실성 표본 수가 minimum_robust_samples보다 작아 robust 판정을 허용하지 않습니다."
        )

    risks_a: list[float] = []
    risks_b: list[float] = []
    improvements: list[float] = []
    material_wins_a = material_wins_b = material_ties = material_inconclusive = 0
    for global_index, (
        tension_index,
        material_index,
        assumptions,
        result_a,
        result_b,
    ) in enumerate(uncertainty_pairs):
        scenario_winner, basis, pass_a, pass_b = _compare_results(
            result_a,
            result_b,
            assumptions.bottom_completion_ratio,
            assumptions.tie_tolerance_percent,
        )
        improvement = _a_improvement(
            result_a.peak_top_risk, result_b.peak_top_risk
        )
        risks_a.append(result_a.peak_top_risk)
        risks_b.append(result_b.peak_top_risk)
        improvements.append(improvement)
        material_wins_a += int(scenario_winner == "a")
        material_wins_b += int(scenario_winner == "b")
        material_ties += int(scenario_winner == "tie")
        material_inconclusive += int(scenario_winner == "inconclusive")
        material_records.append(
            UncertaintyScenarioResult(
                index=global_index,
                material_index=material_index,
                tension_scenario_index=tension_index if nested else None,
                assumptions=assumptions,
                peak_risk_a=result_a.peak_top_risk,
                peak_risk_b=result_b.peak_top_risk,
                bottom_gate_pass_a=pass_a,
                bottom_gate_pass_b=pass_b,
                winner=scenario_winner,
                decision_basis=basis,
                a_vs_b_peak_risk_change_percent=improvement,
            )
        )

    material_count = len(material_records)
    material_a_rate = material_wins_a / max(material_count, 1)
    material_b_rate = material_wins_b / max(material_count, 1)
    rank_flips = [
        item.index
        for item in material_records
        if nominal_scenario.winner in ("a", "b")
        and item.winner in ("a", "b")
        and item.winner != nominal_scenario.winner
    ]
    if nested:
        estimated_count = len(cases) * len(material_sets) * 2
    else:
        estimated_count = len(cases) * 2
        if project.run_uncertainty:
            estimated_count += max(len(material_sets) - 1, 0) * 2

    if not config.enabled:
        overall_winner = nominal_scenario.winner
        overall_classification, overall_verdict = _nominal_verdict(
            nominal_scenario.winner, nominal_scenario.decision_basis
        )
    else:
        overall_winner = tension_winner
        overall_classification = tension_classification
        overall_verdict = tension_verdict
    if insufficient_material_samples and overall_classification in {"robust_a", "robust_b"}:
        overall_winner = "inconclusive"
        overall_classification = "inconclusive"
        overall_verdict = (
            "장력 조합에서는 한쪽이 우세하지만 재료 불확실성 표본 수가 부족해 "
            "장력 조건 간 우열이 엇갈려 결론을 보류합니다."
        )

    return ComparisonResult(
        project_schema_version=project.schema_version,
        result_a=nominal_a,
        result_b=nominal_b,
        winner=overall_winner,
        verdict=overall_verdict,
        classification=overall_classification,
        decision_basis=(
            nominal_scenario.decision_basis
            if not config.enabled
            else (
                f"장력 조합 {len(sweep_results)}개: A {tension_wins_a}, "
                f"B {tension_wins_b}, 동률 {tension_ties}, 보류 {tension_inconclusive}"
            )
        ),
        bottom_completion_threshold=project.assumptions.bottom_completion_ratio,
        bottom_gate_pass_a=nominal_scenario.bottom_gate_pass_a,
        bottom_gate_pass_b=nominal_scenario.bottom_gate_pass_b,
        pareto_metrics=list(PARETO_METRICS),
        a_vs_b_peak_risk_change_percent=nominal_change,
        tension_sweep_enabled=config.enabled,
        tension_mode=config.mode,
        tension_advisory_only=advisory_only,
        tension_wins_a=tension_wins_a,
        tension_wins_b=tension_wins_b,
        tension_ties=tension_ties,
        tension_inconclusive=tension_inconclusive,
        tension_a_win_rate=tension_a_rate,
        tension_b_win_rate=tension_b_rate,
        tension_classification=tension_classification,
        tension_verdict=tension_verdict,
        tension_scenario_results=sweep_results,
        estimated_simulation_count=estimated_count,
        material_uncertainty_nested=nested,
        material_uncertainty_enabled=project.run_uncertainty,
        material_uncertainty_scenario_count=material_count,
        uncertainty_enabled=project.run_uncertainty,
        uncertainty_wins_a=material_wins_a,
        uncertainty_wins_b=material_wins_b,
        uncertainty_ties=material_ties,
        uncertainty_inconclusive=material_inconclusive,
        uncertainty_a_win_rate=material_a_rate,
        uncertainty_b_win_rate=material_b_rate,
        median_a_vs_b_peak_risk_change_percent=(
            float(np.median(improvements)) if improvements else nominal_change
        ),
        risk_quantiles_a=_quantiles(risks_a),
        risk_quantiles_b=_quantiles(risks_b),
        scenario_results=material_records,
        rank_flip_scenarios=rank_flips,
        warnings=warnings,
    )
