"""Nominal and paired-uncertainty A/B comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import AssumptionSet, ProjectV1
from .simulation import SimulationResult, simulate


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
    assumptions: AssumptionSet
    peak_risk_a: float
    peak_risk_b: float
    bottom_gate_pass_a: bool
    bottom_gate_pass_b: bool
    winner: ScenarioWinner
    decision_basis: str
    a_vs_b_peak_risk_change_percent: float


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
    lhs = _latin_hypercube(count, 7, base.random_seed)
    rng = np.random.default_rng(base.random_seed + 1)
    angle_values = np.asarray(([90.0, 180.0] * ((count + 1) // 2))[:count])
    rng.shuffle(angle_values)
    scenarios: list[AssumptionSet] = []
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
            max_pull_force_n=5.0 + row[6] * 30.0,
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


def compare(project: ProjectV1) -> ComparisonResult:
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
