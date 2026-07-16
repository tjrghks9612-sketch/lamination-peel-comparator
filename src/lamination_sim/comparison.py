"""Nominal and paired-uncertainty A/B comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import AssumptionSet, ProjectV1
from .simulation import SimulationResult, simulate


Winner = Literal["a", "b", "tie", "inconclusive"]


class UncertaintyScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    assumptions: AssumptionSet
    peak_risk_a: float
    peak_risk_b: float
    winner: Literal["a", "b", "tie"]
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
    a_vs_b_peak_risk_change_percent: float

    uncertainty_enabled: bool
    uncertainty_wins_a: int = 0
    uncertainty_wins_b: int = 0
    uncertainty_ties: int = 0
    uncertainty_a_win_rate: float = 0.0
    uncertainty_b_win_rate: float = 0.0
    median_a_vs_b_peak_risk_change_percent: float = 0.0
    risk_quantiles_a: dict[str, float] = Field(default_factory=dict)
    risk_quantiles_b: dict[str, float] = Field(default_factory=dict)
    scenario_results: list[UncertaintyScenarioResult] = Field(default_factory=list)
    rank_flip_scenarios: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _winner(a: float, b: float, tolerance_percent: float) -> Literal["a", "b", "tie"]:
    scale = max(abs(a), abs(b), 1.0e-15)
    difference_percent = abs(a - b) / scale * 100.0
    if difference_percent <= tolerance_percent:
        return "tie"
    return "a" if a < b else "b"


def _a_improvement(a: float, b: float) -> float:
    """Positive means A has lower peak risk; negative means B is lower."""

    denominator = max(abs(b), 1.0e-15)
    return float((b - a) / denominator * 100.0)


def _latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((samples, dimensions), dtype=float)
    for dimension in range(dimensions):
        values[:, dimension] = (rng.permutation(samples) + rng.random(samples)) / samples
    return values


def _uncertainty_assumptions(base: AssumptionSet) -> list[AssumptionSet]:
    count = base.uncertainty_samples
    lhs = _latin_hypercube(count, 6, base.random_seed)
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
            # PSA stiffness spans two decades, so sample it logarithmically.
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


def _nominal_verdict(winner: Literal["a", "b", "tie"]) -> tuple[str, str]:
    if winner == "a":
        return "weak_a", "명목 가정에서는 조건 A의 상면 역박리 지표가 더 낮습니다."
    if winner == "b":
        return "weak_b", "명목 가정에서는 조건 B의 상면 역박리 지표가 더 낮습니다."
    return "tie", "명목 가정에서 두 조건의 차이는 동률 허용범위 안입니다."


def compare(project: ProjectV1) -> ComparisonResult:
    """Compare A and B with identical nominal and uncertainty assumptions."""

    nominal_a = simulate(project.condition_a, project.assumptions, "normal")
    nominal_b = simulate(project.condition_b, project.assumptions, "normal")
    nominal_winner = _winner(
        nominal_a.peak_top_risk,
        nominal_b.peak_top_risk,
        project.assumptions.tie_tolerance_percent,
    )
    nominal_change = _a_improvement(
        nominal_a.peak_top_risk, nominal_b.peak_top_risk
    )
    classification, verdict = _nominal_verdict(nominal_winner)
    warnings = [
        "우세 판정은 가정 정규화 위험도의 상대 순위이며 실제 불량률 예측이 아닙니다."
    ]

    if not project.run_uncertainty:
        return ComparisonResult(
            project_schema_version=project.schema_version,
            result_a=nominal_a,
            result_b=nominal_b,
            winner=nominal_winner,
            verdict=verdict,
            classification=classification,
            a_vs_b_peak_risk_change_percent=nominal_change,
            uncertainty_enabled=False,
            warnings=warnings,
        )

    scenario_results: list[UncertaintyScenarioResult] = []
    risks_a: list[float] = []
    risks_b: list[float] = []
    improvements: list[float] = []
    wins_a = wins_b = ties = 0
    for index, assumptions in enumerate(_uncertainty_assumptions(project.assumptions)):
        result_a = simulate(project.condition_a, assumptions, "coarse")
        result_b = simulate(project.condition_b, assumptions, "coarse")
        scenario_winner = _winner(
            result_a.peak_top_risk,
            result_b.peak_top_risk,
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
        scenario_results.append(
            UncertaintyScenarioResult(
                index=index,
                assumptions=assumptions,
                peak_risk_a=result_a.peak_top_risk,
                peak_risk_b=result_b.peak_top_risk,
                winner=scenario_winner,
                a_vs_b_peak_risk_change_percent=improvement,
            )
        )

    sample_count = len(scenario_results)
    a_rate = wins_a / sample_count
    b_rate = wins_b / sample_count
    median_improvement = float(np.median(improvements))
    robust_a = a_rate >= 0.80 and median_improvement >= 10.0
    robust_b = b_rate >= 0.80 and median_improvement <= -10.0
    if robust_a:
        winner: Winner = "a"
        classification = "robust_a"
        verdict = "조건 A가 가정 변화의 80% 이상에서 10% 이상의 중앙 개선을 보여 강건하게 유리합니다."
    elif robust_b:
        winner = "b"
        classification = "robust_b"
        verdict = "조건 B가 가정 변화의 80% 이상에서 10% 이상의 중앙 개선을 보여 강건하게 유리합니다."
    elif a_rate >= 0.60:
        winner = "a"
        classification = "weak_a"
        verdict = "조건 A가 더 자주 우세하지만 가정에 민감하므로 약한 우세로 판정합니다."
    elif b_rate >= 0.60:
        winner = "b"
        classification = "weak_b"
        verdict = "조건 B가 더 자주 우세하지만 가정에 민감하므로 약한 우세로 판정합니다."
    elif ties == sample_count:
        winner = "tie"
        classification = "tie"
        verdict = "모든 가정 조합에서 두 조건이 동률 허용범위 안입니다."
    else:
        winner = "inconclusive"
        classification = "inconclusive"
        verdict = "가정에 따라 우세 조건이 바뀌어 현재 정보만으로는 판정을 보류합니다."

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
        a_vs_b_peak_risk_change_percent=nominal_change,
        uncertainty_enabled=True,
        uncertainty_wins_a=wins_a,
        uncertainty_wins_b=wins_b,
        uncertainty_ties=ties,
        uncertainty_a_win_rate=a_rate,
        uncertainty_b_win_rate=b_rate,
        median_a_vs_b_peak_risk_change_percent=median_improvement,
        risk_quantiles_a=_quantiles(risks_a),
        risk_quantiles_b=_quantiles(risks_b),
        scenario_results=scenario_results,
        rank_flip_scenarios=rank_flips,
        warnings=warnings,
    )

