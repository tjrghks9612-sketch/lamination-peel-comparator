"""Cross-cutting invariants for the public simulation and comparison APIs.

The numerical modules are imported lazily so model-only development remains
testable while the solver is being assembled. Once the public modules exist,
these tests execute normally and no compatibility branch changes assertions.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from lamination_sim.models import Condition, ProjectV1, SweepLevel, TrajectoryPoint
from lamination_sim.presets import default_condition, default_project
from lamination_sim.trajectory import waypoint_times


def _load_solver_api():
    try:
        from lamination_sim.comparison import compare
        from lamination_sim.simulation import simulate
    except ModuleNotFoundError as exc:
        if exc.name in {"lamination_sim.comparison", "lamination_sim.simulation"}:
            pytest.skip(f"public solver API is not available yet: {exc}")
        raise
    return simulate, compare


def _get_value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    for container_name in ("metrics", "summary", "nominal"):
        container = getattr(obj, container_name, None)
        if container is not None:
            for name in names:
                if isinstance(container, dict) and name in container:
                    return container[name]
                if hasattr(container, name):
                    return getattr(container, name)
    raise AssertionError(f"none of {names!r} were exposed by {type(obj).__name__}")


def _plain(value: Any) -> Any:
    """Convert a result to a deterministic, equality-friendly tree."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="python"))
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _result_pair(comparison: Any) -> tuple[Any, Any]:
    result_a = _get_value(comparison, "result_a", "simulation_a", "condition_a_result")
    result_b = _get_value(comparison, "result_b", "simulation_b", "condition_b_result")
    return result_a, result_b


def _peak_top_risk(result: Any) -> float:
    return float(
        _get_value(
            result,
            "peak_top_risk",
            "top_peak_risk",
            "peak_rtop",
            "max_top_risk",
        )
    )


def _use_single_high_tension(project: ProjectV1) -> None:
    project.tension_sweep.enabled = False
    project.tension_sweep.preload_levels = [SweepLevel(label="test", value=20.0)]
    project.tension_sweep.stiffness_levels = [SweepLevel(label="test", value=1.0)]


def test_condition_rejects_any_trajectory_length_other_than_six() -> None:
    condition = default_condition()
    payload = condition.model_dump(mode="python")
    payload["trajectory"] = payload["trajectory"][:-1]

    with pytest.raises(ValidationError, match="exactly 6"):
        Condition.model_validate(payload)


def test_point_speed_is_a_waypoint_target_for_adjacent_segments() -> None:
    points = [
        TrajectoryPoint(x_mm=float(x), y_mm=0.0, z_mm=0.0, speed_mm_s=speed)
        for x, speed in zip((0, 10, 30, 60, 100, 150), (5, 10, 15, 20, 25, 999))
    ]

    times = waypoint_times(points)

    distances = np.diff(np.asarray((0, 10, 30, 60, 100, 150), dtype=float))
    speeds = np.asarray((5, 10, 15, 20, 25, 999), dtype=float)
    expected_durations = 2.0 * distances / (speeds[:-1] + speeds[1:])
    np.testing.assert_allclose(np.diff(times), expected_durations)


def test_nominal_simulation_is_deterministic() -> None:
    simulate, _ = _load_solver_api()
    project = default_project()

    first = simulate(project.condition_a, project.assumptions, "coarse")
    second = simulate(project.condition_a, project.assumptions, "coarse")

    assert _plain(first) == _plain(second)


def test_peel_progress_and_top_damage_never_reverse() -> None:
    simulate, _ = _load_solver_api()
    project = default_project()

    result = simulate(project.condition_a, project.assumptions, "coarse")

    peel_ratio = np.asarray(_get_value(result, "bottom_peel_ratio"), dtype=float)
    damage_area = np.asarray(_get_value(result, "top_damage_area_mm2"), dtype=float)
    assert np.all(np.diff(peel_ratio) >= -1.0e-12)
    assert np.all(np.diff(damage_area) >= -1.0e-12)
    assert 0.0 < peel_ratio[-1] < 1.0


def test_identical_conditions_have_identical_nominal_risk() -> None:
    _, compare = _load_solver_api()
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)

    comparison = compare(project)
    result_a, result_b = _result_pair(comparison)

    assert _peak_top_risk(result_a) == pytest.approx(_peak_top_risk(result_b))
    assert _get_value(comparison, "winner") == "tie"
    assert _get_value(comparison, "classification") == "tie"


def test_swapping_conditions_swaps_nominal_results() -> None:
    _, compare = _load_solver_api()
    project = default_project()
    project.run_uncertainty = False
    _use_single_high_tension(project)
    project.condition_b.top_film.adhesion_gf *= 1.5

    forward = compare(project)
    swapped_project = ProjectV1(
        condition_a=project.condition_b,
        condition_b=project.condition_a,
        assumptions=project.assumptions,
        run_uncertainty=False,
    )
    _use_single_high_tension(swapped_project)
    reverse = compare(swapped_project)

    forward_a, forward_b = _result_pair(forward)
    reverse_a, reverse_b = _result_pair(reverse)
    assert _peak_top_risk(forward_a) == pytest.approx(_peak_top_risk(reverse_b))
    assert _peak_top_risk(forward_b) == pytest.approx(_peak_top_risk(reverse_a))
    assert _get_value(forward, "winner") == "b"
    assert _get_value(reverse, "winner") == "a"


def test_stronger_top_adhesion_does_not_raise_normalized_risk() -> None:
    simulate, _ = _load_solver_api()
    project = default_project()
    baseline = project.condition_a.model_copy(deep=True)
    stronger = baseline.model_copy(deep=True)
    stronger.top_film.adhesion_gf *= 1.5

    baseline_result = simulate(baseline, project.assumptions, "coarse")
    stronger_result = simulate(stronger, project.assumptions, "coarse")

    assert _peak_top_risk(stronger_result) <= _peak_top_risk(baseline_result)
