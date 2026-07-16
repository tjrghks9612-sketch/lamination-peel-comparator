"""Loose coupling between the GUI and the numerical core.

The core is intentionally imported lazily.  This keeps UI module imports useful for
designer/smoke tests and gives readable diagnostics if an installation is missing
an optional numerical dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class RunBundle:
    project: Any
    comparison: Any
    result_a: Any = None
    result_b: Any = None


def get_value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def model_from_payload(model: type, payload: Mapping[str, Any]) -> Any:
    if hasattr(model, "model_validate"):
        return model.model_validate(dict(payload))
    return model(**dict(payload))


def build_project(payload: Mapping[str, Any]) -> Any:
    """Construct ``ProjectV1`` from the editor's JSON-compatible payload."""

    from lamination_sim.models import ProjectV1

    return model_from_payload(ProjectV1, payload)


def validate_project_payload(payload: Mapping[str, Any]) -> tuple[bool, str]:
    try:
        build_project(payload)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def run_project(project: Any) -> RunBundle:
    """Run the public comparison API with a conservative compatibility fallback."""

    from lamination_sim.comparison import compare

    try:
        comparison = compare(project)
    except TypeError as original_error:
        # Early core prototypes accepted two simulation results.  Retaining this
        # fallback costs little and lets saved UI workspaces survive that change.
        from lamination_sim.simulation import simulate

        condition_a = get_value(project, "condition_a", "a")
        condition_b = get_value(project, "condition_b", "b")
        assumptions = get_value(project, "assumptions", "assumption_set")
        result_a = _call_simulate(simulate, condition_a, assumptions)
        result_b = _call_simulate(simulate, condition_b, assumptions)
        try:
            comparison = compare(result_a, result_b)
        except TypeError:
            raise original_error
        return RunBundle(project, comparison, result_a, result_b)

    result_a = get_value(
        comparison,
        "result_a",
        "simulation_a",
        "condition_a_result",
        default=None,
    )
    result_b = get_value(
        comparison,
        "result_b",
        "simulation_b",
        "condition_b_result",
        default=None,
    )
    return RunBundle(project, comparison, result_a, result_b)


def _call_simulate(func, condition: Any, assumptions: Any) -> Any:
    params = signature(func).parameters
    if len(params) >= 2:
        return func(condition, assumptions)
    return func(condition)


def sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, Iterable):
        try:
            return list(value)
        except TypeError:
            return []
    return []


SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time_s", "times_s", "time", "times"),
    "top_risk": (
        "top_risk",
        "top_peak_risk",
        "top_risk_history",
        "peak_rtop",
        "r_top",
    ),
    "bottom_peel": (
        "bottom_peel_ratio",
        "bottom_peel_history",
        "peel_ratio",
        "peel_fraction",
    ),
    "panel_lift": (
        "panel_max_lift_mm",
        "panel_lift_history_mm",
        "max_lift_mm",
        "panel_lift",
    ),
    "force": (
        "reaction_force_n",
        "force_resultant_n",
        "force_history_n",
        "pull_force_n",
        "force",
    ),
    "twist": ("panel_twist_mm", "twist_history_mm", "panel_twist", "twist"),
}


def result_series(result: Any, key: str) -> list[float]:
    """Extract a numeric series from mapping, object, metrics, or frame records."""

    aliases = SERIES_ALIASES.get(key, (key,))
    containers = [
        result,
        get_value(result, "history", "histories"),
        get_value(result, "metrics", "series"),
    ]
    for container in containers:
        candidate = get_value(container, *aliases, default=None)
        values = _numeric_sequence(candidate)
        if values:
            return values

    frames = sequence(get_value(result, "frames", "states", "steps", default=[]))
    values: list[float] = []
    for frame in frames:
        candidate = get_value(frame, *aliases, default=None)
        number = _to_number(candidate)
        if number is not None:
            values.append(number)
    return values


def scalar_metric(obj: Any, *names: str, default: float | None = None) -> float | None:
    containers = [obj, get_value(obj, "metrics", "summary", "nominal", default=None)]
    for container in containers:
        value = get_value(container, *names, default=None)
        number = _to_number(value)
        if number is not None:
            return number
    return default


def _numeric_sequence(value: Any) -> list[float]:
    result: list[float] = []
    for item in sequence(value):
        if isinstance(item, (list, tuple)) and item:
            item = item[-1]
        number = _to_number(item)
        if number is not None:
            result.append(number)
    return result


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

