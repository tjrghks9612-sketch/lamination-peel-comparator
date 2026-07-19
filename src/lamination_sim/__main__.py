"""Launch the desktop application with ``python -m lamination_sim``."""

from __future__ import annotations

import math
import sys

from lamination_sim.ui import run_app


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""

    arguments = list(sys.argv if argv is None else argv)
    if "--self-test" in arguments:
        return _self_test()
    return run_app(arguments)


def _self_test() -> int:
    """Exercise the packaged numerical stack without opening a window."""

    from lamination_sim.comparison import compare
    from lamination_sim.presets import default_project

    project = default_project()
    project.run_uncertainty = False
    result = compare(project)
    values = (
        result.result_a.peak_top_risk,
        result.result_b.peak_top_risk,
        result.result_a.final_bottom_peel_ratio,
        result.result_b.final_bottom_peel_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        return 2
    scenarios = result.tension_scenario_results
    if len(scenarios) != 3:
        return 3
    if any(item.winner not in {"tie", "inconclusive"} for item in scenarios):
        return 3
    if any(
        not math.isclose(item.final_bottom_peel_ratio_a, item.final_bottom_peel_ratio_b)
        or not math.isclose(item.peak_top_risk_a, item.peak_top_risk_b)
        for item in scenarios
    ):
        return 3
    if (
        result.tension_wins_a
        + result.tension_wins_b
        + result.tension_ties
        + result.tension_inconclusive
        != 3
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
