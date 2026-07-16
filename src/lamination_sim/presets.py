"""Editable panel presets and convenient, valid project factories."""

from __future__ import annotations

from copy import deepcopy

from .models import (
    Condition,
    FilmConfig,
    PanelConfig,
    ProjectV1,
    PullTapeConfig,
    TrajectoryPoint,
)


PANEL_PRESETS: dict[str, dict[str, float | str]] = {
    "pro": {
        "preset": "pro",
        "width_mm": 71.5,
        "height_mm": 149.6,
        "thickness_mm": 0.7,
        "corner_radius_mm": 0.0,
    },
    "pro_max": {
        "preset": "pro_max",
        "width_mm": 77.6,
        "height_mm": 163.0,
        "thickness_mm": 0.7,
        "corner_radius_mm": 0.0,
    },
}


def panel_from_preset(name: str = "pro", **overrides: float | str) -> PanelConfig:
    if name not in PANEL_PRESETS:
        raise ValueError(f"unknown panel preset: {name!r}")
    values = deepcopy(PANEL_PRESETS[name])
    values.update(overrides)
    return PanelConfig.model_validate(values)


def default_trajectory(
    panel: PanelConfig | None = None,
    start_corner: str = "bottom_left",
) -> list[TrajectoryPoint]:
    """Return a six-point diagonal demonstration trajectory.

    The path is deliberately conservative: it starts close to the panel and
    raises Z progressively.  It is sample data, not a recommended process.
    """

    panel = panel or panel_from_preset("pro")
    corner_signs = {
        "bottom_left": (0.0, 0.0, 1.0, 1.0),
        "bottom_right": (panel.width_mm, 0.0, -1.0, 1.0),
        "top_left": (0.0, panel.height_mm, 1.0, -1.0),
        "top_right": (panel.width_mm, panel.height_mm, -1.0, -1.0),
    }
    if start_corner not in corner_signs:
        raise ValueError(f"unknown start corner: {start_corner!r}")
    x0, y0, sx, sy = corner_signs[start_corner]
    fractions = (0.0, 0.08, 0.25, 0.50, 0.75, 1.0)
    z_values = (0.0, 1.0, 3.0, 6.0, 10.0, 15.0)
    speeds = (5.0, 10.0, 20.0, 30.0, 40.0, 40.0)
    return [
        TrajectoryPoint(
            x_mm=x0 + sx * panel.width_mm * fraction,
            y_mm=y0 + sy * panel.height_mm * fraction,
            z_mm=z_mm,
            speed_mm_s=speed,
        )
        for fraction, z_mm, speed in zip(fractions, z_values, speeds)
    ]


def default_condition(
    name: str = "Condition A",
    panel_preset: str = "pro",
    start_corner: str = "bottom_left",
) -> Condition:
    panel = panel_from_preset(panel_preset)
    return Condition(
        name=name,
        panel=panel,
        top_film=FilmConfig(adhesion_gf=2.0),
        bottom_film=FilmConfig(adhesion_gf=1.5),
        pull_tape=PullTapeConfig(start_corner=start_corner),
        trajectory=default_trajectory(panel, start_corner),
    )


def default_project() -> ProjectV1:
    condition_a = default_condition("Condition A")
    condition_b = condition_a.model_copy(deep=True)
    condition_b.name = "Condition B"
    return ProjectV1(condition_a=condition_a, condition_b=condition_b)

