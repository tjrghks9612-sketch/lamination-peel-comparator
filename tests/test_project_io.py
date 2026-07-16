from __future__ import annotations

from pathlib import Path

from lamination_sim.presets import default_project
from lamination_sim.project_io import (
    load_project,
    load_trajectory_csv,
    save_project,
    save_trajectory_csv,
)


def test_project_json_round_trip(tmp_path):
    original = default_project()
    path = save_project(original, tmp_path / "project.json")
    restored = load_project(path)
    assert restored == original


def test_trajectory_csv_round_trip(tmp_path):
    original = default_project().condition_a.trajectory
    path = save_trajectory_csv(original, tmp_path / "trajectory.csv")
    restored = load_trajectory_csv(path)
    assert restored == original


def test_supplied_measured_project_file_loads_with_zero_p1_speed() -> None:
    path = Path(__file__).parents[1] / "examples" / "measured_ab.peel.json"

    project = load_project(path)

    assert project.condition_a.trajectory[0].z_mm == 4.0
    assert project.condition_b.trajectory[0].z_mm == 14.0
    assert project.condition_a.trajectory[0].speed_mm_s == 0.0
    assert project.condition_b.trajectory[0].speed_mm_s == 0.0
