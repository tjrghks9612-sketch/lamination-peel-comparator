from __future__ import annotations

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

