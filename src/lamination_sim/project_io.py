"""Versioned project and trajectory file I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def save_project(project: Any, path: str | Path) -> Path:
    """Write a Pydantic project model as stable, UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(project, "model_dump"):
        payload = project.model_dump(mode="json")
    elif isinstance(project, dict):
        payload = project
    else:
        raise TypeError("project must be a Pydantic model or dictionary")
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_project(path: str | Path):
    """Load and validate a ProjectV1 JSON document."""
    from .models import ProjectV1

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectV1.model_validate(payload)


def load_trajectory_csv(path: str | Path):
    """Load exactly six trajectory rows from the documented CSV schema."""
    from .models import TrajectoryPoint

    rows: list[TrajectoryPoint] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x_mm", "y_mm", "z_mm", "speed_mm_s"}
        if not required.issubset(reader.fieldnames or []):
            missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
            raise ValueError(f"trajectory CSV is missing columns: {missing}")
        for row in reader:
            rows.append(
                TrajectoryPoint(
                    x_mm=float(row["x_mm"]),
                    y_mm=float(row["y_mm"]),
                    z_mm=float(row["z_mm"]),
                    speed_mm_s=float(row["speed_mm_s"]),
                )
            )
    if len(rows) != 6:
        raise ValueError(f"trajectory must contain exactly 6 rows, got {len(rows)}")
    return rows


def save_trajectory_csv(points: Iterable[Any], path: str | Path) -> Path:
    """Write six trajectory points using the public import schema."""
    values = list(points)
    if len(values) != 6:
        raise ValueError(f"trajectory must contain exactly 6 points, got {len(values)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["point", "x_mm", "y_mm", "z_mm", "speed_mm_s"],
        )
        writer.writeheader()
        for index, point in enumerate(values, start=1):
            getter = (
                (lambda key: point[key])
                if isinstance(point, dict)
                else (lambda key: getattr(point, key))
            )
            writer.writerow(
                {
                    "point": f"P{index}",
                    "x_mm": getter("x_mm"),
                    "y_mm": getter("y_mm"),
                    "z_mm": getter("z_mm"),
                    "speed_mm_s": getter("speed_mm_s"),
                }
            )
    return target

