"""Lamination peel trajectory comparator public API."""

from .comparison import ComparisonResult, compare
from .models import (
    PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
    AssumptionSet,
    Condition,
    FilmConfig,
    PanelConfig,
    ProjectV1,
    PullTapeConfig,
    TrimGeometryConfig,
    TrajectoryPoint,
)
from .simulation import SimulationResult, simulate

__version__ = "0.6.0"

__all__ = [
    "AssumptionSet",
    "ComparisonResult",
    "Condition",
    "FilmConfig",
    "PanelConfig",
    "PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM",
    "ProjectV1",
    "PullTapeConfig",
    "SimulationResult",
    "TrimGeometryConfig",
    "TrajectoryPoint",
    "compare",
    "simulate",
]
