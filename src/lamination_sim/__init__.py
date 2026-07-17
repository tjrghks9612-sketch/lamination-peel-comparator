"""Lamination peel trajectory comparator public API."""

from .comparison import ComparisonResult, compare
from .models import (
    AssumptionSet,
    Condition,
    FilmConfig,
    PanelConfig,
    ProjectV1,
    PullTapeConfig,
    TrajectoryPoint,
)
from .simulation import SimulationResult, simulate

__version__ = "0.4.0"

__all__ = [
    "AssumptionSet",
    "ComparisonResult",
    "Condition",
    "FilmConfig",
    "PanelConfig",
    "ProjectV1",
    "PullTapeConfig",
    "SimulationResult",
    "TrajectoryPoint",
    "compare",
    "simulate",
]
