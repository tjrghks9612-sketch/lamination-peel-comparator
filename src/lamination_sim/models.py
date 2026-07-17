"""Validated public input models for the reverse-peel comparator.

The models intentionally keep measured inputs separate from uncertain material
assumptions.  This makes an A/B result auditable: values in ``Condition`` are
operator inputs, while values in ``AssumptionSet`` are applied identically to
both conditions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PanelPreset = Literal["pro", "pro_max", "custom"]
StartCorner = Literal["bottom_left", "bottom_right", "top_left", "top_right"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PanelConfig(_StrictModel):
    preset: PanelPreset = "pro"
    width_mm: float = Field(default=71.5, gt=0.0, le=500.0)
    height_mm: float = Field(default=149.6, gt=0.0, le=500.0)
    thickness_mm: float = Field(default=0.7, gt=0.0, le=20.0)
    corner_radius_mm: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _radius_fits_panel(self) -> "PanelConfig":
        if self.corner_radius_mm > min(self.width_mm, self.height_mm) / 2.0:
            raise ValueError("corner_radius_mm must not exceed half the short side")
        return self


class FilmConfig(_StrictModel):
    pet_thickness_um: float = Field(default=50.0, gt=0.0, le=2000.0)
    psa_thickness_um: float = Field(default=20.0, gt=0.0, le=2000.0)
    adhesion_gf: float = Field(default=1.5, gt=0.0, le=10000.0)


class PullTapeConfig(_StrictModel):
    start_corner: StartCorner = "bottom_left"
    width_mm: float = Field(default=10.0, gt=0.0, le=200.0)
    length_mm: float = Field(default=10.0, gt=0.0, le=200.0)


class TrajectoryPoint(_StrictModel):
    x_mm: float
    y_mm: float
    z_mm: float
    speed_mm_s: float = Field(ge=0.0, le=10000.0)


class Condition(_StrictModel):
    name: str = Field(default="Condition", min_length=1, max_length=100)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    top_film: FilmConfig = Field(
        default_factory=lambda: FilmConfig(adhesion_gf=2.0)
    )
    bottom_film: FilmConfig = Field(
        default_factory=lambda: FilmConfig(adhesion_gf=1.5)
    )
    pull_tape: PullTapeConfig = Field(default_factory=PullTapeConfig)
    trajectory: list[TrajectoryPoint]
    initial_approach: list[TrajectoryPoint] | None = None

    @field_validator("trajectory")
    @classmethod
    def _exactly_six_distinct_points(
        cls, points: list[TrajectoryPoint]
    ) -> list[TrajectoryPoint]:
        if len(points) != 6:
            raise ValueError("trajectory must contain exactly 6 points")
        for index, (left, right) in enumerate(zip(points, points[1:]), start=1):
            distance_sq = (
                (right.x_mm - left.x_mm) ** 2
                + (right.y_mm - left.y_mm) ** 2
                + (right.z_mm - left.z_mm) ** 2
            )
            if distance_sq <= 1.0e-18:
                raise ValueError(
                    f"trajectory points {index} and {index + 1} must be distinct"
                )
            if left.speed_mm_s + right.speed_mm_s <= 0.0:
                raise ValueError(
                    f"trajectory segment {index}-{index + 1} cannot have zero "
                    "target speed at both endpoints"
                )
        return points

    @field_validator("initial_approach")
    @classmethod
    def _valid_initial_approach(
        cls, points: list[TrajectoryPoint] | None
    ) -> list[TrajectoryPoint] | None:
        if points is None:
            return None
        if len(points) < 2:
            raise ValueError("initial_approach must contain at least 2 points")
        for index, (left, right) in enumerate(zip(points, points[1:]), start=1):
            distance_sq = (
                (right.x_mm - left.x_mm) ** 2
                + (right.y_mm - left.y_mm) ** 2
                + (right.z_mm - left.z_mm) ** 2
            )
            if distance_sq <= 1.0e-18:
                raise ValueError(
                    f"initial_approach points {index} and {index + 1} must be distinct"
                )
            if left.speed_mm_s + right.speed_mm_s <= 0.0:
                raise ValueError(
                    f"initial_approach segment {index}-{index + 1} cannot have "
                    "zero target speed at both endpoints"
                )
        return points

    @model_validator(mode="after")
    def _approach_joins_p1(self) -> "Condition":
        if self.initial_approach is None:
            return self
        endpoint = self.initial_approach[-1]
        p1 = self.trajectory[0]
        endpoint_values = (
            endpoint.x_mm,
            endpoint.y_mm,
            endpoint.z_mm,
            endpoint.speed_mm_s,
        )
        p1_values = (p1.x_mm, p1.y_mm, p1.z_mm, p1.speed_mm_s)
        if any(
            abs(left - right) > 1.0e-9
            for left, right in zip(endpoint_values, p1_values)
        ):
            raise ValueError(
                "initial_approach must end at the main trajectory P1 position "
                "and target speed"
            )
        return self


class AssumptionSet(_StrictModel):
    """Shared uncertain inputs used by both sides of a comparison."""

    calibration_version: str = Field(
        default="uncalibrated", min_length=1, max_length=100
    )
    test_width_mm: float = Field(default=25.0, gt=0.0, le=500.0)
    test_angle_deg: float = Field(default=180.0, gt=0.0, le=180.0)
    panel_young_modulus_gpa: float = Field(default=70.0, gt=0.0, le=1000.0)
    panel_poisson_ratio: float = Field(default=0.22, ge=0.0, lt=0.5)
    pet_young_modulus_gpa: float = Field(default=3.0, gt=0.0, le=100.0)
    pet_poisson_ratio: float = Field(default=0.38, ge=0.0, lt=0.5)
    psa_modulus_mpa: float = Field(default=0.5, gt=0.0, le=1000.0)
    max_pull_force_n: float = Field(default=20.0, gt=0.0, le=10000.0)
    mixed_mode_shear_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    vertical_front_reach_factor: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        description=(
            "Deprecated compatibility field. cohesive-v4-film-payout ignores "
            "this value because vertical travel does not create free interface reach."
        ),
    )
    cohesive_stiffness_floor_ratio: float = Field(
        default=0.01, gt=0.0, le=0.25
    )
    torsion_coupling: float = Field(default=0.25, ge=0.0, le=10.0)
    damage_convergence_tolerance: float = Field(
        default=1.0e-3, gt=0.0, le=0.1
    )
    damage_max_iterations: int = Field(default=100, ge=2, le=200)
    bottom_completion_ratio: float = Field(default=0.98, ge=0.5, le=1.0)
    speed_exponent: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_speed_mm_s: float = Field(default=5.0, gt=0.0, le=10000.0)
    quasi_static_speed_mm_s: float = Field(default=0.1, gt=0.0, le=10000.0)
    grip_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    mesh_size_mm: float = Field(default=2.0, ge=0.25, le=20.0)
    fine_mesh_size_mm: float = Field(default=1.0, ge=0.25, le=20.0)
    coarse_mesh_size_mm: float = Field(default=4.0, ge=0.25, le=40.0)
    time_steps_normal: int = Field(default=101, ge=21, le=1001)
    time_steps_fine: int = Field(default=161, ge=21, le=2001)
    time_steps_coarse: int = Field(default=41, ge=11, le=501)
    uncertainty_samples: int = Field(default=24, ge=1, le=200)
    minimum_robust_samples: int = Field(default=20, ge=2, le=200)
    random_seed: int = 20260716
    tie_tolerance_percent: float = Field(default=2.0, ge=0.0, le=100.0)


class ProjectV1(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    condition_a: Condition
    condition_b: Condition
    assumptions: AssumptionSet = Field(default_factory=AssumptionSet)
    run_uncertainty: bool = True
