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
TensionMode = Literal["equal_preload", "shared_rest_length"]
RestLengthReference = Literal["condition_a", "condition_b", "custom"]

# A fixed reduced-order estimate for the PET pull tape used by common film
# peeling fixtures.  It is intentionally not part of the sensitivity sweep;
# only the unknown initial preload remains a scenario variable.
#
# Estimate: E ~= 4.55 GPa, 50 um PET backing, 10 mm tape width and a 1 m
# effective compliant length for the tape + free film path:
#   k = E * (width * thickness) / effective_length ~= 2.25 N/mm.
PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM = 2.25
PREDICTED_PULL_TAPE_STIFFNESS_LABEL = "PET estimate (fixed)"

PANEL_THICKNESS_MM = 0.056
PULL_TAPE_STIFFNESS_N_PER_MM = 2.25
INITIAL_TENSION_CONDITIONS_N = (0.0, 0.5, 1.5)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrimGeometryConfig(_StrictModel):
    """Future trim features inside the current rectangular laminate sheet.

    These dimensions are display/process references only.  The cohesive mesh
    continues to cover the full sharp-cornered pre-trim panel rectangle.
    """

    pretrim_margin_mm: float = Field(default=1.5, gt=0.0, le=50.0)
    cell_corner_radius_mm: float = Field(default=6.0, ge=0.0, le=100.0)
    pad_height_mm: float = Field(default=3.0, gt=0.0, le=100.0)
    island_width_mm: float = Field(default=22.0, gt=0.0, le=100.0)
    island_height_mm: float = Field(default=6.0, gt=0.0, le=50.0)


class PanelConfig(_StrictModel):
    preset: PanelPreset = "pro"
    width_mm: float = Field(default=71.5, gt=0.0, le=500.0)
    height_mm: float = Field(default=149.6, gt=0.0, le=500.0)
    thickness_mm: float = Field(default=PANEL_THICKNESS_MM, gt=0.0, le=20.0)
    corner_radius_mm: float = Field(default=0.0, ge=0.0)
    trim_geometry: TrimGeometryConfig = Field(default_factory=TrimGeometryConfig)

    @model_validator(mode="after")
    def _radius_fits_panel(self) -> "PanelConfig":
        if self.corner_radius_mm > min(self.width_mm, self.height_mm) / 2.0:
            raise ValueError("corner_radius_mm must not exceed half the short side")
        trim = self.trim_geometry
        cell_width = self.width_mm - 2.0 * trim.pretrim_margin_mm
        cell_height = (
            self.height_mm
            - 2.0 * trim.pretrim_margin_mm
            - trim.pad_height_mm
        )
        if cell_width <= 0.0 or cell_height <= 0.0:
            raise ValueError(
                "trim margin and pad height must leave a positive finished-cell area"
            )
        if trim.cell_corner_radius_mm > min(cell_width, cell_height) / 2.0:
            raise ValueError(
                "cell_corner_radius_mm must fit inside the finished-cell area"
            )
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
        if self.trajectory[0].z_mm < 0.0:
            raise ValueError(
                "main trajectory P1 Z must be at or above the panel surface (Z=0)"
            )
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
            "Deprecated compatibility field. cohesive-v5-tension-sweep ignores "
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
    damage_max_iterations: int = Field(default=150, ge=2, le=300)
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


class SweepLevel(_StrictModel):
    label: str = Field(min_length=1, max_length=40)
    value: float = Field(ge=0.0, le=10000.0)


def _default_preload_levels() -> list[SweepLevel]:
    return [
        SweepLevel(label="Low", value=0.0),
        SweepLevel(label="Mid", value=0.5),
        SweepLevel(label="High", value=1.5),
    ]


def _default_stiffness_levels() -> list[SweepLevel]:
    return [
        SweepLevel(
            label=PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
            value=PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
        )
    ]


class TensionCase(_StrictModel):
    """Resolved tension law for one condition and one sensitivity cell."""

    initial_preload_n: float = Field(default=0.5, ge=0.0, le=10000.0)
    tape_stiffness_n_per_mm: float = Field(
        default=PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
        ge=0.0,
        le=10000.0,
    )


class TensionSweepConfig(_StrictModel):
    enabled: bool = True
    mode: TensionMode = "equal_preload"
    preload_levels: list[SweepLevel] = Field(default_factory=_default_preload_levels)
    tape_stiffness_n_per_mm: float = Field(
        default=PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
        gt=0.0,
        le=10000.0,
        description=(
            "Fixed equivalent pull-tape stiffness estimated from common PET "
            "film data; not swept with initial preload."
        ),
    )
    # Kept only so projects written by v0.5.4 can still be opened.  The
    # comparison engine intentionally ignores this legacy list and uses the
    # fixed tape_stiffness_n_per_mm above.
    stiffness_levels: list[SweepLevel] = Field(
        default_factory=lambda: [
            SweepLevel(
                label=PREDICTED_PULL_TAPE_STIFFNESS_LABEL,
                value=PREDICTED_PULL_TAPE_STIFFNESS_N_PER_MM,
            )
        ],
        description="Deprecated compatibility field; no longer a sweep axis.",
    )
    rest_length_reference: RestLengthReference = "condition_a"
    custom_rest_length_mm: float | None = Field(default=None, gt=0.0, le=10000.0)
    nest_material_uncertainty: bool = False

    @model_validator(mode="after")
    def _valid_sweep(self) -> "TensionSweepConfig":
        if not self.preload_levels:
            raise ValueError("tension sweep requires at least one preload level")
        for name, levels in (("preload", self.preload_levels),):
            labels = [level.label.casefold() for level in levels]
            if len(labels) != len(set(labels)):
                raise ValueError(f"{name} level labels must be unique")
        if (
            self.mode == "shared_rest_length"
            and self.rest_length_reference == "custom"
            and self.custom_rest_length_mm is None
        ):
            raise ValueError("custom shared rest length mode requires custom_rest_length_mm")
        return self


class ProjectV1(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    condition_a: Condition
    condition_b: Condition
    assumptions: AssumptionSet = Field(default_factory=AssumptionSet)
    tension_sweep: TensionSweepConfig = Field(default_factory=TensionSweepConfig)
    run_uncertainty: bool = False
