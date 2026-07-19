"""Headless reduced-order reverse-peel simulation.

The commanded XYZ path is an input to, rather than a substitute for, interface
failure.  A node-wise bottom cohesive state advances from the pull-tape corner
only when its Kendall-style energy release rate reaches the local fracture
energy and the actuator supplies the required work.  The resulting 3-D force
and moment load an elastic plate whose top PSA foundation loses stiffness as
cohesive damage grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse
from scipy.sparse.linalg import splu

from .models import AssumptionSet, Condition, TensionCase
from .trajectory import interpolate_trajectory


Resolution = Literal["coarse", "normal", "fine"]
MODEL_VERSION = "cohesive-v6-visual-loads"
FloatArray = NDArray[np.float64]


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str
    calibration_version: str
    condition_name: str
    resolution: Resolution
    input_hash: str
    initial_state_mode: Literal["p1_equilibrium", "specified_approach"]
    main_trajectory_start_index: int
    trajectory_waypoint_indices: list[int]

    time_s: list[float]
    position_xyz_mm: list[list[float]]
    speed_mm_s: list[float]
    trajectory_progress: list[float]
    peel_progress: list[float]
    peel_angle_deg: list[float]
    force_xyz_n: list[list[float]]
    force_resultant_n: list[float]
    moment_xyz_n_mm: list[list[float]]
    moment_resultant_n_mm: list[float]
    pull_force_activation: list[float]
    tension_n: list[float]
    tape_span_length_mm: list[float]
    p1_span_length_mm: float
    initial_preload_n: float
    tape_stiffness_n_per_mm: float
    bottom_peel_ratio: list[float]
    top_peak_risk: list[float]
    top_risk_area_mm2: list[float]
    top_damage_area_mm2: list[float]
    top_min_foundation_retention: list[float]
    top_interface_normal_force_n: list[float]
    top_interface_reaction_centroid_xy_mm: list[list[float]]
    panel_max_lift_mm: list[float]
    panel_twist_mm: list[float]
    bottom_damage_iterations: list[int]
    top_damage_iterations: list[int]
    bottom_damage_converged: list[bool]
    top_damage_converged: list[bool]

    peak_top_risk: float
    peak_top_risk_time_s: float
    final_bottom_peel_ratio: float
    max_top_risk_area_mm2: float
    max_top_damage_area_mm2: float
    final_top_damage_area_mm2: float
    top_risk_exceedance_duration_s: float
    max_panel_lift_mm: float
    max_panel_twist_mm: float
    max_moment_resultant_n_mm: float
    max_tension_n: float
    max_top_interface_normal_force_n: float

    mesh_shape: tuple[int, int]
    mesh_x_mm: list[float]
    mesh_y_mm: list[float]
    frame_indices: list[int]
    panel_z_frames_mm: list[list[float]]
    bottom_damage_frames: list[list[float]]
    top_damage_frames: list[list[float]]
    top_risk_frames: list[list[float]]
    top_interface_reaction_frames_n: list[list[float]]
    front_segments_mm: list[list[float]]
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class _FrontMechanics:
    indices: NDArray[np.intp]
    drive_ratio: FloatArray
    priority: FloatArray
    force_xyz: FloatArray
    scalar_reaction_n: float
    peel_angle_deg: float


def _resolution_values(
    assumptions: AssumptionSet, resolution: Resolution
) -> tuple[float, int, int]:
    if resolution == "coarse":
        return assumptions.coarse_mesh_size_mm, assumptions.time_steps_coarse, 9
    if resolution == "fine":
        return assumptions.fine_mesh_size_mm, assumptions.time_steps_fine, 31
    if resolution == "normal":
        return assumptions.mesh_size_mm, assumptions.time_steps_normal, 21
    raise ValueError(f"unknown resolution: {resolution!r}")


def _mesh(width: float, height: float, target_size: float) -> tuple[FloatArray, ...]:
    nx = max(4, int(np.ceil(width / target_size)) + 1)
    ny = max(4, int(np.ceil(height / target_size)) + 1)
    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return x, y, xx.ravel(), yy.ravel()


def _nodal_areas(x: FloatArray, y: FloatArray) -> FloatArray:
    """Trapezoidal nodal areas whose sum is the exact panel area."""

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    x_weights = np.full(len(x), dx)
    y_weights = np.full(len(y), dy)
    x_weights[[0, -1]] *= 0.5
    y_weights[[0, -1]] *= 0.5
    return np.outer(y_weights, x_weights).ravel()


def _neumann_second_derivative(count: int, spacing: float) -> sparse.csr_matrix:
    main = np.full(count, -2.0)
    main[0] = main[-1] = -1.0
    off = np.ones(count - 1)
    return sparse.diags((off, main, off), (-1, 0, 1), format="csr") / spacing**2


def _laplacian(x: FloatArray, y: FloatArray) -> sparse.csr_matrix:
    lx = _neumann_second_derivative(len(x), float(x[1] - x[0]))
    ly = _neumann_second_derivative(len(y), float(y[1] - y[0]))
    return (
        sparse.kron(sparse.eye(len(y), format="csr"), lx)
        + sparse.kron(ly, sparse.eye(len(x), format="csr"))
    ).tocsr()


def _plate_components(
    condition: Condition,
    assumptions: AssumptionSet,
    x: FloatArray,
    y: FloatArray,
    laplacian: sparse.csr_matrix,
    nodal_areas: FloatArray,
) -> tuple[sparse.csr_matrix, float, float]:
    cell_area = float(x[1] - x[0]) * float(y[1] - y[0])
    panel_e = assumptions.panel_young_modulus_gpa * 1000.0
    thickness = condition.panel.thickness_mm
    nu = assumptions.panel_poisson_ratio
    bending = panel_e * thickness**3 / (12.0 * (1.0 - nu**2))
    bending_matrix = (
        bending * cell_area * (laplacian.T @ laplacian)
    ).tocsr()
    psa_thickness = condition.top_film.psa_thickness_um / 1000.0
    foundation = assumptions.psa_modulus_mpa / psa_thickness
    reference_diagonal = bending_matrix.diagonal() + foundation * nodal_areas
    regularization = max(float(reference_diagonal.max(initial=0.0)), 1.0) * 1.0e-12
    return bending_matrix, foundation, regularization


def _factor_top(
    bending: sparse.csr_matrix,
    foundation: float,
    nodal_areas: FloatArray,
    damage: FloatArray,
    assumptions: AssumptionSet,
    regularization: float,
):
    retention = (
        assumptions.cohesive_stiffness_floor_ratio
        + (1.0 - assumptions.cohesive_stiffness_floor_ratio) * (1.0 - damage)
    )
    diagonal = foundation * retention * nodal_areas + regularization
    return splu((bending + sparse.diags(diagonal, format="csr")).tocsc())


def _adhesion_energy_n_per_mm(
    adhesion_gf: float,
    pet_thickness_um: float,
    assumptions: AssumptionSet,
    speed_mm_s: float,
) -> float:
    """Infer cohesive fracture energy from a reported peel-force value."""

    measured_force_n = adhesion_gf * 0.00980665
    line_force = measured_force_n / assumptions.test_width_mm
    angle = np.deg2rad(assumptions.test_angle_deg)
    pet_e = assumptions.pet_young_modulus_gpa * 1000.0
    pet_t = pet_thickness_um / 1000.0
    gamma = line_force * (1.0 - np.cos(angle)) + line_force**2 / (
        2.0 * pet_e * pet_t
    )
    effective_speed = max(speed_mm_s, assumptions.quasi_static_speed_mm_s)
    speed_ratio = effective_speed / assumptions.reference_speed_mm_s
    return float(gamma * speed_ratio**assumptions.speed_exponent)


def _energy_release_n_per_mm(
    line_force: FloatArray,
    peel_angle: FloatArray,
    pet_thickness_um: float,
    assumptions: AssumptionSet,
) -> FloatArray:
    extensional_stiffness = (
        assumptions.pet_young_modulus_gpa
        * 1000.0
        * (pet_thickness_um / 1000.0)
    )
    angular = np.maximum(1.0 - np.cos(peel_angle), 0.0)
    stretching = (
        assumptions.mixed_mode_shear_weight
        * line_force**2
        / (2.0 * extensional_stiffness)
    )
    return line_force * angular + stretching


def _corner_distances(
    mesh_x: FloatArray,
    mesh_y: FloatArray,
    width: float,
    height: float,
    corner: str,
) -> tuple[FloatArray, FloatArray]:
    local_x = width - mesh_x if "right" in corner else mesh_x
    local_y = height - mesh_y if "top" in corner else mesh_y
    return local_x, local_y


def _seed_mask(
    local_x: FloatArray,
    local_y: FloatArray,
    x: FloatArray,
    y: FloatArray,
    tape_width: float,
) -> NDArray[np.bool_]:
    edge_x = 0.5 * float(x[1] - x[0]) + 1.0e-12
    edge_y = 0.5 * float(y[1] - y[0]) + 1.0e-12
    return (
        (local_x <= edge_x) & (local_y <= tape_width + 1.0e-12)
    ) | (
        (local_y <= edge_y) & (local_x <= tape_width + 1.0e-12)
    )


def _frontier_mask(
    damage: FloatArray,
    nx: int,
    ny: int,
    seed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    full = damage.reshape(ny, nx) >= 1.0 - 1.0e-10
    partial = (damage.reshape(ny, nx) > 1.0e-12) & ~full
    if not np.any(full) and not np.any(partial):
        return seed.copy()
    padded = np.pad(full, 1, mode="constant", constant_values=False)
    adjacent = np.zeros_like(full)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            adjacent |= padded[
                row_offset : row_offset + ny,
                column_offset : column_offset + nx,
            ]
    intact = damage.reshape(ny, nx) <= 1.0e-12
    return (partial | (intact & adjacent)).ravel()


def _film_reach_mask(
    local_x: FloatArray,
    local_y: FloatArray,
    cumulative_in_plane_travel_mm: float,
    tape_width: float,
) -> NDArray[np.bool_]:
    """Causal inextensible-film payout envelope from the attached corner.

    The old rectangular mask independently converted positive X, Y and Z
    travel into peeled dimensions.  That made vertical motion create free
    interfacial reach and made the answer depend strongly on panel axes.  The
    reduced-order replacement only pays out the measured cumulative in-plane
    gripper travel.  Cohesive damage and the connected frontier still decide
    which nodes inside this isotropic geodesic envelope actually peel.
    """

    reach = max(tape_width + cumulative_in_plane_travel_mm, tape_width)
    return np.hypot(local_x, local_y) <= reach + 1.0e-12


def _actuator_work_n_mm(force_xyz_n: FloatArray, displacement_xyz_mm: FloatArray) -> float:
    """Positive incremental actuator work; transverse/reverse motion adds none."""

    return max(float(np.dot(force_xyz_n, displacement_xyz_mm)), 0.0)


def tension_from_span(
    span_length_mm: float,
    p1_span_length_mm: float,
    initial_preload_n: float,
    tape_stiffness_n_per_mm: float,
    max_pull_force_n: float,
) -> float:
    """Unilateral capped tension law; a tape never carries compression."""

    tension = initial_preload_n + tape_stiffness_n_per_mm * (
        span_length_mm - p1_span_length_mm
    )
    return min(max(float(tension), 0.0), float(max_pull_force_n))


def p1_span_length_mm(condition: Condition) -> float:
    """Distance from the initially attached corner/front to the P1 gripper."""

    corner = _corner_coordinate(
        condition.panel.width_mm,
        condition.panel.height_mm,
        condition.pull_tape.start_corner,
    )
    p1 = condition.trajectory[0]
    return float(np.linalg.norm(np.asarray((p1.x_mm, p1.y_mm, p1.z_mm)) - np.asarray((*corner, 0.0))))


def _front_mechanics(
    damage: FloatArray,
    seed: NDArray[np.bool_],
    kinematic_mask: NDArray[np.bool_],
    nx: int,
    ny: int,
    mesh_x: FloatArray,
    mesh_y: FloatArray,
    nodal_areas: FloatArray,
    gripper: FloatArray,
    available_force_n: float,
    gamma: float,
    pet_thickness_um: float,
    tape_width: float,
    panel_diagonal: float,
    assumptions: AssumptionSet,
) -> _FrontMechanics:
    frontier = _frontier_mask(damage, nx, ny, seed)
    indices = np.flatnonzero(frontier)
    if len(indices) == 0 or available_force_n <= 0.0:
        return _FrontMechanics(
            indices=indices,
            drive_ratio=np.zeros(len(indices), dtype=float),
            priority=np.ones(len(indices), dtype=float),
            force_xyz=np.zeros(3, dtype=float),
            scalar_reaction_n=0.0,
            peel_angle_deg=0.0,
        )

    front_xyz = np.column_stack(
        (mesh_x[indices], mesh_y[indices], np.zeros(len(indices), dtype=float))
    )
    vectors = gripper[None, :] - front_xyz
    lengths = np.linalg.norm(vectors, axis=1)
    zero_length = lengths <= 1.0e-12
    if np.any(zero_length):
        vectors[zero_length, 2] = 1.0
        lengths[zero_length] = 1.0
    unit_directions = vectors / lengths[:, None]
    horizontal = np.linalg.norm(vectors[:, :2], axis=1)
    angles = np.arctan2(np.abs(vectors[:, 2]), np.maximum(horizontal, 1.0e-12))

    distance_xy = np.linalg.norm(vectors[:, :2], axis=1)
    priority_scale = max(tape_width, 0.25 * panel_diagonal, 1.0e-9)
    priority = 0.25 + 0.75 * np.exp(-distance_xy / priority_scale)
    edge_width = np.sqrt(np.maximum(nodal_areas[indices], 1.0e-15))
    normalization = float(np.dot(priority, edge_width))
    available_line_force = (
        available_force_n * priority / max(normalization, 1.0e-15)
    )
    release = _energy_release_n_per_mm(
        available_line_force, angles, pet_thickness_um, assumptions
    )
    drive_ratio = release / max(gamma, 1.0e-15)
    drive_ratio = np.where(kinematic_mask[indices], drive_ratio, 0.0)
    # The spring law supplies the actual tensile force.  Adhesion controls
    # crack advance through ``drive_ratio``; it must not silently clip the
    # externally transmitted reaction below that tension.
    reaction_line_force = available_line_force
    node_force = (
        reaction_line_force[:, None]
        * edge_width[:, None]
        * unit_directions
    )
    scalar_reaction = float(np.dot(reaction_line_force, edge_width))
    weighted_angle = float(
        np.rad2deg(
            np.average(
                angles,
                weights=np.maximum(reaction_line_force * edge_width, 1.0e-15),
            )
        )
    )
    return _FrontMechanics(
        indices=indices,
        drive_ratio=drive_ratio,
        priority=priority,
        force_xyz=np.sum(node_force, axis=0),
        scalar_reaction_n=scalar_reaction,
        peel_angle_deg=weighted_angle,
    )


def _advance_bottom_damage(
    damage: FloatArray,
    mechanics: _FrontMechanics,
    work_budget_n_mm: float,
    gamma: float,
    nodal_areas: FloatArray,
) -> tuple[float, float]:
    eligible = np.flatnonzero(mechanics.drive_ratio >= 1.0)
    if len(eligible) == 0 or work_budget_n_mm <= 1.0e-15:
        return work_budget_n_mm, 0.0
    score = mechanics.drive_ratio[eligible] * mechanics.priority[eligible]
    ordered = eligible[np.argsort(-score, kind="stable")]
    changed = 0.0
    for local_index in ordered:
        node = int(mechanics.indices[local_index])
        capacity = max(1.0 - float(damage[node]), 0.0)
        if capacity <= 1.0e-15:
            continue
        full_cost = max(gamma * float(nodal_areas[node]), 1.0e-15)
        increment = min(capacity, work_budget_n_mm / full_cost)
        if increment <= 0.0:
            break
        damage[node] += increment
        energy_used = increment * full_cost
        work_budget_n_mm -= energy_used
        changed = max(changed, increment)
        if work_budget_n_mm <= 1.0e-15:
            break
    return max(work_budget_n_mm, 0.0), changed


def _damage_candidate(energy_ratio: FloatArray) -> FloatArray:
    """Top-interface softening, activated only after G/Gamma exceeds one."""

    safe_ratio = np.maximum(energy_ratio, 1.0)
    return np.where(energy_ratio > 1.0, 1.0 - 1.0 / safe_ratio, 0.0)


def _corner_coordinate(width: float, height: float, corner: str) -> FloatArray:
    return np.asarray(
        (
            width if "right" in corner else 0.0,
            height if "top" in corner else 0.0,
        ),
        dtype=float,
    )


def _damage_front(
    damage: FloatArray,
    x: FloatArray,
    y: FloatArray,
    corner: str,
) -> tuple[FloatArray, FloatArray]:
    """Reduce the current damage-boundary contour to a display/load segment."""

    ny, nx = len(y), len(x)
    grid = damage.reshape(ny, nx)
    grad_y, grad_x = np.gradient(grid, y, x, edge_order=1)
    strength = np.hypot(grad_x, grad_y)
    maximum = float(strength.max(initial=0.0))
    width = float(x[-1])
    height = float(y[-1])
    if maximum <= 1.0e-10:
        start = _corner_coordinate(width, height, corner)
        point = (
            np.asarray((width, height), dtype=float) - start
            if float(np.mean(damage)) >= 0.5
            else start
        )
        return np.asarray((*point, *point), dtype=float), point

    yy, xx = np.meshgrid(y, x, indexing="ij")
    selected = strength >= maximum * 0.15
    points = np.column_stack((xx[selected], yy[selected]))
    weights = strength[selected]
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    if len(points) == 1 or float(np.linalg.norm(covariance)) <= 1.0e-15:
        return np.asarray((*center, *center), dtype=float), center
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ direction
    first = center + direction * float(projections.min(initial=0.0))
    second = center + direction * float(projections.max(initial=0.0))
    first = np.clip(first, (0.0, 0.0), (width, height))
    second = np.clip(second, (0.0, 0.0), (width, height))
    return np.asarray((*first, *second), dtype=float), center


def _equivalent_panel_load(
    force_xyz: FloatArray,
    moment_xyz: FloatArray,
    center_xy: FloatArray,
    mesh_x: FloatArray,
    mesh_y: FloatArray,
    nodal_areas: FloatArray,
    sigma: float,
    torsion_coupling: float,
) -> FloatArray:
    """Map a 3-D resultant and moment to a transverse plate load vector."""

    dx = mesh_x - center_xy[0]
    dy = mesh_y - center_xy[1]
    weights = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2)) * nodal_areas
    weight_sum = float(weights.sum())
    if weight_sum <= 1.0e-15:
        weights[int(np.argmin(dx**2 + dy**2))] = 1.0
        weight_sum = 1.0
    load = float(force_xyz[2]) * weights / weight_sum

    def add_couple(pattern: FloatArray, lever: FloatArray, moment: float) -> None:
        nonlocal load
        zero_sum = pattern - weights * (float(pattern.sum()) / weight_sum)
        denominator = float(np.dot(lever, zero_sum))
        if abs(denominator) > 1.0e-15 and abs(moment) > 0.0:
            load = load + moment * zero_sum / denominator

    add_couple(dy * weights, dy, float(moment_xyz[0]))
    add_couple(dx * weights, dx, -float(moment_xyz[1]))

    torsion_pattern = dx * dy * weights
    torsion_pattern -= weights * (float(torsion_pattern.sum()) / weight_sum)
    torsion_norm = float(np.abs(torsion_pattern).sum())
    if torsion_norm > 1.0e-15 and abs(float(moment_xyz[2])) > 0.0:
        equivalent_force = (
            torsion_coupling * float(moment_xyz[2]) / max(sigma, 1.0e-12)
        )
        load = load + equivalent_force * torsion_pattern / torsion_norm
    return load


def _corner_values(displacement: FloatArray, nx: int, ny: int) -> tuple[float, ...]:
    grid = displacement.reshape(ny, nx)
    return (
        float(grid[0, 0]),
        float(grid[0, -1]),
        float(grid[-1, 0]),
        float(grid[-1, -1]),
    )


def _input_hash(
    condition: Condition,
    assumptions: AssumptionSet,
    tension_case: TensionCase,
    resolution: str,
) -> str:
    payload = (
        condition.model_dump_json()
        + assumptions.model_dump_json()
        + tension_case.model_dump_json()
        + resolution
        + MODEL_VERSION
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _command_series(condition: Condition, requested_steps: int):
    """Build a fixed-per-segment, P1-referenced command history."""

    main_segment_count = len(condition.trajectory) - 1
    quotient, remainder = divmod(requested_steps - 1, main_segment_count)
    main_intervals = np.full(main_segment_count, quotient, dtype=np.int64)
    main_intervals[:remainder] += 1

    approach = condition.initial_approach
    if approach is None:
        combined_points = condition.trajectory
        approach_segments = 0
        all_intervals = main_intervals
        initial_state_mode = "p1_equilibrium"
    else:
        approach_segments = len(approach) - 1
        combined_points = [*approach[:-1], *condition.trajectory]
        approach_intervals = np.resize(main_intervals, approach_segments)
        all_intervals = np.concatenate((approach_intervals, main_intervals))
        initial_state_mode = "specified_approach"

    series = interpolate_trajectory(
        combined_points,
        segment_intervals=all_intervals,
    )
    main_waypoint_indices = series.waypoint_indices[
        approach_segments : approach_segments + len(condition.trajectory)
    ]
    main_start_index = int(main_waypoint_indices[0])
    time_s = series.time_s - series.time_s[main_start_index]
    main_progress = np.clip(
        (series.path_parameter - approach_segments) / main_segment_count,
        0.0,
        1.0,
    )
    return (
        series,
        time_s,
        main_progress,
        main_start_index,
        main_waypoint_indices,
        initial_state_mode,
    )


def simulate(
    condition: Condition,
    assumptions: AssumptionSet,
    resolution: Resolution = "normal",
    tension_case: TensionCase | None = None,
) -> SimulationResult:
    """Run one deterministic stateful cohesive-zone comparison simulation."""

    tension_case = tension_case or TensionCase()
    mesh_size, requested_steps, requested_frames = _resolution_values(
        assumptions, resolution
    )
    (
        trajectory,
        time_s,
        trajectory_progress,
        main_start_index,
        main_waypoint_indices,
        initial_state_mode,
    ) = _command_series(condition, requested_steps)
    steps = len(trajectory.time_s)
    x, y, mesh_x, mesh_y = _mesh(
        condition.panel.width_mm, condition.panel.height_mm, mesh_size
    )
    nx, ny = len(x), len(y)
    node_count = nx * ny
    nodal_areas = _nodal_areas(x, y)
    panel_area = float(nodal_areas.sum())
    panel_diagonal = float(
        np.hypot(condition.panel.width_mm, condition.panel.height_mm)
    )
    laplacian = _laplacian(x, y)
    bending, top_foundation, top_regularization = _plate_components(
        condition, assumptions, x, y, laplacian, nodal_areas
    )

    bottom_damage = np.zeros(node_count, dtype=float)
    top_damage = np.zeros(node_count, dtype=float)
    top_solver = _factor_top(
        bending,
        top_foundation,
        nodal_areas,
        top_damage,
        assumptions,
        top_regularization,
    )

    tape_width = condition.pull_tape.width_mm * assumptions.grip_scale
    local_x, local_y = _corner_distances(
        mesh_x,
        mesh_y,
        condition.panel.width_mm,
        condition.panel.height_mm,
        condition.pull_tape.start_corner,
    )
    seed = _seed_mask(local_x, local_y, x, y, tape_width)
    frame_indices = np.unique(
        np.linspace(0, steps - 1, min(requested_frames, steps), dtype=int)
    )
    frame_lookup = set(int(index) for index in frame_indices)
    panel_frames: list[list[float]] = []
    bottom_damage_frames: list[list[float]] = []
    top_damage_frames: list[list[float]] = []
    top_risk_frames: list[list[float]] = []
    top_interface_reaction_frames: list[list[float]] = []

    force_xyz = np.zeros((steps, 3), dtype=float)
    force_resultant = np.zeros(steps, dtype=float)
    moment_xyz = np.zeros((steps, 3), dtype=float)
    moment_resultant = np.zeros(steps, dtype=float)
    force_activation = np.zeros(steps, dtype=float)
    tension_history = np.zeros(steps, dtype=float)
    span_history = np.zeros(steps, dtype=float)
    peel_angles = np.zeros(steps, dtype=float)
    bottom_ratio = np.zeros(steps, dtype=float)
    peak_risk = np.zeros(steps, dtype=float)
    risk_area = np.zeros(steps, dtype=float)
    damage_area = np.zeros(steps, dtype=float)
    min_foundation_retention = np.ones(steps, dtype=float)
    top_interface_force = np.zeros(steps, dtype=float)
    top_reaction_centroid = np.zeros((steps, 2), dtype=float)
    max_lift = np.zeros(steps, dtype=float)
    twist = np.zeros(steps, dtype=float)
    fronts = np.zeros((steps, 4), dtype=float)
    bottom_iterations = np.ones(steps, dtype=int)
    top_iterations = np.ones(steps, dtype=int)
    bottom_converged = np.ones(steps, dtype=bool)
    top_converged = np.ones(steps, dtype=bool)

    tolerance = assumptions.damage_convergence_tolerance
    max_iterations = assumptions.damage_max_iterations
    cumulative_in_plane_travel = 0.0
    p1_span = p1_span_length_mm(condition)

    def current_tension_and_span(
        current_damage: FloatArray, current_gripper: FloatArray
    ) -> tuple[float, float]:
        _segment, center = _damage_front(
            current_damage, x, y, condition.pull_tape.start_corner
        )
        span = float(
            np.linalg.norm(
                current_gripper - np.asarray((*center, 0.0), dtype=float)
            )
        )
        return (
            tension_from_span(
                span,
                p1_span,
                tension_case.initial_preload_n,
                tension_case.tape_stiffness_n_per_mm,
                assumptions.max_pull_force_n,
            ),
            span,
        )

    for index in range(steps):
        gripper = trajectory.xyz_mm[index]
        previous_gripper = (
            trajectory.xyz_mm[index - 1] if index else trajectory.xyz_mm[0]
        )
        gripper_increment = gripper - previous_gripper
        cumulative_in_plane_travel += float(
            np.linalg.norm(gripper_increment[:2])
        )
        # P1 is a held, tape-attached equilibrium state. Unknown pre-P1 motion
        # is not inferred; only the selected preload exists at the zero-work
        # first frame. max_pull_force_n is an equipment cap, not a permanent
        # force source.
        available_force, _span = current_tension_and_span(bottom_damage, gripper)
        kinematic_mask = _film_reach_mask(
            local_x,
            local_y,
            cumulative_in_plane_travel,
            tape_width,
        )
        gamma_bottom = _adhesion_energy_n_per_mm(
            condition.bottom_film.adhesion_gf,
            condition.bottom_film.pet_thickness_um,
            assumptions,
            trajectory.speed_mm_s[index],
        )
        mechanics = _front_mechanics(
            bottom_damage,
            seed,
            kinematic_mask,
            nx,
            ny,
            mesh_x,
            mesh_y,
            nodal_areas,
            gripper,
            available_force,
            gamma_bottom,
            condition.bottom_film.pet_thickness_um,
            tape_width,
            panel_diagonal,
            assumptions,
        )
        work_remaining = _actuator_work_n_mm(
            mechanics.force_xyz, gripper_increment
        )
        converged = True
        for iteration in range(1, max_iterations + 1):
            bottom_iterations[index] = iteration
            work_remaining, changed = _advance_bottom_damage(
                bottom_damage,
                mechanics,
                work_remaining,
                gamma_bottom,
                nodal_areas,
            )
            if changed <= tolerance or work_remaining <= 1.0e-15:
                break
            available_force, _span = current_tension_and_span(
                bottom_damage, gripper
            )
            mechanics = _front_mechanics(
                bottom_damage,
                seed,
                kinematic_mask,
                nx,
                ny,
                mesh_x,
                mesh_y,
                nodal_areas,
                gripper,
                available_force,
                gamma_bottom,
                condition.bottom_film.pet_thickness_um,
                tape_width,
                panel_diagonal,
                assumptions,
            )
            if not np.any(mechanics.drive_ratio >= 1.0):
                break
        else:
            available_force, _span = current_tension_and_span(
                bottom_damage, gripper
            )
            mechanics = _front_mechanics(
                bottom_damage,
                seed,
                kinematic_mask,
                nx,
                ny,
                mesh_x,
                mesh_y,
                nodal_areas,
                gripper,
                available_force,
                gamma_bottom,
                condition.bottom_film.pet_thickness_um,
                tape_width,
                panel_diagonal,
                assumptions,
            )
            converged = not (
                work_remaining > 1.0e-15
                and np.any(mechanics.drive_ratio >= 1.0)
            )
        # Treat a residual smaller than the declared damage tolerance as the
        # fully failed cohesive state.  This avoids reporting 99.975% solely
        # because the last node retained a sub-tolerance numerical sliver.
        bottom_damage[bottom_damage >= 1.0 - tolerance] = 1.0
        unresolved_area = float(np.dot(1.0 - bottom_damage, nodal_areas))
        if (
            unresolved_area / panel_area <= tolerance
            or unresolved_area
            <= 2.0 * float(nodal_areas.max(initial=0.0)) + 1.0e-12
        ):
            # A sub-tolerance area fraction (or residual below two nodal
            # control areas) cannot span a resolved front. Close it as
            # mesh-complete.
            bottom_damage.fill(1.0)
        bottom_converged[index] = converged

        front, front_center = _damage_front(
            bottom_damage, x, y, condition.pull_tape.start_corner
        )
        available_force, current_span = current_tension_and_span(
            bottom_damage, gripper
        )
        mechanics = _front_mechanics(
            bottom_damage,
            seed,
            kinematic_mask,
            nx,
            ny,
            mesh_x,
            mesh_y,
            nodal_areas,
            gripper,
            available_force,
            gamma_bottom,
            condition.bottom_film.pet_thickness_um,
            tape_width,
            panel_diagonal,
            assumptions,
        )
        force = mechanics.force_xyz
        tension_history[index] = available_force
        span_history[index] = current_span
        force_activation[index] = available_force / assumptions.max_pull_force_n
        force_xyz[index] = force
        force_resultant[index] = float(np.linalg.norm(force))
        peel_angles[index] = mechanics.peel_angle_deg
        bottom_ratio[index] = float(
            np.clip(np.dot(bottom_damage, nodal_areas) / panel_area, 0.0, 1.0)
        )

        fronts[index] = front
        front_xyz = np.asarray((*front_center, 0.0), dtype=float)
        lever = gripper - front_xyz
        surface_offset = np.asarray(
            (0.0, 0.0, -0.5 * condition.panel.thickness_mm), dtype=float
        )
        applied_moment = np.cross(lever, force) + np.cross(
            surface_offset, force
        )
        moment_xyz[index] = applied_moment
        moment_resultant[index] = float(np.linalg.norm(applied_moment))

        # The equivalent load footprint is a physical tape-width scale, not a
        # mesh scale.  Letting sigma shrink with refinement made peak Rtop
        # artificially rise as the grid became finer.
        sigma = max(tape_width * 0.5, 1.0)
        panel_load = _equivalent_panel_load(
            force,
            applied_moment,
            front_center,
            mesh_x,
            mesh_y,
            nodal_areas,
            sigma,
            assumptions.torsion_coupling,
        )
        gamma_top = _adhesion_energy_n_per_mm(
            condition.top_film.adhesion_gf,
            condition.top_film.pet_thickness_um,
            assumptions,
            trajectory.speed_mm_s[index],
        )

        displacement = np.zeros(node_count, dtype=float)
        local_risk = np.zeros(node_count, dtype=float)
        converged = False
        for iteration in range(1, max_iterations + 1):
            displacement = np.asarray(top_solver.solve(panel_load), dtype=float)
            opening = np.maximum(displacement, 0.0)
            top_energy = 0.5 * top_foundation * opening**2
            local_risk = top_energy / max(gamma_top, 1.0e-15)
            candidate = np.maximum(top_damage, _damage_candidate(local_risk))
            change = float(np.max(candidate - top_damage, initial=0.0))
            top_iterations[index] = iteration
            if change <= tolerance:
                converged = True
                break
            top_damage = candidate
            top_solver = _factor_top(
                bending,
                top_foundation,
                nodal_areas,
                top_damage,
                assumptions,
                top_regularization,
            )
        if not converged:
            displacement = np.asarray(top_solver.solve(panel_load), dtype=float)
            opening = np.maximum(displacement, 0.0)
            top_energy = 0.5 * top_foundation * opening**2
            local_risk = top_energy / max(gamma_top, 1.0e-15)
        top_converged[index] = converged

        peak_risk[index] = float(local_risk.max(initial=0.0))
        risk_area[index] = float(
            np.dot((local_risk >= 1.0).astype(float), nodal_areas)
        )
        damage_area[index] = float(np.dot(top_damage, nodal_areas))
        top_retention = (
            assumptions.cohesive_stiffness_floor_ratio
            + (1.0 - assumptions.cohesive_stiffness_floor_ratio)
            * (1.0 - top_damage)
        )
        # Normal reaction transmitted through the damage-softened top PSA.
        # Winkler springs do not resolve in-plane interface shear, therefore
        # this is stored explicitly as a normal-only field and must not be
        # presented as the full 3-D top-interface traction.
        top_reaction_n = (
            top_foundation * top_retention * np.maximum(displacement, 0.0) * nodal_areas
        )
        top_interface_force[index] = float(top_reaction_n.sum())
        if top_interface_force[index] > 1.0e-15:
            top_reaction_centroid[index, 0] = float(
                np.dot(top_reaction_n, mesh_x) / top_interface_force[index]
            )
            top_reaction_centroid[index, 1] = float(
                np.dot(top_reaction_n, mesh_y) / top_interface_force[index]
            )
        else:
            top_reaction_centroid[index] = front_center
        min_foundation_retention[index] = float(top_retention.min(initial=1.0))
        max_lift[index] = float(np.maximum(displacement, 0.0).max(initial=0.0))
        bl, br, tl, tr = _corner_values(displacement, nx, ny)
        twist[index] = abs((bl + tr) - (br + tl)) * 0.5

        if index in frame_lookup:
            panel_frames.append(displacement.tolist())
            bottom_damage_frames.append(bottom_damage.tolist())
            top_damage_frames.append(top_damage.tolist())
            top_risk_frames.append(local_risk.tolist())
            top_interface_reaction_frames.append(top_reaction_n.tolist())

    peak_index = int(np.argmax(peak_risk))
    exceedance_duration = float(
        np.trapezoid((peak_risk >= 1.0).astype(float), time_s)
    )
    warnings = [
        "이 결과는 보정 전 상대 비교용 축약 모델이며 실제 불량률 또는 안전 판정이 아닙니다.",
        "하면 박리는 노드별 에너지·일·연결성 기준, 상면은 손상 연성 Winkler 기초로 근사합니다.",
        "gf 시험 폭·각도·속도와 장력-신장 가정을 실제 계측값으로 보정해야 합니다.",
        "Z는 패널 표면 Z=0 기준 절대 좌표이며 P1은 테이프 부착 후 정적 평형으로 해석합니다.",
    ]
    if not bool(np.all(bottom_converged)):
        warnings.append("일부 시간 단계에서 하면 손상 전파가 설정된 최대 반복 횟수에 도달했습니다.")
    if not bool(np.all(top_converged)):
        warnings.append("일부 시간 단계에서 상면 손상-강성 반복이 설정된 최대 횟수 안에 수렴하지 않았습니다.")

    return SimulationResult(
        model_version=MODEL_VERSION,
        calibration_version=assumptions.calibration_version,
        condition_name=condition.name,
        resolution=resolution,
        input_hash=_input_hash(condition, assumptions, tension_case, resolution),
        initial_state_mode=initial_state_mode,
        main_trajectory_start_index=main_start_index,
        trajectory_waypoint_indices=main_waypoint_indices.tolist(),
        time_s=time_s.tolist(),
        position_xyz_mm=trajectory.xyz_mm.tolist(),
        speed_mm_s=trajectory.speed_mm_s.tolist(),
        trajectory_progress=trajectory_progress.tolist(),
        peel_progress=bottom_ratio.tolist(),
        peel_angle_deg=peel_angles.tolist(),
        force_xyz_n=force_xyz.tolist(),
        force_resultant_n=force_resultant.tolist(),
        moment_xyz_n_mm=moment_xyz.tolist(),
        moment_resultant_n_mm=moment_resultant.tolist(),
        pull_force_activation=force_activation.tolist(),
        tension_n=tension_history.tolist(),
        tape_span_length_mm=span_history.tolist(),
        p1_span_length_mm=p1_span,
        initial_preload_n=tension_case.initial_preload_n,
        tape_stiffness_n_per_mm=tension_case.tape_stiffness_n_per_mm,
        bottom_peel_ratio=bottom_ratio.tolist(),
        top_peak_risk=peak_risk.tolist(),
        top_risk_area_mm2=risk_area.tolist(),
        top_damage_area_mm2=damage_area.tolist(),
        top_min_foundation_retention=min_foundation_retention.tolist(),
        top_interface_normal_force_n=top_interface_force.tolist(),
        top_interface_reaction_centroid_xy_mm=top_reaction_centroid.tolist(),
        panel_max_lift_mm=max_lift.tolist(),
        panel_twist_mm=twist.tolist(),
        bottom_damage_iterations=bottom_iterations.tolist(),
        top_damage_iterations=top_iterations.tolist(),
        bottom_damage_converged=bottom_converged.tolist(),
        top_damage_converged=top_converged.tolist(),
        peak_top_risk=float(peak_risk[peak_index]),
        peak_top_risk_time_s=float(time_s[peak_index]),
        final_bottom_peel_ratio=float(bottom_ratio[-1]),
        max_top_risk_area_mm2=float(risk_area.max(initial=0.0)),
        max_top_damage_area_mm2=float(damage_area.max(initial=0.0)),
        final_top_damage_area_mm2=float(damage_area[-1]),
        top_risk_exceedance_duration_s=exceedance_duration,
        max_panel_lift_mm=float(max_lift.max(initial=0.0)),
        max_panel_twist_mm=float(twist.max(initial=0.0)),
        max_moment_resultant_n_mm=float(moment_resultant.max(initial=0.0)),
        max_tension_n=float(tension_history.max(initial=0.0)),
        max_top_interface_normal_force_n=float(top_interface_force.max(initial=0.0)),
        mesh_shape=(ny, nx),
        mesh_x_mm=mesh_x.tolist(),
        mesh_y_mm=mesh_y.tolist(),
        frame_indices=frame_indices.tolist(),
        panel_z_frames_mm=panel_frames,
        bottom_damage_frames=bottom_damage_frames,
        top_damage_frames=top_damage_frames,
        top_risk_frames=top_risk_frames,
        top_interface_reaction_frames_n=top_interface_reaction_frames,
        front_segments_mm=fronts.tolist(),
        warnings=warnings,
    )
