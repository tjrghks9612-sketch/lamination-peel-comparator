"""Headless reduced-order reverse-peel simulation.

This is deliberately a comparison model rather than a calibrated FEA solver.
It combines a Kendall-inspired peel force, a diagonal moving peel front, a
sparse elastic plate on a PSA foundation, and an irreversible cohesive damage
indicator.  All uncertain parameters are explicit in :class:`AssumptionSet`.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse
from scipy.sparse.linalg import factorized

from .models import AssumptionSet, Condition
from .trajectory import interpolate_trajectory, projected_progress


Resolution = Literal["coarse", "normal", "fine"]


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_name: str
    resolution: Resolution
    input_hash: str

    time_s: list[float]
    position_xyz_mm: list[list[float]]
    speed_mm_s: list[float]
    peel_progress: list[float]
    peel_angle_deg: list[float]
    force_xyz_n: list[list[float]]
    force_resultant_n: list[float]
    bottom_peel_ratio: list[float]
    top_peak_risk: list[float]
    top_risk_area_mm2: list[float]
    top_damage_area_mm2: list[float]
    panel_max_lift_mm: list[float]
    panel_twist_mm: list[float]

    peak_top_risk: float
    peak_top_risk_time_s: float
    final_bottom_peel_ratio: float
    max_top_risk_area_mm2: float
    max_panel_lift_mm: float
    max_panel_twist_mm: float

    mesh_shape: tuple[int, int]
    mesh_x_mm: list[float]
    mesh_y_mm: list[float]
    frame_indices: list[int]
    panel_z_frames_mm: list[list[float]]
    top_damage_frames: list[list[float]]
    front_segments_mm: list[list[float]]
    warnings: list[str] = Field(default_factory=list)


FloatArray = NDArray[np.float64]


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


def _neumann_second_derivative(count: int, spacing: float) -> sparse.csr_matrix:
    main = np.full(count, -2.0)
    main[0] = main[-1] = -1.0
    off = np.ones(count - 1)
    return sparse.diags((off, main, off), (-1, 0, 1), format="csr") / spacing**2


def _plate_factor(
    condition: Condition,
    assumptions: AssumptionSet,
    x: FloatArray,
    y: FloatArray,
) -> tuple[object, float, float]:
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    area = dx * dy
    lx = _neumann_second_derivative(nx, dx)
    ly = _neumann_second_derivative(ny, dy)
    laplacian = sparse.kron(sparse.eye(ny), lx) + sparse.kron(
        ly, sparse.eye(nx)
    )
    panel_e = assumptions.panel_young_modulus_gpa * 1000.0  # N/mm^2
    thickness = condition.panel.thickness_mm
    nu = assumptions.panel_poisson_ratio
    bending = panel_e * thickness**3 / (12.0 * (1.0 - nu**2))  # N mm
    psa_thickness = condition.top_film.psa_thickness_um / 1000.0
    foundation = assumptions.psa_modulus_mpa / psa_thickness  # N/mm^3
    stiffness = (
        bending * area * (laplacian.T @ laplacian)
        + sparse.eye(nx * ny, format="csr") * foundation * area
    )
    # Tiny regularization protects unusual user inputs without affecting the
    # physical foundation stiffness at displayed precision.
    regularization = max(float(stiffness.diagonal().max()), 1.0) * 1.0e-12
    stiffness = stiffness + sparse.eye(nx * ny, format="csr") * regularization
    return factorized(stiffness.tocsc()), foundation, area


def _adhesion_energy_n_per_mm(
    adhesion_gf: float,
    pet_thickness_um: float,
    assumptions: AssumptionSet,
    speed_mm_s: float,
) -> float:
    """Infer a shared-scale fracture energy from the reported peel force.

    The unknown test width/angle remain explicit assumptions.  Units are N/mm,
    numerically equivalent to the interface energy density mJ/mm^2.
    """

    measured_force_n = adhesion_gf * 0.00980665
    p = measured_force_n / assumptions.test_width_mm  # N/mm
    angle = np.deg2rad(assumptions.test_angle_deg)
    pet_e = assumptions.pet_young_modulus_gpa * 1000.0
    pet_t = pet_thickness_um / 1000.0
    gamma = p * (1.0 - np.cos(angle)) + p**2 / (2.0 * pet_e * pet_t)
    speed_ratio = max(speed_mm_s, 1.0e-9) / assumptions.reference_speed_mm_s
    return float(gamma * speed_ratio**assumptions.speed_exponent)


def _kendall_line_force(
    gamma: float,
    peel_angle_rad: float,
    pet_thickness_um: float,
    assumptions: AssumptionSet,
) -> float:
    pet_e = assumptions.pet_young_modulus_gpa * 1000.0
    pet_t = pet_thickness_um / 1000.0
    extensional_stiffness = pet_e * pet_t  # N/mm
    angular = max(1.0 - np.cos(peel_angle_rad), 0.0)
    discriminant = (extensional_stiffness * angular) ** 2 + (
        2.0 * extensional_stiffness * gamma
    )
    return float(
        max(
            np.sqrt(max(discriminant, 0.0))
            - extensional_stiffness * angular,
            0.0,
        )
    )


def _to_local_corner(
    u: float, v: float, width: float, height: float, corner: str
) -> tuple[float, float]:
    x = u * width
    y = v * height
    if "right" in corner:
        x = width - x
    if "top" in corner:
        y = height - y
    return x, y


def _front_segment(
    progress: float, width: float, height: float, corner: str
) -> tuple[float, float, float, float]:
    diagonal_level = 2.0 * float(np.clip(progress, 0.0, 1.0))
    if diagonal_level <= 1.0:
        local_a = (0.0, diagonal_level)
        local_b = (diagonal_level, 0.0)
    else:
        local_a = (diagonal_level - 1.0, 1.0)
        local_b = (1.0, diagonal_level - 1.0)
    a = _to_local_corner(*local_a, width, height, corner)
    b = _to_local_corner(*local_b, width, height, corner)
    return (*a, *b)


def _corner_values(displacement: FloatArray, nx: int, ny: int) -> tuple[float, ...]:
    grid = displacement.reshape(ny, nx)
    return float(grid[0, 0]), float(grid[0, -1]), float(grid[-1, 0]), float(
        grid[-1, -1]
    )


def _input_hash(condition: Condition, assumptions: AssumptionSet, resolution: str) -> str:
    payload = condition.model_dump_json() + assumptions.model_dump_json() + resolution
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def simulate(
    condition: Condition,
    assumptions: AssumptionSet,
    resolution: Resolution = "normal",
) -> SimulationResult:
    """Run one deterministic reduced-order simulation.

    ``top_peak_risk`` is an assumption-normalized comparison index.  A value of
    one means the inferred cohesive energy was reached under the selected
    assumptions; it is not a calibrated pass/fail probability.
    """

    mesh_size, steps, requested_frames = _resolution_values(assumptions, resolution)
    trajectory = interpolate_trajectory(condition.trajectory, samples=steps)
    progress = projected_progress(trajectory.xyz_mm)
    x, y, mesh_x, mesh_y = _mesh(
        condition.panel.width_mm, condition.panel.height_mm, mesh_size
    )
    nx, ny = len(x), len(y)
    solve_plate, foundation, nodal_area = _plate_factor(
        condition, assumptions, x, y
    )

    frame_indices = np.unique(
        np.linspace(0, steps - 1, min(requested_frames, steps), dtype=int)
    )
    frame_lookup = set(int(index) for index in frame_indices)
    panel_frames: list[list[float]] = []
    damage_frames: list[list[float]] = []

    force_xyz = np.zeros((steps, 3), dtype=float)
    force_resultant = np.zeros(steps, dtype=float)
    peel_angles = np.zeros(steps, dtype=float)
    peak_risk = np.zeros(steps, dtype=float)
    risk_area = np.zeros(steps, dtype=float)
    damage_area = np.zeros(steps, dtype=float)
    max_lift = np.zeros(steps, dtype=float)
    twist = np.zeros(steps, dtype=float)
    damage = np.zeros(nx * ny, dtype=float)
    fronts = np.zeros((steps, 4), dtype=float)

    width = condition.panel.width_mm
    height = condition.panel.height_mm
    corner = condition.pull_tape.start_corner
    tape_width = condition.pull_tape.width_mm * assumptions.grip_scale

    for index in range(steps):
        front = _front_segment(progress[index], width, height, corner)
        fronts[index] = front
        center_x = 0.5 * (front[0] + front[2])
        center_y = 0.5 * (front[1] + front[3])
        gripper = trajectory.xyz_mm[index]
        direction = gripper - np.asarray((center_x, center_y, 0.0))
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            if index + 1 < steps:
                direction = trajectory.xyz_mm[index + 1] - gripper
            elif index:
                direction = gripper - trajectory.xyz_mm[index - 1]
            direction_norm = max(float(np.linalg.norm(direction)), 1.0e-12)
        unit_direction = direction / direction_norm
        horizontal = float(np.linalg.norm(direction[:2]))
        angle = float(np.arctan2(abs(direction[2]), max(horizontal, 1.0e-12)))
        peel_angles[index] = np.rad2deg(angle)

        gamma_bottom = _adhesion_energy_n_per_mm(
            condition.bottom_film.adhesion_gf,
            condition.bottom_film.pet_thickness_um,
            assumptions,
            trajectory.speed_mm_s[index],
        )
        line_force = _kendall_line_force(
            gamma_bottom,
            angle,
            condition.bottom_film.pet_thickness_um,
            assumptions,
        )
        front_width = float(np.hypot(front[2] - front[0], front[3] - front[1]))
        effective_width = max(front_width, tape_width * (1.0 - progress[index]))
        # Initial Z lift can open the corner before there is measurable XY
        # progress, so activation follows actual gripper displacement from P1.
        # Only the fully detached end is faded by projected peel progress.
        start_travel = float(
            np.linalg.norm(gripper - trajectory.xyz_mm[0])
        )
        start_activation = np.clip(
            start_travel / max(0.5, 0.1 * tape_width), 0.0, 1.0
        )
        end_activation = np.clip((1.0 - progress[index]) / 0.02, 0.0, 1.0)
        total_force = line_force * effective_width * start_activation * end_activation
        force_xyz[index] = total_force * unit_direction
        force_resultant[index] = total_force

        vertical_load = max(float(force_xyz[index, 2]), 0.0)
        sigma = max(tape_width * 0.5, mesh_size * 1.5)
        weights = np.exp(
            -((mesh_x - center_x) ** 2 + (mesh_y - center_y) ** 2)
            / (2.0 * sigma**2)
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 1.0e-15 or vertical_load <= 0.0:
            displacement = np.zeros(nx * ny, dtype=float)
        else:
            load = vertical_load * weights / weight_sum
            displacement = np.asarray(solve_plate(load), dtype=float)
            displacement = np.maximum(displacement, 0.0)

        gamma_top = _adhesion_energy_n_per_mm(
            condition.top_film.adhesion_gf,
            condition.top_film.pet_thickness_um,
            assumptions,
            trajectory.speed_mm_s[index],
        )
        local_risk = 0.5 * foundation * displacement**2 / max(gamma_top, 1.0e-15)
        peak_risk[index] = float(local_risk.max(initial=0.0))
        risk_area[index] = float(np.count_nonzero(local_risk >= 1.0) * nodal_area)
        separation_ratio = np.sqrt(np.maximum(local_risk, 0.0))
        candidate_damage = np.clip((separation_ratio - 0.8) / 0.2, 0.0, 1.0)
        damage = np.maximum(damage, candidate_damage)
        damage_area[index] = float(np.count_nonzero(damage > 0.0) * nodal_area)
        max_lift[index] = float(displacement.max(initial=0.0))
        bl, br, tl, tr = _corner_values(displacement, nx, ny)
        twist[index] = abs((bl + tr) - (br + tl)) * 0.5

        if index in frame_lookup:
            panel_frames.append(displacement.tolist())
            damage_frames.append(damage.tolist())

    peak_index = int(np.argmax(peak_risk))
    warnings = [
        "상대 비교용 축약 모델이며 실제 불량률 또는 안전 판정이 아닙니다.",
        "하면 필름 곡면과 박리 전선은 대각선 기구학 근사입니다.",
        "gf 시험 폭·각도·속도가 불명확하면 절대 힘과 위험도 크기는 보정되지 않습니다.",
    ]
    return SimulationResult(
        condition_name=condition.name,
        resolution=resolution,
        input_hash=_input_hash(condition, assumptions, resolution),
        time_s=trajectory.time_s.tolist(),
        position_xyz_mm=trajectory.xyz_mm.tolist(),
        speed_mm_s=trajectory.speed_mm_s.tolist(),
        peel_progress=progress.tolist(),
        peel_angle_deg=peel_angles.tolist(),
        force_xyz_n=force_xyz.tolist(),
        force_resultant_n=force_resultant.tolist(),
        bottom_peel_ratio=progress.tolist(),
        top_peak_risk=peak_risk.tolist(),
        top_risk_area_mm2=risk_area.tolist(),
        top_damage_area_mm2=damage_area.tolist(),
        panel_max_lift_mm=max_lift.tolist(),
        panel_twist_mm=twist.tolist(),
        peak_top_risk=float(peak_risk[peak_index]),
        peak_top_risk_time_s=float(trajectory.time_s[peak_index]),
        final_bottom_peel_ratio=float(progress[-1]),
        max_top_risk_area_mm2=float(risk_area.max(initial=0.0)),
        max_panel_lift_mm=float(max_lift.max(initial=0.0)),
        max_panel_twist_mm=float(twist.max(initial=0.0)),
        mesh_shape=(ny, nx),
        mesh_x_mm=mesh_x.tolist(),
        mesh_y_mm=mesh_y.tolist(),
        frame_indices=frame_indices.tolist(),
        panel_z_frames_mm=panel_frames,
        top_damage_frames=damage_frames,
        front_segments_mm=fronts.tolist(),
        warnings=warnings,
    )
