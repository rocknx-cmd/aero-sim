from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

from .config import FlowConfig, GridConfig


@dataclass
class FlowSolution:
    """Results from an inviscid potential-flow solve."""

    mesh: trimesh.Trimesh
    velocity_field: np.ndarray  # (nx, ny, nz, 3)
    phi_field: np.ndarray  # (nx, ny, nz)
    grid_origin: np.ndarray
    grid_spacing: float
    grid_shape: tuple[int, int, int]
    fluid_mask: np.ndarray  # (nx, ny, nz) bool
    u_inf: np.ndarray
    surface_velocity: np.ndarray  # (n_vertices, 3)
    surface_cp: np.ndarray  # (n_vertices,)
    reynolds_number: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Flow direction must be non-zero.")
    return vec / norm


def _build_grid(mesh: trimesh.Trimesh, grid: GridConfig) -> tuple[np.ndarray, float, np.ndarray]:
    bounds = mesh.bounds
    span = bounds[1] - bounds[0]
    center = 0.5 * (bounds[0] + bounds[1])
    padded_span = span * (1.0 + grid.padding)
    max_span = float(np.max(padded_span))
    spacing = max_span / grid.resolution
    origin = center - 0.5 * padded_span
    shape = np.ceil(padded_span / spacing).astype(int) + 1
    shape = np.maximum(shape, 16)
    return origin, spacing, shape


def _cell_centers(origin: np.ndarray, spacing: float, shape: np.ndarray) -> np.ndarray:
    xs = origin[0] + spacing * np.arange(shape[0])
    ys = origin[1] + spacing * np.arange(shape[1])
    zs = origin[2] + spacing * np.arange(shape[2])
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)


def _solve_potential_flow(
    mesh: trimesh.Trimesh,
    flow: FlowConfig,
    grid: GridConfig,
) -> FlowSolution:
    """Solve steady incompressible inviscid flow: ∇²φ = 0 with u·n = 0 on the body."""
    u_inf = _normalize(np.asarray(flow.direction, dtype=float)) * flow.speed
    origin, spacing, shape = _build_grid(mesh, grid)
    centers = _cell_centers(origin, spacing, shape)

    solid = mesh.contains(centers.reshape(-1, 3)).reshape(shape)
    fluid = ~solid

    nx, ny, nz = shape
    n_total = nx * ny * nz
    index = -np.ones(n_total, dtype=np.int64)
    fluid_idx = np.flatnonzero(fluid.ravel())
    index[fluid_idx] = np.arange(fluid_idx.size)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(fluid_idx.size, dtype=float)

    def add_entry(row: int, col: int, value: float) -> None:
        if value != 0.0:
            rows.append(row)
            cols.append(col)
            data.append(value)

    def boundary_phi(i: int, j: int, k: int) -> float:
        pos = origin + spacing * np.array([i, j, k], dtype=float)
        return float(np.dot(u_inf, pos))

    for flat in fluid_idx:
        i, j, k = np.unravel_index(flat, shape)
        row = int(index[flat])
        diag = 0.0

        neighbors = (
            (i - 1, j, k, -1),
            (i + 1, j, k, 1),
            (i, j - 1, k, -1),
            (i, j + 1, k, 1),
            (i, j, k - 1, -1),
            (i, j, k + 1, 1),
        )

        for ni, nj, nk, _ in neighbors:
            on_boundary = ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz
            if on_boundary:
                phi_bc = boundary_phi(
                    max(0, min(nx - 1, ni)),
                    max(0, min(ny - 1, nj)),
                    max(0, min(nz - 1, nk)),
                )
                coeff = 1.0 / spacing**2
                diag -= coeff
                rhs[row] -= coeff * phi_bc
                continue

            nflat = np.ravel_multi_index((ni, nj, nk), shape)
            if solid[ni, nj, nk]:
                # Impermeable wall: mirror ghost enforces ∂φ/∂n = 0
                coeff = 1.0 / spacing**2
                diag -= coeff
                add_entry(row, row, coeff)
            else:
                ncol = int(index[nflat])
                coeff = 1.0 / spacing**2
                diag -= coeff
                add_entry(row, ncol, coeff)

        add_entry(row, row, diag)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(fluid_idx.size, fluid_idx.size),
        dtype=float,
    )
    phi_flat = spla.spsolve(matrix, rhs)
    phi = np.zeros(n_total, dtype=float)
    phi[fluid_idx] = phi_flat
    phi_field = phi.reshape(shape)

    velocity = np.zeros((*shape, 3), dtype=float)
    for axis in range(3):
        velocity[..., axis] = np.gradient(phi_field, spacing, axis=axis)

    # Characteristic length for Reynolds number
    char_length = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    re = flow.density * flow.speed * char_length / flow.viscosity

    surface_velocity = _sample_velocity(
        mesh, mesh.vertices, velocity, origin, spacing, shape, fluid, u_inf
    )
    speed_ratio = np.clip(np.linalg.norm(surface_velocity, axis=1) / flow.speed, 0.0, 3.0)
    surface_cp = 1.0 - speed_ratio**2

    return FlowSolution(
        mesh=mesh,
        velocity_field=velocity,
        phi_field=phi_field,
        grid_origin=origin,
        grid_spacing=spacing,
        grid_shape=tuple(int(s) for s in shape),
        fluid_mask=fluid,
        u_inf=u_inf,
        surface_velocity=surface_velocity,
        surface_cp=surface_cp,
        reynolds_number=re,
    )


def _sample_velocity(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    velocity: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    shape: np.ndarray,
    fluid: np.ndarray,
    u_inf: np.ndarray | None = None,
) -> np.ndarray:
    """Trilinear interpolation of the velocity field at arbitrary points (vectorized)."""
    import trimesh.proximity as proximity

    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 3))

    fallback = _normalize(u_inf) if u_inf is not None else np.array([1.0, 0.0, 0.0])
    nx, ny, nz, _ = velocity.shape

    _, _, face_ids = proximity.closest_point(mesh, points)
    normals = mesh.face_normals[face_ids].astype(float, copy=True)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    offset_points = points + 1.5 * spacing * normals
    rel = (offset_points - origin) / spacing
    np.nan_to_num(rel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    base = np.floor(rel).astype(np.int64)
    frac = rel - base

    sampled = np.tile(fallback, (len(points), 1))
    valid = (
        (base[:, 0] >= 0)
        & (base[:, 0] < nx - 1)
        & (base[:, 1] >= 0)
        & (base[:, 1] < ny - 1)
        & (base[:, 2] >= 0)
        & (base[:, 2] < nz - 1)
    )
    if not np.any(valid):
        return sampled

    idx = np.where(valid)[0]
    i, j, k = base[idx, 0], base[idx, 1], base[idx, 2]
    tx, ty, tz = frac[idx, 0:1], frac[idx, 1:2], frac[idx, 2:3]

    c000 = velocity[i, j, k]
    c100 = velocity[i + 1, j, k]
    c010 = velocity[i, j + 1, k]
    c110 = velocity[i + 1, j + 1, k]
    c001 = velocity[i, j, k + 1]
    c101 = velocity[i + 1, j, k + 1]
    c011 = velocity[i, j + 1, k + 1]
    c111 = velocity[i + 1, j + 1, k + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    sampled[idx] = c0 * (1 - tz) + c1 * tz

    # Fallback where interpolation landed in blocked/solid cells
    solid_mask = ~fluid[i, j, k]
    if np.any(solid_mask):
        bad = idx[solid_mask]
        sampled[bad] = fallback

    return sampled


def _outward_normal(mesh: trimesh.Trimesh, point: np.ndarray) -> np.ndarray:
    """Approximate outward normal at the nearest surface point."""
    _, _, face_id = mesh.nearest.on_surface([point])
    normal = mesh.face_normals[int(face_id[0])].astype(float)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def solve_flow(mesh: trimesh.Trimesh, flow: FlowConfig, grid: GridConfig) -> FlowSolution:
    return _solve_potential_flow(mesh, flow, grid)


def interpolate_velocity(solution: FlowSolution, points: np.ndarray) -> np.ndarray:
    """Sample the velocity field at arbitrary 3D points."""
    return _sample_velocity(
        solution.mesh,
        points,
        solution.velocity_field,
        solution.grid_origin,
        solution.grid_spacing,
        np.array(solution.grid_shape),
        solution.fluid_mask,
        solution.u_inf,
    )
