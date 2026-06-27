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

    surface_velocity = _sample_velocity(mesh.vertices, velocity, origin, spacing, shape, fluid)
    speed_ratio = np.linalg.norm(surface_velocity, axis=1) / flow.speed
    surface_cp = 1.0 - speed_ratio**2

    return FlowSolution(
        mesh=mesh,
        velocity_field=velocity,
        phi_field=phi_field,
        grid_origin=origin,
        grid_spacing=spacing,
        u_inf=u_inf,
        surface_velocity=surface_velocity,
        surface_cp=surface_cp,
        reynolds_number=re,
    )


def _sample_velocity(
    points: np.ndarray,
    velocity: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    shape: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of the velocity field at arbitrary points."""
    nx, ny, nz, _ = velocity.shape
    sampled = np.zeros((len(points), 3), dtype=float)

    for idx, point in enumerate(points):
        rel = (point - origin) / spacing
        base = np.floor(rel).astype(int)
        frac = rel - base

        if np.any(base < 0) or base[0] >= nx - 1 or base[1] >= ny - 1 or base[2] >= nz - 1:
            sampled[idx] = np.array([1.0, 0.0, 0.0])
            continue

        i, j, k = base
        if not fluid[i, j, k]:
            # Walk outward to nearest fluid cell for robust surface sampling
            sampled[idx] = _nearest_fluid_velocity(point, velocity, origin, spacing, shape, fluid)
            continue

        i1, j1, k1 = i + 1, j + 1, k + 1
        c000 = velocity[i, j, k]
        c100 = velocity[i1, j, k]
        c010 = velocity[i, j1, k]
        c110 = velocity[i1, j1, k]
        c001 = velocity[i, j, k1]
        c101 = velocity[i1, j, k1]
        c011 = velocity[i, j1, k1]
        c111 = velocity[i1, j1, k1]

        tx, ty, tz = frac
        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        sampled[idx] = c0 * (1 - tz) + c1 * tz

    return sampled


def _nearest_fluid_velocity(
    point: np.ndarray,
    velocity: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    shape: np.ndarray,
    fluid: np.ndarray,
    max_radius: int = 6,
) -> np.ndarray:
    rel = np.round((point - origin) / spacing).astype(int)
    for radius in range(1, max_radius + 1):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    i, j, k = rel + np.array([di, dj, dk])
                    if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                        if fluid[i, j, k]:
                            return velocity[i, j, k]
    return np.array([1.0, 0.0, 0.0])


def solve_flow(mesh: trimesh.Trimesh, flow: FlowConfig, grid: GridConfig) -> FlowSolution:
    return _solve_potential_flow(mesh, flow, grid)
