from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import VizConfig
from .objects import mesh_to_pyvista
from .solver import FlowSolution, interpolate_velocity


def render_solution(
    solution: FlowSolution,
    viz: VizConfig,
    output_path: Path | None = None,
    save_path: Path | None = None,
) -> None:
    """Render cow-style aerodynamics visualization: surface Cp + velocity glyphs."""
    screenshot = save_path or output_path
    try:
        _render_pyvista(
            solution,
            viz,
            interactive=viz.interactive,
            save_path=screenshot,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        if viz.interactive:
            raise RuntimeError(
                "Interactive 3D viewer requires PyVista/VTK. "
                "Install dependencies and retry, or use --no-interactive --save output.png"
            ) from exc
        if screenshot is None:
            raise RuntimeError(
                "PyVista is unavailable. Use --save to write a matplotlib PNG instead."
            ) from exc
        _render_matplotlib(solution, viz, screenshot)
    except ValueError as exc:
        raise RuntimeError(f"Visualization error: {exc}") from exc


def _render_pyvista(
    solution: FlowSolution,
    viz: VizConfig,
    interactive: bool,
    save_path: Path | None,
) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        off = _create_plotter(solution, viz, off_screen=True, interactive=False)
        off.show(screenshot=str(save_path))
        off.close()

    if interactive:
        pl = _create_plotter(solution, viz, off_screen=False, interactive=True)
        pl.show()
    elif not save_path:
        pl = _create_plotter(solution, viz, off_screen=True, interactive=False)
        pl.show()


def _create_checker_texture(cells: int = 16, pixels_per_cell: int = 64) -> np.ndarray:
    """Black grid lines on white — classic CFD / wind-tunnel floor look."""
    size = cells * pixels_per_cell
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    line = max(2, pixels_per_cell // 24)
    for i in range(cells + 1):
        pos = min(i * pixels_per_cell, size - 1)
        end = min(pos + line, size)
        image[pos:end, :, :] = 0
        image[:, pos:end, :] = 0
    return image


def _add_checker_floor(pl, bounds: np.ndarray, padding: float = 1.4, cells: int = 12) -> None:
    """Add a tiled grid floor beneath the object."""
    import pyvista as pv

    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]
    span_x = max(float(xmax - xmin), 1e-6)
    span_z = max(float(zmax - zmin), 1e-6)
    floor_y = ymin - 0.02 * max(ymax - ymin, 1e-6)

    half_x = span_x * padding / 2.0
    half_z = span_z * padding / 2.0
    cx = (xmin + xmax) / 2.0
    cz = (zmin + zmax) / 2.0

    x0, x1 = cx - half_x, cx + half_x
    z0, z1 = cz - half_z, cz + half_z

    points = np.array(
        [
            [x0, floor_y, z0],
            [x1, floor_y, z0],
            [x1, floor_y, z1],
            [x0, floor_y, z1],
        ],
        dtype=float,
    )
    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    floor = pv.PolyData(points, faces)

    # Manual UV tiling — avoids PyVista texture_map_to_plane failures on Plane meshes
    floor.active_texture_coordinates = np.array(
        [[0.0, 0.0], [cells, 0.0], [cells, cells], [0.0, cells]],
        dtype=np.float32,
    )

    texture = pv.numpy_to_texture(_create_checker_texture(cells=cells))
    pl.add_mesh(floor, texture=texture, lighting=False, show_scalar_bar=False)


def _center_camera(pl, bounds: np.ndarray) -> None:
    """Point the camera at the object center with a balanced iso view."""
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    span = float(max(xmax - xmin, ymax - ymin, zmax - zmin))

    pl.reset_camera(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
    pl.camera.focal_point = center
    pl.camera.position = center + np.array([1.4, 0.9, 1.2]) * span
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.clipping_range = (span * 0.05, span * 20.0)


def _configure_interaction(pl) -> None:
    """Left drag rotates; Ctrl+left drag pans focal point; scroll zooms."""
    pl.enable_custom_trackball_style(
        left="rotate",
        control_left="pan",
        shift_left="pan",
        middle="pan",
        right="dolly",
    )


def _create_plotter(
    solution: FlowSolution,
    viz: VizConfig,
    off_screen: bool,
    interactive: bool,
) -> "pv.Plotter":
    import pyvista as pv

    mesh = mesh_to_pyvista(solution.mesh)
    mesh["Cp"] = solution.surface_cp
    mesh["velocity"] = solution.surface_velocity
    mesh.set_active_scalars("Cp")

    bounds = solution.mesh.bounds
    span = float(np.max(bounds[1] - bounds[0]))
    arrow_len = viz.arrow_size * span
    thin_arrow = pv.Arrow(
        tip_length=0.35,
        tip_radius=0.12,
        shaft_radius=0.04,
    )

    pl = pv.Plotter(off_screen=off_screen, window_size=(1280, 900))
    pl.set_background(viz.background)

    if viz.checker_floor:
        _add_checker_floor(pl, bounds)

    pl.add_mesh(
        mesh,
        scalars="Cp",
        cmap=viz.surface_colormap,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={"title": "Pressure coefficient Cp"},
    )

    near_field = _near_field_glyph_cloud(solution, viz)
    if near_field.n_points > 0:
        flow_arrows = near_field.glyph(
            orient="velocity",
            scale=False,
            factor=arrow_len,
            geom=thin_arrow,
        )
        pl.add_mesh(flow_arrows, color="black", opacity=0.92)

    if viz.show_streamlines:
        stream_mesh = _build_streamline_seed_mesh(solution)
        if stream_mesh.n_points > 1:
            streamlines = stream_mesh.streamlines_from_source(
                _streamline_source(solution),
                vectors="velocity",
                max_length=span * 4.0,
            )
            if streamlines.n_points > 0:
                pl.add_mesh(streamlines, color=[0.1, 0.1, 0.1], line_width=1.0, opacity=0.25)

    pl.add_axes()
    _center_camera(pl, bounds)

    if interactive:
        _configure_interaction(pl)
        pl.add_text(
            "Drag=rotate  Ctrl+drag=pan  Scroll=zoom  q=close",
            position="upper_left",
            font_size=10,
            color="black",
        )
    return pl


def _render_matplotlib(
    solution: FlowSolution,
    viz: VizConfig,
    output_path: Path,
) -> None:
    """Fallback renderer when VTK/PyVista is blocked (e.g. Windows App Control)."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = solution.mesh
    vertices = mesh.vertices
    faces = mesh.faces
    cp = solution.surface_cp
    velocity = solution.surface_velocity

    fig = plt.figure(figsize=(12, 9), facecolor=viz.background)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(viz.background)

    cmap = colormaps.get_cmap(viz.surface_colormap)
    norm = Normalize(vmin=float(np.min(cp)), vmax=float(np.max(cp)))

    triangles = vertices[faces]
    face_cp = cp[faces].mean(axis=1)
    colors = cmap(norm(face_cp))

    collection = Poly3DCollection(triangles, facecolors=colors, edgecolor="none", alpha=1.0)
    ax.add_collection3d(collection)

    span = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    arrow_len = viz.arrow_size * span

    near_pts, near_dirs = _near_field_points(solution, viz)
    if len(near_pts) > 0:
        ax.quiver(
            near_pts[:, 0],
            near_pts[:, 1],
            near_pts[:, 2],
            near_dirs[:, 0],
            near_dirs[:, 1],
            near_dirs[:, 2],
            length=arrow_len,
            normalize=True,
            color="black",
            linewidth=0.25,
            alpha=0.85,
        )

    ax.set_title("Aerodynamics — surface Cp with near-field flow vectors")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Pressure coefficient Cp")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=viz.background)
    plt.close(fig)


def _near_field_points(solution: FlowSolution, viz: VizConfig) -> tuple[np.ndarray, np.ndarray]:
    """Seed points in a few fluid shells upstream and alongside the body."""
    import trimesh.proximity as proximity

    mesh = solution.mesh
    span = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    u_hat = solution.u_inf / np.linalg.norm(solution.u_inf)
    center = mesh.centroid

    sample_count = viz.vector_density**2
    idx = np.linspace(0, len(mesh.vertices) - 1, sample_count, dtype=int)
    anchor_points = mesh.vertices[idx]

    _, _, face_ids = proximity.closest_point(mesh, anchor_points)
    normals = mesh.face_normals[face_ids]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    layer_fracs = np.linspace(
        0.06,
        viz.near_field_depth,
        max(viz.near_field_layers, 1),
    )

    seeds: list[np.ndarray] = []
    for frac in layer_fracs:
        offset = frac * span
        seeds.append(anchor_points + normals * offset)

    points = np.vstack(seeds)
    rel = points - center
    upstream = rel @ u_hat
    lateral = rel - np.outer(upstream, u_hat)
    lateral_mag = np.linalg.norm(lateral, axis=1)
    # Front + sides only — skip deep wake behind the body
    keep = (upstream > -0.12 * span) | (lateral_mag > 0.25 * span)
    points = points[keep]

    inside = mesh.contains(points)
    points = points[~inside]
    if len(points) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    velocity = interpolate_velocity(solution, points)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    speed[speed == 0] = 1.0
    directions = velocity / speed
    return points, directions


def _near_field_glyph_cloud(solution: FlowSolution, viz: VizConfig):
    import pyvista as pv

    points, directions = _near_field_points(solution, viz)
    if len(points) == 0:
        return pv.PolyData()

    cloud = pv.PolyData(points)
    cloud["velocity"] = directions
    return cloud


def _streamline_source(solution: FlowSolution):
    import pyvista as pv

    bounds = solution.mesh.bounds
    xmin, ymin, zmin = bounds[0] - 0.5
    xmax, _, zmax = bounds[1] + 0.5
    ymid = 0.5 * (bounds[0][1] + bounds[1][1])

    xs = np.linspace(xmin, xmax, 8)
    zs = np.linspace(zmin, zmax, 8)
    xx, zz = np.meshgrid(xs, zs)
    yy = np.full_like(xx, ymid)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return pv.PolyData(points)


def _build_streamline_seed_mesh(solution: FlowSolution):
    import pyvista as pv

    velocity = solution.velocity_field
    origin = solution.grid_origin
    spacing = solution.grid_spacing
    nx, ny, nz, _ = velocity.shape

    x = origin[0] + spacing * np.arange(nx)
    y = origin[1] + spacing * np.arange(ny)
    z = origin[2] + spacing * np.arange(nz)
    grid = pv.RectilinearGrid(x, y, z)
    vectors = velocity.reshape(-1, 3, order="F")
    grid["velocity"] = vectors
    return grid
