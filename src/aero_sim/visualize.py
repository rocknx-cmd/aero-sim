from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import VizConfig
from .solver import FlowSolution


def render_solution(
    solution: FlowSolution,
    viz: VizConfig,
    output_path: Path | None = None,
) -> None:
    """Render cow-style aerodynamics visualization: surface Cp + velocity glyphs."""
    try:
        _render_pyvista(solution, viz, output_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if output_path is None:
            raise RuntimeError(
                "Interactive PyVista rendering is unavailable on this system. "
                "Set viz.screenshot in the config for matplotlib fallback output."
            ) from exc
        _render_matplotlib(solution, viz, output_path)


def _render_pyvista(
    solution: FlowSolution,
    viz: VizConfig,
    output_path: Path | None,
) -> None:
    import pyvista as pv

    from .objects import mesh_to_pyvista

    mesh = mesh_to_pyvista(solution.mesh)
    mesh["Cp"] = solution.surface_cp
    mesh["velocity"] = solution.surface_velocity
    mesh.set_active_scalars("Cp")

    off_screen = bool(output_path and output_path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background(viz.background)

    pl.add_mesh(
        mesh,
        scalars="Cp",
        cmap=viz.surface_colormap,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={"title": "Pressure coefficient Cp"},
    )

    surface_points = mesh.points
    vectors = solution.surface_velocity
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(magnitudes > 0, magnitudes, 1.0)
    unit_vectors = vectors / safe

    stride = max(1, len(surface_points) // (viz.vector_density**2))
    glyph_points = surface_points[::stride]
    glyph_vectors = unit_vectors[::stride]
    glyph_mags = magnitudes[::stride].ravel()

    glyph_cloud = pv.PolyData(glyph_points)
    glyph_cloud["velocity"] = glyph_vectors
    glyph_cloud["magnitude"] = glyph_mags

    span = float(np.max(solution.mesh.bounds[1] - solution.mesh.bounds[0]))
    arrow_scale = 0.08 * span

    arrows = glyph_cloud.glyph(
        orient="velocity",
        scale="magnitude",
        factor=arrow_scale,
    )
    pl.add_mesh(arrows, color="black", opacity=0.85)

    volume_glyphs = _volume_glyph_cloud(solution, density=viz.vector_density)
    if volume_glyphs.n_points > 0:
        vol_arrows = volume_glyphs.glyph(
            orient="velocity",
            scale="magnitude",
            factor=0.06 * span,
        )
        pl.add_mesh(vol_arrows, color=[0.2, 0.2, 0.2], opacity=0.55)

    if viz.show_streamlines:
        stream_mesh = _build_streamline_seed_mesh(solution)
        if stream_mesh.n_points > 1:
            streamlines = stream_mesh.streamlines_from_source(
                _streamline_source(solution),
                vectors="velocity",
                max_time=100.0,
            )
            if streamlines.n_points > 0:
                pl.add_mesh(streamlines, color=[0.1, 0.1, 0.1], line_width=1.5, opacity=0.35)

    pl.add_axes()
    pl.camera_position = "iso"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.show(screenshot=str(output_path))
    else:
        pl.show()


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

    stride = max(1, len(vertices) // (viz.vector_density**2))
    pts = vertices[::stride]
    vecs = velocity[::stride]
    mags = np.linalg.norm(vecs, axis=1, keepdims=True)
    mags[mags == 0] = 1.0
    dirs = vecs / mags
    span = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    arrow_len = 0.06 * span

    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        dirs[:, 0],
        dirs[:, 1],
        dirs[:, 2],
        length=arrow_len,
        normalize=True,
        color="black",
        linewidth=0.4,
        alpha=0.85,
    )

    ax.set_title("Aerodynamics — surface Cp with velocity vectors")
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


def _volume_glyph_cloud(solution: FlowSolution, density: int):
    import pyvista as pv

    velocity = solution.velocity_field
    origin = solution.grid_origin
    spacing = solution.grid_spacing
    nx, ny, nz, _ = velocity.shape

    xs = np.linspace(0, nx - 1, density)
    ys = np.linspace(0, ny - 1, density)
    zs = np.linspace(0, nz - 1, density)

    points: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    magnitudes: list[float] = []

    for xi in xs:
        for yi in ys:
            for zi in zs:
                i, j, k = int(xi), int(yi), int(zi)
                vec = velocity[i, j, k]
                mag = float(np.linalg.norm(vec))
                if mag < 1e-8:
                    continue
                pos = origin + spacing * np.array([i, j, k], dtype=float)
                if solution.mesh.contains([pos])[0]:
                    continue
                points.append(pos)
                vectors.append(vec / mag)
                magnitudes.append(mag)

    if not points:
        return pv.PolyData()

    cloud = pv.PolyData(np.asarray(points))
    cloud["velocity"] = np.asarray(vectors)
    cloud["magnitude"] = np.asarray(magnitudes)
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
