from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .config import ObjectConfig


def _unit_direction(direction: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Flow direction must be non-zero.")
    return vec / norm


def build_mesh(config: ObjectConfig) -> trimesh.Trimesh:
    """Create or load a watertight surface mesh for the test object."""
    kind = config.kind.lower()
    params = config.params
    scale = config.scale

    if kind == "sphere":
        radius = float(params.get("radius", 1.0))
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    elif kind == "box":
        extents = (
            float(params.get("width", 1.0)),
            float(params.get("depth", 1.0)),
            float(params.get("height", 1.0)),
        )
        mesh = trimesh.creation.box(extents=extents)
    elif kind == "cylinder":
        radius = float(params.get("radius", 0.5))
        height = float(params.get("height", 2.0))
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    elif kind == "ellipsoid":
        radii = (
            float(params.get("rx", 1.0)),
            float(params.get("ry", 0.6)),
            float(params.get("rz", 0.8)),
        )
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        mesh.apply_scale(radii)
    elif kind == "stl":
        if not config.path:
            raise ValueError("STL objects require `object.path`.")
        mesh = trimesh.load_mesh(Path(config.path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    else:
        raise ValueError(f"Unsupported object kind: {config.kind}")

    mesh.apply_scale(scale)

    if not mesh.is_watertight:
        mesh.fill_holes()
        mesh.remove_unreferenced_vertices()

    mesh.process(validate=True)
    return mesh


def mesh_to_pyvista(mesh: trimesh.Trimesh):
    """Convert a trimesh surface to a PyVista PolyData mesh."""
    import pyvista as pv

    faces = np.hstack(
        [np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)
