from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .config import ObjectConfig

SUPPORTED_EXTENSIONS = {".stl", ".obj", ".ply", ".glb", ".gltf", ".off", ".dae", ".3mf"}


def build_mesh(config: ObjectConfig) -> trimesh.Trimesh:
    """Create or load a surface mesh for the test object."""
    kind = config.kind.lower()
    params = config.params
    scale = config.scale

    if kind in {"file", "stl", "obj", "model", "import"}:
        mesh = _load_external_mesh(config.path)
    elif kind == "sphere":
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
    else:
        raise ValueError(
            f"Unsupported object kind: {config.kind}. "
            f"Use sphere, box, cylinder, ellipsoid, or file."
        )

    mesh = _apply_rotation(mesh, config.rotation)
    mesh.apply_scale(scale)

    if config.center:
        mesh = _center_mesh(mesh)

    if config.target_size is not None:
        mesh = _normalize_size(mesh, config.target_size)

    if config.max_faces is not None and len(mesh.faces) > config.max_faces:
        mesh = _decimate_mesh(mesh, config.max_faces)

    mesh = _repair_mesh(mesh)
    mesh.process(validate=True)
    return mesh


def _load_external_mesh(path: str | None) -> trimesh.Trimesh:
    if not path:
        raise ValueError("Imported objects require `object.path` or `--model`.")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported model format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loaded = trimesh.load(file_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {file_path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Could not load mesh from {file_path}")

    return mesh


def _apply_rotation(
    mesh: trimesh.Trimesh,
    rotation_deg: tuple[float, float, float],
) -> trimesh.Trimesh:
    rx, ry, rz = rotation_deg
    if rx:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))
    if ry:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]))
    if rz:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]))
    return mesh


def _center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    return mesh


def _normalize_size(mesh: trimesh.Trimesh, target_size: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    span = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if span <= 0:
        return mesh
    mesh.apply_scale(target_size / span)
    return mesh


def _decimate_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(max_faces)
        if len(simplified.faces) > 0:
            return simplified
    except Exception:
        pass
    # Fallback: uniform face subsampling for environments without fast-simplification
    stride = max(1, len(mesh.faces) // max_faces)
    keep = mesh.faces[::stride][:max_faces]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=keep, process=False)


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()
    return mesh


def mesh_to_pyvista(mesh: trimesh.Trimesh):
    """Convert a trimesh surface to a PyVista PolyData mesh."""
    import pyvista as pv

    faces = np.hstack(
        [np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)
