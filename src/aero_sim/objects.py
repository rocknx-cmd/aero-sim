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
    try:
        mesh.process(validate=False)
    except Exception:
        pass
    _validate_mesh(mesh, config.path or kind)
    return mesh


def _validate_mesh(mesh: trimesh.Trimesh, label: str) -> None:
    if mesh.bounds is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh '{label}' is empty after loading.")
    span = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if not np.isfinite(span) or span <= 0:
        raise ValueError(
            f"Mesh '{label}' has invalid bounds. "
            "The file may use an unsupported compression format."
        )


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

    if suffix in {".glb", ".gltf"}:
        return _load_gltf_mesh(file_path)

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


def _load_gltf_mesh(file_path: Path) -> trimesh.Trimesh:
    """Load GLB/GLTF with Draco support (required for NASA models)."""
    from pygltflib import GLTF2

    if file_path.suffix.lower() == ".glb":
        gltf = GLTF2().load_binary(str(file_path))
    else:
        gltf = GLTF2().load(str(file_path))

    blob = gltf.binary_blob()
    if blob is None:
        raise ValueError(f"No binary payload in {file_path}")

    meshes: list[trimesh.Trimesh] = []
    scene_index = gltf.scene if gltf.scene is not None else 0
    root_nodes = gltf.scenes[scene_index].nodes or []

    for node_index in root_nodes:
        _collect_gltf_node(gltf, blob, node_index, np.eye(4), meshes)

    if not meshes:
        raise ValueError(f"No geometry found in {file_path}")

    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def _collect_gltf_node(
    gltf,
    blob: bytes,
    node_index: int,
    parent_matrix: np.ndarray,
    meshes: list[trimesh.Trimesh],
) -> None:
    node = gltf.nodes[node_index]
    world_matrix = parent_matrix @ _node_local_matrix(node)

    if node.mesh is not None:
        gltf_mesh = gltf.meshes[node.mesh]
        for primitive in gltf_mesh.primitives:
            piece = _decode_gltf_primitive(gltf, blob, primitive)
            if piece is None:
                continue
            vertices, faces = piece
            if len(vertices) == 0 or len(faces) == 0:
                continue
            homog = np.column_stack([vertices, np.ones(len(vertices))])
            vertices = (world_matrix @ homog.T).T[:, :3]
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

    for child_index in node.children or []:
        _collect_gltf_node(gltf, blob, child_index, world_matrix, meshes)


def _node_local_matrix(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, dtype=float).reshape(4, 4).T

    matrix = np.eye(4)
    if node.translation:
        matrix[:3, 3] = np.asarray(node.translation, dtype=float)
    if node.rotation:
        matrix[:3, :3] = _quaternion_to_matrix(node.rotation)
    if node.scale:
        scale = np.diag([*node.scale, 1.0])
        matrix = matrix @ scale
    return matrix


def _quaternion_to_matrix(quat: list[float]) -> np.ndarray:
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _decode_gltf_primitive(gltf, blob: bytes, primitive):
    extensions = primitive.extensions or {}
    if "KHR_draco_mesh_compression" in extensions:
        return _decode_draco_primitive(gltf, blob, primitive)

    if primitive.attributes.POSITION is None:
        return None

    vertices = _read_accessor(gltf, blob, primitive.attributes.POSITION)
    faces = _read_faces(gltf, blob, primitive)
    return vertices, faces


def _decode_draco_primitive(gltf, blob: bytes, primitive):
    import DracoPy

    ext = primitive.extensions["KHR_draco_mesh_compression"]
    view = gltf.bufferViews[ext["bufferView"]]
    start = view.byteOffset or 0
    end = start + view.byteLength
    decoded = DracoPy.decode(blob[start:end])

    vertices = np.asarray(decoded.points, dtype=float).reshape(-1, 3)
    faces = np.asarray(decoded.faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _read_accessor(gltf, blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)

    dtype = {
        5126: np.float32,
        5123: np.uint16,
        5125: np.uint32,
    }[accessor.componentType]

    component_sizes = {5126: 4, 5123: 2, 5125: 4}
    num_components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor.type]
    count = accessor.count
    itemsize = component_sizes[accessor.componentType] * num_components
    data = blob[start : start + count * itemsize]
    array = np.frombuffer(data, dtype=dtype)
    return array.reshape(count, num_components) if num_components > 1 else array


def _read_faces(gltf, blob: bytes, primitive) -> np.ndarray:
    if primitive.indices is None:
        raise ValueError("Non-indexed GLTF primitives are not supported.")

    indices = _read_accessor(gltf, blob, primitive.indices).astype(np.int64).reshape(-1)
    mode = primitive.mode if primitive.mode is not None else 4
    if mode == 4:
        return indices.reshape(-1, 3)
    if mode == 5:
        return np.column_stack([indices[0::2], indices[1::2], indices[2::2]])
    raise ValueError(f"Unsupported GLTF primitive mode: {mode}")


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
    center = mesh.centroid
    if not np.all(np.isfinite(center)):
        center = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.vertices -= center
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
    stride = max(1, len(mesh.faces) // max_faces)
    keep = mesh.faces[::stride][:max_faces]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=keep, process=False)


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    try:
        if not mesh.is_watertight and len(mesh.faces) < 200_000:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
            mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


def mesh_to_pyvista(mesh: trimesh.Trimesh):
    """Convert a trimesh surface to a PyVista PolyData mesh."""
    import pyvista as pv

    faces = np.hstack(
        [np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)
