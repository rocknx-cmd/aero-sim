from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FlowConfig:
    """Free-stream and fluid properties."""

    speed: float = 10.0  # m/s
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    density: float = 1.225  # kg/m^3 (air at sea level)
    viscosity: float = 1.81e-5  # Pa·s


@dataclass
class GridConfig:
    """Numerical grid for the potential-flow solver."""

    resolution: int = 48
    padding: float = 0.5  # fraction of object span added around bounding box


@dataclass
class ObjectConfig:
    """Geometry definition for the test body."""

    kind: str = "sphere"
    # sphere | box | cylinder | ellipsoid | file | stl
    path: str | None = None
    scale: float = 1.0
    center: bool = True
    target_size: float | None = None  # normalize max span to this length
    max_faces: int | None = 50_000  # decimate large CAD meshes (e.g. aircraft)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees XYZ
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class VizConfig:
    """Visualization settings."""

    surface_colormap: str = "jet"
    vector_density: int = 24
    arrow_size: float = 0.016  # fraction of object span
    near_field_layers: int = 3
    near_field_depth: float = 0.22  # max offset from surface (× object span)
    show_streamlines: bool = False
    checker_floor: bool = True
    background: str = "white"
    interactive: bool = True
    screenshot: str | None = None


@dataclass
class SimulationConfig:
    name: str = "run"
    object: ObjectConfig = field(default_factory=ObjectConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    output_dir: str = "output"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        obj = data.get("object", {})
        flow = data.get("flow", {})
        grid = data.get("grid", {})
        viz = data.get("viz", {})

        direction = tuple(flow.get("direction", [1.0, 0.0, 0.0]))
        rotation = tuple(obj.get("rotation", [0.0, 0.0, 0.0]))
        max_faces = obj.get("max_faces", 50_000)
        target_size = obj.get("target_size")

        return cls(
            name=data.get("name", "run"),
            object=ObjectConfig(
                kind=obj.get("kind", "sphere"),
                path=obj.get("path"),
                scale=float(obj.get("scale", 1.0)),
                center=bool(obj.get("center", True)),
                target_size=float(target_size) if target_size is not None else None,
                max_faces=int(max_faces) if max_faces is not None else None,
                rotation=rotation,  # type: ignore[arg-type]
                params={k: float(v) for k, v in obj.get("params", {}).items()},
            ),
            flow=FlowConfig(
                speed=float(flow.get("speed", 10.0)),
                direction=direction,  # type: ignore[arg-type]
                density=float(flow.get("density", 1.225)),
                viscosity=float(flow.get("viscosity", 1.81e-5)),
            ),
            grid=GridConfig(
                resolution=int(grid.get("resolution", 48)),
                padding=float(grid.get("padding", 0.5)),
            ),
            viz=VizConfig(
                surface_colormap=viz.get("surface_colormap", "jet"),
                vector_density=int(viz.get("vector_density", 24)),
                arrow_size=float(viz.get("arrow_size", 0.016)),
                near_field_layers=int(viz.get("near_field_layers", 3)),
                near_field_depth=float(viz.get("near_field_depth", 0.22)),
                show_streamlines=bool(viz.get("show_streamlines", False)),
                checker_floor=bool(viz.get("checker_floor", True)),
                background=viz.get("background", "white"),
                interactive=bool(viz.get("interactive", True)),
                screenshot=viz.get("screenshot"),
            ),
            output_dir=data.get("output_dir", "output"),
        )
