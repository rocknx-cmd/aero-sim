from __future__ import annotations

from pathlib import Path

import click

from .config import ObjectConfig, SimulationConfig, VizConfig
from .objects import build_mesh
from .solver import solve_flow
from .visualize import render_solution


@click.group()
def cli() -> None:
    """Aerodynamics simulation CLI."""


def _resolve_output_path(cfg: SimulationConfig, save: Path | None) -> Path | None:
    if save is not None:
        return save
    if cfg.viz.screenshot:
        return Path(cfg.viz.screenshot)
    return None


@cli.command("run")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Import a 3D model (.stl, .obj, .glb, …) and run interactively.",
)
@click.option("--interactive/--no-interactive", default=True, help="Open interactive 3D viewer.")
@click.option("--save", type=click.Path(path_type=Path), default=None, help="Also save a PNG screenshot.")
@click.option("--resolution", type=int, default=None, help="Override grid resolution.")
def run(
    config: Path | None,
    model: Path | None,
    interactive: bool,
    save: Path | None,
    resolution: int | None,
) -> None:
    """Run a simulation from a YAML config and/or an imported model file."""
    if config is None and model is None:
        raise click.UsageError("Provide --config and/or --model.")

    if config is not None:
        cfg = SimulationConfig.from_yaml(config)
    else:
        cfg = SimulationConfig(name=model.stem)

    if model is not None:
        cfg.object.kind = "file"
        cfg.object.path = str(model)

    cfg.viz.interactive = interactive
    if resolution is not None:
        cfg.grid.resolution = resolution

    mesh = build_mesh(cfg.object)
    click.echo(
        f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces "
        f"(span {float(mesh.bounds[1].max() - mesh.bounds[0].min()):.2f} units)"
    )

    click.echo("Solving flow field…")
    solution = solve_flow(mesh, cfg.flow, cfg.grid)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = _resolve_output_path(cfg, save)

    if cfg.viz.interactive:
        click.echo("Opening interactive viewer — drag to rotate, scroll to zoom, q to close.")
        render_solution(solution, cfg.viz, output_path=None, save_path=screenshot_path)
    elif screenshot_path is not None:
        render_solution(solution, cfg.viz, output_path=screenshot_path)
    else:
        raise click.ClickException("Use --interactive or --save to produce output.")

    npz_path = output_dir / f"{cfg.name}_fields.npz"
    import numpy as np

    np.savez(
        npz_path,
        surface_cp=solution.surface_cp,
        surface_velocity=solution.surface_velocity,
        velocity_field=solution.velocity_field,
        phi_field=solution.phi_field,
        grid_origin=solution.grid_origin,
        grid_spacing=solution.grid_spacing,
        u_inf=solution.u_inf,
        reynolds_number=solution.reynolds_number,
    )

    click.echo(f"Reynolds number: {solution.reynolds_number:,.0f}")
    if screenshot_path is not None:
        click.echo(f"Saved screenshot: {screenshot_path}")
    click.echo(f"Saved field data: {npz_path}")


@cli.command("import-model")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--resolution", type=int, default=40, show_default=True)
@click.option("--target-size", type=float, default=4.0, show_default=True)
@click.option("--max-faces", type=int, default=50_000, show_default=True)
@click.option("--save", type=click.Path(path_type=Path), default=None)
def import_model(
    model_path: Path,
    resolution: int,
    target_size: float,
    max_faces: int,
    save: Path | None,
) -> None:
    """Quick interactive run on any imported 3D model (aircraft, STL, OBJ, GLB, …)."""
    cfg = SimulationConfig(name=model_path.stem)
    cfg.object = ObjectConfig(
        kind="file",
        path=str(model_path),
        center=True,
        target_size=target_size,
        max_faces=max_faces,
    )
    cfg.grid.resolution = resolution
    cfg.viz.interactive = True
    cfg.viz.screenshot = str(save) if save else None

    ctx = click.get_current_context()
    ctx.invoke(
        run,
        config=None,
        model=model_path,
        interactive=True,
        save=save,
        resolution=resolution,
    )


@cli.command("list-objects")
def list_objects() -> None:
    """List built-in object kinds."""
    kinds = ["sphere", "box", "cylinder", "ellipsoid", "file"]
    for kind in kinds:
        click.echo(kind)


@cli.command("list-formats")
def list_formats() -> None:
    """List supported import file formats."""
    from .objects import SUPPORTED_EXTENSIONS

    for ext in sorted(SUPPORTED_EXTENSIONS):
        click.echo(ext)


if __name__ == "__main__":
    cli()
