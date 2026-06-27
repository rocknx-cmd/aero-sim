from __future__ import annotations

from pathlib import Path

import click

from .config import SimulationConfig
from .objects import build_mesh
from .solver import solve_flow
from .visualize import render_solution


@click.group()
def cli() -> None:
    """Aerodynamics simulation CLI."""


@cli.command("run")
@click.option("--config", "-c", required=True, type=click.Path(exists=True, path_type=Path))
def run(config: Path) -> None:
    """Run a simulation from a YAML config file."""
    cfg = SimulationConfig.from_yaml(config)
    mesh = build_mesh(cfg.object)
    solution = solve_flow(mesh, cfg.flow, cfg.grid)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshot = cfg.viz.screenshot or str(output_dir / f"{cfg.name}.png")
    render_solution(solution, cfg.viz, output_path=Path(screenshot))

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
    click.echo(f"Saved visualization: {screenshot}")
    click.echo(f"Saved field data: {npz_path}")


@cli.command("list-objects")
def list_objects() -> None:
    """List built-in object kinds."""
    kinds = ["sphere", "box", "cylinder", "ellipsoid", "stl"]
    for kind in kinds:
        click.echo(kind)


if __name__ == "__main__":
    cli()
