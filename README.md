# aero-sim

Physically grounded 3D aerodynamics simulation with cow-style visualization: surface pressure maps and velocity arrow fields on variable objects.

## What it does

- Simulates **steady inviscid incompressible flow** by solving the Laplace equation for velocity potential (`∇²φ = 0`)
- Enforces **no flow through the body** (`u·n = 0`) using an immersed-boundary finite-difference solver
- Computes **pressure coefficient** on the surface via Bernoulli: `Cp = 1 - |V|²/U²`
- Renders **surface heatmaps + vector glyphs** (inspired by the classic "aerodynamics of a cow" visualization)
- Supports **variable objects**: sphere, box, cylinder, ellipsoid, or custom STL meshes

## Physics scope

This solver is physically accurate for **potential flow** (high Reynolds number, attached flow, no viscosity effects). It does not model:

- Boundary layers, separation, or drag from viscosity
- Compressibility or unsteady effects

Reynolds number is reported for reference. A viscous Navier-Stokes backend can be added later in `sandbox/integration/`.

## Setup

```powershell
cd C:\Users\DevT\Dev\personal\aero-sim
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Run a simulation

```powershell
aero-sim run --config configs/sphere.yaml
aero-sim run --config configs/box.yaml
aero-sim run --config configs/cylinder.yaml
```

Outputs land in `output/` as PNG screenshots and `.npz` field data.

## Configure your own object

Edit or copy a file in `configs/`:

```yaml
name: my-test

object:
  kind: ellipsoid        # sphere | box | cylinder | ellipsoid | stl
  scale: 1.0
  params:
    rx: 1.2
    ry: 0.5
    rz: 0.8
  # path: models/custom.stl   # required for kind: stl

flow:
  speed: 15.0
  direction: [1.0, 0.0, 0.0]

grid:
  resolution: 56        # higher = more accurate, slower
  padding: 0.6

viz:
  surface_colormap: jet
  vector_density: 12
  screenshot: output/my-test.png
```

## Project layout

```
aero-sim/
├── configs/          # simulation configs (one per object/test)
├── models/           # place custom STL/OBJ files here
├── output/           # rendered images and field data
├── src/aero_sim/     # solver, visualization, CLI
└── scripts/          # helper scripts
```

## List built-in shapes

```powershell
aero-sim list-objects
```
