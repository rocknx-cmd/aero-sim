# aero-sim

Physically grounded 3D aerodynamics simulation with cow-style visualization: surface pressure maps and velocity arrow fields on variable objects.

## What it does

- Simulates **steady inviscid incompressible flow** by solving the Laplace equation for velocity potential (`∇²φ = 0`)
- Enforces **no flow through the body** (`u·n = 0`) using an immersed-boundary finite-difference solver
- Computes **pressure coefficient** on the surface via Bernoulli: `Cp = 1 - |V|²/U²`
- Opens an **interactive 3D viewer** — drag to rotate, scroll to zoom, `q` to close
- Supports **imported models**: STL, OBJ, GLB, PLY, and more (aircraft, vehicles, CAD exports)

## Setup

```cmd
cd C:\Users\DevT\Dev\personal\aero-sim
.venv\Scripts\activate.bat
pip install -e .
```

## Interactive run (built-in shapes)

```cmd
aero-sim run --config configs\sphere.yaml
aero-sim run --config configs\box.yaml
```

## Import any 3D model (B2 bomber, car, etc.)

Bundled **NASA models** are in `models/nasa/` — X-57 aircraft, Space Shuttle, Ingenuity helicopter, Parker Solar Probe, Apollo Lunar Module. See `models/nasa/README.md`.

```cmd
aero-sim run --config configs\nasa-x57.yaml
aero-sim run --config configs\nasa-space-shuttle.yaml
```

For your own files:

1. Download a model (`.stl`, `.obj`, `.glb`, …) into `models\`
2. Run interactively:

```cmd
aero-sim import-model models\b2-bomber.stl
```

Or pass the file directly:

```cmd
aero-sim run --model models\b2-bomber.stl
```

With a config file for fine control, copy `configs\import-model.yaml` and set `object.path`.

### Supported formats

`.stl` `.obj` `.glb` `.gltf` `.ply` `.off` `.dae` `.3mf`

### Tips for aircraft / heavy CAD meshes

- Large models are **auto-simplified** (`max_faces: 50000` by default)
- Models are **centered and scaled** to fit the solver (`target_size: 4.0`)
- Start with `--resolution 32`, increase once it runs smoothly
- Point `flow.direction` along the nose of the aircraft

## Save a screenshot too

```cmd
aero-sim run --config configs\sphere.yaml --save output\sphere.png
aero-sim import-model models\plane.stl --save output\plane.png
```

## Headless / image-only (no window)

```cmd
aero-sim run --config configs\sphere.yaml --no-interactive --save output\sphere.png
```

## Project layout

```
aero-sim/
├── configs/          # simulation configs
├── models/           # drop imported 3D files here
├── output/           # saved screenshots and field data
└── src/aero_sim/     # solver, visualization, CLI
```

## Commands

```cmd
aero-sim list-objects     # built-in shapes
aero-sim list-formats     # supported import formats
```

## Physics and math

See **[docs/physics-and-math.md](docs/physics-and-math.md)** for the full derivation: Laplace equation, Bernoulli, \(C_p\), boundary conditions, numerical discretization, and limitations.
