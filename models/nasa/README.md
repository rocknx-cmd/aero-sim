# NASA 3D models (bundled)

Free public-domain models from [NASA 3D Resources](https://github.com/nasa/NASA-3D-Resources).

| File | Model | Run config |
|------|-------|------------|
| `x57.glb` | NASA X-57 electric aircraft | `configs/nasa-x57.yaml` |
| `space-shuttle.glb` | Space Shuttle orbiter | `configs/nasa-space-shuttle.yaml` |
| `ingenuity-helicopter.glb` | Ingenuity Mars Helicopter | `configs/nasa-ingenuity.yaml` |
| `parker-solar-probe.glb` | Parker Solar Probe | `configs/nasa-parker-probe.yaml` |
| `apollo-lunar-module.glb` | Apollo Lunar Module | `configs/nasa-apollo-lm.yaml` |

## Quick run

```cmd
aero-sim run --config configs\nasa-x57.yaml
aero-sim run --config configs\nasa-space-shuttle.yaml
aero-sim import-model models\nasa\ingenuity-helicopter.glb
```

## Re-download

```powershell
.\scripts\download_nasa_models.ps1
```

## Credit

NASA 3D Resources — free to use per [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
