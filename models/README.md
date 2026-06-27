# 3D models folder

Drop imported models here — aircraft, vehicles, CAD exports, etc.

## Supported formats

`.stl` `.obj` `.glb` `.gltf` `.ply` `.off` `.dae` `.3mf`

## Where to find models

- [Sketchfab](https://sketchfab.com/search?q=b2+bomber&type=models) — search "B2 bomber", filter downloadable
- [Thingiverse](https://www.thingiverse.com/) — STL downloads
- [NASA 3D Resources](https://nasa3d.arc.nasa.gov/) — spacecraft and aircraft
- Export from Blender, Fusion 360, SolidWorks, etc.

## Quick test (interactive viewer)

```cmd
cd C:\Users\DevT\Dev\personal\aero-sim
.venv\Scripts\activate.bat
aero-sim import-model models\b2-bomber.stl
```

Or with a config:

```cmd
aero-sim run --config configs\import-model.yaml
```

## Tips for aircraft / large models

- Use `max_faces: 50000` in the config — heavy meshes are auto-simplified
- Set `target_size: 4.0` so the solver grid fits the model
- Point `flow.direction` along the aircraft nose (e.g. `[1, 0, 0]`)
- Start with `resolution: 32`, increase once it runs smoothly
