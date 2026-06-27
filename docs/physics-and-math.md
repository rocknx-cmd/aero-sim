# Physics and Mathematics of aero-sim

This document explains the fluid model, boundary conditions, numerical method, and post-processing used in **aero-sim**. It is written to be readable without opening the source code.

---

## 1. What problem are we solving?

**aero-sim** computes **steady, inviscid, incompressible flow** around a solid 3D body. The fluid is treated as **irrotational**, which means the velocity field can be derived from a **scalar potential** \(\phi\):

\[
\mathbf{u} = \nabla \phi
\]

This is called **potential flow**. It is a classical aerodynamics model that captures:

- How flow accelerates and decelerates around a body
- **Pressure variations** on the surface (via Bernoulli)
- Overall streamlining patterns at **high Reynolds number** when viscosity effects are thin (attached flow)

It does **not** capture:

- Viscous **boundary layers**
- **Flow separation** and wake turbulence
- **Drag** dominated by skin friction or separated regions
- **Compressibility** (Mach effects)

Those require Navier–Stokes (or experimental/CFD tools like OpenFOAM). Potential flow is still **physically correct within its assumptions** — it is not an ad-hoc fake visualization.

---

## 2. Governing equations

### 2.1 Continuity (incompressible)

\[
\nabla \cdot \mathbf{u} = 0
\]

Since \(\mathbf{u} = \nabla\phi\):

\[
\nabla^2 \phi = 0
\]

This is the **Laplace equation**. It is elliptic: boundary conditions at the body and far field fully determine the solution.

### 2.2 Irrotationality

\[
\nabla \times \mathbf{u} = \mathbf{0}
\]

Potential flow automatically satisfies this because mixed partial derivatives commute (for simply connected domains).

### 2.3 Free-stream condition

Far from the body, the flow approaches a uniform velocity \(\mathbf{U}_\infty\):

\[
\mathbf{u} \to \mathbf{U}_\infty, \quad \|\mathbf{x}\| \to \infty
\]

The corresponding potential (up to an additive constant) is:

\[
\phi_\infty = \mathbf{U}_\infty \cdot \mathbf{x}
\]

In code, \(\mathbf{U}_\infty\) is built from `flow.speed` and `flow.direction`:

\[
\mathbf{U}_\infty = U_\infty \, \hat{\mathbf{n}}, \quad U_\infty = \text{speed},\ \hat{\mathbf{n}} = \frac{\text{direction}}{\|\text{direction}\|}
\]

### 2.4 Impermeable wall (body surface)

Fluid cannot pass through the solid surface. With outward unit normal \(\hat{\mathbf{n}}\) on the body:

\[
\mathbf{u} \cdot \hat{\mathbf{n}} = 0
\]

Substituting \(\mathbf{u} = \nabla\phi\):

\[
\frac{\partial \phi}{\partial n} = \nabla\phi \cdot \hat{\mathbf{n}} = 0
\]

This is a **Neumann (no-flux) boundary condition** on the body.

---

## 3. Pressure and pressure coefficient

### 3.1 Bernoulli equation (steady, inviscid)

Along a streamline:

\[
p + \frac{1}{2}\rho \|\mathbf{u}\|^2 = \text{constant}
\]

Subtracting the free-stream value \(p_\infty + \frac{1}{2}\rho U_\infty^2\):

\[
p - p_\infty = \frac{1}{2}\rho \left(U_\infty^2 - \|\mathbf{u}\|^2\right)
\]

### 3.2 Pressure coefficient \(C_p\)

Non-dimensional pressure, standard in aerodynamics:

\[
C_p = \frac{p - p_\infty}{\frac{1}{2}\rho U_\infty^2}
\]

Using Bernoulli:

\[
\boxed{C_p = 1 - \frac{\|\mathbf{u}\|^2}{U_\infty^2}}
\]

Interpretation:

| \(C_p\) | Meaning |
|--------|---------|
| \(C_p = 1\) | Stagnation point (\(\mathbf{u} = \mathbf{0}\)) |
| \(C_p = 0\) | Local speed equals free-stream speed |
| \(C_p < 0\) | Local speed **greater** than free stream (suction / low pressure) |

The viewer colors the mesh by \(C_p\) and draws small arrows in the direction of \(\mathbf{u}\).

---

## 4. Reynolds number (reported, not solved)

Viscosity \(\mu\) does not enter the potential-flow solve, but we report:

\[
\mathrm{Re} = \frac{\rho U_\infty L}{\mu}
\]

where \(L\) is a characteristic length (max bounding-box span of the body), \(\rho\) is `flow.density`, and \(\mu\) is `flow.viscosity`.

| Regime | Typical meaning |
|--------|-----------------|
| \(\mathrm{Re} \ll 1\) | Viscous forces dominate — potential flow is **not** valid |
| \(\mathrm{Re} \gg 1\) | Inertia dominates — potential flow is a useful **outer** approximation when flow stays attached |

---

## 5. Numerical method

### 5.1 Domain and grid

1. Build an axis-aligned bounding box around the mesh (with padding).
2. Discretize it into a uniform 3D Cartesian grid of size \(N_x \times N_y \times N_z\) (`grid.resolution`).
3. Mark cells whose **centers** lie inside the solid as **blocked** using ray/triangle tests (`trimesh.contains`).

### 5.2 Discrete Laplace equation

For each **fluid** cell, approximate:

\[
\nabla^2 \phi \approx \sum_{k \in \{x,y,z\}} \frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{\Delta^2} = 0
\]

This yields a **sparse linear system** \(\mathbf{A}\boldsymbol{\phi} = \mathbf{b}\).

### 5.3 Boundary conditions on the grid

**Far-field (domain faces):** Dirichlet data from the free stream:

\[
\phi = \mathbf{U}_\infty \cdot \mathbf{x}
\]

**Solid–fluid interface:** ghost-cell / mirror treatment enforces \(\partial\phi/\partial n = 0\) on faces adjacent to blocked cells (impermeable wall).

The system is solved with SciPy’s sparse direct solver (`scipy.sparse.linalg.spsolve`).

### 5.4 Recovering velocity

After \(\phi\) is known on fluid nodes, velocity components are computed by **central finite differences**:

\[
u_x \approx \frac{\partial \phi}{\partial x}, \quad
u_y \approx \frac{\partial \phi}{\partial y}, \quad
u_z \approx \frac{\partial \phi}{\partial z}
\]

### 5.5 Surface sampling

\(C_p\) is evaluated at mesh vertices by interpolating \(\mathbf{u}\) from the grid at a point slightly **offset outward** from the surface (avoids numerical artifacts exactly on the wall).

---

## 6. Accuracy and limitations

### What improves accuracy

- Higher `grid.resolution` (finer Cartesian mesh)
- Larger `grid.padding` (far field farther from body)
- Watertight, reasonably smooth surface meshes

### Known limitations

| Effect | In this model? |
|--------|----------------|
| Attached high-Re streamlining | Yes (qualitatively) |
| Stagnation / suction peaks on smooth bodies | Yes (approximate) |
| Boundary layer | No |
| Separation & wake | No |
| Form drag from separation | No |
| Skin-friction drag | No |

For a **sphere** in exact potential flow, theory gives \(C_p = 1 - \frac{9}{4}\sin^2\theta\) on the surface (in 3D axisymmetric potential flow around a sphere). Our finite-difference + immersed-boundary solution approximates this; coarser grids smear sharp gradients.

---

## 7. Geometry pipeline

Imported models (STL, OBJ, GLB, …) go through:

1. **Load** — multi-object scenes are merged into one mesh
2. **Rotate** — optional `rotation: [rx, ry, rz]` in degrees
3. **Scale** — user `scale` factor
4. **Center** — centroid moved to origin (`center: true`)
5. **Normalize** — optional `target_size` (longest axis set to fixed length)
6. **Decimate** — if face count exceeds `max_faces`, mesh is simplified (quadric decimation) for speed
7. **Repair** — hole filling, normal fixing for simulation

---

## 8. Visualization mapping

| Visual element | Quantity |
|----------------|----------|
| Surface color | \(C_p\) |
| Arrow direction | \(\mathbf{u} / \|\mathbf{u}\|\) |
| Arrow length | Fixed small size (direction-only glyphs, cow-style) |
| Checker floor | Reference ground plane (no physics) |

Interactive controls:

- **Drag** — orbit camera around focal point
- **Ctrl + drag** — pan focal point (shift view center)
- **Scroll** — zoom

---

## 9. References (conceptual)

- Anderson, *Fundamentals of Aerodynamics* — potential flow, Bernoulli, \(C_p\)
- Batchelor, *An Introduction to Fluid Dynamics* — irrotational flow, Laplace equation
- Katz & Plotkin, *Low-Speed Aerodynamics* — panel methods and potential theory

---

## 10. Possible extensions

Future backends could add:

- **Panel method** (boundary-element) for smoother surface BCs
- **Navier–Stokes** (OpenFOAM / lattice Boltzmann) for viscous effects and separation
- **Lift/drag integration** from surface pressure and shear

Those belong in `Dev/sandbox/integration/` when you are ready to go beyond potential flow.
