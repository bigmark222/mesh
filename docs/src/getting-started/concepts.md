# Basic Concepts

Understanding these fundamental concepts will help you work effectively with mesh-repair.

## What Is a Triangle Mesh?

A triangle mesh is a 3D surface represented as:
- **Vertices**: Points in 3D space (x, y, z coordinates)
- **Faces**: Triangles connecting three vertices each

```
    v2
    /\
   /  \
  /    \
 /______\
v0      v1
```

mesh-repair uses an indexed representation:
- Vertices stored in a list: `[v0, v1, v2, v3, ...]`
- Faces reference vertices by index: `[0, 1, 2]` means "triangle using v0, v1, v2"

```rust
use mesh_repair::{Mesh, Vertex};

let mut mesh = Mesh::new();

// Add vertices
mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));

// Add a face (indices into vertex list)
mesh.faces.push([0, 1, 2]);
```

## Mesh Properties

### Manifold

A **manifold** mesh has well-defined inside/outside:
- Every edge is shared by exactly 1 or 2 faces
- No edge is shared by 3+ faces (non-manifold)

```
Manifold edge (2 faces):     Non-manifold edge (3+ faces):
    ┌───┐                        ┌───┐
    │   │                        │ / │
────┼───┼────                ────┼───┼────
    │   │                        │ \ │
    └───┘                        └───┘
```

### Watertight

A **watertight** mesh is completely closed:
- No boundary edges (edges with only 1 face)
- No holes in the surface

Watertight meshes have a well-defined volume and are required for most 3D printing.

### Winding Order

**Winding order** determines which side of a face is "outside":
- **Counter-clockwise (CCW)**: Standard convention, normal points toward viewer
- **Clockwise (CW)**: Normal points away from viewer

```
Counter-clockwise (outside up):    Clockwise (outside down):
      v2                                 v2
      /\                                 /\
     /  \  → Normal UP                  /  \  → Normal DOWN
    /    \                             /    \
   v0────v1                           v1────v0
```

Consistent winding order is critical for:
- Correct rendering (backface culling)
- Volume calculations
- Shell generation (inside vs outside)

## Common Mesh Issues

### Duplicate Vertices

Multiple vertices at the same (or nearly same) position:

```
Before welding:        After welding:
v0──v1  v2──v3        v0──v1──v2
│    │  │    │   →    │       │
v4──v5  v6──v7        v3──────v4
(8 vertices)          (5 vertices)
```

Use `weld_vertices()` to merge duplicates.

### Holes (Boundary Edges)

Missing faces leave gaps in the surface:

```
    ┌───┬───┐
    │   │   │
    ├───┼───┤
    │   │ ← Hole (missing face)
    └───┘
```

Use `fill_holes()` to close gaps.

### Degenerate Triangles

Triangles with zero or near-zero area:
- Collinear vertices (all three on a line)
- Duplicate vertices in the same face

```
Degenerate (collinear):    Degenerate (duplicate):
v0──────v1──────v2         v0 = v1
     (area ≈ 0)                  \
                                  v2
```

Use `remove_degenerate_triangles()` to clean these.

### Inconsistent Winding

Adjacent faces with opposite winding orders:

```
Normal UP ↑    Normal DOWN ↓
   ┌───┬───┐
   │ ↑ │ ↓ │  ← Inconsistent!
   └───┴───┘
```

Use `fix_winding_order()` to make winding consistent.

## Mesh Quality Metrics

### Surface Area

Total area of all triangles. Used for:
- Material estimation
- Coating calculations
- Quality validation

```rust
let area = mesh.surface_area();
println!("Surface area: {} mm²", area);
```

### Volume

Enclosed volume (only valid for watertight meshes):

```rust
let volume = mesh.volume();
println!("Volume: {} mm³", volume.abs());
```

Note: Volume can be negative if winding is inverted.

### Bounding Box

Axis-aligned bounding box containing the mesh:

```rust
if let Some((min, max)) = mesh.bounds() {
    println!("Bounds: ({}, {}, {}) to ({}, {}, {})",
        min.x, min.y, min.z,
        max.x, max.y, max.z);
}
```

## Validation Report

The `validate_mesh()` function returns a comprehensive report:

```rust
use mesh_repair::{load_mesh, validate_mesh};

let mesh = load_mesh("model.stl")?;
let report = validate_mesh(&mesh);

// Topology checks
println!("Is manifold: {}", report.is_manifold);
println!("Is watertight: {}", report.is_watertight);
println!("Boundary edges: {}", report.boundary_edge_count);
println!("Non-manifold edges: {}", report.non_manifold_edge_count);

// Geometry stats
println!("Components: {}", report.component_count);
println!("Degenerate faces: {}", report.degenerate_face_count);

// 3D printing readiness
if report.is_printable() {
    println!("Ready for 3D printing!");
}
```

## Coordinate Systems

mesh-repair uses right-handed coordinates:
- **X**: Right
- **Y**: Up (or forward, depending on source)
- **Z**: Forward (or up, depending on source)

Units are not specified - use whatever your source data uses (usually millimeters for 3D printing).

## Next Steps

- [Loading and Saving Meshes](../guide/io.md) - File format details
- [Mesh Validation](../guide/validation.md) - Detailed validation guide
- [Repair Operations](../guide/repair.md) - Fixing mesh issues
