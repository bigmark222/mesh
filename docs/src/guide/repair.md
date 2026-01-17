# Repair Operations

mesh-repair provides tools to fix common mesh issues.

## Overview

| Operation | Function | Fixes |
|-----------|----------|-------|
| Weld vertices | `weld_vertices()` | Duplicate vertices |
| Fill holes | `fill_holes()` | Boundary edges, gaps |
| Fix winding | `fix_winding_order()` | Inconsistent normals |
| Remove degenerates | `remove_degenerate_triangles()` | Zero-area faces |
| Remove small components | `remove_small_components()` | Noise, debris |

## Weld Vertices

Merge vertices that are within a tolerance distance:

```rust
use mesh_repair::{load_mesh, weld_vertices};

let mut mesh = load_mesh("model.stl")?;
let original = mesh.vertex_count();

// Merge vertices within 1 micron (0.001mm)
weld_vertices(&mut mesh, 1e-3);

let merged = original - mesh.vertex_count();
println!("Merged {} duplicate vertices", merged);
```

**When to use**:
- After loading STL files (which don't share vertices)
- When mesh has "cracks" between faces
- To reduce vertex count

**Tolerance guidelines**:
- `1e-6`: Very tight, only exact duplicates
- `1e-3`: 1 micron, good default for mm-scale models
- `0.1`: Aggressive, may merge intentional geometry

## Fill Holes

Close gaps in the mesh surface:

```rust
use mesh_repair::{load_mesh, fill_holes, validate_mesh};

let mut mesh = load_mesh("model.stl")?;

let report = validate_mesh(&mesh);
println!("Boundary edges before: {}", report.boundary_edge_count);

fill_holes(&mut mesh)?;

let report = validate_mesh(&mesh);
println!("Boundary edges after: {}", report.boundary_edge_count);
```

**How it works**:
1. Finds boundary edge loops
2. Triangulates each hole
3. Uses ear-clipping algorithm for clean triangulation

**Limitations**:
- Very large holes may produce poor triangulation
- Complex hole shapes may need manual intervention
- Cannot fill non-manifold boundaries

## Fix Winding Order

Make all face normals consistent:

```rust
use mesh_repair::{load_mesh, fix_winding_order};

let mut mesh = load_mesh("model.stl")?;
fix_winding_order(&mut mesh)?;
```

**How it works**:
1. Picks a seed face
2. Propagates orientation to neighbors
3. Flips inconsistent faces

**After fixing**:
- All faces wind counter-clockwise when viewed from outside
- Volume calculation will be correct sign
- Shell generation will work correctly

## Remove Degenerate Triangles

Remove triangles with zero or near-zero area:

```rust
use mesh_repair::{load_mesh, remove_degenerate_triangles, validate_mesh};

let mut mesh = load_mesh("model.stl")?;

let report = validate_mesh(&mesh);
println!("Degenerate faces: {}", report.degenerate_face_count);

// Remove triangles with area < 1e-10
remove_degenerate_triangles(&mut mesh, 1e-10);
```

**Types of degenerates**:
- Collinear vertices (all three on a line)
- Duplicate vertex indices in same face
- Very small triangles (below tolerance)

## Remove Small Components

Remove disconnected pieces below a face count threshold:

```rust
use mesh_repair::{load_mesh, remove_small_components, validate_mesh};

let mut mesh = load_mesh("scan.stl")?;

let report = validate_mesh(&mesh);
println!("Components: {}", report.component_count);

// Remove components with fewer than 100 faces
let removed = remove_small_components(&mut mesh, 100);
println!("Removed {} small components", removed);
```

**Use cases**:
- Removing scan noise/artifacts
- Cleaning up boolean operation debris
- Isolating main geometry

## Keep Largest Component

Keep only the largest connected component:

```rust
use mesh_repair::{load_mesh, keep_largest_component};

let mut mesh = load_mesh("scan.stl")?;
keep_largest_component(&mut mesh)?;

// Now mesh contains only the largest piece
```

## Combined Repair Pipeline

For typical repair workflow, use the Pipeline API:

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};

let mesh = load_mesh("broken.stl")?;

let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)              // 1. Merge duplicates
    .remove_degenerate_triangles(1e-10) // 2. Clean degenerates
    .fill_holes()                     // 3. Close holes
    .fix_winding()                    // 4. Fix normals
    .remove_small_components(50)      // 5. Remove noise
    .validate()                       // 6. Final check
    .finish();

save_mesh(&result.mesh, "repaired.stl")?;
```

## Repair Parameters

For fine-grained control:

```rust
use mesh_repair::{load_mesh, repair_mesh, RepairParams};

let mesh = load_mesh("model.stl")?;

let params = RepairParams {
    weld_tolerance: 1e-6,
    fill_holes: true,
    fix_winding: true,
    remove_degenerates: true,
    degenerate_threshold: 1e-10,
    ..Default::default()
};

let repaired = repair_mesh(&mesh, &params)?;
```

## CLI Repair

```bash
# Auto-repair with defaults
mesh-cli repair input.stl -o output.stl

# Specific operations
mesh-cli repair input.stl -o output.stl --weld --fill-holes

# With custom tolerance
mesh-cli repair input.stl -o output.stl --weld-tolerance 0.001
```

## Repair Order Matters

Recommended order for best results:

1. **Weld vertices** - Creates proper connectivity
2. **Remove degenerates** - Cleans up zero-area faces
3. **Fill holes** - Needs valid connectivity first
4. **Fix winding** - Works better on closed mesh
5. **Remove small components** - Final cleanup

## What Repair Cannot Fix

Some issues require manual intervention:
- Self-intersecting geometry
- Heavily corrupted topology
- Missing large portions of mesh
- Inverted normals on entire mesh (use explicit flip)

For self-intersections, see [Boolean Operations](./boolean.md) for mesh splitting.

## Next Steps

- [Decimation](./decimation.md) - Reduce triangle count
- [Remeshing](./remeshing.md) - Improve triangle quality
- [Pipeline API](./pipeline.md) - Chain operations
