# Mesh Validation

Validation detects issues that could cause problems in downstream processing or manufacturing.

## Quick Validation

```rust
use mesh_repair::{load_mesh, validate_mesh};

let mesh = load_mesh("model.stl")?;
let report = validate_mesh(&mesh);

println!("Valid: {}", report.is_valid());
println!("Printable: {}", report.is_printable());
```

## Validation Report

The `MeshReport` struct contains comprehensive mesh analysis:

```rust
use mesh_repair::{load_mesh, validate_mesh};

let mesh = load_mesh("model.stl")?;
let report = validate_mesh(&mesh);

// Basic counts
println!("Vertices: {}", report.vertex_count);
println!("Faces: {}", report.face_count);
println!("Components: {}", report.component_count);

// Topology
println!("Manifold: {}", report.is_manifold);
println!("Watertight: {}", report.is_watertight);
println!("Boundary edges: {}", report.boundary_edge_count);
println!("Non-manifold edges: {}", report.non_manifold_edge_count);

// Quality
println!("Degenerate faces: {}", report.degenerate_face_count);
```

## What Gets Checked

### Manifold Topology

A manifold mesh has well-defined inside/outside:

| Check | Description | Impact |
|-------|-------------|--------|
| Edge valence | Each edge shared by 1-2 faces | Non-manifold edges cause slicing errors |
| Vertex valence | Vertices form proper fan | Non-manifold vertices cause rendering issues |

```rust
if !report.is_manifold {
    println!("Non-manifold edges: {}", report.non_manifold_edge_count);
    // May need manual repair or mesh simplification
}
```

### Watertightness

A watertight mesh is completely closed:

```rust
if !report.is_watertight {
    println!("Boundary edges: {}", report.boundary_edge_count);
    // Use fill_holes() to close gaps
}
```

### Connected Components

Separate pieces in the mesh:

```rust
if report.component_count > 1 {
    println!("Mesh has {} separate pieces", report.component_count);
    // May want to keep_largest_component() or process separately
}
```

### Degenerate Faces

Zero-area or near-zero-area triangles:

```rust
if report.degenerate_face_count > 0 {
    println!("Degenerate faces: {}", report.degenerate_face_count);
    // Use remove_degenerate_triangles() to clean
}
```

## Printability Check

The `is_printable()` method checks 3D printing readiness:

```rust
if report.is_printable() {
    println!("Ready for 3D printing!");
} else {
    println!("Needs repair before printing:");
    if !report.is_manifold {
        println!("  - Fix non-manifold geometry");
    }
    if !report.is_watertight {
        println!("  - Fill holes");
    }
    if report.degenerate_face_count > 0 {
        println!("  - Remove degenerate triangles");
    }
}
```

## Component Analysis

Analyze individual connected components:

```rust
use mesh_repair::{load_mesh, find_connected_components};

let mesh = load_mesh("model.stl")?;
let analysis = find_connected_components(&mesh);

for (i, component) in analysis.components.iter().enumerate() {
    println!("Component {}: {} faces", i, component.face_count);
}
```

## Geometry Metrics

Beyond topology, check geometric properties:

```rust
let mesh = load_mesh("model.stl")?;

// Bounding box
if let Some((min, max)) = mesh.bounds() {
    let size_x = max.x - min.x;
    let size_y = max.y - min.y;
    let size_z = max.z - min.z;
    println!("Size: {:.2} x {:.2} x {:.2}", size_x, size_y, size_z);
}

// Surface area
println!("Surface area: {:.2} mm²", mesh.surface_area());

// Volume (only meaningful for watertight meshes)
let volume = mesh.volume();
println!("Volume: {:.2} mm³", volume.abs());
```

## Validation Workflow

Recommended validation workflow:

```rust
use mesh_repair::{load_mesh, validate_mesh, Pipeline};

fn process_mesh(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load
    let mesh = load_mesh(path)?;

    // Initial validation
    let report = validate_mesh(&mesh);
    println!("Initial state:");
    println!("  Manifold: {}", report.is_manifold);
    println!("  Watertight: {}", report.is_watertight);

    // Repair if needed
    let result = Pipeline::new(mesh)
        .validate()
        .weld_vertices(1e-6)
        .remove_degenerate_triangles(1e-10)
        .fill_holes()
        .finish();

    // Final validation
    let final_report = validate_mesh(&result.mesh);
    println!("\nAfter repair:");
    println!("  Manifold: {}", final_report.is_manifold);
    println!("  Watertight: {}", final_report.is_watertight);
    println!("  Printable: {}", final_report.is_printable());

    Ok(())
}
```

## CLI Validation

```bash
# Basic validation
mesh-cli validate model.stl

# Verbose output
mesh-cli validate model.stl --verbose

# JSON output for scripting
mesh-cli validate model.stl --json
```

## Performance

Validation is fast:
- O(V + F) complexity
- ~256M elements/second on modern hardware
- Can validate 1M triangle mesh in ~4ms

## Next Steps

- [Repair Operations](./repair.md) - Fix detected issues
- [Decimation](./decimation.md) - Reduce complexity
