# Quick Start

This guide walks you through processing your first mesh with mesh-repair.

## Loading a Mesh

```rust
use mesh_repair::load_mesh;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load from any supported format
    let mesh = load_mesh("model.stl")?;

    println!("Loaded mesh:");
    println!("  Vertices: {}", mesh.vertex_count());
    println!("  Faces: {}", mesh.face_count());

    Ok(())
}
```

Supported formats:
- **STL** (`.stl`) - Binary and ASCII
- **OBJ** (`.obj`) - Wavefront OBJ
- **PLY** (`.ply`) - Polygon File Format
- **3MF** (`.3mf`) - 3D Manufacturing Format

## Validating a Mesh

```rust
use mesh_repair::{load_mesh, validate_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("model.stl")?;
    let report = validate_mesh(&mesh);

    println!("Validation Report:");
    println!("  Manifold: {}", report.is_manifold);
    println!("  Watertight: {}", report.is_watertight);
    println!("  Boundary edges: {}", report.boundary_edge_count);
    println!("  Components: {}", report.component_count);

    if report.is_printable() {
        println!("Mesh is ready for 3D printing!");
    }

    Ok(())
}
```

## Basic Repair

```rust
use mesh_repair::{load_mesh, save_mesh, weld_vertices, fill_holes, fix_winding_order};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh = load_mesh("broken_model.stl")?;

    // Weld duplicate vertices (tolerance in model units)
    weld_vertices(&mut mesh, 1e-6);

    // Fill any holes
    fill_holes(&mut mesh)?;

    // Fix inconsistent face orientation
    fix_winding_order(&mut mesh)?;

    // Save the repaired mesh
    save_mesh(&mesh, "repaired_model.stl")?;

    println!("Mesh repaired and saved!");
    Ok(())
}
```

## Using the Pipeline API

The Pipeline API provides a fluent interface for chaining operations:

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("scan.stl")?;

    let result = Pipeline::new(mesh)
        .validate()                      // Check initial state
        .weld_vertices(1e-6)             // Merge duplicate vertices
        .fill_holes()                    // Close any gaps
        .remove_small_components(100)    // Remove noise
        .decimate_to_ratio(0.5)          // Reduce to 50% triangles
        .finish();

    println!("Pipeline completed in {} stages", result.stages_executed);
    println!("Final mesh: {} faces", result.mesh.face_count());

    save_mesh(&result.mesh, "processed.stl")?;
    Ok(())
}
```

## Decimation (Simplification)

Reduce triangle count while preserving shape:

```rust
use mesh_repair::{load_mesh, decimate_mesh, DecimateParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("high_poly.stl")?;
    println!("Original: {} faces", mesh.face_count());

    // Reduce to 10,000 triangles
    let params = DecimateParams::with_target_triangles(10_000);
    let result = decimate_mesh(&mesh, &params);

    println!("Decimated: {} faces", result.mesh.face_count());
    Ok(())
}
```

## Shell Generation

Create hollow shells for manufacturing:

```rust
use mesh_repair::load_mesh;
use mesh_shell::{generate_shell, ShellParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("solid_part.stl")?;

    let params = ShellParams {
        wall_thickness_mm: 2.0,
        ..Default::default()
    };

    let shell = generate_shell(&mesh, &params)?;

    println!("Generated shell with {} faces", shell.face_count());
    Ok(())
}
```

## Command-Line Usage

Process meshes without writing code:

```bash
# Validate a mesh
mesh-cli validate model.stl

# Repair a mesh
mesh-cli repair input.stl -o output.stl

# Decimate to 50% triangles
mesh-cli decimate input.stl -o output.stl --ratio 0.5

# Convert between formats
mesh-cli convert model.stl model.obj

# Generate a shell
mesh-cli shell solid.stl -o hollow.stl --thickness 2.0
```

## Next Steps

- [Basic Concepts](./concepts.md) - Understand mesh fundamentals
- [User Guide](../guide/io.md) - Detailed operation guides
- [Examples](../examples/gallery.md) - Real-world examples
