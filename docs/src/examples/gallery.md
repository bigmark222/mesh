# Example Gallery

Curated examples demonstrating common mesh processing workflows.

## Basic Examples

### Load, Validate, Save

```rust
use mesh_repair::{load_mesh, save_mesh, validate_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load any supported format
    let mesh = load_mesh("input.stl")?;

    // Validate
    let report = validate_mesh(&mesh);
    println!("Vertices: {}", report.vertex_count);
    println!("Faces: {}", report.face_count);
    println!("Manifold: {}", report.is_manifold);
    println!("Watertight: {}", report.is_watertight);

    // Save (format from extension)
    save_mesh(&mesh, "output.obj")?;

    Ok(())
}
```

### Quick Repair

```rust
use mesh_repair::{load_mesh, save_mesh, weld_vertices, fill_holes, fix_winding_order};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh = load_mesh("broken.stl")?;

    // Basic repair sequence
    weld_vertices(&mut mesh, 1e-6);
    fill_holes(&mut mesh)?;
    fix_winding_order(&mut mesh)?;

    save_mesh(&mesh, "fixed.stl")?;
    Ok(())
}
```

### Simplify High-Poly Mesh

```rust
use mesh_repair::{load_mesh, save_mesh, decimate_mesh, DecimateParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("high_poly.stl")?;
    println!("Original: {} faces", mesh.face_count());

    // Reduce to 10,000 triangles
    let params = DecimateParams::with_target_triangles(10_000);
    let result = decimate_mesh(&mesh, &params);

    println!("Simplified: {} faces", result.mesh.face_count());
    save_mesh(&result.mesh, "low_poly.stl")?;

    Ok(())
}
```

## Pipeline Examples

### Scan Cleanup Pipeline

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scan = load_mesh("raw_scan.stl")?;

    let result = Pipeline::new(scan)
        .weld_vertices(1e-6)
        .remove_degenerate_triangles(1e-10)
        .remove_small_components(100)
        .fill_holes()
        .fix_winding()
        .validate()
        .finish();

    let report = validate_mesh(&result.mesh);
    if report.is_printable() {
        println!("Scan is ready for use!");
        save_mesh(&result.mesh, "clean_scan.stl")?;
    }

    Ok(())
}
```

### Optimize for Real-Time Display

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("detailed_model.stl")?;
    println!("Original: {} faces", mesh.face_count());

    // Create LOD versions
    let lod0 = Pipeline::new(mesh.clone())
        .decimate_to_ratio(0.5)
        .finish();

    let lod1 = Pipeline::new(mesh.clone())
        .decimate_to_ratio(0.25)
        .finish();

    let lod2 = Pipeline::new(mesh.clone())
        .decimate_to_ratio(0.1)
        .finish();

    println!("LOD0: {} faces", lod0.mesh.face_count());
    println!("LOD1: {} faces", lod1.mesh.face_count());
    println!("LOD2: {} faces", lod2.mesh.face_count());

    save_mesh(&lod0.mesh, "model_lod0.stl")?;
    save_mesh(&lod1.mesh, "model_lod1.stl")?;
    save_mesh(&lod2.mesh, "model_lod2.stl")?;

    Ok(())
}
```

## Shell Generation Examples

### Basic Shell

```rust
use mesh_repair::load_mesh;
use mesh_shell::{generate_shell, ShellParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let solid = load_mesh("solid_part.stl")?;

    let params = ShellParams {
        wall_thickness_mm: 2.0,
        ..Default::default()
    };

    let shell = generate_shell(&solid, &params)?;
    println!("Shell: {} faces", shell.face_count());

    Ok(())
}
```

### Shell with SDF Method

```rust
use mesh_repair::load_mesh;
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let solid = load_mesh("complex_shape.stl")?;

    let params = ShellParams {
        wall_thickness_mm: 2.0,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.5,
        ..Default::default()
    };

    let shell = generate_shell(&solid, &params)?;
    println!("Shell: {} faces", shell.face_count());

    Ok(())
}
```

## Boolean Operations

### Combine Two Parts

```rust
use mesh_repair::{load_mesh, save_mesh, boolean_operation, BooleanOp, BooleanParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let part_a = load_mesh("part_a.stl")?;
    let part_b = load_mesh("part_b.stl")?;

    let params = BooleanParams::default();
    let combined = boolean_operation(&part_a, &part_b, BooleanOp::Union, &params)?;

    save_mesh(&combined.mesh, "combined.stl")?;
    Ok(())
}
```

### Create Cutout

```rust
use mesh_repair::{load_mesh, save_mesh, boolean_operation, BooleanOp, BooleanParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let solid = load_mesh("solid.stl")?;
    let cutter = load_mesh("hole_shape.stl")?;

    let params = BooleanParams::default();
    let with_hole = boolean_operation(&solid, &cutter, BooleanOp::Difference, &params)?;

    save_mesh(&with_hole.mesh, "with_cutout.stl")?;
    Ok(())
}
```

## Analysis Examples

### Mesh Statistics

```rust
use mesh_repair::{load_mesh, validate_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("model.stl")?;

    println!("=== Mesh Statistics ===");
    println!("Vertices: {}", mesh.vertex_count());
    println!("Faces: {}", mesh.face_count());

    if let Some((min, max)) = mesh.bounds() {
        println!("Bounds:");
        println!("  Min: ({:.2}, {:.2}, {:.2})", min.x, min.y, min.z);
        println!("  Max: ({:.2}, {:.2}, {:.2})", max.x, max.y, max.z);
        println!("  Size: {:.2} x {:.2} x {:.2}",
            max.x - min.x, max.y - min.y, max.z - min.z);
    }

    println!("Surface Area: {:.2} mm²", mesh.surface_area());
    println!("Volume: {:.2} mm³", mesh.volume().abs());

    let report = validate_mesh(&mesh);
    println!("\n=== Topology ===");
    println!("Manifold: {}", report.is_manifold);
    println!("Watertight: {}", report.is_watertight);
    println!("Components: {}", report.component_count);
    println!("Boundary Edges: {}", report.boundary_edge_count);
    println!("Printable: {}", report.is_printable());

    Ok(())
}
```

### Component Analysis

```rust
use mesh_repair::{load_mesh, find_connected_components};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh("multi_part.stl")?;

    let analysis = find_connected_components(&mesh);

    println!("Found {} components:", analysis.components.len());
    for (i, comp) in analysis.components.iter().enumerate() {
        println!("  Component {}: {} faces", i, comp.face_count);
    }

    Ok(())
}
```

## More Examples

- [Basic Repair](./basic-repair.md) - Step-by-step repair guide
- [Scan to Shell](./scan-to-shell.md) - Complete custom-fit workflow
- [Batch Processing](./batch.md) - Process multiple files
