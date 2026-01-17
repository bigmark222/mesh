# Basic Repair Example

Step-by-step guide to repairing a mesh with common issues.

## The Problem Mesh

Let's work with a mesh that has typical issues:
- Duplicate vertices (from STL import)
- Small holes
- Inconsistent winding
- Scanner noise (small disconnected pieces)

## Step 1: Load and Analyze

```rust
use mesh_repair::{load_mesh, validate_mesh};

let mesh = load_mesh("problem_mesh.stl")?;

println!("Initial state:");
println!("  Vertices: {}", mesh.vertex_count());
println!("  Faces: {}", mesh.face_count());

let report = validate_mesh(&mesh);
println!("  Manifold: {}", report.is_manifold);
println!("  Watertight: {}", report.is_watertight);
println!("  Boundary edges: {}", report.boundary_edge_count);
println!("  Components: {}", report.component_count);
println!("  Degenerate faces: {}", report.degenerate_face_count);
```

Example output:
```
Initial state:
  Vertices: 15432
  Faces: 5144
  Manifold: true
  Watertight: false
  Boundary edges: 156
  Components: 12
  Degenerate faces: 3
```

## Step 2: Weld Vertices

STL files store each triangle independently, creating duplicate vertices:

```rust
use mesh_repair::weld_vertices;

let original_verts = mesh.vertex_count();
weld_vertices(&mut mesh, 1e-6);  // 1 micron tolerance

println!("After welding:");
println!("  Vertices: {} (was {})", mesh.vertex_count(), original_verts);
```

Output:
```
After welding:
  Vertices: 2573 (was 15432)
```

## Step 3: Remove Degenerate Faces

Zero-area triangles cause problems:

```rust
use mesh_repair::remove_degenerate_triangles;

remove_degenerate_triangles(&mut mesh, 1e-10);

let report = validate_mesh(&mesh);
println!("After removing degenerates:");
println!("  Degenerate faces: {}", report.degenerate_face_count);
```

## Step 4: Remove Small Components

Scanner noise creates tiny disconnected pieces:

```rust
use mesh_repair::{remove_small_components, validate_mesh};

let removed = remove_small_components(&mut mesh, 100);

println!("Removed {} small components", removed);

let report = validate_mesh(&mesh);
println!("  Remaining components: {}", report.component_count);
```

Output:
```
Removed 11 small components
  Remaining components: 1
```

## Step 5: Fill Holes

Close any gaps in the surface:

```rust
use mesh_repair::fill_holes;

fill_holes(&mut mesh)?;

let report = validate_mesh(&mesh);
println!("After filling holes:");
println!("  Boundary edges: {}", report.boundary_edge_count);
println!("  Watertight: {}", report.is_watertight);
```

Output:
```
After filling holes:
  Boundary edges: 0
  Watertight: true
```

## Step 6: Fix Winding Order

Ensure consistent face orientation:

```rust
use mesh_repair::fix_winding_order;

fix_winding_order(&mut mesh)?;
```

## Step 7: Final Validation

```rust
let report = validate_mesh(&mesh);

println!("\nFinal state:");
println!("  Vertices: {}", report.vertex_count);
println!("  Faces: {}", report.face_count);
println!("  Manifold: {}", report.is_manifold);
println!("  Watertight: {}", report.is_watertight);
println!("  Printable: {}", report.is_printable());
```

Output:
```
Final state:
  Vertices: 2561
  Faces: 5118
  Manifold: true
  Watertight: true
  Printable: true
```

## Complete Code

```rust
use mesh_repair::{
    load_mesh, save_mesh, validate_mesh,
    weld_vertices, remove_degenerate_triangles,
    remove_small_components, fill_holes, fix_winding_order,
};

fn repair_mesh(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load
    let mut mesh = load_mesh(input)?;
    println!("Loaded: {} vertices, {} faces", mesh.vertex_count(), mesh.face_count());

    // Repair sequence
    weld_vertices(&mut mesh, 1e-6);
    remove_degenerate_triangles(&mut mesh, 1e-10);
    remove_small_components(&mut mesh, 100);
    fill_holes(&mut mesh)?;
    fix_winding_order(&mut mesh)?;

    // Validate
    let report = validate_mesh(&mesh);
    if report.is_printable() {
        println!("Repair successful!");
        save_mesh(&mesh, output)?;
        println!("Saved to: {}", output);
    } else {
        println!("Warning: Mesh still has issues");
    }

    Ok(())
}

fn main() {
    if let Err(e) = repair_mesh("broken.stl", "fixed.stl") {
        eprintln!("Error: {}", e);
    }
}
```

## Using the Pipeline

The same repairs using the Pipeline API:

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};

fn repair_with_pipeline(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh(input)?;

    let result = Pipeline::new(mesh)
        .weld_vertices(1e-6)
        .remove_degenerate_triangles(1e-10)
        .remove_small_components(100)
        .fill_holes()
        .fix_winding()
        .validate()
        .finish();

    let report = validate_mesh(&result.mesh);
    if report.is_printable() {
        save_mesh(&result.mesh, output)?;
    }

    Ok(())
}
```
