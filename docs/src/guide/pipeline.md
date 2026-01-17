# Pipeline API

The Pipeline API provides a fluent interface for chaining mesh operations.

## Basic Usage

```rust
use mesh_repair::{load_mesh, Pipeline};

let mesh = load_mesh("scan.stl")?;

let result = Pipeline::new(mesh)
    .validate()
    .weld_vertices(1e-6)
    .fill_holes()
    .decimate_to_ratio(0.5)
    .finish();

println!("Processed mesh: {} faces", result.mesh.face_count());
println!("Stages executed: {}", result.stages_executed);
```

## Available Operations

### Validation

```rust
Pipeline::new(mesh)
    .validate()  // Runs validation, stores report
    .finish();
```

### Repair Operations

```rust
Pipeline::new(mesh)
    .weld_vertices(1e-6)             // Merge duplicate vertices
    .remove_degenerate_triangles(1e-10)  // Remove zero-area faces
    .fill_holes()                    // Close gaps
    .fix_winding()                   // Consistent normals
    .remove_small_components(100)    // Remove noise (min 100 faces)
    .keep_largest_component()        // Keep only biggest piece
    .finish();
```

### Simplification

```rust
Pipeline::new(mesh)
    .decimate_to_ratio(0.5)          // Keep 50% of triangles
    .decimate_to_target(10_000)      // Target specific count
    .finish();
```

### Quality Improvement

```rust
Pipeline::new(mesh)
    .remesh(0.5)                     // Isotropic remesh, 0.5mm edges
    .subdivide(1)                    // Loop subdivision, 1 iteration
    .finish();
```

### Transform

```rust
Pipeline::new(mesh)
    .scale(2.0)                      // Double size
    .translate(10.0, 0.0, 0.0)       // Move 10 units in X
    .finish();
```

## Pipeline Result

```rust
let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .decimate_to_ratio(0.5)
    .finish();

// Access final mesh
let final_mesh = result.mesh;

// Check execution stats
println!("Stages executed: {}", result.stages_executed);
```

## Chaining Examples

### Scan Cleanup Pipeline

```rust
let result = Pipeline::new(scan_mesh)
    .weld_vertices(1e-6)           // 1. Fix connectivity
    .remove_degenerate_triangles(1e-10)  // 2. Remove bad faces
    .remove_small_components(50)   // 3. Remove scan noise
    .fill_holes()                  // 4. Close gaps
    .fix_winding()                 // 5. Consistent normals
    .validate()                    // 6. Check result
    .finish();
```

### Optimization Pipeline

```rust
let result = Pipeline::new(mesh)
    .decimate_to_ratio(0.3)        // Reduce to 30%
    .remesh(0.5)                   // Uniform 0.5mm triangles
    .finish();
```

### Smooth and Simplify

```rust
let result = Pipeline::new(mesh)
    .subdivide(1)                  // Smooth (4× faces)
    .decimate_to_target(10_000)    // Back down to 10k
    .finish();
```

## Custom Parameters

For fine control, use parameter structs:

```rust
use mesh_repair::{Pipeline, DecimateParams, RemeshParams};

let decimate_params = DecimateParams {
    target_ratio: 0.5,
    preserve_boundaries: true,
    ..Default::default()
};

let remesh_params = RemeshParams {
    target_edge_length: Some(0.5),
    iterations: 5,
    preserve_features: true,
    ..Default::default()
};

let result = Pipeline::new(mesh)
    .decimate_with_params(&decimate_params)
    .remesh_with_params(&remesh_params)
    .finish();
```

## Error Handling

Pipeline operations that can fail return `Result`:

```rust
use mesh_repair::Pipeline;

let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .fill_holes()       // May fail
    .fix_winding()      // May fail
    .finish();

// Or handle errors explicitly:
match Pipeline::new(mesh).fill_holes().try_finish() {
    Ok(result) => println!("Success: {} faces", result.mesh.face_count()),
    Err(e) => eprintln!("Pipeline failed: {}", e),
}
```

## Pipeline vs Individual Functions

| Approach | Use When |
|----------|----------|
| Pipeline | Multiple operations, cleaner code |
| Individual functions | Single operation, need intermediate results |

```rust
// Pipeline approach
let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .fill_holes()
    .finish();

// Individual approach (more control)
let mut mesh = load_mesh("model.stl")?;
weld_vertices(&mut mesh, 1e-6);
println!("After weld: {} vertices", mesh.vertex_count());
fill_holes(&mut mesh)?;
println!("After fill: {} faces", mesh.face_count());
```

## Best Practices

1. **Order matters**: Weld before fill, fill before fix winding
2. **Validate last**: Check final result
3. **Keep intermediate meshes**: For debugging, break into steps
4. **Don't over-process**: Each step adds computation time

## Typical Pipeline Order

```rust
// Recommended order for scan processing
let result = Pipeline::new(scan)
    // 1. Fix connectivity
    .weld_vertices(1e-6)

    // 2. Clean up
    .remove_degenerate_triangles(1e-10)
    .remove_small_components(50)

    // 3. Close holes
    .fill_holes()

    // 4. Fix orientation
    .fix_winding()

    // 5. Optimize (optional)
    .decimate_to_ratio(0.5)
    .remesh(0.5)

    // 6. Validate
    .validate()

    .finish();
```

## Next Steps

- [Tutorials](../tutorials/overview.md) - Complete workflows
- [Examples](../examples/gallery.md) - Real-world code
