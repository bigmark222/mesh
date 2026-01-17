# Decimation

Decimation reduces triangle count while preserving shape. Essential for optimizing scan data.

## Basic Usage

```rust
use mesh_repair::{load_mesh, decimate_mesh, DecimateParams};

let mesh = load_mesh("high_poly.stl")?;
println!("Original: {} faces", mesh.face_count());

// Reduce to target triangle count
let params = DecimateParams::with_target_triangles(10_000);
let result = decimate_mesh(&mesh, &params);

println!("Decimated: {} faces", result.mesh.face_count());
```

## Target Methods

### Target Triangle Count

Specify exact number of triangles:

```rust
let params = DecimateParams::with_target_triangles(5_000);
let result = decimate_mesh(&mesh, &params);
```

### Target Ratio

Specify fraction to keep:

```rust
// Keep 50% of triangles
let params = DecimateParams::with_target_ratio(0.5);
let result = decimate_mesh(&mesh, &params);

// Keep 10% of triangles (90% reduction)
let params = DecimateParams::with_target_ratio(0.1);
let result = decimate_mesh(&mesh, &params);
```

## Algorithm

mesh-repair uses **Quadric Error Metrics (QEM)**:

1. Computes error quadric for each vertex
2. Evaluates cost of collapsing each edge
3. Collapses lowest-cost edges first
4. Repeats until target reached

**Advantages**:
- Preserves sharp features
- Minimizes geometric error
- O(F log F) complexity

## Decimation Result

```rust
let result = decimate_mesh(&mesh, &params);

println!("Final faces: {}", result.mesh.face_count());
println!("Edges collapsed: {}", result.edges_collapsed);
```

## Pipeline Integration

```rust
use mesh_repair::{load_mesh, Pipeline};

let mesh = load_mesh("scan.stl")?;

let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .decimate_to_ratio(0.5)  // Reduce by half
    .finish();
```

## Preserving Features

### Boundary Preservation

Edges on mesh boundaries are preserved by default:

```rust
let params = DecimateParams {
    preserve_boundaries: true,  // Default
    ..DecimateParams::with_target_ratio(0.5)
};
```

### Quality Threshold

Prevent creating poor-quality triangles:

```rust
let params = DecimateParams {
    min_quality: 0.1,  // Minimum triangle quality (0-1)
    ..DecimateParams::with_target_ratio(0.5)
};
```

## Performance

| Mesh Size | 50% Reduction | Throughput |
|-----------|---------------|------------|
| 320 tri | ~580 µs | 550K tri/s |
| 1,280 tri | ~7.3 ms | 175K tri/s |
| 5,120 tri | ~105 ms | 48K tri/s |

Decimation scales well for large meshes.

## CLI Usage

```bash
# Reduce to 50% triangles
mesh-cli decimate input.stl -o output.stl --ratio 0.5

# Reduce to specific count
mesh-cli decimate input.stl -o output.stl --target 10000
```

## When to Decimate

- **After scanning**: Raw scans are often over-tessellated
- **Before boolean operations**: Fewer faces = faster operations
- **For real-time display**: Reduce for preview/LOD
- **Before export**: Smaller files, faster slicing

## Quality vs Size Trade-off

```rust
// Aggressive decimation (may lose detail)
let params = DecimateParams::with_target_ratio(0.1);

// Conservative decimation (preserves more detail)
let params = DecimateParams::with_target_ratio(0.8);
```

Test on your specific models to find the right balance.

## Next Steps

- [Remeshing](./remeshing.md) - Improve triangle quality
- [Pipeline API](./pipeline.md) - Chain with other operations
