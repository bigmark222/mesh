# Remeshing

Remeshing creates a new mesh with uniform, well-shaped triangles.

## Why Remesh?

- **Uniform triangles**: Better for simulation and analysis
- **Controlled density**: Specify target edge length
- **Quality improvement**: Replace poor triangles
- **Noise reduction**: Smooths scan artifacts

## Basic Usage

```rust
use mesh_repair::{load_mesh, remesh_isotropic, RemeshParams};

let mesh = load_mesh("scan.stl")?;

let params = RemeshParams {
    target_edge_length: Some(0.5),  // 0.5mm edges
    iterations: 5,
    ..Default::default()
};

let result = remesh_isotropic(&mesh, &params);
println!("Remeshed: {} faces", result.mesh.face_count());
```

## Parameters

### Target Edge Length

Controls triangle size:

```rust
let params = RemeshParams {
    target_edge_length: Some(0.5),  // Smaller = more triangles
    ..Default::default()
};
```

**Guidelines**:
- `0.1 mm`: Very fine, for detailed features
- `0.5 mm`: Good for most custom-fit products
- `1.0 mm`: Coarse, for large parts

### Iterations

More iterations = better quality, longer time:

```rust
let params = RemeshParams {
    target_edge_length: Some(0.5),
    iterations: 5,  // Default: 5
    ..Default::default()
};
```

### Feature Preservation

Preserve sharp edges and corners:

```rust
let params = RemeshParams {
    target_edge_length: Some(0.5),
    preserve_features: true,
    feature_angle: 30.0,  // Degrees
    ..Default::default()
};
```

## Algorithm

Isotropic remeshing uses iterative operations:

1. **Split long edges**: Edges > 4/3 × target
2. **Collapse short edges**: Edges < 4/5 × target
3. **Flip edges**: Improve vertex valence
4. **Smooth vertices**: Tangential relaxation

Each iteration improves triangle quality.

## Remeshing Result

```rust
let result = remesh_isotropic(&mesh, &params);

println!("Final mesh: {} faces", result.mesh.face_count());
println!("Iterations: {}", result.iterations_performed);
```

## Pipeline Integration

```rust
use mesh_repair::{load_mesh, Pipeline};

let mesh = load_mesh("scan.stl")?;

let result = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .fill_holes()
    .remesh(0.5)  // 0.5mm target edge length
    .finish();
```

## Performance

| Mesh Size | 5 Iterations | Throughput |
|-----------|--------------|------------|
| 320 tri | ~38 ms | 8.4K tri/s |
| 1,280 tri | ~48 ms | 26K tri/s |

Remeshing is computationally intensive but scales linearly.

## When to Remesh

- **After scan import**: Clean up scanner artifacts
- **Before simulation**: Uniform elements for FEA
- **After boolean operations**: Clean up intersection regions
- **For visualization**: Consistent shading

## Remesh vs Decimate

| Operation | Use When |
|-----------|----------|
| Decimate | Need fewer triangles, preserve shape |
| Remesh | Need uniform triangles, quality matters |
| Both | Decimate first, then remesh |

## CLI Usage

```bash
# Remesh with 0.5mm edges
mesh-cli remesh input.stl -o output.stl --edge-length 0.5

# With more iterations
mesh-cli remesh input.stl -o output.stl --edge-length 0.5 --iterations 10
```

## Tips

1. **Weld vertices first**: Remeshing needs proper connectivity
2. **Fill holes first**: Boundary handling is limited
3. **Start with fewer iterations**: Increase if quality insufficient
4. **Match edge length to features**: Don't go smaller than smallest detail

## Next Steps

- [Subdivision](./subdivision.md) - Smooth surface
- [Shell Generation](./shell.md) - Create hollow parts
