# Subdivision

Subdivision smooths meshes by adding triangles and adjusting vertices.

## Basic Usage

```rust
use mesh_repair::{load_mesh, subdivide_mesh, SubdivideParams};

let mesh = load_mesh("model.stl")?;
println!("Original: {} faces", mesh.face_count());

let params = SubdivideParams::with_iterations(2);
let result = subdivide_mesh(&mesh, &params);

println!("Subdivided: {} faces", result.mesh.face_count());
```

## How It Works

mesh-repair implements **Loop subdivision**:

1. Each triangle becomes 4 triangles
2. New vertices placed at edge midpoints
3. All vertices moved toward neighbors
4. Result is smoother surface

```
Original:           After 1 subdivision:
    /\                    /\
   /  \                  /--\
  /    \                /\  /\
 /______\              /__\/__\
```

**Face count growth**: F × 4^n (where n = iterations)
- 1 iteration: 4× faces
- 2 iterations: 16× faces
- 3 iterations: 64× faces

## Parameters

### Iterations

```rust
// Single subdivision (4× faces)
let params = SubdivideParams::with_iterations(1);

// Double subdivision (16× faces)
let params = SubdivideParams::with_iterations(2);
```

**Warning**: Face count grows exponentially. Use sparingly.

## Subdivision Result

```rust
let result = subdivide_mesh(&mesh, &params);

println!("Final faces: {}", result.mesh.face_count());
```

## Pipeline Integration

```rust
use mesh_repair::{load_mesh, Pipeline};

let mesh = load_mesh("model.stl")?;

let result = Pipeline::new(mesh)
    .subdivide(1)           // Smooth
    .decimate_to_ratio(0.5) // Reduce back down
    .finish();
```

## When to Use

- **Smoothing low-poly models**: Add surface detail
- **Before shell generation**: Smoother inner surface
- **For visualization**: Better rendering quality

## Volume Changes

Loop subdivision **changes volume**:
- Convex shapes shrink (corners smoothed inward)
- Concave shapes grow (corners smoothed outward)

For volume preservation, consider remeshing instead.

## Subdivision vs Remeshing

| Operation | Result | Volume | Use Case |
|-----------|--------|--------|----------|
| Subdivision | Smooth surface, more faces | Changes | Smoothing |
| Remeshing | Uniform triangles, configurable density | Preserved | Quality improvement |

## Tips

1. **Start with clean mesh**: Fix topology issues first
2. **Use minimal iterations**: 1-2 is usually enough
3. **Decimate after**: Control final triangle count
4. **Watch memory**: High iterations consume lots of memory

## Example: Smooth Then Reduce

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};

let mesh = load_mesh("blocky.stl")?;

let result = Pipeline::new(mesh)
    .subdivide(2)              // Smooth (16× faces)
    .decimate_to_ratio(0.1)    // Reduce to 10%
    .finish();

// Result: smooth surface, reasonable face count
save_mesh(&result.mesh, "smooth.stl")?;
```

## CLI Usage

```bash
# Subdivide once
mesh-cli subdivide input.stl -o output.stl

# Subdivide twice
mesh-cli subdivide input.stl -o output.stl --iterations 2
```

## Next Steps

- [Boolean Operations](./boolean.md) - Combine meshes
- [Shell Generation](./shell.md) - Create hollow parts
