# Boolean Operations

Boolean operations combine two meshes using set operations: union, intersection, or difference.

## Operations

| Operation | Result | Use Case |
|-----------|--------|----------|
| **Union** | A ∪ B | Combine parts |
| **Intersection** | A ∩ B | Find overlap |
| **Difference** | A - B | Cut/subtract |

## Basic Usage

```rust
use mesh_repair::{load_mesh, boolean_operation, BooleanOp, BooleanParams};

let mesh_a = load_mesh("part_a.stl")?;
let mesh_b = load_mesh("part_b.stl")?;

let params = BooleanParams::default();

// Union: combine both meshes
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Union, &params)?;

// Intersection: keep only overlapping region
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Intersection, &params)?;

// Difference: subtract B from A
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Difference, &params)?;

println!("Result: {} faces", result.mesh.face_count());
```

## Requirements

Both input meshes must be:
- **Watertight**: No holes or boundary edges
- **Manifold**: Well-defined inside/outside
- **Overlapping**: Some spatial intersection (for non-union operations)

## Parameters

```rust
let params = BooleanParams {
    tolerance: 1e-6,  // Intersection tolerance
    ..Default::default()
};
```

## Boolean Result

```rust
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Union, &params)?;

println!("Result faces: {}", result.mesh.face_count());
```

## Use Cases

### Combining Parts

```rust
// Merge two separate pieces
let combined = boolean_operation(&part1, &part2, BooleanOp::Union, &params)?;
```

### Creating Cutouts

```rust
// Cut a hole using a cylinder
let with_hole = boolean_operation(&solid, &cylinder, BooleanOp::Difference, &params)?;
```

### Finding Overlap

```rust
// Find where two parts intersect
let overlap = boolean_operation(&part1, &part2, BooleanOp::Intersection, &params)?;
```

## Performance Considerations

Boolean operations are computationally expensive:
- O(F1 × F2) worst case
- Benefits from spatial acceleration
- Consider decimating inputs first

```rust
// Decimate before boolean for speed
let mesh_a = decimate_mesh(&original_a, &DecimateParams::with_target_ratio(0.5)).mesh;
let mesh_b = decimate_mesh(&original_b, &DecimateParams::with_target_ratio(0.5)).mesh;

let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Union, &params)?;
```

## Error Handling

```rust
match boolean_operation(&mesh_a, &mesh_b, BooleanOp::Union, &params) {
    Ok(result) => {
        println!("Success: {} faces", result.mesh.face_count());
    }
    Err(e) => {
        eprintln!("Boolean failed: {}", e);
        // Common issues:
        // - Non-watertight input
        // - Non-manifold geometry
        // - No intersection region
    }
}
```

## Tips

1. **Validate inputs**: Ensure both meshes are watertight and manifold
2. **Repair first**: Use repair operations before boolean
3. **Simplify**: Decimate complex meshes for speed
4. **Clean after**: Boolean results may need cleanup

## Example: Create Shell with Cutout

```rust
use mesh_repair::{load_mesh, boolean_operation, BooleanOp, BooleanParams};

let outer = load_mesh("outer_shell.stl")?;
let inner = load_mesh("inner_shell.stl")?;
let cutout = load_mesh("ventilation_hole.stl")?;

let params = BooleanParams::default();

// Create hollow shell
let shell = boolean_operation(&outer, &inner, BooleanOp::Difference, &params)?;

// Add ventilation cutout
let final_shell = boolean_operation(&shell.mesh, &cutout, BooleanOp::Difference, &params)?;
```

## CLI Usage

```bash
# Union
mesh-cli boolean union a.stl b.stl -o result.stl

# Difference
mesh-cli boolean difference a.stl b.stl -o result.stl

# Intersection
mesh-cli boolean intersection a.stl b.stl -o result.stl
```

## Limitations

- Requires watertight, manifold inputs
- May produce small artifacts at intersection edges
- Complex intersections may fail or produce degenerate faces

Consider [Shell Generation](./shell.md) for creating hollow parts from solid meshes.

## Next Steps

- [Shell Generation](./shell.md) - Create hollow shells
- [Pipeline API](./pipeline.md) - Chain operations
