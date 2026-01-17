# Shell Generation

Shell generation creates hollow parts with uniform wall thickness. This is the core operation for custom-fit products.

## Overview

Shell generation takes a solid mesh and creates a hollow version:

```
Solid input:        Shell output:
┌─────────┐         ┌─────────┐
│█████████│         │░░░░░░░░░│
│█████████│   →     │░░┌───┐░░│
│█████████│         │░░│   │░░│
└─────────┘         └──┴───┴──┘
                    (hollow inside)
```

## Basic Usage

```rust
use mesh_repair::load_mesh;
use mesh_shell::{generate_shell, ShellParams};

let mesh = load_mesh("solid_part.stl")?;

let params = ShellParams {
    wall_thickness_mm: 2.0,  // 2mm walls
    ..Default::default()
};

let shell = generate_shell(&mesh, &params)?;

println!("Shell: {} faces", shell.face_count());
```

## Parameters

### Wall Thickness

The most important parameter:

```rust
let params = ShellParams {
    wall_thickness_mm: 2.0,  // 2mm thick walls
    ..Default::default()
};
```

**Typical values**:
- `1.5 mm`: Flexible, lightweight
- `2.0 mm`: Balanced (good default)
- `3.0 mm`: Rigid, durable
- `4.0+ mm`: Very rigid, heavy

### Generation Method

Two methods for creating offset surfaces:

```rust
use mesh_shell::{ShellParams, WallGenerationMethod};

// Normal offset (fast, may have issues with complex geometry)
let params = ShellParams {
    wall_thickness_mm: 2.0,
    wall_generation_method: WallGenerationMethod::Normal,
    ..Default::default()
};

// SDF-based offset (robust, handles complex geometry)
let params = ShellParams {
    wall_thickness_mm: 2.0,
    wall_generation_method: WallGenerationMethod::Sdf,
    sdf_voxel_size_mm: 0.5,  // Controls resolution
    ..Default::default()
};
```

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| Normal | Fast | May self-intersect | Simple shapes |
| SDF | Slower | Robust | Complex geometry |

### SDF Resolution

For SDF method, control voxel size:

```rust
let params = ShellParams {
    wall_generation_method: WallGenerationMethod::Sdf,
    sdf_voxel_size_mm: 0.5,  // Smaller = higher quality, slower
    ..Default::default()
};
```

**Guidelines**:
- `0.25 mm`: High quality, slow
- `0.5 mm`: Balanced (good default)
- `1.0 mm`: Fast, coarser detail

## Shell Generation Process

1. **Compute offset surface**: Inner surface at wall_thickness distance
2. **Generate walls**: Connect inner and outer surfaces at boundaries
3. **Cap openings**: Close any remaining holes
4. **Validate**: Ensure watertight result

## Pipeline Integration

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};
use mesh_shell::{ShellParams, WallGenerationMethod};

let mesh = load_mesh("foot_scan.stl")?;

// First: repair and prepare the scan
let repaired = Pipeline::new(mesh)
    .weld_vertices(1e-6)
    .fill_holes()
    .remove_small_components(100)
    .decimate_to_ratio(0.5)
    .finish();

// Then: generate shell
let shell_params = ShellParams {
    wall_thickness_mm: 2.0,
    wall_generation_method: WallGenerationMethod::Sdf,
    ..Default::default()
};

let shell = generate_shell(&repaired.mesh, &shell_params)?;

save_mesh(&shell, "foot_shell.stl")?;
```

## Rim Generation

Shells often need rims (flanges) for attachment:

```rust
use mesh_shell::{generate_shell_with_rim, ShellParams, RimParams};

let shell_params = ShellParams {
    wall_thickness_mm: 2.0,
    ..Default::default()
};

let rim_params = RimParams {
    width_mm: 5.0,       // Rim extends 5mm
    thickness_mm: 3.0,   // Rim is 3mm thick
    ..Default::default()
};

let shell = generate_shell_with_rim(&mesh, &shell_params, &rim_params)?;
```

## GPU Acceleration

For faster SDF computation:

```rust
use mesh_gpu::GpuContext;
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

// Create GPU context
let gpu = GpuContext::new()?;

let params = ShellParams {
    wall_generation_method: WallGenerationMethod::Sdf,
    gpu_context: Some(&gpu),
    ..Default::default()
};

let shell = generate_shell(&mesh, &params)?;
```

GPU provides 3-68× speedup for SDF operations.

## Quality Checklist

After shell generation, verify:

```rust
use mesh_repair::validate_mesh;

let shell = generate_shell(&mesh, &params)?;
let report = validate_mesh(&shell);

assert!(report.is_watertight, "Shell should be watertight");
assert!(report.is_manifold, "Shell should be manifold");
println!("Shell is ready for 3D printing!");
```

## CLI Usage

```bash
# Generate shell with 2mm walls
mesh-cli shell solid.stl -o hollow.stl --thickness 2.0

# Use SDF method
mesh-cli shell solid.stl -o hollow.stl --thickness 2.0 --method sdf

# With rim
mesh-cli shell solid.stl -o hollow.stl --thickness 2.0 --rim-width 5.0
```

## Common Issues

### Self-Intersections

Problem: Normal offset creates self-intersecting geometry on complex shapes.

Solution: Use SDF method with appropriate voxel size.

### Thin Features Lost

Problem: Features thinner than wall thickness disappear.

Solution: Reduce wall thickness or use adaptive offset.

### Poor Quality at Boundaries

Problem: Rough edges where inner/outer surfaces meet.

Solution: Increase SDF resolution or post-process with remeshing.

## Use Cases

- **Shoe insoles**: Scan foot → generate shell → 3D print
- **Helmet liners**: Scan head → generate shell → add ventilation
- **Orthotics**: Scan limb → generate shell → customize fit
- **Protective gear**: Scan body part → generate padded shell

## Next Steps

- [Pipeline API](./pipeline.md) - Complete workflows
- [Tutorials](../tutorials/overview.md) - Real-world examples
