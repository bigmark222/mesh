# Tutorial: Custom Shoe Insoles

This tutorial walks through creating custom-fit shoe insoles from 3D foot scans.

## Overview

**Input**: 3D scan of foot (bottom surface)
**Output**: Hollow insole shell ready for 3D printing

```
Foot scan        →      Repaired mesh       →      Insole shell
(noisy, holes)          (clean, watertight)        (hollow, printable)
```

## Step 1: Load the Foot Scan

```rust
use mesh_repair::{load_mesh, validate_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the foot scan
    let scan = load_mesh("foot_scan.stl")?;

    println!("Loaded foot scan:");
    println!("  Vertices: {}", scan.vertex_count());
    println!("  Faces: {}", scan.face_count());

    // Check initial quality
    let report = validate_mesh(&scan);
    println!("  Watertight: {}", report.is_watertight);
    println!("  Components: {}", report.component_count);

    Ok(())
}
```

**Typical scan characteristics**:
- 50,000 - 200,000 triangles
- Multiple components (noise)
- Holes in occluded areas
- Non-manifold edges

## Step 2: Clean Up the Scan

```rust
use mesh_repair::{Pipeline, validate_mesh};

let repaired = Pipeline::new(scan)
    // Merge duplicate vertices (scanner often creates duplicates)
    .weld_vertices(0.1)  // 0.1mm tolerance

    // Remove scanner noise (small disconnected pieces)
    .remove_small_components(500)  // Keep only pieces > 500 faces

    // Close any holes
    .fill_holes()

    // Ensure consistent normals (outward-facing)
    .fix_winding()

    // Simplify to manageable size
    .decimate_to_target(20_000)

    .finish();

println!("After cleanup: {} faces", repaired.mesh.face_count());

// Verify
let report = validate_mesh(&repaired.mesh);
assert!(report.is_watertight, "Must be watertight for shell generation");
```

## Step 3: Prepare Insole Region

For insoles, we typically need just the bottom surface:

```rust
// Get mesh bounds to understand orientation
if let Some((min, max)) = repaired.mesh.bounds() {
    println!("Foot bounds:");
    println!("  Length (Y): {:.1}mm", max.y - min.y);
    println!("  Width (X): {:.1}mm", max.x - min.x);
    println!("  Height (Z): {:.1}mm", max.z - min.z);
}

// For a plantar (bottom) scan, the mesh is typically already
// oriented correctly. If not, you may need to rotate or clip.
```

## Step 4: Generate Insole Shell

```rust
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

let shell_params = ShellParams {
    // Wall thickness: 2mm for comfort + durability
    wall_thickness_mm: 2.0,

    // Use SDF method for robust results
    wall_generation_method: WallGenerationMethod::Sdf,

    // 0.5mm voxels for good detail
    sdf_voxel_size_mm: 0.5,

    ..Default::default()
};

let insole = generate_shell(&repaired.mesh, &shell_params)?;

println!("Generated insole: {} faces", insole.face_count());
```

## Step 5: Add Support Features

Insoles often need additional features:

```rust
// Example: Add heel cup rim
use mesh_shell::{RimParams, generate_shell_with_rim};

let rim_params = RimParams {
    width_mm: 8.0,        // 8mm rim width
    thickness_mm: 3.0,    // 3mm rim thickness
    // Add rim only to heel area (you'd implement region detection)
    ..Default::default()
};

let insole_with_rim = generate_shell_with_rim(
    &repaired.mesh,
    &shell_params,
    &rim_params
)?;
```

## Step 6: Validate and Export

```rust
use mesh_repair::{save_mesh, validate_mesh};

// Final validation
let report = validate_mesh(&insole);
println!("Final insole:");
println!("  Faces: {}", report.face_count);
println!("  Watertight: {}", report.is_watertight);
println!("  Manifold: {}", report.is_manifold);

if report.is_printable() {
    println!("Ready for 3D printing!");

    // Export as 3MF (includes units, preferred by modern slicers)
    save_mesh(&insole, "custom_insole.3mf")?;

    // Or STL for maximum compatibility
    save_mesh(&insole, "custom_insole.stl")?;
} else {
    eprintln!("Insole needs manual repair before printing");
}
```

## Complete Example

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

fn create_insole(scan_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating custom insole from: {}", scan_path);

    // Load
    let scan = load_mesh(scan_path)?;
    println!("  Loaded: {} faces", scan.face_count());

    // Repair
    let repaired = Pipeline::new(scan)
        .weld_vertices(0.1)
        .remove_small_components(500)
        .fill_holes()
        .fix_winding()
        .decimate_to_target(20_000)
        .finish();

    let report = validate_mesh(&repaired.mesh);
    if !report.is_watertight {
        return Err("Scan could not be made watertight".into());
    }
    println!("  Repaired: {} faces", repaired.mesh.face_count());

    // Generate shell
    let shell_params = ShellParams {
        wall_thickness_mm: 2.0,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.5,
        ..Default::default()
    };

    let insole = generate_shell(&repaired.mesh, &shell_params)?;
    println!("  Shell: {} faces", insole.face_count());

    // Export
    save_mesh(&insole, output_path)?;
    println!("  Saved to: {}", output_path);

    Ok(())
}

fn main() {
    if let Err(e) = create_insole("foot_scan.stl", "custom_insole.3mf") {
        eprintln!("Error: {}", e);
    }
}
```

## Material Recommendations

| Material | Properties | Use Case |
|----------|------------|----------|
| TPU 95A | Flexible, durable | Everyday comfort |
| TPU 85A | Softer, more cushion | Running, high impact |
| Nylon | Rigid, supportive | Orthotics, arch support |
| EVA Foam | Lightweight, soft | Athletic insoles |

## Common Issues

### Hole in Arch Area

**Problem**: Scanner missed the arch.
**Solution**: Increase `fill_holes()` tolerance or use manual patching.

### Too Many Triangles

**Problem**: Slow shell generation, huge files.
**Solution**: Decimate more aggressively (target 10-20k faces).

### Shell Self-Intersects

**Problem**: Complex geometry causes normal offset issues.
**Solution**: Use SDF method with smaller voxel size.

## Next Steps

- [Helmet Liners](./helmets.md) - Multi-component shells
- [Protective Equipment](./protective.md) - Impact-resistant designs
