# Tutorial: Protective Equipment

This tutorial covers creating custom protective equipment like shin guards, elbow pads, and body armor.

## Overview

Protective equipment requires:
- Body-conforming fit
- Impact-resistant structure
- Flexibility for movement
- Segmented design for articulation

## Segmented Shell Design

Unlike single-piece shells, protective equipment often needs segments:

```rust
use mesh_repair::{load_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams};

// Load body part scan (e.g., shin)
let scan = load_mesh("shin_scan.stl")?;

// Prepare mesh
let prepared = Pipeline::new(scan)
    .weld_vertices(0.1)
    .remove_small_components(200)
    .fill_holes()
    .fix_winding()
    .decimate_to_target(15_000)
    .finish();

// Generate protective shell
let shell_params = ShellParams {
    wall_thickness_mm: 3.0,  // Thicker for impact
    ..Default::default()
};

let shell = generate_shell(&prepared.mesh, &shell_params)?;
```

## Creating Articulated Guards

For joints (knees, elbows), create segmented designs:

```rust
// Conceptual example of segmented knee guard

struct Segment {
    mesh: Mesh,
    hinge_location: [f64; 3],
}

fn create_knee_guard(knee_scan: &Mesh) -> Vec<Segment> {
    // Split into segments based on knee anatomy
    // - Upper thigh segment
    // - Knee cap segment
    // - Lower shin segment

    // Each segment would be generated separately
    // with hinge points for articulation

    todo!("Implement segment splitting")
}
```

## Lattice Structures for Impact Absorption

For advanced protection, use lattice infill:

```rust
use mesh_repair::generate_lattice;
use mesh_repair::LatticeParams;

let lattice_params = LatticeParams {
    cell_size_mm: 8.0,        // 8mm lattice cells
    strut_diameter_mm: 1.5,   // 1.5mm struts
    pattern: LatticePattern::Gyroid,  // Good for impact
    ..Default::default()
};

let protective_lattice = generate_lattice(&shell_bounds, &lattice_params)?;

// Combine shell with lattice
let guard = boolean_operation(&shell, &protective_lattice, BooleanOp::Union, &params)?;
```

## Ventilation and Weight Reduction

Add ventilation while maintaining protection:

```rust
// Create ventilation pattern
fn create_vent_holes(shell: &Mesh, spacing: f64) -> Vec<Mesh> {
    // Generate array of small holes in non-critical areas
    // Avoid impact zones
    todo!()
}

// Subtract vents from shell
for vent in create_vent_holes(&shell, 15.0) {
    shell = boolean_operation(&shell, &vent, BooleanOp::Difference, &params)?.mesh;
}
```

## Complete Shin Guard Example

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

fn create_shin_guard(
    scan_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load scan
    let scan = load_mesh(scan_path)?;
    println!("Loaded shin scan: {} faces", scan.face_count());

    // Prepare mesh
    let prepared = Pipeline::new(scan)
        .weld_vertices(0.1)
        .remove_small_components(200)
        .fill_holes()
        .fix_winding()
        .decimate_to_target(10_000)
        .finish();

    // Validate
    let report = validate_mesh(&prepared.mesh);
    if !report.is_watertight {
        return Err("Scan could not be made watertight".into());
    }

    // Generate protective shell
    // Use thicker walls for impact protection
    let shell_params = ShellParams {
        wall_thickness_mm: 4.0,  // 4mm for protection
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.5,
        ..Default::default()
    };

    let shell = generate_shell(&prepared.mesh, &shell_params)?;

    // Final validation
    let final_report = validate_mesh(&shell);
    if !final_report.is_printable() {
        return Err("Generated shell is not printable".into());
    }

    // Export
    save_mesh(&shell, output_path)?;
    println!("Created shin guard: {}", output_path);

    Ok(())
}
```

## Material Selection for Protection

| Material | Impact Absorption | Flexibility | Weight | Use Case |
|----------|-------------------|-------------|--------|----------|
| TPU 95A | Good | High | Light | General protection |
| Nylon 12 | Excellent | Low | Medium | Hard shell guards |
| PETG | Good | Medium | Medium | Budget option |
| PA12 (SLS) | Excellent | Medium | Light | Professional gear |

## Safety Testing

**Critical**: All protective equipment must be tested:

1. **Impact testing** per relevant standards
2. **Coverage verification** for protected areas
3. **Fit testing** for secure attachment
4. **Durability testing** over repeated use

## Next Steps

- [Medical/Orthotics](./medical.md) - Precision medical applications
- [Example Gallery](../examples/gallery.md) - More examples
