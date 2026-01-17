# Tutorial: Custom Helmet Liners

This tutorial covers creating custom-fit helmet liners for optimal protection and comfort.

## Overview

Helmet liners require:
- Precise fit to head shape
- Impact absorption zones
- Ventilation channels
- Consistent wall thickness

```
Head scan    →    Liner shell    →    With ventilation    →    Final liner
```

## Step 1: Process Head Scan

```rust
use mesh_repair::{load_mesh, Pipeline, validate_mesh};

let scan = load_mesh("head_scan.stl")?;

// Head scans are typically larger and noisier than foot scans
let prepared = Pipeline::new(scan)
    .weld_vertices(0.2)              // Slightly larger tolerance
    .remove_small_components(1000)   // Remove more noise
    .fill_holes()
    .fix_winding()
    .decimate_to_target(30_000)      // Keep more detail
    .finish();

println!("Prepared head scan: {} faces", prepared.mesh.face_count());
```

## Step 2: Define Impact Zones

Different areas need different protection levels:

```rust
// Impact zones (conceptual - implement based on your needs)
struct ImpactZone {
    name: &'static str,
    min_thickness_mm: f64,
    density: &'static str,
}

let zones = [
    ImpactZone { name: "crown", min_thickness_mm: 15.0, density: "high" },
    ImpactZone { name: "temporal", min_thickness_mm: 12.0, density: "medium" },
    ImpactZone { name: "rear", min_thickness_mm: 10.0, density: "medium" },
    ImpactZone { name: "front", min_thickness_mm: 8.0, density: "low" },
];
```

## Step 3: Generate Multi-Layer Shell

For helmets, you often need multiple shells:

```rust
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

// Inner comfort layer (soft, thin)
let comfort_params = ShellParams {
    wall_thickness_mm: 3.0,
    wall_generation_method: WallGenerationMethod::Sdf,
    sdf_voxel_size_mm: 0.5,
    ..Default::default()
};

// Outer impact layer (thicker, energy absorbing)
let impact_params = ShellParams {
    wall_thickness_mm: 10.0,
    wall_generation_method: WallGenerationMethod::Sdf,
    sdf_voxel_size_mm: 1.0,
    ..Default::default()
};

let comfort_layer = generate_shell(&prepared.mesh, &comfort_params)?;
let impact_layer = generate_shell(&comfort_layer, &impact_params)?;
```

## Step 4: Add Ventilation

Ventilation improves comfort but requires careful design:

```rust
use mesh_repair::{boolean_operation, BooleanOp, BooleanParams};

// Create ventilation channels (simplified example)
fn create_vent_channel(position: [f64; 3], direction: [f64; 3]) -> Mesh {
    // Create elongated cylinder for vent channel
    // Implementation depends on your channel design
    todo!("Create vent channel geometry")
}

// Subtract vents from liner
let vent_positions = [
    ([0.0, 80.0, 0.0], [0.0, 0.0, 1.0]),   // Top vents
    ([50.0, 60.0, 30.0], [1.0, 0.0, 0.0]), // Side vents
    // ... more vents
];

let mut liner_with_vents = impact_layer;
let params = BooleanParams::default();

for (pos, dir) in &vent_positions {
    let vent = create_vent_channel(*pos, *dir);
    liner_with_vents = boolean_operation(&liner_with_vents, &vent, BooleanOp::Difference, &params)?.mesh;
}
```

## Step 5: Validate Structural Integrity

After adding vents, ensure the liner is still printable:

```rust
let report = validate_mesh(&liner_with_vents);

// Check critical properties
assert!(report.is_manifold, "Liner must be manifold");
assert!(report.is_watertight, "Liner must be watertight");
assert!(report.component_count == 1, "Liner should be single piece");

// Verify minimum wall thickness
// (would need custom thickness analysis implementation)
```

## Complete Helmet Liner Pipeline

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

fn create_helmet_liner(
    scan_path: &str,
    output_path: &str,
    comfort_thickness: f64,
    impact_thickness: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load and prepare scan
    let scan = load_mesh(scan_path)?;
    let prepared = Pipeline::new(scan)
        .weld_vertices(0.2)
        .remove_small_components(1000)
        .fill_holes()
        .fix_winding()
        .decimate_to_target(30_000)
        .remesh(1.0)  // Uniform 1mm triangles
        .finish();

    // Validate base mesh
    let report = validate_mesh(&prepared.mesh);
    if !report.is_watertight {
        return Err("Head scan could not be made watertight".into());
    }

    // Generate comfort layer
    let comfort_params = ShellParams {
        wall_thickness_mm: comfort_thickness,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.5,
        ..Default::default()
    };
    let comfort_layer = generate_shell(&prepared.mesh, &comfort_params)?;

    // Generate impact layer
    let impact_params = ShellParams {
        wall_thickness_mm: impact_thickness,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 1.0,
        ..Default::default()
    };
    let impact_layer = generate_shell(&comfort_layer, &impact_params)?;

    // Export
    save_mesh(&impact_layer, output_path)?;

    Ok(())
}
```

## Safety Considerations

**Important**: Custom helmet liners are safety-critical. Always:

1. **Test impact absorption** according to relevant standards
2. **Verify fit** with physical prototypes
3. **Consult with safety engineers**
4. **Follow certification requirements** (DOT, ECE, Snell, etc.)

This library provides geometry processing - actual safety testing requires specialized equipment and expertise.

## Next Steps

- [Protective Equipment](./protective.md) - More complex protective gear
- [Medical/Orthotics](./medical.md) - Precision medical applications
