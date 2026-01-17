# Scan to Shell Example

Complete workflow from 3D scan to manufacturable shell.

## Overview

This example demonstrates the full custom-fit product pipeline:

1. Load raw 3D scan
2. Clean up scan artifacts
3. Validate mesh quality
4. Generate hollow shell
5. Export for manufacturing

## Complete Implementation

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

/// Process a 3D scan into a manufacturable shell.
fn scan_to_shell(
    scan_path: &str,
    output_path: &str,
    wall_thickness: f64,
) -> Result<ShellResult, Box<dyn std::error::Error>> {
    println!("=== Scan to Shell Pipeline ===\n");

    // Step 1: Load scan
    println!("1. Loading scan: {}", scan_path);
    let scan = load_mesh(scan_path)?;
    println!("   Loaded: {} vertices, {} faces", scan.vertex_count(), scan.face_count());

    // Step 2: Initial validation
    println!("\n2. Initial validation");
    let initial_report = validate_mesh(&scan);
    println!("   Watertight: {}", initial_report.is_watertight);
    println!("   Components: {}", initial_report.component_count);
    println!("   Boundary edges: {}", initial_report.boundary_edge_count);

    // Step 3: Repair pipeline
    println!("\n3. Repairing scan");
    let repaired = Pipeline::new(scan)
        .weld_vertices(0.1)              // 0.1mm tolerance for scan data
        .remove_degenerate_triangles(1e-10)
        .remove_small_components(200)    // Remove noise
        .fill_holes()
        .fix_winding()
        .decimate_to_target(25_000)      // Reasonable resolution
        .finish();

    println!("   After repair: {} faces", repaired.mesh.face_count());

    // Step 4: Validate repaired mesh
    println!("\n4. Post-repair validation");
    let repair_report = validate_mesh(&repaired.mesh);

    if !repair_report.is_watertight {
        return Err("Scan could not be made watertight".into());
    }
    if !repair_report.is_manifold {
        return Err("Scan has non-manifold geometry".into());
    }

    println!("   Watertight: {}", repair_report.is_watertight);
    println!("   Manifold: {}", repair_report.is_manifold);

    // Step 5: Generate shell
    println!("\n5. Generating shell");
    println!("   Wall thickness: {}mm", wall_thickness);

    let shell_params = ShellParams {
        wall_thickness_mm: wall_thickness,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.5,
        ..Default::default()
    };

    let shell = generate_shell(&repaired.mesh, &shell_params)?;
    println!("   Shell generated: {} faces", shell.face_count());

    // Step 6: Final validation
    println!("\n6. Final validation");
    let final_report = validate_mesh(&shell);

    if !final_report.is_printable() {
        println!("   WARNING: Shell may not be printable");
    } else {
        println!("   Shell is ready for manufacturing!");
    }

    // Step 7: Export
    println!("\n7. Exporting");
    save_mesh(&shell, output_path)?;
    println!("   Saved to: {}", output_path);

    // Return results
    Ok(ShellResult {
        original_faces: scan.face_count(),
        repaired_faces: repaired.mesh.face_count(),
        shell_faces: shell.face_count(),
        wall_thickness,
        is_printable: final_report.is_printable(),
    })
}

struct ShellResult {
    original_faces: usize,
    repaired_faces: usize,
    shell_faces: usize,
    wall_thickness: f64,
    is_printable: bool,
}

fn main() {
    match scan_to_shell("foot_scan.stl", "custom_insole.3mf", 2.0) {
        Ok(result) => {
            println!("\n=== Summary ===");
            println!("Original faces: {}", result.original_faces);
            println!("Repaired faces: {}", result.repaired_faces);
            println!("Shell faces: {}", result.shell_faces);
            println!("Wall thickness: {}mm", result.wall_thickness);
            println!("Ready for printing: {}", result.is_printable);
        }
        Err(e) => {
            eprintln!("Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
```

## Example Output

```
=== Scan to Shell Pipeline ===

1. Loading scan: foot_scan.stl
   Loaded: 156432 vertices, 52144 faces

2. Initial validation
   Watertight: false
   Components: 8
   Boundary edges: 234

3. Repairing scan
   After repair: 25000 faces

4. Post-repair validation
   Watertight: true
   Manifold: true

5. Generating shell
   Wall thickness: 2mm
   Shell generated: 48234 faces

6. Final validation
   Shell is ready for manufacturing!

7. Exporting
   Saved to: custom_insole.3mf

=== Summary ===
Original faces: 52144
Repaired faces: 25000
Shell faces: 48234
Wall thickness: 2mm
Ready for printing: true
```

## CLI Version

Achieve similar results with the command line:

```bash
# Repair scan
mesh-cli repair foot_scan.stl -o repaired.stl \
    --weld-tolerance 0.1 \
    --remove-small 200 \
    --fill-holes

# Decimate
mesh-cli decimate repaired.stl -o decimated.stl \
    --target 25000

# Generate shell
mesh-cli shell decimated.stl -o insole.3mf \
    --thickness 2.0 \
    --method sdf
```

## Configuration File

For repeatable processing, use a config file:

```yaml
# insole_config.yaml
repair:
  weld_tolerance: 0.1
  remove_small_components: 200
  fill_holes: true
  fix_winding: true

decimate:
  target_triangles: 25000

shell:
  wall_thickness_mm: 2.0
  method: sdf
  voxel_size_mm: 0.5

output:
  format: 3mf
```

## Error Handling

Handle common failures gracefully:

```rust
fn process_scan(path: &str) -> Result<Mesh, ProcessError> {
    let scan = load_mesh(path).map_err(|e| ProcessError::LoadFailed(e.to_string()))?;

    let repaired = Pipeline::new(scan)
        .weld_vertices(0.1)
        .fill_holes()
        .fix_winding()
        .finish();

    let report = validate_mesh(&repaired.mesh);
    if !report.is_watertight {
        return Err(ProcessError::NotWatertight);
    }

    let shell_params = ShellParams {
        wall_thickness_mm: 2.0,
        ..Default::default()
    };

    generate_shell(&repaired.mesh, &shell_params)
        .map_err(|e| ProcessError::ShellFailed(e.to_string()))
}

enum ProcessError {
    LoadFailed(String),
    NotWatertight,
    ShellFailed(String),
}
```
