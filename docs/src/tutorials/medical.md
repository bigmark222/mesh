# Tutorial: Orthotics & Medical Devices

This tutorial covers precision requirements for medical applications including orthotics, prosthetics, and custom medical devices.

## Medical Application Requirements

Medical devices have strict requirements:
- **Dimensional accuracy**: ±0.5mm or better
- **Surface quality**: Smooth, no sharp edges
- **Material biocompatibility**: Must be certified
- **Documentation**: Full traceability

## Precision Scan Processing

```rust
use mesh_repair::{load_mesh, Pipeline, validate_mesh};

fn process_medical_scan(scan_path: &str) -> Result<Mesh, Box<dyn std::error::Error>> {
    let scan = load_mesh(scan_path)?;

    // Use tighter tolerances for medical applications
    let prepared = Pipeline::new(scan)
        // Very tight weld tolerance - preserve accuracy
        .weld_vertices(0.01)  // 0.01mm = 10 microns

        // Remove only definite noise
        .remove_small_components(2000)

        // Careful hole filling
        .fill_holes()

        // Fix normals
        .fix_winding()

        // Minimal decimation to preserve accuracy
        .decimate_to_target(50_000)

        // High-quality remesh for smooth surface
        .remesh(0.3)  // 0.3mm edges

        .finish();

    // Rigorous validation
    let report = validate_mesh(&prepared.mesh);
    if !report.is_manifold || !report.is_watertight {
        return Err("Medical scan does not meet quality requirements".into());
    }

    Ok(prepared.mesh)
}
```

## Orthotic Insole with Arch Support

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams, WallGenerationMethod};

fn create_orthotic_insole(
    foot_scan_path: &str,
    output_path: &str,
    arch_support_height: f64,  // mm of arch support
) -> Result<(), Box<dyn std::error::Error>> {
    // Load and prepare foot scan
    let scan = load_mesh(foot_scan_path)?;

    let prepared = Pipeline::new(scan)
        .weld_vertices(0.01)
        .remove_small_components(1000)
        .fill_holes()
        .fix_winding()
        .decimate_to_target(30_000)
        .remesh(0.5)
        .finish();

    // Verify dimensions match expected foot size
    if let Some((min, max)) = prepared.mesh.bounds() {
        let length = max.y - min.y;
        let width = max.x - min.x;
        println!("Foot dimensions: {:.1}mm x {:.1}mm", length, width);

        // Sanity check
        if length < 200.0 || length > 350.0 {
            println!("Warning: Unusual foot length");
        }
    }

    // Generate orthotic shell
    let shell_params = ShellParams {
        wall_thickness_mm: 2.5,  // Appropriate for rigid orthotics
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.25,  // High resolution for medical
        ..Default::default()
    };

    let orthotic = generate_shell(&prepared.mesh, &shell_params)?;

    // Final medical-grade validation
    validate_medical_device(&orthotic)?;

    save_mesh(&orthotic, output_path)?;
    Ok(())
}

fn validate_medical_device(mesh: &Mesh) -> Result<(), Box<dyn std::error::Error>> {
    let report = validate_mesh(mesh);

    // Strict requirements
    if !report.is_manifold {
        return Err("Medical device must be manifold".into());
    }
    if !report.is_watertight {
        return Err("Medical device must be watertight".into());
    }
    if report.degenerate_face_count > 0 {
        return Err("Medical device cannot have degenerate faces".into());
    }
    if report.component_count != 1 {
        return Err("Medical device must be single component".into());
    }

    Ok(())
}
```

## Prosthetic Socket Design

```rust
// Prosthetic sockets require precise fit and load distribution

fn create_prosthetic_socket(
    residual_limb_scan: &Mesh,
    socket_thickness: f64,
) -> Result<Mesh, Box<dyn std::error::Error>> {
    // 1. Prepare scan with minimal modification
    let prepared = Pipeline::new(residual_limb_scan.clone())
        .weld_vertices(0.01)
        .fill_holes()
        .fix_winding()
        .finish();

    // 2. Apply pressure relief modifications
    // (This would be custom to your prosthetic design methodology)

    // 3. Generate socket shell
    let shell_params = ShellParams {
        wall_thickness_mm: socket_thickness,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.2,  // Very high resolution
        ..Default::default()
    };

    let socket = generate_shell(&prepared.mesh, &shell_params)?;

    // 4. Add structural features
    // - Suspension mechanism
    // - Alignment features
    // - Trim line

    Ok(socket)
}
```

## Regulatory Considerations

**Important**: Medical devices are heavily regulated.

### FDA (United States)
- Class I, II, or III depending on risk
- 510(k) clearance or PMA approval may be required
- Quality Management System (21 CFR Part 820)

### CE Marking (Europe)
- Medical Device Regulation (MDR) 2017/745
- Conformity assessment required
- Technical documentation

### ISO Standards
- ISO 13485: Quality Management for Medical Devices
- ISO 10993: Biocompatibility
- ISO 14708: Implants (if applicable)

## Documentation Requirements

For medical applications, maintain:

```rust
struct MedicalDeviceRecord {
    patient_id: String,
    scan_date: DateTime,
    scan_parameters: ScanParams,
    processing_parameters: ProcessingParams,
    validation_results: ValidationReport,
    output_file_hash: String,
    operator_id: String,
}

// Log all processing parameters
fn log_medical_processing(
    record: &MedicalDeviceRecord,
    log_path: &str,
) -> Result<(), std::io::Error> {
    // Write to immutable audit log
    // Include all parameters for reproducibility
    todo!()
}
```

## Quality Control Checklist

- [ ] Scan captured complete anatomy
- [ ] No holes or missing data
- [ ] Dimensional accuracy verified
- [ ] Surface quality acceptable
- [ ] Shell wall thickness uniform
- [ ] No self-intersections
- [ ] Single connected component
- [ ] Final dimensions match prescription

## Next Steps

- [Example Gallery](../examples/gallery.md) - Code examples
- [Best Practices](../best-practices/performance.md) - Optimization tips
