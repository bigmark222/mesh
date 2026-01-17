//! Example: Pipeline Configuration Serialization
//!
//! This example demonstrates how to use `PipelineConfig` to define
//! mesh processing workflows as TOML or JSON files.
//!
//! Run with: `cargo run --example pipeline_config --features pipeline-config`

#[cfg(not(feature = "pipeline-config"))]
fn main() {
    eprintln!("This example requires the 'pipeline-config' feature.");
    eprintln!("Run with: cargo run --example pipeline_config --features pipeline-config");
}

#[cfg(feature = "pipeline-config")]
use mesh_repair::{Mesh, Pipeline, PipelineConfig, PipelineStep, Vertex};

#[cfg(feature = "pipeline-config")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // =========================================================================
    // 1. Create a config programmatically
    // =========================================================================

    let config = PipelineConfig::with_name("scan-to-print")
        .description("Prepare 3D scan data for 3D printing")
        .add_step(PipelineStep::RepairForScans)
        .add_step(PipelineStep::Remesh {
            target_edge_length: 2.0,
        })
        .add_step(PipelineStep::DecimateToRatio { ratio: 0.5 })
        .add_step(PipelineStep::Validate);

    // =========================================================================
    // 2. Serialize to TOML
    // =========================================================================

    let toml_str = config.to_toml()?;
    println!("=== TOML Output ===\n{}", toml_str);

    // =========================================================================
    // 3. Serialize to JSON
    // =========================================================================

    let json_str = config.to_json()?;
    println!("=== JSON Output ===\n{}", json_str);

    // =========================================================================
    // 4. Parse config from TOML string
    // =========================================================================

    let toml_input = r#"
        name = "custom-workflow"
        description = "Custom mesh processing workflow"

        [[steps]]
        operation = "repair"

        [[steps]]
        operation = "fill_holes"
        max_edges = 100

        [[steps]]
        operation = "fix_winding"

        [[steps]]
        operation = "remesh"
        target_edge_length = 1.5

        [[steps]]
        operation = "decimate_to_count"
        target_count = 50000

        [[steps]]
        operation = "compute_normals"

        [[steps]]
        operation = "validate"
    "#;

    let parsed_config = PipelineConfig::from_toml(toml_input)?;
    println!(
        "\n=== Parsed Config ===\nName: {:?}\nSteps: {}",
        parsed_config.name,
        parsed_config.steps.len()
    );

    // =========================================================================
    // 5. Use preset configurations
    // =========================================================================

    println!("\n=== Preset Configurations ===");

    let scan_preset = PipelineConfig::preset_scan_to_print();
    println!("scan-to-print: {} steps", scan_preset.steps.len());

    let simplify_preset = PipelineConfig::preset_simplify(0.25);
    println!("simplify (25%): {} steps", simplify_preset.steps.len());

    let refine_preset = PipelineConfig::preset_refine(1, 1.0);
    println!("refine: {} steps", refine_preset.steps.len());

    // =========================================================================
    // 6. Run a config on a test mesh
    // =========================================================================

    let mesh = create_test_cube();
    println!(
        "\n=== Running Pipeline ===\nInput: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    let result = Pipeline::new(mesh).run_config(&parsed_config)?.finish();

    println!(
        "Output: {} vertices, {} faces",
        result.mesh.vertices.len(),
        result.mesh.faces.len()
    );
    println!("Stages executed: {}", result.stages_executed);
    println!("\nOperation log:");
    for entry in &result.operation_log {
        println!("  - {}", entry);
    }

    if let Some(validation) = &result.validation {
        println!("\nValidation:");
        println!("  Watertight: {}", validation.is_watertight);
        println!("  Manifold: {}", validation.is_manifold);
    }

    // =========================================================================
    // 7. Save/load config to file (commented out to avoid filesystem writes)
    // =========================================================================

    // config.save_toml("workflow.toml")?;
    // let loaded = PipelineConfig::from_toml_file("workflow.toml")?;

    Ok(())
}

/// Create a simple test cube mesh
#[cfg(feature = "pipeline-config")]
fn create_test_cube() -> Mesh {
    let mut mesh = Mesh::new();

    mesh.vertices = vec![
        Vertex::from_coords(0.0, 0.0, 0.0),
        Vertex::from_coords(10.0, 0.0, 0.0),
        Vertex::from_coords(10.0, 10.0, 0.0),
        Vertex::from_coords(0.0, 10.0, 0.0),
        Vertex::from_coords(0.0, 0.0, 10.0),
        Vertex::from_coords(10.0, 0.0, 10.0),
        Vertex::from_coords(10.0, 10.0, 10.0),
        Vertex::from_coords(0.0, 10.0, 10.0),
    ];

    mesh.faces = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];

    mesh
}
