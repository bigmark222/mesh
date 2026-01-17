# Custom-Fit Products Overview

This section provides tutorials for building custom-fit product applications using mesh-repair.

## The Custom-Fit Pipeline

All custom-fit products follow a similar pattern:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  3D Scan    │ ──▶ │   Repair    │ ──▶ │  Generate   │ ──▶ │ Manufacture │
│             │     │             │     │   Shell     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
   Raw scan          Clean mesh          Hollow part         3D print/mill
```

## Typical Workflow

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use mesh_shell::{generate_shell, ShellParams};

fn create_custom_product(scan_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load scan
    let scan = load_mesh(scan_path)?;
    println!("Loaded scan: {} faces", scan.face_count());

    // 2. Repair and prepare
    let prepared = Pipeline::new(scan)
        .weld_vertices(1e-6)
        .remove_small_components(100)
        .fill_holes()
        .fix_winding()
        .decimate_to_ratio(0.5)
        .finish();

    println!("Prepared mesh: {} faces", prepared.mesh.face_count());

    // 3. Validate for shell generation
    let report = validate_mesh(&prepared.mesh);
    if !report.is_watertight {
        return Err("Mesh must be watertight for shell generation".into());
    }

    // 4. Generate shell
    let shell_params = ShellParams {
        wall_thickness_mm: 2.0,
        ..Default::default()
    };
    let shell = generate_shell(&prepared.mesh, &shell_params)?;

    println!("Generated shell: {} faces", shell.face_count());

    // 5. Export for manufacturing
    save_mesh(&shell, output_path)?;

    Ok(())
}
```

## Product-Specific Tutorials

Each product type has unique considerations:

| Product | Key Considerations | Tutorial |
|---------|-------------------|----------|
| [Shoe Insoles](./shoes.md) | Foot anatomy, pressure points, flexibility | Scan → Shell → Print |
| [Helmet Liners](./helmets.md) | Impact absorption, ventilation, fit | Multi-density shells |
| [Protective Equipment](./protective.md) | Impact zones, coverage, mobility | Segmented shells |
| [Medical/Orthotics](./medical.md) | Anatomical accuracy, compliance, materials | Precision fitting |

## Common Challenges

### 1. Scan Quality

Raw 3D scans often have:
- Missing data (occlusions)
- Noise and artifacts
- Inconsistent density
- Multiple disconnected pieces

**Solution**: Robust repair pipeline before shell generation.

### 2. Wall Thickness

Too thin: Breaks during use
Too thick: Heavy, doesn't flex properly

**Solution**: Test with material and use case.

### 3. Fit Accuracy

Shell must match body contour precisely.

**Solution**: Minimal decimation in critical areas, validate dimensions.

### 4. Manufacturing Constraints

- Minimum wall thickness for printer
- Maximum build volume
- Support requirements
- Material properties

**Solution**: Design with manufacturing in mind.

## Reference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Application                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Scanner   │  │   Web UI    │  │  Order Mgmt │             │
│  │   Driver    │  │             │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐             │
│  │              mesh-repair / mesh-shell          │             │
│  │  - Load scan                                   │             │
│  │  - Repair topology                             │             │
│  │  - Generate shell                              │             │
│  │  - Export for manufacturing                    │             │
│  └────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Getting Started

1. Start with [Shoe Insoles](./shoes.md) - simplest workflow
2. Progress to [Helmet Liners](./helmets.md) - multi-component
3. Try [Protective Equipment](./protective.md) - segmented designs
4. Learn [Medical/Orthotics](./medical.md) - precision requirements
