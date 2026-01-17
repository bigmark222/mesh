# mesh-repair

**A pure Rust library for 3D mesh processing, repair, and shell generation.**

mesh-repair is designed for custom-fit product manufacturing - think custom shoe insoles, helmet liners, protective equipment, and orthotics. It provides a complete pipeline from 3D scan to manufacturable shell.

## Key Features

- **Mesh I/O**: Load and save STL, OBJ, PLY, and 3MF files
- **Validation**: Detect topology issues, holes, non-manifold edges
- **Repair**: Fix winding order, weld vertices, fill holes, remove degenerates
- **Decimation**: Reduce triangle count while preserving shape (QEM algorithm)
- **Remeshing**: Isotropic remeshing for uniform triangle quality
- **Subdivision**: Loop subdivision for smooth surfaces
- **Boolean Operations**: Union, intersection, difference of meshes
- **Shell Generation**: Create hollow shells with uniform wall thickness
- **Pipeline API**: Chain operations fluently

## Quick Example

```rust
use mesh_repair::{load_mesh, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a 3D scan
    let mesh = load_mesh("foot_scan.stl")?;

    // Process it through a repair pipeline
    let result = Pipeline::new(mesh)
        .validate()
        .weld_vertices(1e-6)
        .fill_holes()
        .decimate_to_ratio(0.5)
        .finish();

    println!("Processed mesh: {} vertices, {} faces",
        result.mesh.vertex_count(),
        result.mesh.face_count());

    Ok(())
}
```

## Who Is This For?

This library is for **Rust developers building custom-fit product applications**:

- 3D scanning app backends
- Mass customization platforms
- Medical device software
- Protective equipment manufacturers
- Anyone processing 3D scan data programmatically

## What This Library Is NOT

- **Not an application** - No REST API, no web server, no GUI
- **Not a service** - You build the service, this is the engine
- **Not multi-language** - Rust only (use the CLI for other languages)

## Getting Started

1. [Installation](./getting-started/installation.md) - Add to your project
2. [Quick Start](./getting-started/quick-start.md) - Your first mesh processing
3. [Basic Concepts](./getting-started/concepts.md) - Understanding meshes

## Crates

| Crate | Description |
|-------|-------------|
| `mesh-repair` | Core mesh processing library |
| `mesh-shell` | Shell generation for hollow parts |
| `mesh-cli` | Command-line interface |
| `mesh-gpu` | Optional GPU acceleration |

## Performance

mesh-repair is designed for production use:

- **Fast**: Native Rust performance, SIMD-optimized where applicable
- **Memory efficient**: ~40 bytes per triangle (vs 120+ in many libraries)
- **Parallel**: Rayon-based parallelization for large meshes
- **GPU ready**: Optional GPU acceleration for SDF operations

See [Benchmarks](https://github.com/bigmark222/mesh/blob/main/BENCHMARKS.md) for detailed performance data.

## License

Licensed under MIT OR Apache-2.0, at your option.
