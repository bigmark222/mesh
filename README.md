# mesh

A Rust workspace for 3D triangle mesh processing, providing tools for mesh repair and shell generation.

## Crates

### mesh-repair

[![crates.io](https://img.shields.io/crates/v/mesh-repair.svg)](https://crates.io/crates/mesh-repair)

Triangle mesh repair utilities:

- **I/O**: Load and save STL, OBJ, and 3MF formats
- **Validation**: Detect holes, non-manifold edges, inconsistent winding
- **Repair**: Fill holes, fix winding order, weld vertices, remove degenerate triangles

```rust
use mesh_repair::Mesh;

let mut mesh = Mesh::load("model.stl")?;

// Check for issues
let report = mesh.validate();
println!("{}", report);

// Repair and save
mesh.repair()?;
mesh.save("repaired.3mf")?;
```

### mesh-shell

[![crates.io](https://img.shields.io/crates/v/mesh-shell.svg)](https://crates.io/crates/mesh-shell)

Generate shells around 3D meshes using SDF-based offset:

- **SDF offset**: Voxelize mesh, compute signed distance field, extract isosurface
- **Variable offset**: Per-vertex offset distances via interpolation
- **Shell generation**: Create inner/outer surfaces with rim closure

```rust
use mesh_shell::{apply_sdf_offset, generate_shell, SdfOffsetParams, ShellParams};
use mesh_repair::Mesh;

let mesh = Mesh::load("model.stl")?;

// Apply SDF-based offset
let params = SdfOffsetParams::default();
let result = apply_sdf_offset(&mesh, &params)?;

// Or generate a complete shell (inner + outer + rim)
let shell_params = ShellParams::default();
let shell = generate_shell(&mesh, &shell_params)?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mesh-repair = "0.1"
mesh-shell = "0.1"  # includes mesh-repair
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
