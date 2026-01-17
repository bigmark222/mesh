# mesh-repair API Reference

The core mesh processing library.

## Quick Links

- [Full API Docs](https://docs.rs/mesh-repair) (docs.rs)
- [Source Code](https://github.com/bigmark222/mesh/tree/main/crates/mesh-repair)

## Main Types

### Mesh

The core mesh type representing a triangle mesh.

```rust
use mesh_repair::{Mesh, Vertex};

let mut mesh = Mesh::new();
mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
mesh.faces.push([0, 1, 2]);

// Geometry queries
println!("Vertices: {}", mesh.vertex_count());
println!("Faces: {}", mesh.face_count());
println!("Surface area: {}", mesh.surface_area());
println!("Volume: {}", mesh.volume());

if let Some((min, max)) = mesh.bounds() {
    println!("Bounds: {:?} to {:?}", min, max);
}
```

### MeshReport

Validation report containing mesh analysis.

```rust
use mesh_repair::{validate_mesh, Mesh};

let mesh = Mesh::new();
let report = validate_mesh(&mesh);

// Counts
report.vertex_count;
report.face_count;
report.component_count;

// Topology
report.is_manifold;
report.is_watertight;
report.boundary_edge_count;
report.non_manifold_edge_count;
report.degenerate_face_count;

// Helper methods
report.is_valid();
report.is_printable();
```

## Core Functions

### I/O

```rust
// Load from file (format from extension)
let mesh = load_mesh("model.stl")?;
let mesh = load_mesh("model.obj")?;
let mesh = load_mesh("model.ply")?;
let mesh = load_mesh("model.3mf")?;

// Save to file
save_mesh(&mesh, "output.stl")?;
```

### Validation

```rust
// Full validation
let report = validate_mesh(&mesh);

// Component analysis
let analysis = find_connected_components(&mesh);
```

### Repair

```rust
// Weld duplicate vertices
weld_vertices(&mut mesh, tolerance);

// Fill holes
fill_holes(&mut mesh)?;

// Fix winding order
fix_winding_order(&mut mesh)?;

// Remove degenerate triangles
remove_degenerate_triangles(&mut mesh, area_threshold);

// Component operations
remove_small_components(&mut mesh, min_faces);
keep_largest_component(&mut mesh)?;
```

### Simplification

```rust
// Decimation
let params = DecimateParams::with_target_triangles(10_000);
let result = decimate_mesh(&mesh, &params);

// Remeshing
let params = RemeshParams { target_edge_length: Some(0.5), ..Default::default() };
let result = remesh_isotropic(&mesh, &params);

// Subdivision
let params = SubdivideParams::with_iterations(1);
let result = subdivide_mesh(&mesh, &params);
```

### Boolean Operations

```rust
let params = BooleanParams::default();
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Union, &params)?;
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Intersection, &params)?;
let result = boolean_operation(&mesh_a, &mesh_b, BooleanOp::Difference, &params)?;
```

## Pipeline API

```rust
let result = Pipeline::new(mesh)
    .validate()
    .weld_vertices(1e-6)
    .fill_holes()
    .fix_winding()
    .decimate_to_ratio(0.5)
    .remesh(0.5)
    .finish();
```

## Parameter Structs

### DecimateParams

```rust
DecimateParams {
    target_triangles: Option<usize>,  // Target face count
    target_ratio: f64,                 // Target ratio (0.0-1.0)
    preserve_boundaries: bool,         // Keep boundary edges
    min_quality: f64,                  // Min triangle quality
}
```

### RemeshParams

```rust
RemeshParams {
    target_edge_length: Option<f64>,  // Target edge length
    iterations: usize,                 // Number of iterations
    preserve_features: bool,           // Keep sharp features
    feature_angle: f64,                // Angle threshold for features
}
```

### SubdivideParams

```rust
SubdivideParams {
    iterations: usize,  // Number of subdivisions
}
```

## Error Types

```rust
pub enum MeshError {
    IoError(std::io::Error),
    ParseError { path: String, message: String },
    InvalidTopology(String),
    EmptyMesh(String),
    RepairFailed(String),
    HoleFillFailed(String),
    BooleanFailed { operation: String, reason: String },
    DecimationFailed(String),
    RemeshingFailed(String),
}
```

## Feature Flags

```toml
[dependencies]
mesh-repair = { version = "0.1", features = ["pipeline-config"] }
```

- `pipeline-config`: Enable YAML/JSON pipeline configuration
