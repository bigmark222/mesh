# Installation

## Requirements

- Rust 1.70 or later
- Cargo (comes with Rust)

## Adding to Your Project

Add mesh-repair to your `Cargo.toml`:

```toml
[dependencies]
mesh-repair = "0.1"
```

For shell generation capabilities:

```toml
[dependencies]
mesh-repair = "0.1"
mesh-shell = "0.1"
```

## Optional Features

### GPU Acceleration

For GPU-accelerated SDF computation (requires wgpu):

```toml
[dependencies]
mesh-repair = "0.1"
mesh-gpu = "0.1"
```

GPU acceleration provides 3-68x speedup for SDF operations on supported hardware.

### Pipeline Configuration

For YAML/JSON pipeline configuration support:

```toml
[dependencies]
mesh-repair = { version = "0.1", features = ["pipeline-config"] }
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/bigmark222/mesh.git
cd mesh

# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Build documentation
cargo doc --workspace --no-deps --open
```

## Command-Line Interface

Install the CLI tool:

```bash
cargo install mesh-cli
```

Or build from source:

```bash
cd crates/mesh-cli
cargo install --path .
```

Verify installation:

```bash
mesh-cli --version
mesh-cli --help
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Fully supported | Primary development platform |
| macOS x86_64 | Fully supported | Intel Macs |
| macOS ARM64 | Fully supported | Apple Silicon |
| Windows x86_64 | Fully supported | MSVC toolchain |
| WebAssembly | Experimental | No GPU support |

## Verifying Installation

Create a test file `src/main.rs`:

```rust
use mesh_repair::{Mesh, validate_mesh};

fn main() {
    let mesh = Mesh::new();
    let report = validate_mesh(&mesh);
    println!("Empty mesh validation: {:?}", report);
    println!("mesh-repair installed successfully!");
}
```

Run it:

```bash
cargo run
```

## Next Steps

- [Quick Start](./quick-start.md) - Process your first mesh
- [Basic Concepts](./concepts.md) - Understand mesh fundamentals
