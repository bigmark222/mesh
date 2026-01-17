# Development Setup

Guide for setting up a development environment for contributing to mesh-repair.

## Prerequisites

### Required

- **Rust 1.70+**: Install via [rustup](https://rustup.rs/)
- **Git**: For version control

### Optional

- **GPU with Vulkan**: For mesh-gpu development
- **mdBook**: For documentation (`cargo install mdbook`)
- **cargo-llvm-cov**: For coverage reports (`cargo install cargo-llvm-cov`)

## Clone and Build

```bash
# Clone the repository
git clone https://github.com/bigmark222/mesh.git
cd mesh

# Build all crates in debug mode
cargo build --workspace

# Build in release mode (faster, but slower compilation)
cargo build --workspace --release
```

## Project Structure

```
mesh/
├── crates/
│   ├── mesh-repair/      # Core library
│   │   ├── src/          # Source code
│   │   ├── tests/        # Integration tests
│   │   └── benches/      # Benchmarks
│   ├── mesh-shell/       # Shell generation
│   ├── mesh-cli/         # Command-line tool
│   └── mesh-gpu/         # GPU acceleration
├── docs/                 # mdBook documentation
├── tests/
│   └── fixtures/         # Test mesh files
├── CONTRIBUTING.md
├── ROADMAP_v2.md
└── BENCHMARKS.md
```

## Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p mesh-repair

# Specific test
cargo test -p mesh-repair test_weld_vertices

# With output
cargo test -p mesh-repair -- --nocapture

# Property-based tests
cargo test --workspace -- proptest
```

## Running Benchmarks

```bash
# All benchmarks
cargo bench -p mesh-repair

# Specific group
cargo bench -p mesh-repair -- Validation

# Save baseline
cargo bench -p mesh-repair -- --save-baseline main
```

## Code Quality

```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace -- -D warnings

# Run all checks (like CI)
cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace
```

## Building Documentation

### API Docs (rustdoc)

```bash
cargo doc --workspace --no-deps --open
```

### Book (mdBook)

```bash
cd docs
mdbook build
mdbook serve --open  # Preview locally
```

## IDE Setup

### VS Code

Recommended extensions:
- rust-analyzer
- Even Better TOML
- CodeLLDB (for debugging)

### IntelliJ/CLion

Use the Rust plugin with rust-analyzer backend.

## Debugging

### With VS Code

1. Install CodeLLDB extension
2. Add launch configuration:

```json
{
    "type": "lldb",
    "request": "launch",
    "name": "Debug unit tests",
    "cargo": {
        "args": ["test", "-p", "mesh-repair", "--no-run"],
        "filter": {
            "kind": "lib"
        }
    },
    "args": [],
    "cwd": "${workspaceFolder}"
}
```

### With println debugging

```bash
cargo test -p mesh-repair test_name -- --nocapture
```

## Common Tasks

### Adding a new function

1. Add implementation in appropriate module
2. Add public export in `lib.rs`
3. Add unit tests
4. Add documentation with examples
5. Run `cargo test` and `cargo clippy`

### Adding a new test file

Place in `crates/{crate}/tests/` and it will be auto-discovered.

### Adding test fixtures

Place mesh files in `tests/fixtures/`. Keep files small.

## Troubleshooting

### Build fails with "can't find crate"

```bash
cargo clean && cargo build
```

### Tests timeout

Some tests are slow. Run specific tests:

```bash
cargo test -p mesh-repair fast_test_name
```

### GPU tests fail

Ensure Vulkan drivers are installed. GPU tests can be skipped:

```bash
cargo test -p mesh-gpu --no-default-features
```
