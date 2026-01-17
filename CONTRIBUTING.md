# Contributing to mesh-repair

Thank you for your interest in contributing to mesh-repair! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are ready for community contributions
- Check the [ROADMAP_v2.md](ROADMAP_v2.md) for larger features being planned

### Before You Start

1. Check if an issue already exists for your proposed change
2. For significant changes, open an issue first to discuss the approach
3. Comment on the issue to let others know you're working on it

## Development Setup

### Prerequisites

- Rust 1.70 or later
- Git
- (Optional) GPU with Vulkan support for mesh-gpu development

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/bigmark222/mesh.git
cd mesh

# Build all crates
cargo build

# Run tests
cargo test --workspace

# Build documentation
cargo doc --workspace --no-deps --open
```

### Project Structure

```
mesh/
├── crates/
│   ├── mesh-repair/     # Core mesh processing library
│   ├── mesh-shell/      # Shell generation
│   ├── mesh-cli/        # Command-line interface
│   └── mesh-gpu/        # GPU acceleration (optional)
├── docs/                # mdBook documentation
├── tests/               # Integration tests
│   └── fixtures/        # Test mesh files
└── benches/             # Benchmarks (in each crate)
```

### Running Specific Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p mesh-repair

# Run a specific test
cargo test -p mesh-repair test_weld_vertices

# Run tests with output
cargo test -p mesh-repair -- --nocapture

# Run property-based tests
cargo test --workspace -- proptest
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p mesh-repair

# Run specific benchmark group
cargo bench -p mesh-repair -- Validation

# Save baseline for comparison
cargo bench -p mesh-repair -- --save-baseline main

# Compare against baseline
cargo bench -p mesh-repair -- --baseline main
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-lattice-generation` - New features
- `fix/hole-filling-crash` - Bug fixes
- `docs/update-api-reference` - Documentation
- `refactor/simplify-decimation` - Code improvements
- `test/add-boolean-tests` - Test additions

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(repair): add vertex welding with spatial hashing

Implements O(n) vertex welding using a spatial hash grid instead of
the previous O(n²) brute force approach.

Closes #42
```

```
fix(io): handle binary STL files with no faces

Previously, loading an empty binary STL would panic. Now returns
an empty mesh with a warning.

Fixes #87
```

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Follow existing code patterns in the crate

## Pull Request Process

### Before Submitting

1. **Update tests**: Add tests for new functionality
2. **Run the test suite**: `cargo test --workspace`
3. **Run clippy**: `cargo clippy --workspace -- -D warnings`
4. **Format code**: `cargo fmt --all`
5. **Update documentation**: If changing public API
6. **Update CHANGELOG**: For user-visible changes

### PR Description

Include in your PR description:

- **What**: Brief description of changes
- **Why**: Motivation and context
- **How**: Implementation approach (for complex changes)
- **Testing**: How you tested the changes
- **Related Issues**: Link to related issues

### Review Process

1. CI must pass (tests, clippy, format)
2. At least one maintainer approval required
3. All review comments addressed
4. PR squashed or rebased as needed

## Coding Standards

### API Design

- Follow Rust API guidelines: https://rust-lang.github.io/api-guidelines/
- Prefer builder patterns for complex configuration
- Use `Result` for fallible operations
- Provide sensible `Default` implementations

### Error Handling

```rust
// Good: Specific error types
pub fn load_mesh(path: &Path) -> Result<Mesh, MeshError> {
    // ...
}

// Good: Descriptive error messages
Err(MeshError::parse_error(path, "Invalid vertex count"))
```

### Documentation

- All public items must have doc comments
- Include examples in doc comments where helpful
- Use `# Examples` section in module docs

```rust
/// Welds duplicate vertices within a tolerance.
///
/// # Arguments
///
/// * `mesh` - The mesh to modify
/// * `tolerance` - Maximum distance to consider vertices as duplicates
///
/// # Example
///
/// ```
/// use mesh_repair::{Mesh, weld_vertices};
///
/// let mut mesh = Mesh::new();
/// // ... add vertices and faces ...
/// weld_vertices(&mut mesh, 1e-6);
/// ```
pub fn weld_vertices(mesh: &mut Mesh, tolerance: f64) {
    // ...
}
```

### Performance

- Profile before optimizing
- Document algorithmic complexity in comments
- Consider memory usage for large meshes
- Use `#[inline]` sparingly and with benchmarks

## Testing

### Test Categories

1. **Unit tests**: In the same file as the code
2. **Integration tests**: In `tests/` directory
3. **Property-based tests**: Using proptest
4. **Benchmarks**: Using criterion

### Writing Good Tests

```rust
#[test]
fn test_weld_vertices_removes_duplicates() {
    // Arrange
    let mut mesh = create_mesh_with_duplicates();
    let original_count = mesh.vertex_count();

    // Act
    weld_vertices(&mut mesh, 1e-6);

    // Assert
    assert!(mesh.vertex_count() < original_count);
    assert!(validate_mesh(&mesh).is_manifold);
}
```

### Test Fixtures

Place test mesh files in `tests/fixtures/`. Keep files small when possible.

## Documentation

### Code Documentation

- Run `cargo doc --open` to preview
- Check for broken links with `cargo doc --no-deps`

### Book Documentation

```bash
# Build the mdBook
cd docs
mdbook build

# Serve locally for preview
mdbook serve --open
```

## Release Process

Releases are managed by maintainers. The process:

1. Update version in `Cargo.toml` files
2. Update `CHANGELOG.md`
3. Create git tag `vX.Y.Z`
4. CI publishes to crates.io

## Questions?

- Open a [GitHub Discussion](https://github.com/bigmark222/mesh/discussions)
- Check existing issues and PRs
- Read the [documentation](https://bigmark222.github.io/mesh/)

Thank you for contributing!
