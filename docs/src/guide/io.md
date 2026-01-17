# Loading and Saving Meshes

mesh-repair supports multiple 3D file formats for input and output.

## Supported Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| STL | `.stl` | Binary & ASCII | Binary | Industry standard for 3D printing |
| OBJ | `.obj` | Yes | Yes | Wavefront format, widely supported |
| PLY | `.ply` | ASCII | ASCII | Stanford format, good for scans |
| 3MF | `.3mf` | Yes | Yes | Modern 3D printing format |

## Loading Meshes

### Basic Loading

```rust
use mesh_repair::load_mesh;

// Format is auto-detected from extension
let mesh = load_mesh("model.stl")?;
let mesh = load_mesh("model.obj")?;
let mesh = load_mesh("model.ply")?;
let mesh = load_mesh("model.3mf")?;
```

### Loading with Path

```rust
use mesh_repair::load_mesh;
use std::path::Path;

let path = Path::new("/path/to/model.stl");
let mesh = load_mesh(path)?;
```

### Error Handling

```rust
use mesh_repair::{load_mesh, MeshError};

match load_mesh("model.stl") {
    Ok(mesh) => println!("Loaded {} faces", mesh.face_count()),
    Err(e) => match e {
        MeshError::IoError(io_err) => println!("File error: {}", io_err),
        MeshError::ParseError { path, message } => {
            println!("Parse error in {}: {}", path, message);
        }
        _ => println!("Error: {}", e),
    }
}
```

## Saving Meshes

### Basic Saving

```rust
use mesh_repair::{load_mesh, save_mesh};

let mesh = load_mesh("input.stl")?;
save_mesh(&mesh, "output.stl")?;  // Format from extension
save_mesh(&mesh, "output.obj")?;  // Convert to OBJ
save_mesh(&mesh, "output.3mf")?;  // Convert to 3MF
```

### Format Conversion

```rust
use mesh_repair::{load_mesh, save_mesh};

// STL to OBJ
let mesh = load_mesh("model.stl")?;
save_mesh(&mesh, "model.obj")?;

// OBJ to 3MF
let mesh = load_mesh("model.obj")?;
save_mesh(&mesh, "model.3mf")?;
```

## Format Details

### STL (Stereolithography)

The most common format for 3D printing.

**Binary STL** (default for saving):
- Compact file size
- Fast to read/write
- No vertex sharing (triangle soup)

**ASCII STL**:
- Human-readable
- Larger file size
- Useful for debugging

```
solid model
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0.5 1 0
    endloop
  endfacet
endsolid model
```

### OBJ (Wavefront)

Widely supported format with vertex sharing.

```
# Wavefront OBJ
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
```

**Features**:
- Vertex sharing (smaller files)
- Optional normals (`vn`)
- Optional texture coordinates (`vt`)
- Material references (`.mtl`)

### PLY (Polygon File Format)

Stanford format, common for 3D scans.

```
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0.5 1 0
3 0 1 2
```

**Features**:
- Vertex colors
- Per-vertex properties
- Good for point clouds

### 3MF (3D Manufacturing Format)

Modern XML-based format designed for 3D printing.

**Features**:
- Built-in compression (ZIP)
- Units specification (default: millimeters)
- Multiple objects per file
- Material and color support
- Metadata

## Memory Considerations

### Large Files

For very large meshes, memory usage is approximately:
- ~24 bytes per vertex
- ~12 bytes per face

A 1 million triangle mesh uses ~36 MB.

### Streaming (Future)

For extremely large files, consider:
1. Decimating after loading to reduce memory
2. Processing in chunks (not yet supported)

## CLI Usage

```bash
# Convert formats
mesh-cli convert input.stl output.obj

# Validate file format
mesh-cli validate model.stl

# Get file info
mesh-cli info model.stl
```

## Best Practices

1. **Use binary STL** for 3D printing output (smaller, faster)
2. **Use OBJ** when vertex sharing matters
3. **Use 3MF** for modern slicers with metadata needs
4. **Validate after loading** to catch format issues early

```rust
use mesh_repair::{load_mesh, validate_mesh};

let mesh = load_mesh("model.stl")?;
let report = validate_mesh(&mesh);

if !report.is_valid() {
    eprintln!("Warning: Mesh has issues after loading");
}
```

## Next Steps

- [Mesh Validation](./validation.md) - Check mesh quality
- [Repair Operations](./repair.md) - Fix common issues
