//! Mesh file I/O for STL, OBJ, and 3MF formats.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use tracing::{debug, info, warn};

use crate::error::{MeshError, MeshResult};
use crate::{Mesh, Vertex};

/// Supported mesh file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    Stl,
    Obj,
    ThreeMf,
}

impl MeshFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .and_then(|ext| match ext.as_str() {
                "stl" => Some(MeshFormat::Stl),
                "obj" => Some(MeshFormat::Obj),
                "3mf" => Some(MeshFormat::ThreeMf),
                _ => None,
            })
    }
}

/// Load a mesh from file, auto-detecting format from extension.
pub fn load_mesh(path: &Path) -> MeshResult<Mesh> {
    let format = MeshFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path.extension().and_then(|e| e.to_str()).map(String::from),
    })?;

    info!("Loading mesh from {:?} (format: {:?})", path, format);

    let mesh = match format {
        MeshFormat::Stl => load_stl(path)?,
        MeshFormat::Obj => load_obj(path)?,
        MeshFormat::ThreeMf => load_3mf(path)?,
    };

    // Log basic stats
    if let Some((min, max)) = mesh.bounds() {
        let dims = max - min;
        info!(
            "Loaded mesh: {} vertices, {} faces",
            mesh.vertex_count(),
            mesh.face_count()
        );
        debug!(
            "Bounding box: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
            min.x, min.y, min.z, max.x, max.y, max.z
        );
        debug!(
            "Dimensions: {:.1} x {:.1} x {:.1}",
            dims.x, dims.y, dims.z
        );

        // Warn if dimensions seem unusually small or large
        let max_dim = dims.x.max(dims.y).max(dims.z);
        if max_dim < 0.1 {
            warn!(
                "Mesh largest dimension is {:.6} - may need scaling",
                max_dim
            );
        }
    }

    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "mesh has no vertices or faces".to_string(),
        });
    }

    Ok(mesh)
}

/// Load mesh from STL file (binary or ASCII).
fn load_stl(path: &Path) -> MeshResult<Mesh> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    // stl_io::read_stl returns an IndexedMesh with vertices and indexed faces
    let stl = stl_io::read_stl(&mut reader).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: e.to_string(),
    })?;

    debug!(
        "STL contains {} vertices, {} triangles",
        stl.vertices.len(),
        stl.faces.len()
    );

    // Convert stl_io types to our types
    let mut mesh = Mesh::with_capacity(stl.vertices.len(), stl.faces.len());

    // Convert vertices (stl_io::Vertex is Vector<f32> with .0 being [f32; 3])
    for v in &stl.vertices {
        mesh.vertices.push(Vertex::from_coords(
            v.0[0] as f64,
            v.0[1] as f64,
            v.0[2] as f64,
        ));
    }

    // Convert faces (stl_io::IndexedTriangle has .vertices: [usize; 3])
    for face in &stl.faces {
        let indices = [
            face.vertices[0] as u32,
            face.vertices[1] as u32,
            face.vertices[2] as u32,
        ];

        // Skip degenerate triangles
        if indices[0] != indices[1] && indices[1] != indices[2] && indices[0] != indices[2] {
            mesh.faces.push(indices);
        }
    }

    debug!(
        "Converted mesh: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    Ok(mesh)
}

/// Load mesh from OBJ file.
fn load_obj(path: &Path) -> MeshResult<Mesh> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: e.to_string(),
    })?;

    if models.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "OBJ file contains no models".to_string(),
        });
    }

    // Merge all models into single mesh
    let mut mesh = Mesh::new();
    let mut vertex_offset = 0u32;

    for model in &models {
        debug!("OBJ model '{}': loading", model.name);

        let obj_mesh = &model.mesh;

        // Add vertices
        for chunk in obj_mesh.positions.chunks(3) {
            if chunk.len() == 3 {
                mesh.vertices.push(Vertex::from_coords(
                    chunk[0] as f64,
                    chunk[1] as f64,
                    chunk[2] as f64,
                ));
            }
        }

        // Add faces (indices are per-model, need offset)
        for chunk in obj_mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                mesh.faces.push([
                    chunk[0] + vertex_offset,
                    chunk[1] + vertex_offset,
                    chunk[2] + vertex_offset,
                ]);
            }
        }

        vertex_offset = mesh.vertices.len() as u32;
    }

    debug!(
        "OBJ loaded: {} vertices, {} faces from {} models",
        mesh.vertices.len(),
        mesh.faces.len(),
        models.len()
    );

    Ok(mesh)
}

/// Save mesh to file, auto-detecting format from extension.
pub fn save_mesh(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    let format = MeshFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path.extension().and_then(|e| e.to_str()).map(String::from),
    })?;

    match format {
        MeshFormat::Stl => save_stl(mesh, path),
        MeshFormat::Obj => save_obj(mesh, path),
        MeshFormat::ThreeMf => save_3mf(mesh, path),
    }
}

/// Save mesh to STL file (binary format).
pub fn save_stl(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?}", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    // Build stl_io triangles
    let triangles: Vec<stl_io::Triangle> = mesh
        .faces
        .iter()
        .map(|&[i0, i1, i2]| {
            let v0 = &mesh.vertices[i0 as usize].position;
            let v1 = &mesh.vertices[i1 as usize].position;
            let v2 = &mesh.vertices[i2 as usize].position;

            stl_io::Triangle {
                normal: stl_io::Normal::new([0.0, 0.0, 0.0]), // Readers recompute
                vertices: [
                    stl_io::Vertex::new([v0.x as f32, v0.y as f32, v0.z as f32]),
                    stl_io::Vertex::new([v1.x as f32, v1.y as f32, v1.z as f32]),
                    stl_io::Vertex::new([v2.x as f32, v2.y as f32, v2.z as f32]),
                ],
            }
        })
        .collect();

    stl_io::write_stl(&mut writer, triangles.iter()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} triangles to {:?}",
        mesh.face_count(),
        path
    );

    Ok(())
}

/// Save mesh to OBJ file (ASCII format).
///
/// OBJ format preserves vertex indices exactly, making it ideal for debugging
/// pipeline stages where vertex tracking is important. Unlike STL which
/// duplicates vertices per-triangle, OBJ maintains the indexed mesh structure.
///
/// The output includes:
/// - Vertex positions (`v x y z`)
/// - Vertex normals if present (`vn nx ny nz`)
/// - Face indices (`f v1 v2 v3` or `f v1//n1 v2//n2 v3//n3` with normals)
/// - Comments with tag and offset info for debugging
pub fn save_obj(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?} (OBJ format)", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    // Header comment
    writeln!(writer, "# OBJ file exported by mesh-repair").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Vertices: {}", mesh.vertices.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Faces: {}", mesh.faces.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Check if we have normals
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());

    // Write vertices
    for (i, v) in mesh.vertices.iter().enumerate() {
        // Write position
        writeln!(writer, "v {:.6} {:.6} {:.6}", v.position.x, v.position.y, v.position.z)
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Add debug comment with vertex attributes (tag, offset)
        if v.tag.is_some() || v.offset.is_some() {
            let tag_str = v.tag.map_or("none".to_string(), |z| format!("{}", z));
            let offset_str = v.offset.map_or("none".to_string(), |c| format!("{:.3}", c));
            writeln!(writer, "# v{} tag={} offset={}", i, tag_str, offset_str)
                .map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                })?;
        }
    }

    // Write normals if present
    if has_normals {
        writeln!(writer).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "# Vertex normals").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        for v in &mesh.vertices {
            if let Some(n) = &v.normal {
                writeln!(writer, "vn {:.6} {:.6} {:.6}", n.x, n.y, n.z)
                    .map_err(|e| MeshError::IoWrite {
                        path: path.to_path_buf(),
                        source: e,
                    })?;
            } else {
                // Write zero normal as placeholder to maintain index correspondence
                writeln!(writer, "vn 0 0 0").map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                })?;
            }
        }
    }

    // Write faces
    writeln!(writer).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Faces").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    for face in &mesh.faces {
        // OBJ uses 1-based indexing
        let i0 = face[0] + 1;
        let i1 = face[1] + 1;
        let i2 = face[2] + 1;

        if has_normals {
            // Format: f v1//n1 v2//n2 v3//n3 (no texture coords)
            writeln!(writer, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2)
                .map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                })?;
        } else {
            writeln!(writer, "f {} {} {}", i0, i1, i2).map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;
        }
    }

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?}",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

/// Load mesh from 3MF file.
///
/// 3MF is a ZIP archive containing XML files. The mesh data is in
/// 3D/3dmodel.model as indexed vertices and triangles.
fn load_3mf(path: &Path) -> MeshResult<Mesh> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut archive = zip::ZipArchive::new(file).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: format!("Invalid 3MF archive: {}", e),
    })?;

    // Find the model file (usually 3D/3dmodel.model)
    let model_path = find_3mf_model_path(&mut archive)?;

    let mut model_file = archive.by_name(&model_path).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: format!("Cannot open model file '{}': {}", model_path, e),
    })?;

    let mut xml_content = String::new();
    model_file.read_to_string(&mut xml_content).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    parse_3mf_model(&xml_content, path)
}

/// Find the model file path in a 3MF archive.
fn find_3mf_model_path(archive: &mut zip::ZipArchive<File>) -> MeshResult<String> {
    // Common locations for the model file
    let candidates = ["3D/3dmodel.model", "3d/3dmodel.model", "3D/3DModel.model"];

    for candidate in candidates {
        if archive.by_name(candidate).is_ok() {
            return Ok(candidate.to_string());
        }
    }

    // Search for any .model file
    for i in 0..archive.len() {
        if let Ok(file) = archive.by_index(i) {
            let name = file.name().to_lowercase();
            if name.ends_with(".model") {
                return Ok(file.name().to_string());
            }
        }
    }

    Err(MeshError::ParseError {
        path: std::path::PathBuf::new(),
        details: "No model file found in 3MF archive".to_string(),
    })
}

/// Parse 3MF model XML content.
fn parse_3mf_model(xml: &str, path: &Path) -> MeshResult<Mesh> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut mesh = Mesh::new();
    let mut in_vertices = false;
    let mut in_triangles = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local_name = e.local_name();
                match local_name.as_ref() {
                    b"vertices" => in_vertices = true,
                    b"triangles" => in_triangles = true,
                    b"vertex" if in_vertices => {
                        let mut x = 0.0f64;
                        let mut y = 0.0f64;
                        let mut z = 0.0f64;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"x" => x = value.parse().unwrap_or(0.0),
                                b"y" => y = value.parse().unwrap_or(0.0),
                                b"z" => z = value.parse().unwrap_or(0.0),
                                _ => {}
                            }
                        }
                        mesh.vertices.push(Vertex::from_coords(x, y, z));
                    }
                    b"triangle" if in_triangles => {
                        let mut v1 = 0u32;
                        let mut v2 = 0u32;
                        let mut v3 = 0u32;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"v1" => v1 = value.parse().unwrap_or(0),
                                b"v2" => v2 = value.parse().unwrap_or(0),
                                b"v3" => v3 = value.parse().unwrap_or(0),
                                _ => {}
                            }
                        }
                        mesh.faces.push([v1, v2, v3]);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                match local_name.as_ref() {
                    b"vertices" => in_vertices = false,
                    b"triangles" => in_triangles = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(MeshError::ParseError {
                    path: path.to_path_buf(),
                    details: format!("XML parse error: {}", e),
                });
            }
            _ => {}
        }
    }

    debug!(
        "3MF loaded: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    Ok(mesh)
}

/// Save mesh to 3MF file.
///
/// 3MF is a modern mesh format that:
/// - Preserves vertex indexing exactly (no deduplication issues)
/// - Uses ZIP compression for smaller files
/// - Is widely supported by slicers (PrusaSlicer, Cura, etc.)
/// - Stores units as millimeters by default
pub fn save_3mf(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?} (3MF format)", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Write content types file (required by 3MF spec)
    zip.start_file("[Content_Types].xml", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write relationships file
    zip.start_file("_rels/.rels", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;
    zip.write_all(RELS_XML.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write the model file
    zip.start_file("3D/3dmodel.model", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    let model_xml = generate_3mf_model_xml(mesh);
    zip.write_all(model_xml.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    zip.finish().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?} (3MF)",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

/// Generate 3MF model XML content.
fn generate_3mf_model_xml(mesh: &Mesh) -> String {
    let mut xml = String::with_capacity(mesh.vertices.len() * 50 + mesh.faces.len() * 40);

    // XML header and model element
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
"#);

    // Write vertices
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    xml.push_str("        </vertices>\n        <triangles>\n");

    // Write triangles
    for face in &mesh.faces {
        xml.push_str(&format!(
            "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
            face[0], face[1], face[2]
        ));
    }

    xml.push_str(r#"        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
"#);

    xml
}

/// 3MF Content Types XML (required by spec).
const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

/// 3MF Relationships XML (required by spec).
const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_stl() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".stl").unwrap();

        // ASCII STL with a single triangle
        writeln!(file, "solid test").unwrap();
        writeln!(file, "  facet normal 0 0 1").unwrap();
        writeln!(file, "    outer loop").unwrap();
        writeln!(file, "      vertex 0 0 0").unwrap();
        writeln!(file, "      vertex 100 0 0").unwrap();
        writeln!(file, "      vertex 0 100 0").unwrap();
        writeln!(file, "    endloop").unwrap();
        writeln!(file, "  endfacet").unwrap();
        writeln!(file, "endsolid test").unwrap();

        file
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            MeshFormat::from_path(Path::new("test.stl")),
            Some(MeshFormat::Stl)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.STL")),
            Some(MeshFormat::Stl)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.obj")),
            Some(MeshFormat::Obj)
        );
        assert_eq!(MeshFormat::from_path(Path::new("test.xyz")), None);
    }

    #[test]
    fn test_load_stl() {
        let file = create_test_stl();
        let mesh = load_mesh(file.path()).expect("should load");

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);

        let (min, max) = mesh.bounds().unwrap();
        assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Point3::new(100.0, 100.0, 0.0));
    }

    #[test]
    fn test_save_and_reload_stl() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to temp file
        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_stl(&mesh, file.path()).expect("should save");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload");

        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);
    }

    #[test]
    fn test_save_and_reload_obj() {
        use nalgebra::Vector3;

        // Create a simple mesh with vertex attributes
        let mut mesh = Mesh::new();

        // Add vertices with attributes
        let mut v0 = Vertex::from_coords(0.0, 0.0, 0.0);
        v0.normal = Some(Vector3::new(0.0, 0.0, 1.0));
        v0.tag = Some(1);
        v0.offset = Some(2.5);
        mesh.vertices.push(v0);

        let mut v1 = Vertex::from_coords(10.0, 0.0, 0.0);
        v1.normal = Some(Vector3::new(1.0, 0.0, 0.0));
        v1.tag = Some(2);
        v1.offset = Some(3.0);
        mesh.vertices.push(v1);

        let mut v2 = Vertex::from_coords(0.0, 10.0, 0.0);
        v2.normal = Some(Vector3::new(0.0, 1.0, 0.0));
        v2.tag = Some(3);
        v2.offset = Some(2.0);
        mesh.vertices.push(v2);

        let mut v3 = Vertex::from_coords(0.0, 0.0, 10.0);
        v3.normal = Some(Vector3::new(-1.0, 0.0, 0.0));
        mesh.vertices.push(v3);

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to OBJ
        let file = NamedTempFile::with_suffix(".obj").unwrap();
        save_obj(&mesh, file.path()).expect("should save");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload");

        // OBJ preserves exact vertex count and order
        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);

        // Verify vertex positions are preserved exactly
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
            let pos_diff = (orig.position - loaded.position).norm();
            assert!(
                pos_diff < 1e-5,
                "Vertex {} position mismatch: {:?} vs {:?}",
                i,
                orig.position,
                loaded.position
            );
        }

        // Verify face indices are preserved
        for (i, (orig, loaded)) in mesh.faces.iter().zip(reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "Face {} indices mismatch", i);
        }
    }

    #[test]
    fn test_obj_vertex_index_preservation() {
        // This test specifically verifies that OBJ preserves vertex indices
        // unlike STL which re-orders vertices during save/load

        let mut mesh = Mesh::new();

        // Create vertices in a specific order
        for i in 0..10 {
            mesh.vertices.push(Vertex::from_coords(
                i as f64 * 10.0,
                (i % 3) as f64 * 5.0,
                (i / 3) as f64 * 7.0,
            ));
        }

        // Create some faces referencing specific vertices
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 4, 5]);
        mesh.faces.push([6, 7, 8]);
        mesh.faces.push([0, 5, 9]);

        // Save and reload OBJ
        let obj_file = NamedTempFile::with_suffix(".obj").unwrap();
        save_obj(&mesh, obj_file.path()).expect("should save obj");
        let obj_reloaded = load_mesh(obj_file.path()).expect("should reload obj");

        // Save and reload STL for comparison
        let stl_file = NamedTempFile::with_suffix(".stl").unwrap();
        save_stl(&mesh, stl_file.path()).expect("should save stl");
        let _stl_reloaded = load_mesh(stl_file.path()).expect("should reload stl");

        // OBJ should preserve exact vertex count
        assert_eq!(
            obj_reloaded.vertex_count(),
            mesh.vertex_count(),
            "OBJ should preserve vertex count"
        );

        // STL may have different vertex count due to deduplication
        // (it duplicates vertices per-triangle, then deduplicates)

        // OBJ should preserve exact face indices
        for (i, (orig, loaded)) in mesh.faces.iter().zip(obj_reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "OBJ face {} indices should match", i);
        }

        // Verify we can track a specific vertex through OBJ save/load
        let target_vertex_idx = 5;
        let orig_pos = mesh.vertices[target_vertex_idx].position;
        let loaded_pos = obj_reloaded.vertices[target_vertex_idx].position;
        let diff = (orig_pos - loaded_pos).norm();
        assert!(
            diff < 1e-5,
            "Vertex {} should be at same index after OBJ reload",
            target_vertex_idx
        );
    }
}
