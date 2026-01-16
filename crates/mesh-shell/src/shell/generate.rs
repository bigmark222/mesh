//! Shell generation algorithm.
//!
//! Generates a printable shell from the inner surface.

use tracing::{debug, info};

use mesh_repair::{compute_vertex_normals, Mesh};

use super::rim::generate_rim;

/// Parameters for shell generation.
#[derive(Debug, Clone)]
pub struct ShellParams {
    /// Uniform wall thickness in mm.
    pub wall_thickness_mm: f64,
    /// Minimum acceptable wall thickness.
    pub min_thickness_mm: f64,
}

impl Default for ShellParams {
    fn default() -> Self {
        Self {
            wall_thickness_mm: 2.5,
            min_thickness_mm: 1.5,
        }
    }
}

/// Result of shell generation.
#[derive(Debug)]
pub struct ShellResult {
    /// Number of inner surface vertices.
    pub inner_vertex_count: usize,
    /// Number of outer surface vertices.
    pub outer_vertex_count: usize,
    /// Number of rim faces generated.
    pub rim_face_count: usize,
    /// Total face count.
    pub total_face_count: usize,
    /// Boundary loop size (number of edges).
    pub boundary_size: usize,
}

/// Generate a printable shell from the inner surface.
///
/// Creates outer surface by offsetting along normals, then connects
/// inner and outer at boundaries with a rim.
///
/// # Arguments
/// * `inner_shell` - The inner surface mesh (from offset stage)
/// * `params` - Shell generation parameters
///
/// # Returns
/// A tuple of (shell mesh, generation result).
pub fn generate_shell(inner_shell: &Mesh, params: &ShellParams) -> (Mesh, ShellResult) {
    info!("Generating shell with thickness={:.2}mm", params.wall_thickness_mm);

    let n = inner_shell.vertices.len();
    let mut shell = Mesh::new();

    // Step 1: Copy inner vertices and ensure normals
    let mut inner_with_normals = inner_shell.clone();
    compute_vertex_normals(&mut inner_with_normals);

    // Step 2: Generate outer vertices by offsetting along normals
    for vertex in &inner_with_normals.vertices {
        // Inner vertex (copy directly)
        shell.vertices.push(vertex.clone());
    }

    for vertex in &inner_with_normals.vertices {
        // Outer vertex (offset by wall thickness)
        let normal = vertex.normal.unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0));
        let outer_pos = vertex.position + normal * params.wall_thickness_mm;

        let mut outer_vertex = vertex.clone();
        outer_vertex.position = outer_pos;
        // Keep normal for outer surface (points outward)
        outer_vertex.normal = Some(normal);

        shell.vertices.push(outer_vertex);
    }

    debug!("Generated {} inner + {} outer vertices", n, n);

    // Step 3: Copy inner faces (reversed winding so normal points inward)
    for face in &inner_shell.faces {
        // Reverse winding so normal points inward
        shell.faces.push([face[0], face[2], face[1]]);
    }

    // Step 4: Generate outer faces with offset indices (original winding for outward normals)
    for face in &inner_shell.faces {
        let n32 = n as u32;
        shell.faces.push([face[0] + n32, face[1] + n32, face[2] + n32]);
    }

    let inner_face_count = inner_shell.faces.len();
    debug!("Added {} inner + {} outer faces", inner_face_count, inner_face_count);

    // Step 5: Find boundary edges and generate rim
    let (rim_faces, boundary_size) = generate_rim(&inner_with_normals, n);

    let rim_face_count = rim_faces.len();
    for face in rim_faces {
        shell.faces.push(face);
    }

    info!(
        "Shell generation complete: {} vertices, {} faces",
        shell.vertices.len(),
        shell.faces.len()
    );

    let result = ShellResult {
        inner_vertex_count: n,
        outer_vertex_count: n,
        rim_face_count,
        total_face_count: shell.faces.len(),
        boundary_size,
    };

    (shell, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_open_box() -> Mesh {
        // A box open on top (5 faces instead of 6)
        let mut mesh = Mesh::new();

        // Bottom corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Top corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // Bottom (2 triangles)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Front
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        // Left
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);
        // Top is OPEN - boundary is 4-5-6-7

        mesh
    }

    #[test]
    fn test_shell_params_default() {
        let params = ShellParams::default();
        assert_eq!(params.wall_thickness_mm, 2.5);
        assert_eq!(params.min_thickness_mm, 1.5);
    }

    #[test]
    fn test_generate_shell_doubles_vertices() {
        let inner = create_open_box();
        let params = ShellParams::default();

        let (shell, result) = generate_shell(&inner, &params);

        // Should have 2x vertices (inner + outer)
        assert_eq!(shell.vertices.len(), inner.vertices.len() * 2);
        assert_eq!(result.inner_vertex_count, inner.vertices.len());
        assert_eq!(result.outer_vertex_count, inner.vertices.len());
    }

    #[test]
    fn test_shell_has_more_faces() {
        let inner = create_open_box();
        let params = ShellParams::default();

        let (shell, result) = generate_shell(&inner, &params);

        // Should have inner + outer + rim faces
        assert!(shell.faces.len() > inner.faces.len() * 2);
        assert!(result.rim_face_count > 0);
    }
}
