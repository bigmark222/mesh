//! Rim generation for connecting inner and outer shell surfaces.

use tracing::debug;

use mesh_repair::{Mesh, MeshAdjacency};

/// Generate rim faces to connect inner and outer surfaces at boundaries.
///
/// Returns the rim faces and the total boundary edge count.
pub fn generate_rim(inner_mesh: &Mesh, vertex_offset: usize) -> (Vec<[u32; 3]>, usize) {
    let adjacency = MeshAdjacency::build(&inner_mesh.faces);
    let boundary_edges: Vec<(u32, u32)> = adjacency.boundary_edges().collect();

    if boundary_edges.is_empty() {
        return (Vec::new(), 0);
    }

    debug!("Generating rim for {} boundary edges", boundary_edges.len());

    let mut faces = Vec::new();
    let n = vertex_offset as u32;

    for &(v0, v1) in &boundary_edges {
        // Create two triangles to connect inner edge to outer edge
        // Inner edge: (v0, v1)
        // Outer edge: (v0+n, v1+n)
        //
        // Rim triangles (normals pointing outward):
        // Triangle 1: v1 -> v0 -> v0+n
        // Triangle 2: v1 -> v0+n -> v1+n
        faces.push([v1, v0, v0 + n]);
        faces.push([v1, v0 + n, v1 + n]);
    }

    let boundary_count = boundary_edges.len();
    debug!("Generated {} rim faces", faces.len());

    (faces, boundary_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_open_square() -> Mesh {
        // A simple square (4 vertices, 2 triangles, 4 boundary edges)
        let mut mesh = Mesh::new();

        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);

        mesh
    }

    #[test]
    fn test_generate_rim() {
        let mesh = create_open_square();
        let (rim_faces, boundary_count) = generate_rim(&mesh, 4);

        // 4 boundary edges * 2 triangles per edge = 8 faces
        assert_eq!(boundary_count, 4);
        assert_eq!(rim_faces.len(), 8);
    }
}
