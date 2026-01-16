//! Mesh topology queries via adjacency structures.

use hashbrown::HashMap;

/// Lightweight topology structure for mesh queries.
///
/// Provides efficient lookups for vertex-to-face and edge-to-face relationships
/// without the overhead of a full half-edge data structure.
#[derive(Debug, Clone)]
pub struct MeshAdjacency {
    /// Maps vertex index → list of face indices that use this vertex.
    pub vertex_to_faces: HashMap<u32, Vec<u32>>,

    /// Maps edge (min_idx, max_idx) → list of face indices that share this edge.
    /// Edge key is always (smaller_index, larger_index) for canonical ordering.
    pub edge_to_faces: HashMap<(u32, u32), Vec<u32>>,
}

impl MeshAdjacency {
    /// Build adjacency structures from a face list.
    pub fn build(faces: &[[u32; 3]]) -> Self {
        let mut vertex_to_faces: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut edge_to_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

        for (face_idx, &[v0, v1, v2]) in faces.iter().enumerate() {
            let face_idx = face_idx as u32;

            // Record vertex → face mappings
            vertex_to_faces.entry(v0).or_default().push(face_idx);
            vertex_to_faces.entry(v1).or_default().push(face_idx);
            vertex_to_faces.entry(v2).or_default().push(face_idx);

            // Record edge → face mappings (canonical edge ordering)
            for &(a, b) in &[(v0, v1), (v1, v2), (v2, v0)] {
                let edge_key = if a < b { (a, b) } else { (b, a) };
                edge_to_faces.entry(edge_key).or_default().push(face_idx);
            }
        }

        Self {
            vertex_to_faces,
            edge_to_faces,
        }
    }

    /// Find boundary edges (edges with exactly 1 adjacent face).
    ///
    /// In a watertight mesh, this returns an empty iterator.
    pub fn boundary_edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.edge_to_faces
            .iter()
            .filter(|(_, faces)| faces.len() == 1)
            .map(|(&edge, _)| edge)
    }

    /// Find non-manifold edges (edges with more than 2 adjacent faces).
    ///
    /// These indicate self-intersections or other topology issues.
    pub fn non_manifold_edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.edge_to_faces
            .iter()
            .filter(|(_, faces)| faces.len() > 2)
            .map(|(&edge, _)| edge)
    }

    /// Check if the mesh is manifold.
    ///
    /// A manifold mesh has at most 2 faces for every edge.
    /// (Edges with 1 face are boundary edges, which is allowed.)
    pub fn is_manifold(&self) -> bool {
        self.edge_to_faces.values().all(|faces| faces.len() <= 2)
    }

    /// Check if the mesh is watertight (no boundary edges).
    ///
    /// A watertight mesh has no edges with only 1 adjacent face.
    pub fn is_watertight(&self) -> bool {
        self.edge_to_faces.values().all(|faces| faces.len() >= 2)
    }

    /// Count boundary edges.
    pub fn boundary_edge_count(&self) -> usize {
        self.edge_to_faces
            .values()
            .filter(|faces| faces.len() == 1)
            .count()
    }

    /// Count non-manifold edges.
    pub fn non_manifold_edge_count(&self) -> usize {
        self.edge_to_faces
            .values()
            .filter(|faces| faces.len() > 2)
            .count()
    }

    /// Get faces adjacent to a vertex.
    pub fn faces_for_vertex(&self, vertex_idx: u32) -> Option<&[u32]> {
        self.vertex_to_faces.get(&vertex_idx).map(|v| v.as_slice())
    }

    /// Get faces adjacent to an edge.
    /// The edge is automatically canonicalized (min, max).
    pub fn faces_for_edge(&self, v0: u32, v1: u32) -> Option<&[u32]> {
        let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.edge_to_faces.get(&edge_key).map(|v| v.as_slice())
    }

    /// Find vertices on boundary edges.
    pub fn boundary_vertices(&self) -> impl Iterator<Item = u32> + '_ {
        self.boundary_edges().flat_map(|(a, b)| [a, b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_triangle() -> Vec<[u32; 3]> {
        vec![[0, 1, 2]]
    }

    fn two_triangles_shared_edge() -> Vec<[u32; 3]> {
        vec![[0, 1, 2], [1, 0, 3]]
    }

    fn tetrahedron() -> Vec<[u32; 3]> {
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    }

    #[test]
    fn test_single_triangle_is_not_watertight() {
        let adj = MeshAdjacency::build(&single_triangle());
        assert!(!adj.is_watertight());
        assert!(adj.is_manifold());
        assert_eq!(adj.boundary_edge_count(), 3);
    }

    #[test]
    fn test_two_triangles_shared_edge() {
        let adj = MeshAdjacency::build(&two_triangles_shared_edge());
        let shared = adj.faces_for_edge(0, 1).expect("edge exists");
        assert_eq!(shared.len(), 2);
        assert_eq!(adj.boundary_edge_count(), 4);
    }

    #[test]
    fn test_tetrahedron_is_watertight() {
        let adj = MeshAdjacency::build(&tetrahedron());
        assert!(adj.is_watertight());
        assert!(adj.is_manifold());
        assert_eq!(adj.boundary_edge_count(), 0);
        assert_eq!(adj.non_manifold_edge_count(), 0);
    }

    #[test]
    fn test_vertex_to_faces() {
        let adj = MeshAdjacency::build(&tetrahedron());
        for v in 0..4u32 {
            let faces = adj.faces_for_vertex(v).expect("vertex exists");
            assert_eq!(faces.len(), 3, "vertex {} should touch 3 faces", v);
        }
    }

    #[test]
    fn test_edge_canonicalization() {
        let adj = MeshAdjacency::build(&two_triangles_shared_edge());
        let e1 = adj.faces_for_edge(0, 1);
        let e2 = adj.faces_for_edge(1, 0);
        assert_eq!(e1, e2);
    }
}
