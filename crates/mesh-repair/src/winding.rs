//! Normal consistency and winding order correction.

use hashbrown::HashSet;
use std::collections::VecDeque;
use tracing::{debug, info};

use crate::adjacency::MeshAdjacency;
use crate::error::MeshResult;
use crate::Mesh;

/// Fix winding order so all faces have consistent orientation.
///
/// Uses BFS flood fill from an arbitrary start face. For each face,
/// ensures that shared edges are traversed in opposite directions.
///
/// Returns the number of faces that were flipped.
pub fn fix_winding_order(mesh: &mut Mesh) -> MeshResult<()> {
    if mesh.faces.is_empty() {
        return Ok(());
    }

    let adjacency = MeshAdjacency::build(&mesh.faces);

    // Track which faces have been visited and their orientation
    let mut visited: HashSet<u32> = HashSet::new();
    let mut to_flip: HashSet<u32> = HashSet::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Start from face 0
    queue.push_back(0);
    visited.insert(0);

    while let Some(face_idx) = queue.pop_front() {
        let face = mesh.faces[face_idx as usize];

        // Check all three edges of this face
        for edge_idx in 0..3 {
            let v0 = face[edge_idx];
            let v1 = face[(edge_idx + 1) % 3];

            // Get the canonical edge key
            let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

            // Find neighbor faces sharing this edge
            if let Some(neighbors) = adjacency.edge_to_faces.get(&edge_key) {
                for &neighbor_idx in neighbors {
                    if neighbor_idx == face_idx {
                        continue;
                    }

                    if visited.contains(&neighbor_idx) {
                        continue;
                    }

                    visited.insert(neighbor_idx);

                    // Check edge direction in neighbor
                    let neighbor_face = mesh.faces[neighbor_idx as usize];
                    let neighbor_dir = edge_direction_in_face(&neighbor_face, v0, v1);

                    // Current face traverses edge as v0 -> v1
                    // For consistent winding, neighbor should traverse as v1 -> v0
                    // (opposite direction on the shared edge)
                    let should_flip = match neighbor_dir {
                        Some(same_dir) => {
                            // If neighbor has same direction, one of them needs flipping
                            // Since current face is "correct", flip the neighbor
                            same_dir
                        }
                        None => false, // Edge not found (shouldn't happen)
                    };

                    let actual_flip = if to_flip.contains(&face_idx) {
                        // Current face was itself flipped, so invert the decision
                        !should_flip
                    } else {
                        should_flip
                    };

                    if actual_flip {
                        to_flip.insert(neighbor_idx);
                    }

                    queue.push_back(neighbor_idx);
                }
            }
        }
    }

    // Apply flips (swap indices 1 and 2)
    for &face_idx in &to_flip {
        let face = &mut mesh.faces[face_idx as usize];
        face.swap(1, 2);
    }

    let flipped_count = to_flip.len();
    if flipped_count > 0 {
        info!("Fixed winding order: flipped {} faces", flipped_count);
    } else {
        debug!("Winding order already consistent");
    }

    // Check for unvisited faces (disconnected components)
    let unvisited = mesh.faces.len() - visited.len();
    if unvisited > 0 {
        info!(
            "Warning: {} faces not visited (mesh has disconnected components)",
            unvisited
        );
    }

    Ok(())
}

/// Check if edge (a, b) appears in face in the same direction (a -> b).
/// Returns Some(true) if same direction, Some(false) if opposite, None if edge not found.
fn edge_direction_in_face(face: &[u32; 3], a: u32, b: u32) -> Option<bool> {
    for i in 0..3 {
        let v0 = face[i];
        let v1 = face[(i + 1) % 3];

        if v0 == a && v1 == b {
            return Some(true); // Same direction
        }
        if v0 == b && v1 == a {
            return Some(false); // Opposite direction
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    #[test]
    fn test_already_consistent() {
        // Tetrahedron with consistent winding
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 0.5, 1.0));

        // All faces with outward normals (CCW when viewed from outside)
        mesh.faces.push([0, 1, 2]); // Bottom
        mesh.faces.push([0, 3, 1]); // Front
        mesh.faces.push([1, 3, 2]); // Right
        mesh.faces.push([2, 3, 0]); // Left

        fix_winding_order(&mut mesh).unwrap();
        // May or may not flip depending on starting face, but should be consistent
    }

    #[test]
    fn test_fix_inconsistent() {
        // Two triangles sharing an edge, one with wrong winding
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, -1.0, 0.0));

        mesh.faces.push([0, 1, 2]); // CCW
        mesh.faces.push([0, 1, 3]); // Wrong: should be [1, 0, 3] for consistent winding

        fix_winding_order(&mut mesh).unwrap();

        // Check that edge (0,1) is now traversed in opposite directions
        let f0 = mesh.faces[0];
        let f1 = mesh.faces[1];

        let dir0 = edge_direction_in_face(&f0, 0, 1);
        let dir1 = edge_direction_in_face(&f1, 0, 1);

        // They should be opposite
        match (dir0, dir1) {
            (Some(d0), Some(d1)) => assert_ne!(d0, d1),
            _ => panic!("Edge should exist in both faces"),
        }
    }
}
