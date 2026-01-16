//! Mesh repair operations: degenerate removal, welding, compaction.

use hashbrown::{HashMap, HashSet};
use nalgebra::Point3;
use tracing::{debug, info};

use crate::adjacency::MeshAdjacency;
use crate::error::MeshResult;
use crate::{Mesh, Triangle};

/// Remove triangles with area below threshold.
///
/// Returns the number of triangles removed.
pub fn remove_degenerate_triangles(mesh: &mut Mesh, area_threshold: f64) -> usize {
    let original_count = mesh.faces.len();

    mesh.faces.retain(|&[i0, i1, i2]| {
        let tri = Triangle::new(
            mesh.vertices[i0 as usize].position,
            mesh.vertices[i1 as usize].position,
            mesh.vertices[i2 as usize].position,
        );
        tri.area() >= area_threshold
    });

    let removed = original_count - mesh.faces.len();
    if removed > 0 {
        info!("Removed {} degenerate triangles (area < {:.6})", removed, area_threshold);
    }
    removed
}

/// Weld vertices that are within epsilon distance of each other.
///
/// Uses spatial hashing for efficiency. Returns the number of vertices merged.
pub fn weld_vertices(mesh: &mut Mesh, epsilon: f64) -> usize {
    let original_count = mesh.vertices.len();
    if original_count == 0 {
        return 0;
    }

    // Cell size for spatial hashing (2x epsilon as recommended)
    let cell_size = epsilon * 2.0;

    // Build spatial hash: cell -> list of vertex indices
    let mut spatial_hash: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();

    for (idx, vertex) in mesh.vertices.iter().enumerate() {
        let cell = pos_to_cell(&vertex.position, cell_size);
        spatial_hash.entry(cell).or_default().push(idx as u32);
    }

    // For each vertex, find its canonical representative (smallest index in cluster)
    let mut vertex_remap: Vec<u32> = (0..mesh.vertices.len() as u32).collect();
    let mut merged_count = 0;

    for (idx, vertex) in mesh.vertices.iter().enumerate() {
        let idx = idx as u32;
        if vertex_remap[idx as usize] != idx {
            // Already merged into another vertex
            continue;
        }

        let cell = pos_to_cell(&vertex.position, cell_size);

        // Check 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);

                    if let Some(candidates) = spatial_hash.get(&neighbor_cell) {
                        for &other_idx in candidates {
                            if other_idx <= idx {
                                continue; // Only merge into smaller indices
                            }
                            if vertex_remap[other_idx as usize] != other_idx {
                                continue; // Already merged
                            }

                            let other_pos = &mesh.vertices[other_idx as usize].position;
                            let dist = (vertex.position - other_pos).norm();

                            if dist < epsilon {
                                vertex_remap[other_idx as usize] = idx;
                                merged_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if merged_count == 0 {
        return 0;
    }

    // Resolve transitive merges (A->B, B->C => A->C)
    for i in 0..vertex_remap.len() {
        let mut target = vertex_remap[i];
        while vertex_remap[target as usize] != target {
            target = vertex_remap[target as usize];
        }
        vertex_remap[i] = target;
    }

    // Remap face indices
    for face in &mut mesh.faces {
        face[0] = vertex_remap[face[0] as usize];
        face[1] = vertex_remap[face[1] as usize];
        face[2] = vertex_remap[face[2] as usize];
    }

    // Remove faces that became degenerate after welding
    mesh.faces.retain(|&[i0, i1, i2]| i0 != i1 && i1 != i2 && i0 != i2);

    info!(
        "Welded {} vertices (epsilon = {:.3}): {} → {}",
        merged_count,
        epsilon,
        original_count,
        original_count - merged_count
    );

    merged_count
}

/// Remove unreferenced vertices and compact the vertex array.
///
/// Returns the number of vertices removed.
pub fn remove_unreferenced_vertices(mesh: &mut Mesh) -> usize {
    let original_count = mesh.vertices.len();

    // Find all referenced vertices
    let mut referenced: HashSet<u32> = HashSet::new();
    for face in &mesh.faces {
        referenced.insert(face[0]);
        referenced.insert(face[1]);
        referenced.insert(face[2]);
    }

    if referenced.len() == original_count {
        return 0; // All vertices are referenced
    }

    // Build compacted vertex list and remap
    let mut new_vertices = Vec::with_capacity(referenced.len());
    let mut remap: HashMap<u32, u32> = HashMap::new();

    for (old_idx, vertex) in mesh.vertices.iter().enumerate() {
        if referenced.contains(&(old_idx as u32)) {
            let new_idx = new_vertices.len() as u32;
            remap.insert(old_idx as u32, new_idx);
            new_vertices.push(vertex.clone());
        }
    }

    // Remap face indices
    for face in &mut mesh.faces {
        face[0] = remap[&face[0]];
        face[1] = remap[&face[1]];
        face[2] = remap[&face[2]];
    }

    let removed = original_count - new_vertices.len();
    mesh.vertices = new_vertices;

    if removed > 0 {
        info!("Removed {} unreferenced vertices", removed);
    }

    removed
}

/// Compute vertex normals as area-weighted average of adjacent face normals.
pub fn compute_vertex_normals(mesh: &mut Mesh) {
    // Reset all normals
    for vertex in &mut mesh.vertices {
        vertex.normal = None;
    }

    // Accumulate face normals weighted by area
    let mut normal_accum: Vec<nalgebra::Vector3<f64>> =
        vec![nalgebra::Vector3::zeros(); mesh.vertices.len()];

    for face in &mesh.faces {
        let tri = Triangle::new(
            mesh.vertices[face[0] as usize].position,
            mesh.vertices[face[1] as usize].position,
            mesh.vertices[face[2] as usize].position,
        );

        // Use unnormalized normal (length = 2*area) for area weighting
        let weighted_normal = tri.normal_unnormalized();

        normal_accum[face[0] as usize] += weighted_normal;
        normal_accum[face[1] as usize] += weighted_normal;
        normal_accum[face[2] as usize] += weighted_normal;
    }

    // Normalize and assign
    for (idx, accum) in normal_accum.into_iter().enumerate() {
        let len_sq = accum.norm_squared();
        if len_sq > f64::EPSILON {
            mesh.vertices[idx].normal = Some(accum / len_sq.sqrt());
        }
    }

    debug!("Computed vertex normals for {} vertices", mesh.vertices.len());
}

/// Convert position to spatial hash cell.
fn pos_to_cell(pos: &Point3<f64>, cell_size: f64) -> (i64, i64, i64) {
    (
        (pos.x / cell_size).floor() as i64,
        (pos.y / cell_size).floor() as i64,
        (pos.z / cell_size).floor() as i64,
    )
}

/// Remove duplicate faces from the mesh.
///
/// Faces are considered duplicate if they have the same set of vertices
/// (regardless of winding order or starting vertex). This function removes
/// all copies except the first occurrence.
///
/// Returns the number of duplicate faces removed.
pub fn remove_duplicate_faces(mesh: &mut Mesh) -> usize {
    let original_count = mesh.faces.len();

    // Normalize face to smallest vertex first, maintaining cyclic order
    fn normalize_face(face: [u32; 3]) -> [u32; 3] {
        let mut min_idx = 0;
        for i in 1..3 {
            if face[i] < face[min_idx] {
                min_idx = i;
            }
        }
        [
            face[min_idx],
            face[(min_idx + 1) % 3],
            face[(min_idx + 2) % 3],
        ]
    }

    let mut seen: HashSet<[u32; 3]> = HashSet::new();
    let mut duplicate_indices: HashSet<usize> = HashSet::new();

    for (i, face) in mesh.faces.iter().enumerate() {
        let fwd = normalize_face(*face);
        let rev = normalize_face([face[0], face[2], face[1]]);

        // Check if we've seen this face (either winding direction)
        if seen.contains(&fwd) || seen.contains(&rev) {
            duplicate_indices.insert(i);
        } else {
            seen.insert(fwd);
        }
    }

    if duplicate_indices.is_empty() {
        return 0;
    }

    // Remove duplicates by retaining only non-duplicate faces
    let mut idx = 0;
    mesh.faces.retain(|_| {
        let keep = !duplicate_indices.contains(&idx);
        idx += 1;
        keep
    });

    let removed = original_count - mesh.faces.len();
    if removed > 0 {
        info!("Removed {} duplicate faces", removed);
    }

    removed
}

/// Fix non-manifold edges by removing excess faces.
///
/// Non-manifold edges are edges shared by more than 2 faces. For each such edge,
/// this function keeps the 2 largest-area faces and removes the rest.
///
/// Returns the number of faces removed.
pub fn fix_non_manifold_edges(mesh: &mut Mesh) -> usize {
    let adjacency = MeshAdjacency::build(&mesh.faces);
    let nm_edges: Vec<(u32, u32)> = adjacency.non_manifold_edges().collect();

    if nm_edges.is_empty() {
        return 0;
    }

    debug!("Found {} non-manifold edges to fix", nm_edges.len());

    let mut faces_to_remove: HashSet<usize> = HashSet::new();

    for (v0, v1) in &nm_edges {
        // Find all faces sharing this edge
        let mut faces_with_edge: Vec<(usize, f64)> = Vec::new();

        for (fi, face) in mesh.faces.iter().enumerate() {
            let has_v0 = face.contains(v0);
            let has_v1 = face.contains(v1);
            if has_v0 && has_v1 {
                // Compute area
                let p0 = mesh.vertices[face[0] as usize].position;
                let p1 = mesh.vertices[face[1] as usize].position;
                let p2 = mesh.vertices[face[2] as usize].position;
                let tri = Triangle::new(p0, p1, p2);
                let area = tri.area();
                faces_with_edge.push((fi, area));
            }
        }

        if faces_with_edge.len() <= 2 {
            continue; // Not actually non-manifold
        }

        // Sort by area descending (keep largest)
        faces_with_edge.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mark all but the 2 largest for removal
        for (fi, _area) in faces_with_edge.iter().skip(2) {
            faces_to_remove.insert(*fi);
        }
    }

    if faces_to_remove.is_empty() {
        return 0;
    }

    let removed_count = faces_to_remove.len();

    // Remove marked faces
    let mut idx = 0;
    mesh.faces.retain(|_| {
        let keep = !faces_to_remove.contains(&idx);
        idx += 1;
        keep
    });

    info!(
        "Fixed {} non-manifold edges by removing {} faces",
        nm_edges.len(),
        removed_count
    );

    removed_count
}

/// Fix inverted triangles by flipping their winding order.
///
/// Compares each face's normal against its original normal direction.
/// If the normal has flipped (due to offset or other operations), the face
/// winding is corrected by swapping indices 1 and 2.
///
/// # Arguments
/// * `mesh` - The mesh to fix (modified in place)
/// * `original` - The original mesh to compare against (same topology expected)
///
/// # Returns
/// The number of faces that were flipped.
pub fn fix_inverted_faces(mesh: &mut Mesh, original: &Mesh) -> usize {
    if mesh.faces.len() != original.faces.len() {
        // Topology mismatch - can't fix
        return 0;
    }

    let mut flipped_count = 0;

    for (i, face) in mesh.faces.iter_mut().enumerate() {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let new_normal = e1.cross(&e2);

        // Skip degenerate faces
        if new_normal.norm_squared() < 1e-20 {
            continue;
        }

        let orig_face = &original.faces[i];
        let ov0 = &original.vertices[orig_face[0] as usize].position;
        let ov1 = &original.vertices[orig_face[1] as usize].position;
        let ov2 = &original.vertices[orig_face[2] as usize].position;

        let orig_e1 = ov1 - ov0;
        let orig_e2 = ov2 - ov0;
        let orig_normal = orig_e1.cross(&orig_e2);

        // Skip degenerate original faces
        if orig_normal.norm_squared() < 1e-20 {
            continue;
        }

        // If normals point opposite directions, face is inverted
        if new_normal.dot(&orig_normal) < 0.0 {
            // Flip winding by swapping indices 1 and 2
            face.swap(1, 2);
            flipped_count += 1;
        }
    }

    if flipped_count > 0 {
        info!("Fixed {} inverted faces by flipping winding order", flipped_count);
    }

    flipped_count
}

/// Run the full repair pipeline on a mesh.
///
/// Repairs in order:
/// 1. Remove degenerate triangles
/// 2. Weld vertices
/// 3. Remove duplicate faces (introduced by welding)
/// 4. Fix non-manifold edges (remove smallest faces causing >2 faces per edge)
/// 5. Remove unreferenced vertices
/// 6. Compute vertex normals
pub fn repair_mesh(mesh: &mut Mesh) -> MeshResult<()> {
    repair_mesh_with_params(mesh, 0.001, 1e-6)
}

/// Run the full repair pipeline on a mesh with custom parameters.
///
/// # Arguments
/// * `mesh` - The mesh to repair
/// * `weld_epsilon` - Distance threshold for welding vertices
/// * `degenerate_threshold` - Area threshold for removing degenerate triangles
pub fn repair_mesh_with_params(
    mesh: &mut Mesh,
    weld_epsilon: f64,
    degenerate_threshold: f64,
) -> MeshResult<()> {
    info!("Starting mesh repair pipeline");

    let initial_verts = mesh.vertex_count();
    let initial_faces = mesh.face_count();

    // 1. Remove degenerate triangles
    remove_degenerate_triangles(mesh, degenerate_threshold);

    // 2. Weld vertices
    weld_vertices(mesh, weld_epsilon);

    // 3. Remove duplicate faces (welding can create duplicates when nearby triangles merge)
    remove_duplicate_faces(mesh);

    // 4. Fix non-manifold edges (welding can create edges shared by >2 faces)
    fix_non_manifold_edges(mesh);

    // 5. Remove unreferenced vertices
    remove_unreferenced_vertices(mesh);

    // 6. Compute vertex normals
    compute_vertex_normals(mesh);

    info!(
        "Repair complete: {} verts → {}, {} faces → {}",
        initial_verts,
        mesh.vertex_count(),
        initial_faces,
        mesh.face_count()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;
    use approx::assert_relative_eq;

    fn simple_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_remove_degenerate_triangles() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Normal triangle
        mesh.faces.push([0, 1, 2]);
        // Degenerate triangle (collinear points)
        mesh.vertices.push(Vertex::from_coords(5.0, 0.0, 0.0));
        mesh.faces.push([0, 1, 3]); // This has zero area

        let removed = remove_degenerate_triangles(&mut mesh, 0.0001);
        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_weld_vertices() {
        let mut mesh = Mesh::new();
        // Two triangles with nearly-coincident vertices
        // 5 vertices total, vertex 3 is a near-duplicate of vertex 1
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));     // 0
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));    // 1
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));    // 2
        mesh.vertices.push(Vertex::from_coords(10.001, 0.0, 0.0));  // 3 (near-duplicate of 1)
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));   // 4

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 2, 4]); // Uses near-duplicate

        let merged = weld_vertices(&mut mesh, 0.01);
        assert_eq!(merged, 1);

        // After merging, vertex 3 should be remapped to vertex 1
        // So all face indices should be valid (< 5, the original vertex count)
        // and the second face should now reference vertex 1 instead of 3
        assert!(mesh.faces.iter().all(|f| f[0] <= 4 && f[1] <= 4 && f[2] <= 4));

        // Second face should have been remapped: [3, 2, 4] -> [1, 2, 4]
        assert_eq!(mesh.faces[1][0], 1); // Vertex 3 was merged into vertex 1
    }

    #[test]
    fn test_remove_unreferenced() {
        let mut mesh = simple_mesh();
        // Add unreferenced vertex
        mesh.vertices.push(Vertex::from_coords(100.0, 100.0, 100.0));

        let removed = remove_unreferenced_vertices(&mut mesh);
        assert_eq!(removed, 1);
        assert_eq!(mesh.vertex_count(), 3);
    }

    #[test]
    fn test_compute_vertex_normals() {
        // Triangle in XY plane
        let mut mesh = simple_mesh();
        compute_vertex_normals(&mut mesh);

        // All vertices should have normal pointing in +Z
        for v in &mesh.vertices {
            let n = v.normal.expect("should have normal");
            assert_relative_eq!(n.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(n.y, 0.0, epsilon = 1e-10);
            assert_relative_eq!(n.z, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fix_inverted_faces() {
        // Create a simple triangle in XY plane
        let mut original = Mesh::new();
        original.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        original.faces.push([0, 1, 2]); // CCW winding, normal points +Z

        // Create an "offset" mesh with inverted winding (simulates bad offset)
        let mut mesh = original.clone();
        // Swap vertices 1 and 2 to invert the winding
        mesh.faces[0] = [0, 2, 1]; // Now CW winding, normal points -Z

        // The face should be inverted (dot product of normals < 0)
        let v0 = &mesh.vertices[mesh.faces[0][0] as usize].position;
        let v1 = &mesh.vertices[mesh.faces[0][1] as usize].position;
        let v2 = &mesh.vertices[mesh.faces[0][2] as usize].position;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let new_normal = e1.cross(&e2);

        let ov0 = &original.vertices[original.faces[0][0] as usize].position;
        let ov1 = &original.vertices[original.faces[0][1] as usize].position;
        let ov2 = &original.vertices[original.faces[0][2] as usize].position;
        let orig_e1 = ov1 - ov0;
        let orig_e2 = ov2 - ov0;
        let orig_normal = orig_e1.cross(&orig_e2);

        // Verify the face is inverted before fix
        assert!(new_normal.dot(&orig_normal) < 0.0, "Face should be inverted before fix");

        // Fix the inverted face
        let fixed_count = fix_inverted_faces(&mut mesh, &original);
        assert_eq!(fixed_count, 1, "Should fix 1 face");

        // After fix, the face should have correct winding
        assert_eq!(mesh.faces[0], [0, 1, 2], "Face should be restored to original winding");
    }

    #[test]
    fn test_fix_inverted_faces_no_change_needed() {
        // Create a simple triangle in XY plane
        let mut original = Mesh::new();
        original.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        original.faces.push([0, 1, 2]);

        // Create a mesh with same winding (no inversion)
        let mut mesh = original.clone();

        // No fix should be needed
        let fixed_count = fix_inverted_faces(&mut mesh, &original);
        assert_eq!(fixed_count, 0, "Should not fix any faces");
        assert_eq!(mesh.faces[0], [0, 1, 2], "Face should remain unchanged");
    }
}
