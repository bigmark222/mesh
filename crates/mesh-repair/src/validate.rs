//! Mesh validation and reporting.

use nalgebra::Point3;
use tracing::{debug, info, warn};

use crate::adjacency::MeshAdjacency;
use crate::Mesh;

/// Validation report for a mesh.
#[derive(Debug, Clone)]
pub struct MeshReport {
    /// Whether the mesh has no boundary edges.
    pub is_watertight: bool,

    /// Whether all edges have at most 2 adjacent faces.
    pub is_manifold: bool,

    /// Number of boundary edges (edges with 1 adjacent face).
    pub boundary_edge_count: usize,

    /// Number of non-manifold edges (edges with >2 adjacent faces).
    pub non_manifold_edge_count: usize,

    /// Total vertex count.
    pub vertex_count: usize,

    /// Total face count.
    pub face_count: usize,

    /// Bounding box as (min_corner, max_corner).
    pub bounds: Option<(Point3<f64>, Point3<f64>)>,

    /// Dimensions (x, y, z).
    pub dimensions: Option<(f64, f64, f64)>,
}

impl MeshReport {
    /// Check if mesh passes basic validity checks.
    pub fn is_valid(&self) -> bool {
        self.vertex_count > 0 && self.face_count > 0
    }

    /// Check if mesh is suitable for 3D printing.
    pub fn is_printable(&self) -> bool {
        self.is_watertight && self.is_manifold
    }
}

impl std::fmt::Display for MeshReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mesh Report:")?;
        writeln!(f, "  Vertices: {}", self.vertex_count)?;
        writeln!(f, "  Faces: {}", self.face_count)?;

        if let Some((min, max)) = &self.bounds {
            writeln!(
                f,
                "  Bounds: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
                min.x, min.y, min.z, max.x, max.y, max.z
            )?;
        }

        if let Some((dx, dy, dz)) = &self.dimensions {
            writeln!(f, "  Dimensions: {:.1} x {:.1} x {:.1}", dx, dy, dz)?;
        }

        writeln!(
            f,
            "  Watertight: {} (boundary edges: {})",
            if self.is_watertight { "yes" } else { "NO" },
            self.boundary_edge_count
        )?;

        writeln!(
            f,
            "  Manifold: {} (non-manifold edges: {})",
            if self.is_manifold { "yes" } else { "NO" },
            self.non_manifold_edge_count
        )?;

        writeln!(
            f,
            "  Printable: {}",
            if self.is_printable() { "yes" } else { "NO" }
        )?;

        Ok(())
    }
}

/// Validate a mesh and return a report.
pub fn validate_mesh(mesh: &Mesh) -> MeshReport {
    let adjacency = MeshAdjacency::build(&mesh.faces);

    let boundary_edge_count = adjacency.boundary_edge_count();
    let non_manifold_edge_count = adjacency.non_manifold_edge_count();

    let bounds = mesh.bounds();
    let dimensions = bounds.map(|(min, max)| (max.x - min.x, max.y - min.y, max.z - min.z));

    let report = MeshReport {
        is_watertight: boundary_edge_count == 0,
        is_manifold: non_manifold_edge_count == 0,
        boundary_edge_count,
        non_manifold_edge_count,
        vertex_count: mesh.vertex_count(),
        face_count: mesh.face_count(),
        bounds,
        dimensions,
    };

    // Log warnings
    if !report.is_watertight {
        warn!(
            "Mesh is not watertight: {} boundary edges",
            boundary_edge_count
        );
    }

    if !report.is_manifold {
        warn!(
            "Mesh is not manifold: {} non-manifold edges",
            non_manifold_edge_count
        );
    }

    debug!("{}", report);

    report
}

/// Log a summary of mesh validation.
pub fn log_validation(report: &MeshReport) {
    info!(
        "Mesh: {} verts, {} faces, {}x{}x{}",
        report.vertex_count,
        report.face_count,
        report.dimensions.map(|d| format!("{:.1}", d.0)).unwrap_or_default(),
        report.dimensions.map(|d| format!("{:.1}", d.1)).unwrap_or_default(),
        report.dimensions.map(|d| format!("{:.1}", d.2)).unwrap_or_default(),
    );

    if report.is_printable() {
        info!("Mesh is watertight and manifold (printable)");
    } else {
        if !report.is_watertight {
            warn!("Not watertight: {} boundary edges", report.boundary_edge_count);
        }
        if !report.is_manifold {
            warn!("Not manifold: {} non-manifold edges", report.non_manifold_edge_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 0.5, 1.0));

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);
        mesh.faces.push([2, 3, 0]);

        mesh
    }

    fn single_triangle() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_validate_watertight_mesh() {
        let mesh = tetrahedron();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(report.is_watertight);
        assert!(report.is_manifold);
        assert!(report.is_printable());
        assert_eq!(report.boundary_edge_count, 0);
        assert_eq!(report.non_manifold_edge_count, 0);
    }

    #[test]
    fn test_validate_open_mesh() {
        let mesh = single_triangle();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(!report.is_watertight); // Has boundary edges
        assert!(report.is_manifold);    // No edge has >2 faces (manifold allows boundaries)
        assert!(!report.is_printable()); // Not printable because not watertight
        assert_eq!(report.boundary_edge_count, 3);
    }

    #[test]
    fn test_report_display() {
        let mesh = tetrahedron();
        let report = validate_mesh(&mesh);
        let output = format!("{}", report);

        assert!(output.contains("Vertices: 4"));
        assert!(output.contains("Faces: 4"));
        assert!(output.contains("Watertight: yes"));
    }
}
