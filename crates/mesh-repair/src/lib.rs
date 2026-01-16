//! Triangle mesh repair utilities.
//!
//! This crate provides tools for loading, validating, and repairing triangle meshes.
//! Common operations include:
//!
//! - **Loading/Saving**: STL, OBJ, and 3MF formats
//! - **Validation**: Check for issues like non-manifold edges, holes, inconsistent winding
//! - **Repair**: Fill holes, fix winding, remove degenerate triangles, weld vertices
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::Mesh;
//!
//! // Load a mesh
//! let mut mesh = Mesh::load("model.stl").unwrap();
//!
//! // Validate and report issues
//! let report = mesh.validate();
//! println!("{}", report);
//!
//! // Repair common issues
//! mesh.repair().unwrap();
//!
//! // Save repaired mesh
//! mesh.save("repaired.3mf").unwrap();
//! ```

mod error;
mod types;

pub mod adjacency;
pub mod holes;
pub mod io;
pub mod repair;
pub mod validate;
pub mod winding;

// Re-export core types at crate root
pub use error::{MeshError, MeshResult};
pub use types::{Mesh, Triangle, Vertex};

// Re-export adjacency at crate root for convenience
pub use adjacency::MeshAdjacency;

// Re-export commonly used functions
pub use io::{load_mesh, save_mesh, save_stl, save_obj, save_3mf, MeshFormat};
pub use repair::{compute_vertex_normals, fix_inverted_faces, remove_duplicate_faces, fix_non_manifold_edges};
pub use validate::{validate_mesh, MeshReport};
pub use holes::{fill_holes, fill_holes_with_max_edges, detect_holes, BoundaryLoop};
pub use winding::fix_winding_order;

// Convenience methods on Mesh
impl Mesh {
    /// Load a mesh from a file, auto-detecting format from extension.
    pub fn load(path: impl AsRef<std::path::Path>) -> MeshResult<Self> {
        io::load_mesh(path.as_ref())
    }

    /// Save the mesh to a file, auto-detecting format from extension.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> MeshResult<()> {
        io::save_mesh(self, path.as_ref())
    }

    /// Validate the mesh and return a report of any issues.
    pub fn validate(&self) -> MeshReport {
        validate::validate_mesh(self)
    }

    /// Repair common mesh issues (holes, inconsistent winding, etc).
    pub fn repair(&mut self) -> MeshResult<()> {
        repair::repair_mesh(self)
    }

    /// Compute vertex normals from face normals (area-weighted average).
    pub fn compute_normals(&mut self) {
        repair::compute_vertex_normals(self)
    }

    /// Fix inconsistent face winding to ensure all faces have consistent orientation.
    pub fn fix_winding(&mut self) -> MeshResult<()> {
        winding::fix_winding_order(self)
    }

    /// Fill holes in the mesh.
    pub fn fill_holes(&mut self) -> MeshResult<usize> {
        holes::fill_holes(self)
    }
}
