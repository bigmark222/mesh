//! Error types for mesh operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for mesh operations.
pub type MeshResult<T> = Result<T, MeshError>;

/// Errors that can occur during mesh operations.
#[derive(Debug, Error)]
pub enum MeshError {
    /// Error reading from a file.
    #[error("failed to read mesh from {path}: {source}")]
    IoRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error writing to a file.
    #[error("failed to write mesh to {path}: {source}")]
    IoWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error parsing mesh file format.
    #[error("failed to parse mesh from {path}: {details}")]
    ParseError { path: PathBuf, details: String },

    /// Unsupported file format.
    #[error("unsupported mesh format: {extension:?}")]
    UnsupportedFormat { extension: Option<String> },

    /// Empty mesh (no vertices or faces).
    #[error("mesh is empty: {details}")]
    EmptyMesh { details: String },

    /// Invalid mesh topology.
    #[error("invalid mesh topology: {details}")]
    InvalidTopology { details: String },

    /// Mesh repair failed.
    #[error("mesh repair failed: {details}")]
    RepairFailed { details: String },
}
