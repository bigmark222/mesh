//! Error types for shell operations.

use thiserror::Error;

/// Result type alias for shell operations.
pub type ShellResult<T> = Result<T, ShellError>;

/// Errors that can occur during shell operations.
#[derive(Debug, Error)]
pub enum ShellError {
    /// Input mesh is empty.
    #[error("input mesh is empty")]
    EmptyMesh,

    /// SDF grid would be too large.
    #[error("SDF grid too large: {dims:?} = {total} voxels exceeds limit of {max}")]
    GridTooLarge {
        dims: [usize; 3],
        total: usize,
        max: usize,
    },

    /// Isosurface extraction failed.
    #[error("isosurface extraction produced empty mesh")]
    EmptyIsosurface,

    /// Tag transfer failed.
    #[error("failed to transfer vertex tags: {details}")]
    TagTransferFailed { details: String },

    /// Shell generation failed.
    #[error("shell generation failed: {details}")]
    ShellGenerationFailed { details: String },
}
