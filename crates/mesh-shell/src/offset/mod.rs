//! SDF-based mesh offset.
//!
//! This module provides the core SDF offset functionality for creating
//! offset surfaces without self-intersections.

mod grid;
mod extract;
mod transfer;
mod sdf;

pub use sdf::{apply_sdf_offset, SdfOffsetResult, SdfOffsetStats};
pub use grid::SdfOffsetParams;
