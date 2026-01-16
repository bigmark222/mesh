//! Shell generation for printable geometry.
//!
//! Transforms the inner surface into a printable shell with walls.

mod generate;
mod rim;

pub use generate::{generate_shell, ShellParams, ShellResult};
