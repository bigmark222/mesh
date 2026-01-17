//! Output formatting utilities for the CLI.

use serde::Serialize;

use crate::OutputFormat;

/// Print output in the specified format.
pub fn print<T: Serialize>(value: &T, format: OutputFormat, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        OutputFormat::Text => {
            // Text output is handled by the caller
        }
        OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string_pretty(value) {
                println!("{}", json);
            }
        }
    }
}

/// Print a success message.
pub fn success(msg: &str, format: OutputFormat, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        OutputFormat::Text => {
            use colored::Colorize;
            println!("{} {}", "✓".green().bold(), msg);
        }
        OutputFormat::Json => {
            // JSON output handles success differently
        }
    }
}

/// Print an info message.
pub fn info(msg: &str, format: OutputFormat, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        OutputFormat::Text => {
            println!("{}", msg);
        }
        OutputFormat::Json => {
            // JSON output handles info differently
        }
    }
}

/// Print a warning message.
pub fn warning(msg: &str, format: OutputFormat, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        OutputFormat::Text => {
            use colored::Colorize;
            eprintln!("{} {}", "⚠".yellow().bold(), msg);
        }
        OutputFormat::Json => {
            // JSON output handles warnings differently
        }
    }
}
