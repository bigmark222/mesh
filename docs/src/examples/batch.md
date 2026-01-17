# Batch Processing Example

Process multiple mesh files efficiently.

## Basic Batch Processing

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline, validate_mesh};
use std::path::Path;

fn process_directory(input_dir: &str, output_dir: &str) -> Result<BatchResult, std::io::Error> {
    let mut results = BatchResult::default();

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Find all STL files
    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "stl").unwrap_or(false) {
            match process_single_file(&path, output_dir) {
                Ok(_) => results.succeeded += 1,
                Err(e) => {
                    eprintln!("Failed to process {:?}: {}", path, e);
                    results.failed += 1;
                }
            }
        }
    }

    Ok(results)
}

fn process_single_file(input: &Path, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mesh = load_mesh(input)?;

    let result = Pipeline::new(mesh)
        .weld_vertices(1e-6)
        .fill_holes()
        .fix_winding()
        .decimate_to_ratio(0.5)
        .finish();

    let output_name = input.file_stem().unwrap().to_string_lossy();
    let output_path = format!("{}/{}_processed.stl", output_dir, output_name);

    save_mesh(&result.mesh, &output_path)?;
    println!("Processed: {:?} -> {}", input, output_path);

    Ok(())
}

#[derive(Default)]
struct BatchResult {
    succeeded: usize,
    failed: usize,
}

fn main() -> Result<(), std::io::Error> {
    let result = process_directory("./input_meshes", "./output_meshes")?;
    println!("\nBatch complete: {} succeeded, {} failed", result.succeeded, result.failed);
    Ok(())
}
```

## Parallel Processing with Rayon

For faster processing of many files:

```rust
use mesh_repair::{load_mesh, save_mesh, Pipeline};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn process_parallel(input_dir: &str, output_dir: &str) {
    std::fs::create_dir_all(output_dir).unwrap();

    // Collect all STL files
    let files: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "stl").unwrap_or(false))
        .collect();

    let total = files.len();
    let processed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);

    // Process in parallel
    files.par_iter().for_each(|path| {
        match process_file(path, output_dir) {
            Ok(_) => {
                let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                println!("[{}/{}] Processed: {:?}", count, total, path.file_name().unwrap());
            }
            Err(e) => {
                failed.fetch_add(1, Ordering::SeqCst);
                eprintln!("Failed: {:?} - {}", path.file_name().unwrap(), e);
            }
        }
    });

    println!("\nComplete: {} processed, {} failed",
        processed.load(Ordering::SeqCst),
        failed.load(Ordering::SeqCst));
}

fn process_file(input: &PathBuf, output_dir: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mesh = load_mesh(input)?;

    let result = Pipeline::new(mesh)
        .weld_vertices(1e-6)
        .fill_holes()
        .fix_winding()
        .finish();

    let output_name = input.file_stem().unwrap().to_string_lossy();
    let output_path = format!("{}/{}_processed.stl", output_dir, output_name);

    save_mesh(&result.mesh, &output_path)?;
    Ok(())
}
```

## Progress Reporting

```rust
use indicatif::{ProgressBar, ProgressStyle};

fn process_with_progress(files: &[PathBuf], output_dir: &str) {
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    for file in files {
        if let Err(e) = process_file(file, output_dir) {
            pb.println(format!("Error: {:?} - {}", file, e));
        }
        pb.inc(1);
    }

    pb.finish_with_message("Batch complete");
}
```

## Batch Validation Report

```rust
use mesh_repair::{load_mesh, validate_mesh};
use std::path::Path;

#[derive(Default)]
struct BatchValidation {
    total: usize,
    manifold: usize,
    watertight: usize,
    printable: usize,
}

fn validate_directory(dir: &str) -> std::io::Result<BatchValidation> {
    let mut stats = BatchValidation::default();

    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map(|e| e == "stl").unwrap_or(false) {
            if let Ok(mesh) = load_mesh(&path) {
                let report = validate_mesh(&mesh);
                stats.total += 1;
                if report.is_manifold { stats.manifold += 1; }
                if report.is_watertight { stats.watertight += 1; }
                if report.is_printable() { stats.printable += 1; }
            }
        }
    }

    println!("Validation Report:");
    println!("  Total files: {}", stats.total);
    println!("  Manifold: {} ({:.1}%)", stats.manifold, 100.0 * stats.manifold as f64 / stats.total as f64);
    println!("  Watertight: {} ({:.1}%)", stats.watertight, 100.0 * stats.watertight as f64 / stats.total as f64);
    println!("  Printable: {} ({:.1}%)", stats.printable, 100.0 * stats.printable as f64 / stats.total as f64);

    Ok(stats)
}
```

## CLI Batch Processing

```bash
# Process all STLs in directory
for f in input/*.stl; do
    mesh-cli repair "$f" -o "output/$(basename "$f" .stl)_fixed.stl"
done

# Parallel with GNU parallel
find input -name "*.stl" | parallel mesh-cli repair {} -o output/{/.}_fixed.stl

# Validate all files
for f in output/*.stl; do
    echo "=== $f ==="
    mesh-cli validate "$f"
done
```
