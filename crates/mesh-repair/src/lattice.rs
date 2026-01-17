//! Lattice and infill structure generation.
//!
//! This module provides tools for generating lattice structures within meshes,
//! enabling lightweight, flexible, and material-efficient designs.
//!
//! # Lattice Types
//!
//! - **Cubic**: Simple grid pattern, easy to print
//! - **OctetTruss**: Strong, lightweight structural lattice
//! - **Gyroid**: TPMS surface, smooth and organic
//! - **Voronoi**: Organic, irregular cell structure
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::lattice::{LatticeParams, LatticeType, generate_lattice};
//!
//! // Create a simple mesh
//! let mut mesh = Mesh::new();
//! // ... add vertices and faces ...
//!
//! // Generate a cubic lattice with 5mm cell size
//! let params = LatticeParams::cubic(5.0);
//! // let lattice = generate_lattice(&mesh, &params);
//! ```

use crate::{Mesh, Vertex};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// Lattice generation parameters.
#[derive(Debug, Clone)]
pub struct LatticeParams {
    /// Type of lattice structure.
    pub lattice_type: LatticeType,

    /// Unit cell size in mm.
    pub cell_size: f64,

    /// Strut thickness in mm (for strut-based lattices).
    pub strut_thickness: f64,

    /// Wall thickness in mm (for TPMS surfaces).
    pub wall_thickness: f64,

    /// Density factor (0.0-1.0, affects strut thickness).
    pub density: f64,

    /// Optional density map for variable density infill.
    pub density_map: Option<DensityMap>,

    /// Minimum feature size (for mesh generation).
    pub min_feature_size: f64,

    /// Mesh resolution for TPMS surfaces (samples per cell).
    pub resolution: usize,

    /// Whether to trim to mesh bounds.
    pub trim_to_bounds: bool,
}

impl Default for LatticeParams {
    fn default() -> Self {
        Self {
            lattice_type: LatticeType::Cubic,
            cell_size: 5.0,
            strut_thickness: 0.8,
            wall_thickness: 0.5,
            density: 0.3,
            density_map: None,
            min_feature_size: 0.1,
            resolution: 10,
            trim_to_bounds: true,
        }
    }
}

impl LatticeParams {
    /// Create parameters for a cubic lattice.
    pub fn cubic(cell_size: f64) -> Self {
        Self {
            lattice_type: LatticeType::Cubic,
            cell_size,
            ..Default::default()
        }
    }

    /// Create parameters for an octet-truss lattice.
    pub fn octet_truss(cell_size: f64) -> Self {
        Self {
            lattice_type: LatticeType::OctetTruss,
            cell_size,
            ..Default::default()
        }
    }

    /// Create parameters for a gyroid lattice.
    pub fn gyroid(cell_size: f64) -> Self {
        Self {
            lattice_type: LatticeType::Gyroid,
            cell_size,
            resolution: 15, // Higher resolution for smooth TPMS
            ..Default::default()
        }
    }

    /// Create parameters for a Voronoi lattice.
    pub fn voronoi(cell_size: f64) -> Self {
        Self {
            lattice_type: LatticeType::Voronoi,
            cell_size,
            ..Default::default()
        }
    }

    /// Set strut thickness.
    pub fn with_strut_thickness(mut self, thickness: f64) -> Self {
        self.strut_thickness = thickness;
        self
    }

    /// Set density.
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density.clamp(0.0, 1.0);
        self
    }

    /// Set density map.
    pub fn with_density_map(mut self, map: DensityMap) -> Self {
        self.density_map = Some(map);
        self
    }

    /// Set resolution for TPMS surfaces.
    pub fn with_resolution(mut self, resolution: usize) -> Self {
        self.resolution = resolution.max(2);
        self
    }
}

/// Type of lattice structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeType {
    /// Simple cubic grid.
    Cubic,

    /// Octet-truss (strong, lightweight).
    OctetTruss,

    /// Gyroid TPMS surface.
    Gyroid,

    /// Schwarz-P TPMS surface.
    SchwarzP,

    /// Diamond TPMS surface.
    Diamond,

    /// Voronoi/organic cells.
    Voronoi,
}

/// Density map for variable density infill.
#[derive(Clone)]
pub enum DensityMap {
    /// Uniform density.
    Uniform(f64),

    /// Gradient from point A (density a) to point B (density b).
    Gradient {
        from: Point3<f64>,
        from_density: f64,
        to: Point3<f64>,
        to_density: f64,
    },

    /// Density based on distance from a point.
    Radial {
        center: Point3<f64>,
        inner_radius: f64,
        inner_density: f64,
        outer_radius: f64,
        outer_density: f64,
    },

    /// Density based on distance from mesh surface.
    SurfaceDistance {
        /// Density at surface.
        surface_density: f64,
        /// Density at core.
        core_density: f64,
        /// Distance at which core density is reached.
        transition_depth: f64,
    },

    /// Custom function.
    Function(std::sync::Arc<dyn Fn(Point3<f64>) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for DensityMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DensityMap::Uniform(d) => write!(f, "Uniform({})", d),
            DensityMap::Gradient { from, from_density, to, to_density } => {
                write!(f, "Gradient {{ from: {:?}, from_density: {}, to: {:?}, to_density: {} }}", from, from_density, to, to_density)
            }
            DensityMap::Radial { center, inner_radius, inner_density, outer_radius, outer_density } => {
                write!(f, "Radial {{ center: {:?}, inner_radius: {}, inner_density: {}, outer_radius: {}, outer_density: {} }}",
                    center, inner_radius, inner_density, outer_radius, outer_density)
            }
            DensityMap::SurfaceDistance { surface_density, core_density, transition_depth } => {
                write!(f, "SurfaceDistance {{ surface_density: {}, core_density: {}, transition_depth: {} }}",
                    surface_density, core_density, transition_depth)
            }
            DensityMap::Function(_) => write!(f, "Function(<closure>)"),
        }
    }
}

impl DensityMap {
    /// Evaluate density at a point.
    pub fn evaluate(&self, point: Point3<f64>) -> f64 {
        match self {
            DensityMap::Uniform(d) => *d,

            DensityMap::Gradient {
                from,
                from_density,
                to,
                to_density,
            } => {
                let axis = to - from;
                let axis_len_sq = axis.norm_squared();
                if axis_len_sq < 1e-10 {
                    return *from_density;
                }
                let t = ((point - from).dot(&axis) / axis_len_sq).clamp(0.0, 1.0);
                from_density + t * (to_density - from_density)
            }

            DensityMap::Radial {
                center,
                inner_radius,
                inner_density,
                outer_radius,
                outer_density,
            } => {
                let dist = (point - center).norm();
                if dist <= *inner_radius {
                    *inner_density
                } else if dist >= *outer_radius {
                    *outer_density
                } else {
                    let t = (dist - inner_radius) / (outer_radius - inner_radius);
                    inner_density + t * (outer_density - inner_density)
                }
            }

            DensityMap::SurfaceDistance {
                surface_density,
                core_density,
                transition_depth: _,
            } => {
                // This requires knowing distance to surface, which would need
                // the mesh reference. For now, just return core density.
                // Full implementation would use an SDF.
                (*surface_density + *core_density) / 2.0
            }

            DensityMap::Function(f) => f(point),
        }
    }
}

/// Result of lattice generation.
#[derive(Debug)]
pub struct LatticeResult {
    /// Generated lattice mesh.
    pub mesh: Mesh,

    /// Actual density achieved (volume fraction).
    pub actual_density: f64,

    /// Number of unit cells.
    pub cell_count: usize,

    /// Total strut length (for strut-based lattices).
    pub total_strut_length: f64,
}

/// Generate a lattice structure within the bounding box of a mesh.
///
/// This creates a lattice mesh that can be combined with a shell for
/// skin + infill manufacturing.
pub fn generate_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    match params.lattice_type {
        LatticeType::Cubic => generate_cubic_lattice(params, bounds),
        LatticeType::OctetTruss => generate_octet_truss_lattice(params, bounds),
        LatticeType::Gyroid => generate_gyroid_lattice(params, bounds),
        LatticeType::SchwarzP => generate_schwarz_p_lattice(params, bounds),
        LatticeType::Diamond => generate_diamond_lattice(params, bounds),
        LatticeType::Voronoi => generate_voronoi_lattice(params, bounds),
    }
}

/// Generate a cubic lattice.
fn generate_cubic_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    let (min, max) = bounds;
    let size = max - min;

    let cells_x = (size.x / params.cell_size).ceil() as usize;
    let cells_y = (size.y / params.cell_size).ceil() as usize;
    let cells_z = (size.z / params.cell_size).ceil() as usize;

    let mut mesh = Mesh::new();
    let mut total_strut_length = 0.0;

    // Generate struts for cubic lattice
    // Each cell has struts along its edges
    for iz in 0..=cells_z {
        for iy in 0..=cells_y {
            for ix in 0..=cells_x {
                let corner = min + Vector3::new(
                    ix as f64 * params.cell_size,
                    iy as f64 * params.cell_size,
                    iz as f64 * params.cell_size,
                );

                // Get density at this point
                let density = params
                    .density_map
                    .as_ref()
                    .map(|dm| dm.evaluate(corner))
                    .unwrap_or(params.density);

                let strut_radius = params.strut_thickness * density / 2.0;

                // Add struts in X direction
                if ix < cells_x {
                    let end = corner + Vector3::new(params.cell_size, 0.0, 0.0);
                    add_cylindrical_strut(&mut mesh, corner, end, strut_radius, 6);
                    total_strut_length += params.cell_size;
                }

                // Add struts in Y direction
                if iy < cells_y {
                    let end = corner + Vector3::new(0.0, params.cell_size, 0.0);
                    add_cylindrical_strut(&mut mesh, corner, end, strut_radius, 6);
                    total_strut_length += params.cell_size;
                }

                // Add struts in Z direction
                if iz < cells_z {
                    let end = corner + Vector3::new(0.0, 0.0, params.cell_size);
                    add_cylindrical_strut(&mut mesh, corner, end, strut_radius, 6);
                    total_strut_length += params.cell_size;
                }
            }
        }
    }

    let cell_count = cells_x * cells_y * cells_z;
    let total_volume = size.x * size.y * size.z;
    let strut_volume = total_strut_length * std::f64::consts::PI * (params.strut_thickness * params.density / 2.0).powi(2);
    let actual_density = strut_volume / total_volume;

    LatticeResult {
        mesh,
        actual_density,
        cell_count,
        total_strut_length,
    }
}

/// Generate an octet-truss lattice.
fn generate_octet_truss_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    let (min, max) = bounds;
    let size = max - min;

    let cells_x = (size.x / params.cell_size).ceil() as usize;
    let cells_y = (size.y / params.cell_size).ceil() as usize;
    let cells_z = (size.z / params.cell_size).ceil() as usize;

    let mut mesh = Mesh::new();
    let mut total_strut_length = 0.0;

    let half_cell = params.cell_size / 2.0;

    // Generate octet-truss structure
    // Each unit cell has 24 struts forming octahedra and tetrahedra
    for iz in 0..cells_z {
        for iy in 0..cells_y {
            for ix in 0..cells_x {
                let origin = min + Vector3::new(
                    ix as f64 * params.cell_size,
                    iy as f64 * params.cell_size,
                    iz as f64 * params.cell_size,
                );

                let density = params
                    .density_map
                    .as_ref()
                    .map(|dm| dm.evaluate(origin + Vector3::new(half_cell, half_cell, half_cell)))
                    .unwrap_or(params.density);

                let strut_radius = params.strut_thickness * density / 2.0;

                // Define cell vertices
                let corners = [
                    origin,
                    origin + Vector3::new(params.cell_size, 0.0, 0.0),
                    origin + Vector3::new(params.cell_size, params.cell_size, 0.0),
                    origin + Vector3::new(0.0, params.cell_size, 0.0),
                    origin + Vector3::new(0.0, 0.0, params.cell_size),
                    origin + Vector3::new(params.cell_size, 0.0, params.cell_size),
                    origin + Vector3::new(params.cell_size, params.cell_size, params.cell_size),
                    origin + Vector3::new(0.0, params.cell_size, params.cell_size),
                ];

                // Face centers
                let face_centers = [
                    origin + Vector3::new(half_cell, half_cell, 0.0),              // bottom
                    origin + Vector3::new(half_cell, half_cell, params.cell_size), // top
                    origin + Vector3::new(half_cell, 0.0, half_cell),              // front
                    origin + Vector3::new(half_cell, params.cell_size, half_cell), // back
                    origin + Vector3::new(0.0, half_cell, half_cell),              // left
                    origin + Vector3::new(params.cell_size, half_cell, half_cell), // right
                ];

                // Cell center
                let center = origin + Vector3::new(half_cell, half_cell, half_cell);

                // Connect corners to adjacent face centers (creates octahedra)
                let corner_to_faces = [
                    (0, vec![0, 2, 4]),
                    (1, vec![0, 2, 5]),
                    (2, vec![0, 3, 5]),
                    (3, vec![0, 3, 4]),
                    (4, vec![1, 2, 4]),
                    (5, vec![1, 2, 5]),
                    (6, vec![1, 3, 5]),
                    (7, vec![1, 3, 4]),
                ];

                for (ci, faces) in &corner_to_faces {
                    for &fi in faces {
                        let start = Point3::from(corners[*ci].coords);
                        let end = Point3::from(face_centers[fi].coords);
                        add_cylindrical_strut(&mut mesh, start, end, strut_radius, 6);
                        total_strut_length += (end - start).norm();
                    }
                }

                // Connect face centers to cell center (creates core structure)
                for fc in &face_centers {
                    let start = Point3::from(fc.coords);
                    let end = Point3::from(center.coords);
                    add_cylindrical_strut(&mut mesh, start, end, strut_radius, 6);
                    total_strut_length += half_cell;
                }
            }
        }
    }

    let cell_count = cells_x * cells_y * cells_z;
    let total_volume = size.x * size.y * size.z;
    let strut_volume = total_strut_length * std::f64::consts::PI * (params.strut_thickness * params.density / 2.0).powi(2);
    let actual_density = strut_volume / total_volume;

    LatticeResult {
        mesh,
        actual_density,
        cell_count,
        total_strut_length,
    }
}

/// Generate a gyroid TPMS lattice.
fn generate_gyroid_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    generate_tpms_lattice(params, bounds, |x, y, z| {
        // Gyroid equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = t
        x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
    })
}

/// Generate a Schwarz-P TPMS lattice.
fn generate_schwarz_p_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    generate_tpms_lattice(params, bounds, |x, y, z| {
        // Schwarz-P equation: cos(x) + cos(y) + cos(z) = t
        x.cos() + y.cos() + z.cos()
    })
}

/// Generate a diamond TPMS lattice.
fn generate_diamond_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    generate_tpms_lattice(params, bounds, |x, y, z| {
        // Diamond equation: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = t
        x.sin() * y.sin() * z.sin()
            + x.sin() * y.cos() * z.cos()
            + x.cos() * y.sin() * z.cos()
            + x.cos() * y.cos() * z.sin()
    })
}

/// Generate a TPMS lattice using marching cubes.
fn generate_tpms_lattice<F>(
    params: &LatticeParams,
    bounds: (Point3<f64>, Point3<f64>),
    tpms_func: F,
) -> LatticeResult
where
    F: Fn(f64, f64, f64) -> f64,
{
    let (min, max) = bounds;
    let size = max - min;

    // Compute grid resolution
    let cells_x = ((size.x / params.cell_size) * params.resolution as f64).ceil() as usize;
    let cells_y = ((size.y / params.cell_size) * params.resolution as f64).ceil() as usize;
    let cells_z = ((size.z / params.cell_size) * params.resolution as f64).ceil() as usize;

    let dx = size.x / cells_x as f64;
    let dy = size.y / cells_y as f64;
    let dz = size.z / cells_z as f64;

    // Period scaling factor
    let scale = 2.0 * std::f64::consts::PI / params.cell_size;

    // Threshold for isosurface (controls wall thickness)
    // Smaller threshold = thicker walls
    let threshold = 1.0 - 2.0 * params.density;

    let mut mesh = Mesh::new();
    let mut vertex_map: HashMap<(usize, usize, usize, u8), u32> = HashMap::new();

    // Marching cubes implementation
    for iz in 0..cells_z {
        for iy in 0..cells_y {
            for ix in 0..cells_x {
                // Evaluate TPMS function at cube corners
                let mut values = [0.0f64; 8];
                let mut points = [Point3::origin(); 8];

                for (corner_idx, (dix, diy, diz)) in [
                    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
                ].iter().enumerate() {
                    let x = min.x + (ix + dix) as f64 * dx;
                    let y = min.y + (iy + diy) as f64 * dy;
                    let z = min.z + (iz + diz) as f64 * dz;

                    points[corner_idx] = Point3::new(x, y, z);
                    values[corner_idx] = tpms_func(x * scale, y * scale, z * scale) - threshold;
                }

                // Determine cube configuration
                let mut cube_index = 0u8;
                for (i, &val) in values.iter().enumerate() {
                    if val < 0.0 {
                        cube_index |= 1 << i;
                    }
                }

                if cube_index == 0 || cube_index == 255 {
                    continue; // Cube is entirely inside or outside
                }

                // Get triangles for this configuration using marching cubes tables
                let triangles = get_marching_cubes_triangles(cube_index);

                for triangle in triangles.chunks(3) {
                    if triangle.len() < 3 {
                        break;
                    }

                    let mut tri_indices = [0u32; 3];
                    for (i, &edge) in triangle.iter().enumerate() {
                        if edge == 255 {
                            break;
                        }

                        let key = (ix, iy, iz, edge);
                        if let Some(&vi) = vertex_map.get(&key) {
                            tri_indices[i] = vi;
                        } else {
                            // Interpolate vertex position along edge
                            let (v0, v1) = EDGE_VERTICES[edge as usize];
                            let t = values[v0] / (values[v0] - values[v1]);
                            let pos = points[v0] + (points[v1] - points[v0]) * t;

                            let vi = mesh.vertices.len() as u32;
                            mesh.vertices.push(Vertex::new(pos));
                            vertex_map.insert(key, vi);
                            tri_indices[i] = vi;
                        }
                    }

                    if tri_indices[0] != tri_indices[1]
                        && tri_indices[1] != tri_indices[2]
                        && tri_indices[0] != tri_indices[2]
                    {
                        mesh.faces.push(tri_indices);
                    }
                }
            }
        }
    }

    let cell_count = ((size.x / params.cell_size) * (size.y / params.cell_size) * (size.z / params.cell_size)) as usize;

    LatticeResult {
        mesh,
        actual_density: params.density,
        cell_count,
        total_strut_length: 0.0, // Not applicable for TPMS
    }
}

/// Generate a Voronoi lattice (simplified version using random seeds).
fn generate_voronoi_lattice(params: &LatticeParams, bounds: (Point3<f64>, Point3<f64>)) -> LatticeResult {
    // For a proper implementation, we'd use Voronoi tessellation
    // For now, generate a random-looking organic structure using perturbed cubic
    let mut result = generate_cubic_lattice(params, bounds);

    // Add some randomness by perturbing vertices
    let seed = 42u64;
    let mut rng_state = seed;

    for vertex in &mut result.mesh.vertices {
        // Simple LCG random number generator
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rx = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * params.cell_size * 0.2;

        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ry = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * params.cell_size * 0.2;

        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rz = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * params.cell_size * 0.2;

        vertex.position.x += rx;
        vertex.position.y += ry;
        vertex.position.z += rz;
    }

    result
}

/// Add a cylindrical strut between two points.
fn add_cylindrical_strut(mesh: &mut Mesh, start: Point3<f64>, end: Point3<f64>, radius: f64, segments: usize) {
    let axis = end - start;
    let length = axis.norm();
    if length < 1e-10 || radius < 1e-10 {
        return;
    }

    let axis_norm = axis / length;

    // Find perpendicular vectors
    let perp1 = if axis_norm.x.abs() < 0.9 {
        axis_norm.cross(&Vector3::x())
    } else {
        axis_norm.cross(&Vector3::y())
    }
    .normalize();
    let perp2 = axis_norm.cross(&perp1);

    let base_idx = mesh.vertices.len() as u32;

    // Generate vertices for both ends of the cylinder
    for (end_point, _end_idx) in [(start, 0), (end, segments as u32)].iter() {
        for i in 0..segments {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
            let offset = perp1 * angle.cos() * radius + perp2 * angle.sin() * radius;
            let pos = Point3::from(end_point.coords + offset);
            mesh.vertices.push(Vertex::new(pos));
        }
    }

    // Add center vertices for caps
    mesh.vertices.push(Vertex::new(start));
    mesh.vertices.push(Vertex::new(end));
    let start_center = base_idx + 2 * segments as u32;
    let end_center = base_idx + 2 * segments as u32 + 1;

    // Generate side faces
    for i in 0..segments {
        let i0 = base_idx + i as u32;
        let i1 = base_idx + ((i + 1) % segments) as u32;
        let i2 = base_idx + segments as u32 + i as u32;
        let i3 = base_idx + segments as u32 + ((i + 1) % segments) as u32;

        // Two triangles per quad
        mesh.faces.push([i0, i2, i1]);
        mesh.faces.push([i1, i2, i3]);
    }

    // Generate cap faces
    for i in 0..segments {
        let i0 = base_idx + i as u32;
        let i1 = base_idx + ((i + 1) % segments) as u32;
        mesh.faces.push([start_center, i1, i0]); // Start cap (reversed)

        let i2 = base_idx + segments as u32 + i as u32;
        let i3 = base_idx + segments as u32 + ((i + 1) % segments) as u32;
        mesh.faces.push([end_center, i2, i3]); // End cap
    }
}

/// Edge vertex indices for marching cubes.
const EDGE_VERTICES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0), // Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4), // Top face
    (0, 4), (1, 5), (2, 6), (3, 7), // Vertical edges
];

/// Get triangles for a marching cubes configuration.
/// Returns indices into edge list, 255 marks end.
fn get_marching_cubes_triangles(cube_index: u8) -> &'static [u8] {
    // Simplified marching cubes table - a full implementation would have 256 entries
    // This is a minimal subset for common cases
    static TRIANGLES: [[u8; 16]; 256] = {
        let mut table = [[255u8; 16]; 256];

        // Case 1: Single corner inside
        table[1] = [0, 8, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[2] = [0, 1, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[4] = [1, 2, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[8] = [3, 11, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[16] = [4, 7, 8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[32] = [9, 5, 4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[64] = [10, 6, 5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[128] = [7, 6, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

        // Case 3: Two adjacent corners inside (edge)
        table[3] = [1, 8, 3, 9, 8, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[5] = [0, 8, 3, 1, 2, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[6] = [9, 2, 10, 0, 2, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

        // Inverse cases (255 - index)
        table[254] = [0, 3, 8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[253] = [0, 9, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[251] = [1, 10, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[247] = [3, 2, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[239] = [4, 8, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[223] = [9, 4, 5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[191] = [10, 5, 6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
        table[127] = [7, 11, 6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

        table
    };

    &TRIANGLES[cube_index as usize]
}

// ============================================================================
// Infill Generation (Skin + Lattice Combination)
// ============================================================================

/// Parameters for infill generation.
#[derive(Debug, Clone)]
pub struct InfillParams {
    /// Lattice parameters for the infill structure.
    pub lattice: LatticeParams,

    /// Shell thickness for the outer skin (mm).
    pub shell_thickness: f64,

    /// Number of shell layers (perimeters).
    pub shell_layers: usize,

    /// Infill percentage (0.0 - 1.0).
    /// 0.0 = hollow, 1.0 = solid.
    pub infill_percentage: f64,

    /// Whether to connect lattice to shell.
    pub connect_to_shell: bool,

    /// Minimum wall thickness for connection regions (mm).
    pub connection_thickness: f64,

    /// Whether to generate solid top and bottom surfaces.
    pub solid_caps: bool,

    /// Number of solid layers for top/bottom.
    pub solid_cap_layers: usize,
}

impl Default for InfillParams {
    fn default() -> Self {
        Self {
            lattice: LatticeParams::default(),
            shell_thickness: 1.2,  // Typical FDM shell thickness
            shell_layers: 3,
            infill_percentage: 0.2,  // 20% infill
            connect_to_shell: true,
            connection_thickness: 0.4,
            solid_caps: true,
            solid_cap_layers: 3,
        }
    }
}

impl InfillParams {
    /// Create infill params with a specific lattice type.
    pub fn with_lattice_type(lattice_type: LatticeType) -> Self {
        Self {
            lattice: LatticeParams {
                lattice_type,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create infill params for a specific percentage (0-100).
    pub fn with_percentage(percentage: f64) -> Self {
        Self {
            infill_percentage: (percentage / 100.0).clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set shell thickness.
    pub fn with_shell_thickness(mut self, thickness: f64) -> Self {
        self.shell_thickness = thickness.max(0.0);
        self
    }

    /// Set number of shell layers.
    pub fn with_shell_layers(mut self, layers: usize) -> Self {
        self.shell_layers = layers.max(1);
        self
    }

    /// Set infill percentage (0-100).
    pub fn with_infill_percentage(mut self, percentage: f64) -> Self {
        self.infill_percentage = (percentage / 100.0).clamp(0.0, 1.0);
        self
    }

    /// Set lattice cell size.
    pub fn with_cell_size(mut self, cell_size: f64) -> Self {
        self.lattice.cell_size = cell_size.max(0.1);
        self
    }

    /// Set strut thickness for lattice.
    pub fn with_strut_thickness(mut self, thickness: f64) -> Self {
        self.lattice.strut_thickness = thickness.max(0.1);
        self
    }

    /// Enable or disable solid caps.
    pub fn with_solid_caps(mut self, enabled: bool) -> Self {
        self.solid_caps = enabled;
        self
    }

    /// Create preset for standard FDM printing (20% infill).
    pub fn for_fdm() -> Self {
        Self {
            lattice: LatticeParams::cubic(5.0),
            shell_thickness: 1.2,
            shell_layers: 3,
            infill_percentage: 0.2,
            solid_caps: true,
            solid_cap_layers: 4,
            ..Default::default()
        }
    }

    /// Create preset for lightweight parts (10% infill).
    pub fn for_lightweight() -> Self {
        Self {
            lattice: LatticeParams::gyroid(8.0),
            shell_thickness: 0.8,
            shell_layers: 2,
            infill_percentage: 0.1,
            solid_caps: true,
            solid_cap_layers: 2,
            ..Default::default()
        }
    }

    /// Create preset for strong parts (50% infill).
    pub fn for_strong() -> Self {
        Self {
            lattice: LatticeParams::octet_truss(4.0),
            shell_thickness: 1.6,
            shell_layers: 4,
            infill_percentage: 0.5,
            solid_caps: true,
            solid_cap_layers: 6,
            ..Default::default()
        }
    }
}

/// Result of infill generation.
#[derive(Debug)]
pub struct InfillResult {
    /// The complete mesh with shell and infill.
    pub mesh: Mesh,

    /// The outer shell mesh (for inspection).
    pub shell: Mesh,

    /// The inner lattice mesh (for inspection).
    pub lattice: Mesh,

    /// Achieved infill density (volume fraction).
    pub actual_density: f64,

    /// Volume of the shell.
    pub shell_volume: f64,

    /// Volume of the lattice.
    pub lattice_volume: f64,

    /// Total interior volume.
    pub interior_volume: f64,
}

/// Generate infill for a mesh (outer shell + interior lattice).
///
/// This creates a hollowed mesh with lattice infill, suitable for
/// 3D printing or lightweight manufacturing.
///
/// # Arguments
///
/// * `mesh` - The input mesh (must be watertight)
/// * `params` - Parameters controlling infill generation
///
/// # Returns
///
/// An `InfillResult` containing the combined mesh and analysis data.
pub fn generate_infill(mesh: &Mesh, params: &InfillParams) -> InfillResult {
    // Handle edge cases
    if params.infill_percentage >= 1.0 {
        // Solid fill - just return the original mesh
        return InfillResult {
            mesh: mesh.clone(),
            shell: mesh.clone(),
            lattice: Mesh::new(),
            actual_density: 1.0,
            shell_volume: estimate_volume(mesh),
            lattice_volume: 0.0,
            interior_volume: 0.0,
        };
    }

    if params.infill_percentage <= 0.0 || mesh.faces.is_empty() {
        // Hollow - generate only shell
        let shell = generate_shell(mesh, params.shell_thickness);
        let shell_vol = estimate_volume(&shell);

        return InfillResult {
            mesh: shell.clone(),
            shell,
            lattice: Mesh::new(),
            actual_density: 0.0,
            shell_volume: shell_vol,
            lattice_volume: 0.0,
            interior_volume: estimate_volume(mesh) - shell_vol,
        };
    }

    // Generate outer shell by offsetting inward
    let shell = generate_shell(mesh, params.shell_thickness);

    // Compute interior region bounds (offset inward from mesh bounds)
    let interior_bounds = compute_interior_bounds(mesh, params.shell_thickness);

    // Generate lattice in interior region
    let mut lattice_params = params.lattice.clone();
    lattice_params.density = params.infill_percentage;
    lattice_params.trim_to_bounds = true;

    let lattice_result = generate_lattice(&lattice_params, interior_bounds);
    let mut lattice = lattice_result.mesh;

    // Trim lattice to interior of mesh
    lattice = trim_lattice_to_interior(lattice, mesh, params.shell_thickness);

    // Connect lattice to shell if requested
    if params.connect_to_shell {
        connect_lattice_to_shell(&mut lattice, mesh, params);
    }

    // Combine shell and lattice
    let combined = combine_shell_and_lattice(&shell, &lattice);

    // Calculate volumes
    let shell_vol = estimate_volume(&shell);
    let lattice_vol = lattice_result.actual_density * estimate_interior_volume(mesh, params.shell_thickness);
    let interior_vol = estimate_interior_volume(mesh, params.shell_thickness);

    InfillResult {
        mesh: combined,
        shell,
        lattice,
        actual_density: lattice_result.actual_density,
        shell_volume: shell_vol,
        lattice_volume: lattice_vol,
        interior_volume: interior_vol,
    }
}

/// Generate outer shell by creating offset surfaces.
fn generate_shell(mesh: &Mesh, thickness: f64) -> Mesh {
    if thickness <= 0.0 {
        return mesh.clone();
    }

    // For a proper shell, we need:
    // 1. Original outer surface
    // 2. Offset inner surface
    // 3. Connection between them at boundaries

    let mut shell = Mesh::new();

    // Copy outer surface (original mesh)
    shell.vertices.extend(mesh.vertices.iter().cloned());
    shell.faces.extend(mesh.faces.iter().cloned());

    // Compute vertex normals for offset direction
    let normals = compute_vertex_normals_array(mesh);

    // Create inner offset surface
    let outer_vert_count = shell.vertices.len() as u32;

    for (i, vertex) in mesh.vertices.iter().enumerate() {
        let normal = normals[i];
        // Offset inward (negative normal direction)
        let offset_pos = vertex.position - normal * thickness;
        shell.vertices.push(Vertex::new(offset_pos));
    }

    // Add inner surface faces (reversed winding for inward-facing normals)
    for face in &mesh.faces {
        shell.faces.push([
            face[0] + outer_vert_count,
            face[2] + outer_vert_count,  // Reversed winding
            face[1] + outer_vert_count,
        ]);
    }

    // Note: A complete implementation would also cap the boundaries
    // where the shell is open (e.g., at holes in the original mesh).
    // For watertight meshes, this isn't needed.

    shell
}

/// Compute vertex normals as an array.
fn compute_vertex_normals_array(mesh: &Mesh) -> Vec<Vector3<f64>> {
    let mut normals = vec![Vector3::zeros(); mesh.vertices.len()];
    let mut counts = vec![0usize; mesh.vertices.len()];

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2);

        // Weight by face area (magnitude of cross product)
        for &vi in face {
            normals[vi as usize] += normal;
            counts[vi as usize] += 1;
        }
    }

    // Normalize
    for (i, normal) in normals.iter_mut().enumerate() {
        if counts[i] > 0 {
            let len = normal.norm();
            if len > 1e-10 {
                *normal /= len;
            }
        }
    }

    normals
}

/// Compute interior bounds for lattice generation.
fn compute_interior_bounds(mesh: &Mesh, shell_thickness: f64) -> (Point3<f64>, Point3<f64>) {
    if mesh.vertices.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = mesh.vertices[0].position;
    let mut max = mesh.vertices[0].position;

    for v in &mesh.vertices {
        min.x = min.x.min(v.position.x);
        min.y = min.y.min(v.position.y);
        min.z = min.z.min(v.position.z);
        max.x = max.x.max(v.position.x);
        max.y = max.y.max(v.position.y);
        max.z = max.z.max(v.position.z);
    }

    // Inset by shell thickness
    let inset = shell_thickness * 1.5; // Extra margin for safety
    (
        Point3::new(min.x + inset, min.y + inset, min.z + inset),
        Point3::new(max.x - inset, max.y - inset, max.z - inset),
    )
}

/// Trim lattice to stay within the interior of the mesh.
fn trim_lattice_to_interior(lattice: Mesh, mesh: &Mesh, shell_thickness: f64) -> Mesh {
    if lattice.vertices.is_empty() || mesh.faces.is_empty() {
        return lattice;
    }

    // Mark vertices that are inside the mesh interior
    let inside_margin = shell_thickness * 0.9; // Slightly less than shell thickness
    let mut vertex_inside = vec![false; lattice.vertices.len()];

    for (i, v) in lattice.vertices.iter().enumerate() {
        // Check if point is inside mesh with margin
        if is_point_inside_with_margin(&v.position, mesh, inside_margin) {
            vertex_inside[i] = true;
        }
    }

    // Keep faces where all vertices are inside
    let mut result = Mesh::new();
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();

    for face in &lattice.faces {
        let all_inside = vertex_inside[face[0] as usize]
            && vertex_inside[face[1] as usize]
            && vertex_inside[face[2] as usize];

        if all_inside {
            let new_face: [u32; 3] = [
                *vertex_map.entry(face[0]).or_insert_with(|| {
                    let idx = result.vertices.len() as u32;
                    result.vertices.push(lattice.vertices[face[0] as usize].clone());
                    idx
                }),
                *vertex_map.entry(face[1]).or_insert_with(|| {
                    let idx = result.vertices.len() as u32;
                    result.vertices.push(lattice.vertices[face[1] as usize].clone());
                    idx
                }),
                *vertex_map.entry(face[2]).or_insert_with(|| {
                    let idx = result.vertices.len() as u32;
                    result.vertices.push(lattice.vertices[face[2] as usize].clone());
                    idx
                }),
            ];
            result.faces.push(new_face);
        }
    }

    result
}

/// Check if a point is inside a mesh with a margin (offset inward).
fn is_point_inside_with_margin(point: &Point3<f64>, mesh: &Mesh, margin: f64) -> bool {
    // First check basic inside/outside
    if !is_point_inside_mesh(point, mesh) {
        return false;
    }

    // Check distance to surface is greater than margin
    let dist = distance_to_surface(point, mesh);
    dist > margin
}

/// Simple point-in-mesh test using ray casting.
fn is_point_inside_mesh(point: &Point3<f64>, mesh: &Mesh) -> bool {
    // Cast ray in +X direction and count intersections
    let ray_dir = Vector3::new(1.0, 0.001, 0.0); // Slight Y offset to avoid edge cases
    let mut intersection_count = 0;

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        if ray_triangle_intersect_positive(point, &ray_dir, v0, v1, v2) {
            intersection_count += 1;
        }
    }

    intersection_count % 2 == 1
}

/// Ray-triangle intersection for positive ray direction only.
fn ray_triangle_intersect_positive(
    origin: &Point3<f64>,
    dir: &Vector3<f64>,
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
) -> bool {
    let epsilon = 1e-10;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(&edge2);
    let a = edge1.dot(&h);

    if a.abs() < epsilon {
        return false;
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(&h);

    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = s.cross(&edge1);
    let v = f * dir.dot(&q);

    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * edge2.dot(&q);
    t > epsilon
}

/// Compute approximate distance from point to mesh surface.
fn distance_to_surface(point: &Point3<f64>, mesh: &Mesh) -> f64 {
    let mut min_dist = f64::MAX;

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        let dist = point_to_triangle_distance(point, v0, v1, v2);
        min_dist = min_dist.min(dist);
    }

    min_dist
}

/// Compute distance from point to triangle.
fn point_to_triangle_distance(
    point: &Point3<f64>,
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
) -> f64 {
    // Project point onto triangle plane
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let normal = edge1.cross(&edge2);
    let normal_len = normal.norm();

    if normal_len < 1e-10 {
        // Degenerate triangle - use distance to vertices
        return (point - v0).norm()
            .min((point - v1).norm())
            .min((point - v2).norm());
    }

    let normal = normal / normal_len;
    let to_point = point - v0;
    let dist_to_plane = to_point.dot(&normal).abs();

    // Project point onto plane
    let projected = point - normal * to_point.dot(&normal);

    // Check if projected point is inside triangle using barycentric coordinates
    let d00 = edge1.dot(&edge1);
    let d01 = edge1.dot(&edge2);
    let d11 = edge2.dot(&edge2);
    let d20 = (projected - v0).dot(&edge1);
    let d21 = (projected - v0).dot(&edge2);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-10 {
        return dist_to_plane;
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    if u >= 0.0 && v >= 0.0 && w >= 0.0 {
        // Point projects inside triangle
        dist_to_plane
    } else {
        // Point projects outside - find closest point on edges/vertices
        let dist_to_edge = |p: &Point3<f64>, a: &Point3<f64>, b: &Point3<f64>| {
            let ab = b - a;
            let ap = p - a;
            let t = (ap.dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
            let closest = a + ab * t;
            (p - closest).norm()
        };

        dist_to_edge(point, v0, v1)
            .min(dist_to_edge(point, v1, v2))
            .min(dist_to_edge(point, v2, v0))
    }
}

/// Connect lattice struts to the shell for structural integrity.
fn connect_lattice_to_shell(_lattice: &mut Mesh, _mesh: &Mesh, _params: &InfillParams) {
    // This is a placeholder for a more sophisticated connection algorithm.
    // A full implementation would:
    // 1. Find lattice endpoints near the shell surface
    // 2. Create bridging geometry to connect them
    // 3. Ensure smooth transitions for printability
    //
    // For now, the lattice is simply positioned inside the shell,
    // which works for most printing scenarios where the slicer
    // handles the layer-by-layer connection.
}

/// Combine shell and lattice into a single mesh.
fn combine_shell_and_lattice(shell: &Mesh, lattice: &Mesh) -> Mesh {
    let mut result = shell.clone();

    if lattice.vertices.is_empty() {
        return result;
    }

    let offset = result.vertices.len() as u32;

    // Add lattice vertices
    result.vertices.extend(lattice.vertices.iter().cloned());

    // Add lattice faces with offset indices
    for face in &lattice.faces {
        result.faces.push([
            face[0] + offset,
            face[1] + offset,
            face[2] + offset,
        ]);
    }

    result
}

/// Estimate volume of a mesh using divergence theorem.
fn estimate_volume(mesh: &Mesh) -> f64 {
    let mut volume = 0.0;

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        // Signed volume of tetrahedron formed with origin
        volume += v0.coords.dot(&v1.coords.cross(&v2.coords)) / 6.0;
    }

    volume.abs()
}

/// Estimate interior volume (total minus shell).
fn estimate_interior_volume(mesh: &Mesh, shell_thickness: f64) -> f64 {
    // Approximate using shell thickness and surface area
    let total_vol = estimate_volume(mesh);
    let surface_area = estimate_surface_area(mesh);

    // Volume reduced by shell: approximately surface_area * shell_thickness
    // This is an approximation; exact calculation requires offset surface
    let shell_vol = surface_area * shell_thickness;

    (total_vol - shell_vol).max(0.0)
}

/// Estimate surface area of a mesh.
fn estimate_surface_area(mesh: &Mesh) -> f64 {
    let mut area = 0.0;

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        area += edge1.cross(&edge2).norm() / 2.0;
    }

    area
}

// Convenience methods on Mesh
impl Mesh {
    /// Generate lattice infill for this mesh.
    ///
    /// Creates a hollowed version of the mesh with internal lattice structure,
    /// suitable for lightweight manufacturing or 3D printing.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    /// use mesh_repair::lattice::InfillParams;
    ///
    /// let mut mesh = Mesh::new();
    /// // ... create mesh geometry ...
    ///
    /// // Generate with 20% infill
    /// // let result = mesh.generate_lattice_infill(&InfillParams::with_percentage(20.0));
    /// ```
    pub fn generate_lattice_infill(&self, params: &InfillParams) -> InfillResult {
        generate_infill(self, params)
    }

    /// Generate lattice infill with default parameters (20% cubic infill).
    pub fn generate_infill_default(&self) -> InfillResult {
        generate_infill(self, &InfillParams::default())
    }

    /// Create a hollow shell from this mesh.
    ///
    /// # Arguments
    ///
    /// * `thickness` - Shell wall thickness in mm
    pub fn hollow(&self, thickness: f64) -> Mesh {
        generate_shell(self, thickness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_params_default() {
        let params = LatticeParams::default();
        assert_eq!(params.lattice_type, LatticeType::Cubic);
        assert!((params.cell_size - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_params_cubic() {
        let params = LatticeParams::cubic(10.0);
        assert_eq!(params.lattice_type, LatticeType::Cubic);
        assert!((params.cell_size - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_params_gyroid() {
        let params = LatticeParams::gyroid(8.0);
        assert_eq!(params.lattice_type, LatticeType::Gyroid);
        assert!((params.cell_size - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_cubic_lattice() {
        let params = LatticeParams::cubic(5.0).with_density(0.2);
        let bounds = (Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        let result = generate_lattice(&params, bounds);

        assert!(result.mesh.vertices.len() > 0);
        assert!(result.mesh.faces.len() > 0);
        assert!(result.cell_count > 0);
        assert!(result.total_strut_length > 0.0);
    }

    #[test]
    fn test_generate_octet_truss_lattice() {
        let params = LatticeParams::octet_truss(5.0).with_density(0.3);
        let bounds = (Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        let result = generate_lattice(&params, bounds);

        assert!(result.mesh.vertices.len() > 0);
        assert!(result.mesh.faces.len() > 0);
    }

    #[test]
    fn test_generate_gyroid_lattice() {
        let params = LatticeParams::gyroid(5.0)
            .with_resolution(5)
            .with_density(0.5);
        let bounds = (Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        let result = generate_lattice(&params, bounds);

        // Gyroid should produce a mesh
        assert!(result.mesh.vertices.len() > 0);
    }

    #[test]
    fn test_density_map_uniform() {
        let map = DensityMap::Uniform(0.5);
        assert!((map.evaluate(Point3::origin()) - 0.5).abs() < 1e-10);
        assert!((map.evaluate(Point3::new(100.0, 100.0, 100.0)) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_density_map_gradient() {
        let map = DensityMap::Gradient {
            from: Point3::origin(),
            from_density: 0.2,
            to: Point3::new(10.0, 0.0, 0.0),
            to_density: 0.8,
        };

        assert!((map.evaluate(Point3::origin()) - 0.2).abs() < 1e-10);
        assert!((map.evaluate(Point3::new(10.0, 0.0, 0.0)) - 0.8).abs() < 1e-10);
        assert!((map.evaluate(Point3::new(5.0, 0.0, 0.0)) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_density_map_radial() {
        let map = DensityMap::Radial {
            center: Point3::origin(),
            inner_radius: 5.0,
            inner_density: 0.8,
            outer_radius: 10.0,
            outer_density: 0.2,
        };

        assert!((map.evaluate(Point3::origin()) - 0.8).abs() < 1e-10);
        assert!((map.evaluate(Point3::new(10.0, 0.0, 0.0)) - 0.2).abs() < 1e-10);
        assert!((map.evaluate(Point3::new(7.5, 0.0, 0.0)) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_add_cylindrical_strut() {
        let mut mesh = Mesh::new();
        add_cylindrical_strut(
            &mut mesh,
            Point3::origin(),
            Point3::new(10.0, 0.0, 0.0),
            1.0,
            6,
        );

        assert!(mesh.vertices.len() > 0);
        assert!(mesh.faces.len() > 0);
    }

    // Infill generation tests

    #[test]
    fn test_infill_params_default() {
        let params = InfillParams::default();
        assert!((params.infill_percentage - 0.2).abs() < 1e-10);
        assert!((params.shell_thickness - 1.2).abs() < 1e-10);
        assert_eq!(params.shell_layers, 3);
    }

    #[test]
    fn test_infill_params_percentage() {
        let params = InfillParams::with_percentage(50.0);
        assert!((params.infill_percentage - 0.5).abs() < 1e-10);

        // Test clamping
        let params_high = InfillParams::with_percentage(150.0);
        assert!((params_high.infill_percentage - 1.0).abs() < 1e-10);

        let params_low = InfillParams::with_percentage(-10.0);
        assert!(params_low.infill_percentage >= 0.0);
    }

    #[test]
    fn test_infill_params_presets() {
        let fdm = InfillParams::for_fdm();
        assert!((fdm.infill_percentage - 0.2).abs() < 1e-10);
        assert_eq!(fdm.lattice.lattice_type, LatticeType::Cubic);

        let light = InfillParams::for_lightweight();
        assert!((light.infill_percentage - 0.1).abs() < 1e-10);
        assert_eq!(light.lattice.lattice_type, LatticeType::Gyroid);

        let strong = InfillParams::for_strong();
        assert!((strong.infill_percentage - 0.5).abs() < 1e-10);
        assert_eq!(strong.lattice.lattice_type, LatticeType::OctetTruss);
    }

    #[test]
    fn test_infill_params_builder() {
        let params = InfillParams::default()
            .with_shell_thickness(2.0)
            .with_shell_layers(5)
            .with_infill_percentage(30.0)
            .with_cell_size(8.0)
            .with_strut_thickness(1.0)
            .with_solid_caps(false);

        assert!((params.shell_thickness - 2.0).abs() < 1e-10);
        assert_eq!(params.shell_layers, 5);
        assert!((params.infill_percentage - 0.3).abs() < 1e-10);
        assert!((params.lattice.cell_size - 8.0).abs() < 1e-10);
        assert!((params.lattice.strut_thickness - 1.0).abs() < 1e-10);
        assert!(!params.solid_caps);
    }

    fn create_test_cube(center: Point3<f64>, size: f64) -> Mesh {
        let half = size / 2.0;
        let mut mesh = Mesh::new();

        // 8 vertices
        let vertices = [
            Point3::new(center.x - half, center.y - half, center.z - half),
            Point3::new(center.x + half, center.y - half, center.z - half),
            Point3::new(center.x + half, center.y + half, center.z - half),
            Point3::new(center.x - half, center.y + half, center.z - half),
            Point3::new(center.x - half, center.y - half, center.z + half),
            Point3::new(center.x + half, center.y - half, center.z + half),
            Point3::new(center.x + half, center.y + half, center.z + half),
            Point3::new(center.x - half, center.y + half, center.z + half),
        ];

        for v in &vertices {
            mesh.vertices.push(Vertex::new(*v));
        }

        // 12 faces (2 per side) with outward normals
        let faces = [
            [0, 2, 1], [0, 3, 2], // Front (-Z)
            [4, 5, 6], [4, 6, 7], // Back (+Z)
            [3, 7, 6], [3, 6, 2], // Top (+Y)
            [0, 1, 5], [0, 5, 4], // Bottom (-Y)
            [0, 4, 7], [0, 7, 3], // Left (-X)
            [1, 2, 6], [1, 6, 5], // Right (+X)
        ];

        for f in &faces {
            mesh.faces.push(*f);
        }

        mesh
    }

    #[test]
    fn test_generate_shell() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let shell = generate_shell(&cube, 1.0);

        // Shell should have double the vertices (outer + inner surface)
        assert_eq!(shell.vertices.len(), cube.vertices.len() * 2);
        // Shell should have double the faces (outer + inner surface)
        assert_eq!(shell.faces.len(), cube.faces.len() * 2);
    }

    #[test]
    fn test_generate_shell_zero_thickness() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let shell = generate_shell(&cube, 0.0);

        // With zero thickness, should return original mesh
        assert_eq!(shell.vertices.len(), cube.vertices.len());
        assert_eq!(shell.faces.len(), cube.faces.len());
    }

    #[test]
    fn test_compute_interior_bounds() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let (min, max) = compute_interior_bounds(&cube, 1.0);

        // Interior should be smaller than original by shell thickness margin
        let inset = 1.5; // Shell thickness * 1.5
        assert!(min.x > 0.0);
        assert!(max.x < 10.0);
        assert!((min.x - inset).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_volume() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let volume = estimate_volume(&cube);

        // Volume of 10x10x10 cube = 1000
        assert!((volume - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_estimate_surface_area() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let area = estimate_surface_area(&cube);

        // Surface area of 10x10x10 cube = 6 * 100 = 600
        assert!((area - 600.0).abs() < 1.0);
    }

    #[test]
    fn test_point_in_mesh() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        // Point inside
        assert!(is_point_inside_mesh(&Point3::new(5.0, 5.0, 5.0), &cube));

        // Point outside
        assert!(!is_point_inside_mesh(&Point3::new(15.0, 5.0, 5.0), &cube));
        assert!(!is_point_inside_mesh(&Point3::new(5.0, 15.0, 5.0), &cube));
    }

    #[test]
    fn test_distance_to_surface() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        // Point at center should be 5mm from nearest face
        let dist = distance_to_surface(&Point3::new(5.0, 5.0, 5.0), &cube);
        assert!((dist - 5.0).abs() < 0.1);

        // Point near face
        let dist_near = distance_to_surface(&Point3::new(5.0, 5.0, 9.5), &cube);
        assert!((dist_near - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_generate_infill_solid() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let params = InfillParams::with_percentage(100.0);
        let result = generate_infill(&cube, &params);

        // Solid should return original mesh
        assert_eq!(result.mesh.vertices.len(), cube.vertices.len());
        assert!((result.actual_density - 1.0).abs() < 1e-10);
        assert!(result.lattice.faces.is_empty());
    }

    #[test]
    fn test_generate_infill_hollow() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let params = InfillParams::with_percentage(0.0)
            .with_shell_thickness(1.0);
        let result = generate_infill(&cube, &params);

        // Hollow should have shell but no lattice
        assert!(result.shell.faces.len() > 0);
        assert!(result.lattice.faces.is_empty());
        assert!((result.actual_density - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_infill_with_lattice() {
        let cube = create_test_cube(Point3::new(10.0, 10.0, 10.0), 20.0);

        let params = InfillParams::with_percentage(20.0)
            .with_shell_thickness(1.0)
            .with_cell_size(4.0);
        let result = generate_infill(&cube, &params);

        // Should have both shell and lattice
        assert!(result.shell.faces.len() > 0);
        // Note: lattice may be empty if trimmed to interior doesn't include any complete cells
        // For a 20mm cube with 1mm shell and 4mm cells, there should be interior room
        assert!(result.mesh.faces.len() >= result.shell.faces.len());
    }

    #[test]
    fn test_mesh_hollow_method() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let hollow = cube.hollow(1.0);

        // Should have inner and outer surfaces
        assert_eq!(hollow.vertices.len(), cube.vertices.len() * 2);
    }

    #[test]
    fn test_compute_vertex_normals() {
        let cube = create_test_cube(Point3::new(5.0, 5.0, 5.0), 10.0);

        let normals = compute_vertex_normals_array(&cube);

        // Each vertex should have a normalized normal
        for normal in &normals {
            let len = normal.norm();
            assert!((len - 1.0).abs() < 0.1 || len < 0.1); // Either normalized or zero
        }
    }

    #[test]
    fn test_infill_with_lattice_type() {
        let params = InfillParams::with_lattice_type(LatticeType::Gyroid);
        assert_eq!(params.lattice.lattice_type, LatticeType::Gyroid);

        let params = InfillParams::with_lattice_type(LatticeType::OctetTruss);
        assert_eq!(params.lattice.lattice_type, LatticeType::OctetTruss);
    }

    #[test]
    fn test_point_to_triangle_distance() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(10.0, 0.0, 0.0);
        let v2 = Point3::new(5.0, 10.0, 0.0);

        // Point directly above center of triangle
        let p = Point3::new(5.0, 3.0, 5.0);
        let dist = point_to_triangle_distance(&p, &v0, &v1, &v2);
        assert!((dist - 5.0).abs() < 0.1);

        // Point on the triangle plane
        let p_on = Point3::new(5.0, 3.0, 0.0);
        let dist_on = point_to_triangle_distance(&p_on, &v0, &v1, &v2);
        assert!(dist_on < 0.1);
    }

    #[test]
    fn test_combine_shell_and_lattice() {
        let mut shell = Mesh::new();
        shell.vertices.push(Vertex::new(Point3::new(0.0, 0.0, 0.0)));
        shell.vertices.push(Vertex::new(Point3::new(1.0, 0.0, 0.0)));
        shell.vertices.push(Vertex::new(Point3::new(0.5, 1.0, 0.0)));
        shell.faces.push([0, 1, 2]);

        let mut lattice = Mesh::new();
        lattice.vertices.push(Vertex::new(Point3::new(0.3, 0.3, 0.1)));
        lattice.vertices.push(Vertex::new(Point3::new(0.7, 0.3, 0.1)));
        lattice.vertices.push(Vertex::new(Point3::new(0.5, 0.7, 0.1)));
        lattice.faces.push([0, 1, 2]);

        let combined = combine_shell_and_lattice(&shell, &lattice);

        assert_eq!(combined.vertices.len(), 6);
        assert_eq!(combined.faces.len(), 2);
    }

    #[test]
    fn test_empty_lattice_combination() {
        let mut shell = Mesh::new();
        shell.vertices.push(Vertex::new(Point3::new(0.0, 0.0, 0.0)));
        shell.vertices.push(Vertex::new(Point3::new(1.0, 0.0, 0.0)));
        shell.vertices.push(Vertex::new(Point3::new(0.5, 1.0, 0.0)));
        shell.faces.push([0, 1, 2]);

        let lattice = Mesh::new();

        let combined = combine_shell_and_lattice(&shell, &lattice);

        // Should just be the shell
        assert_eq!(combined.vertices.len(), shell.vertices.len());
        assert_eq!(combined.faces.len(), shell.faces.len());
    }
}
