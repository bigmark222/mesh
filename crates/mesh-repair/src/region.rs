//! Mesh region definition and management.
//!
//! This module provides tools for defining and working with regions on a mesh,
//! enabling variable thickness, material zones, and selective operations.
//!
//! # Use Cases
//!
//! - Defining thick heel cups and thin arch areas on a skate boot
//! - Marking ventilation zones vs structural zones on a helmet
//! - Specifying material transitions in multi-material prints
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::region::{MeshRegion, RegionMap, RegionSelector};
//! use nalgebra::Point3;
//!
//! // Create a simple mesh
//! let mut mesh = Mesh::new();
//! mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 5.0));
//! mesh.faces.push([0, 1, 2]);
//! mesh.faces.push([0, 1, 3]);
//! mesh.faces.push([1, 2, 3]);
//! mesh.faces.push([2, 0, 3]);
//!
//! // Create a region map
//! let mut regions = RegionMap::new();
//!
//! // Define a region using a sphere selector
//! let top_region = MeshRegion::from_selector(
//!     &mesh,
//!     "top",
//!     RegionSelector::sphere(Point3::new(5.0, 5.0, 5.0), 3.0),
//! );
//! regions.add(top_region);
//!
//! // Define a region using spatial bounds
//! let base_region = MeshRegion::from_selector(
//!     &mesh,
//!     "base",
//!     RegionSelector::bounds(Point3::new(-1.0, -1.0, -1.0), Point3::new(11.0, 11.0, 2.0)),
//! );
//! regions.add(base_region);
//!
//! let names: Vec<_> = regions.names().collect();
//! println!("Regions: {:?}", names);
//! ```

use crate::Mesh;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// A named region of a mesh, defined by face or vertex indices.
#[derive(Debug, Clone)]
pub struct MeshRegion {
    /// Unique name for this region.
    pub name: String,

    /// Vertex indices that belong to this region.
    pub vertices: HashSet<u32>,

    /// Face indices that belong to this region.
    pub faces: HashSet<u32>,

    /// Optional metadata for this region.
    pub metadata: HashMap<String, String>,

    /// Optional color for visualization (RGB, 0-255).
    pub color: Option<(u8, u8, u8)>,
}

impl MeshRegion {
    /// Create an empty region with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            vertices: HashSet::new(),
            faces: HashSet::new(),
            metadata: HashMap::new(),
            color: None,
        }
    }

    /// Create a region from vertex indices.
    pub fn from_vertices(name: impl Into<String>, vertices: impl IntoIterator<Item = u32>) -> Self {
        Self {
            name: name.into(),
            vertices: vertices.into_iter().collect(),
            faces: HashSet::new(),
            metadata: HashMap::new(),
            color: None,
        }
    }

    /// Create a region from face indices.
    pub fn from_faces(name: impl Into<String>, faces: impl IntoIterator<Item = u32>) -> Self {
        Self {
            name: name.into(),
            vertices: HashSet::new(),
            faces: faces.into_iter().collect(),
            metadata: HashMap::new(),
            color: None,
        }
    }

    /// Create a region by applying a selector to a mesh.
    pub fn from_selector(mesh: &Mesh, name: impl Into<String>, selector: RegionSelector) -> Self {
        let name = name.into();
        let (vertices, faces) = selector.select(mesh);
        Self {
            name,
            vertices,
            faces,
            metadata: HashMap::new(),
            color: None,
        }
    }

    /// Add a vertex to this region.
    pub fn add_vertex(&mut self, vertex_index: u32) {
        self.vertices.insert(vertex_index);
    }

    /// Add a face to this region.
    pub fn add_face(&mut self, face_index: u32) {
        self.faces.insert(face_index);
    }

    /// Add multiple vertices to this region.
    pub fn add_vertices(&mut self, indices: impl IntoIterator<Item = u32>) {
        self.vertices.extend(indices);
    }

    /// Add multiple faces to this region.
    pub fn add_faces(&mut self, indices: impl IntoIterator<Item = u32>) {
        self.faces.extend(indices);
    }

    /// Check if a vertex is in this region.
    pub fn contains_vertex(&self, vertex_index: u32) -> bool {
        self.vertices.contains(&vertex_index)
    }

    /// Check if a face is in this region.
    pub fn contains_face(&self, face_index: u32) -> bool {
        self.faces.contains(&face_index)
    }

    /// Get the number of vertices in this region.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces in this region.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if this region is empty.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.faces.is_empty()
    }

    /// Expand the region to include all vertices of contained faces.
    pub fn expand_to_face_vertices(&mut self, mesh: &Mesh) {
        for &face_idx in &self.faces {
            if let Some(face) = mesh.faces.get(face_idx as usize) {
                self.vertices.insert(face[0]);
                self.vertices.insert(face[1]);
                self.vertices.insert(face[2]);
            }
        }
    }

    /// Expand the region to include all faces that contain any region vertex.
    pub fn expand_to_vertex_faces(&mut self, mesh: &Mesh) {
        for (face_idx, face) in mesh.faces.iter().enumerate() {
            if self.vertices.contains(&face[0])
                || self.vertices.contains(&face[1])
                || self.vertices.contains(&face[2])
            {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Set metadata for this region.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata for this region.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Set the visualization color for this region.
    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = Some((r, g, b));
        self
    }

    /// Compute the union with another region.
    pub fn union(&self, other: &MeshRegion) -> MeshRegion {
        MeshRegion {
            name: format!("{}+{}", self.name, other.name),
            vertices: self.vertices.union(&other.vertices).copied().collect(),
            faces: self.faces.union(&other.faces).copied().collect(),
            metadata: HashMap::new(),
            color: self.color.or(other.color),
        }
    }

    /// Compute the intersection with another region.
    pub fn intersection(&self, other: &MeshRegion) -> MeshRegion {
        MeshRegion {
            name: format!("{}&{}", self.name, other.name),
            vertices: self.vertices.intersection(&other.vertices).copied().collect(),
            faces: self.faces.intersection(&other.faces).copied().collect(),
            metadata: HashMap::new(),
            color: self.color.or(other.color),
        }
    }

    /// Compute the difference (self - other).
    pub fn difference(&self, other: &MeshRegion) -> MeshRegion {
        MeshRegion {
            name: format!("{}-{}", self.name, other.name),
            vertices: self.vertices.difference(&other.vertices).copied().collect(),
            faces: self.faces.difference(&other.faces).copied().collect(),
            metadata: HashMap::new(),
            color: self.color,
        }
    }
}

/// A collection of named regions for a mesh.
#[derive(Debug, Clone, Default)]
pub struct RegionMap {
    /// Regions indexed by name.
    regions: HashMap<String, MeshRegion>,
}

impl RegionMap {
    /// Create a new empty region map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a region to the map.
    pub fn add(&mut self, region: MeshRegion) {
        self.regions.insert(region.name.clone(), region);
    }

    /// Get a region by name.
    pub fn get(&self, name: &str) -> Option<&MeshRegion> {
        self.regions.get(name)
    }

    /// Get a mutable reference to a region by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut MeshRegion> {
        self.regions.get_mut(name)
    }

    /// Remove a region by name.
    pub fn remove(&mut self, name: &str) -> Option<MeshRegion> {
        self.regions.remove(name)
    }

    /// Check if a region exists.
    pub fn contains(&self, name: &str) -> bool {
        self.regions.contains_key(name)
    }

    /// Get the number of regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Get an iterator over region names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.regions.keys().map(|s| s.as_str())
    }

    /// Get an iterator over regions.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MeshRegion)> {
        self.regions.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get a mutable iterator over regions.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut MeshRegion)> {
        self.regions.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Find which region(s) contain a vertex.
    pub fn regions_containing_vertex(&self, vertex_index: u32) -> Vec<&str> {
        self.regions
            .iter()
            .filter(|(_, r)| r.contains_vertex(vertex_index))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Find which region(s) contain a face.
    pub fn regions_containing_face(&self, face_index: u32) -> Vec<&str> {
        self.regions
            .iter()
            .filter(|(_, r)| r.contains_face(face_index))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get vertices that are not in any region.
    pub fn unassigned_vertices(&self, mesh: &Mesh) -> HashSet<u32> {
        let assigned: HashSet<u32> = self.regions.values().flat_map(|r| r.vertices.iter().copied()).collect();
        (0..mesh.vertex_count() as u32)
            .filter(|i| !assigned.contains(i))
            .collect()
    }

    /// Get faces that are not in any region.
    pub fn unassigned_faces(&self, mesh: &Mesh) -> HashSet<u32> {
        let assigned: HashSet<u32> = self.regions.values().flat_map(|r| r.faces.iter().copied()).collect();
        (0..mesh.face_count() as u32)
            .filter(|i| !assigned.contains(i))
            .collect()
    }
}

/// Selector for defining regions based on spatial or topological criteria.
#[derive(Debug, Clone)]
pub enum RegionSelector {
    /// Select vertices/faces within an axis-aligned bounding box.
    Bounds { min: Point3<f64>, max: Point3<f64> },

    /// Select vertices/faces within a sphere.
    Sphere { center: Point3<f64>, radius: f64 },

    /// Select vertices/faces within a cylinder.
    Cylinder {
        axis_start: Point3<f64>,
        axis_end: Point3<f64>,
        radius: f64,
    },

    /// Select vertices/faces within a distance of a plane.
    Plane {
        point: Point3<f64>,
        normal: Vector3<f64>,
        tolerance: f64,
    },

    /// Select vertices/faces above a plane (in normal direction).
    HalfSpace {
        point: Point3<f64>,
        normal: Vector3<f64>,
    },

    /// Select by vertex tag value.
    VertexTag(u32),

    /// Select vertices/faces where all vertices have offset above threshold.
    OffsetAbove(f32),

    /// Select vertices/faces where all vertices have offset below threshold.
    OffsetBelow(f32),

    /// Select by explicit vertex indices.
    Vertices(HashSet<u32>),

    /// Select by explicit face indices.
    Faces(HashSet<u32>),

    /// Combine selectors with AND.
    And(Box<RegionSelector>, Box<RegionSelector>),

    /// Combine selectors with OR.
    Or(Box<RegionSelector>, Box<RegionSelector>),

    /// Negate a selector.
    Not(Box<RegionSelector>),
}

impl RegionSelector {
    /// Create a bounding box selector.
    pub fn bounds(min: Point3<f64>, max: Point3<f64>) -> Self {
        RegionSelector::Bounds { min, max }
    }

    /// Create a sphere selector.
    pub fn sphere(center: Point3<f64>, radius: f64) -> Self {
        RegionSelector::Sphere { center, radius }
    }

    /// Create a cylinder selector.
    pub fn cylinder(axis_start: Point3<f64>, axis_end: Point3<f64>, radius: f64) -> Self {
        RegionSelector::Cylinder {
            axis_start,
            axis_end,
            radius,
        }
    }

    /// Create a plane selector (vertices within tolerance of plane).
    pub fn plane(point: Point3<f64>, normal: Vector3<f64>, tolerance: f64) -> Self {
        RegionSelector::Plane {
            point,
            normal: normal.normalize(),
            tolerance,
        }
    }

    /// Create a half-space selector (vertices above plane).
    pub fn half_space(point: Point3<f64>, normal: Vector3<f64>) -> Self {
        RegionSelector::HalfSpace {
            point,
            normal: normal.normalize(),
        }
    }

    /// Create a tag selector.
    pub fn tag(tag: u32) -> Self {
        RegionSelector::VertexTag(tag)
    }

    /// Create a selector for vertices with offset above threshold.
    pub fn offset_above(threshold: f32) -> Self {
        RegionSelector::OffsetAbove(threshold)
    }

    /// Create a selector for vertices with offset below threshold.
    pub fn offset_below(threshold: f32) -> Self {
        RegionSelector::OffsetBelow(threshold)
    }

    /// Create a selector from explicit vertex indices.
    pub fn vertices(indices: impl IntoIterator<Item = u32>) -> Self {
        RegionSelector::Vertices(indices.into_iter().collect())
    }

    /// Create a selector from explicit face indices.
    pub fn faces(indices: impl IntoIterator<Item = u32>) -> Self {
        RegionSelector::Faces(indices.into_iter().collect())
    }

    /// Combine with another selector using AND.
    pub fn and(self, other: RegionSelector) -> Self {
        RegionSelector::And(Box::new(self), Box::new(other))
    }

    /// Combine with another selector using OR.
    pub fn or(self, other: RegionSelector) -> Self {
        RegionSelector::Or(Box::new(self), Box::new(other))
    }

    /// Negate this selector.
    pub fn not(self) -> Self {
        RegionSelector::Not(Box::new(self))
    }

    /// Apply this selector to a mesh, returning (vertices, faces).
    pub fn select(&self, mesh: &Mesh) -> (HashSet<u32>, HashSet<u32>) {
        let vertices = self.select_vertices(mesh);

        // Also select faces where all vertices are selected
        let faces: HashSet<u32> = mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                vertices.contains(&face[0])
                    && vertices.contains(&face[1])
                    && vertices.contains(&face[2])
            })
            .map(|(i, _)| i as u32)
            .collect();

        (vertices, faces)
    }

    /// Select only vertices from this selector.
    pub fn select_vertices(&self, mesh: &Mesh) -> HashSet<u32> {
        match self {
            RegionSelector::Bounds { min, max } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| {
                    v.position.x >= min.x
                        && v.position.x <= max.x
                        && v.position.y >= min.y
                        && v.position.y <= max.y
                        && v.position.z >= min.z
                        && v.position.z <= max.z
                })
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::Sphere { center, radius } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| (v.position - center).norm() <= *radius)
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::Cylinder {
                axis_start,
                axis_end,
                radius,
            } => {
                let axis = axis_end - axis_start;
                let axis_len_sq = axis.norm_squared();
                if axis_len_sq < 1e-10 {
                    return HashSet::new();
                }

                mesh.vertices
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| {
                        let to_point = v.position - axis_start;
                        let t = to_point.dot(&axis) / axis_len_sq;
                        if t < 0.0 || t > 1.0 {
                            return false;
                        }
                        let projection = axis_start + axis * t;
                        (v.position - projection).norm() <= *radius
                    })
                    .map(|(i, _)| i as u32)
                    .collect()
            }

            RegionSelector::Plane {
                point,
                normal,
                tolerance,
            } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| (v.position - point).dot(normal).abs() <= *tolerance)
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::HalfSpace { point, normal } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| (v.position - point).dot(normal) >= 0.0)
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::VertexTag(tag) => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.tag == Some(*tag))
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::OffsetAbove(threshold) => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.offset.map(|o| o > *threshold).unwrap_or(false))
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::OffsetBelow(threshold) => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.offset.map(|o| o < *threshold).unwrap_or(false))
                .map(|(i, _)| i as u32)
                .collect(),

            RegionSelector::Vertices(indices) => indices.clone(),

            RegionSelector::Faces(face_indices) => {
                let mut vertices = HashSet::new();
                for &fi in face_indices {
                    if let Some(face) = mesh.faces.get(fi as usize) {
                        vertices.insert(face[0]);
                        vertices.insert(face[1]);
                        vertices.insert(face[2]);
                    }
                }
                vertices
            }

            RegionSelector::And(a, b) => {
                let va = a.select_vertices(mesh);
                let vb = b.select_vertices(mesh);
                va.intersection(&vb).copied().collect()
            }

            RegionSelector::Or(a, b) => {
                let va = a.select_vertices(mesh);
                let vb = b.select_vertices(mesh);
                va.union(&vb).copied().collect()
            }

            RegionSelector::Not(inner) => {
                let selected = inner.select_vertices(mesh);
                (0..mesh.vertex_count() as u32)
                    .filter(|i| !selected.contains(i))
                    .collect()
            }
        }
    }
}

/// A thickness map that assigns thickness values to vertices or faces.
#[derive(Debug, Clone)]
pub struct ThicknessMap {
    /// Per-vertex thickness values.
    pub vertex_thickness: HashMap<u32, f64>,

    /// Per-face thickness values (takes precedence over vertex if both set).
    pub face_thickness: HashMap<u32, f64>,

    /// Default thickness for vertices/faces not in the map.
    pub default_thickness: f64,
}

impl ThicknessMap {
    /// Create a new thickness map with a default thickness.
    pub fn new(default_thickness: f64) -> Self {
        Self {
            vertex_thickness: HashMap::new(),
            face_thickness: HashMap::new(),
            default_thickness,
        }
    }

    /// Create a uniform thickness map.
    pub fn uniform(thickness: f64) -> Self {
        Self::new(thickness)
    }

    /// Set thickness for a vertex.
    pub fn set_vertex_thickness(&mut self, vertex_index: u32, thickness: f64) {
        self.vertex_thickness.insert(vertex_index, thickness);
    }

    /// Set thickness for a face.
    pub fn set_face_thickness(&mut self, face_index: u32, thickness: f64) {
        self.face_thickness.insert(face_index, thickness);
    }

    /// Set thickness for all vertices in a region.
    pub fn set_region_thickness(&mut self, region: &MeshRegion, thickness: f64) {
        for &vi in &region.vertices {
            self.vertex_thickness.insert(vi, thickness);
        }
        for &fi in &region.faces {
            self.face_thickness.insert(fi, thickness);
        }
    }

    /// Get the thickness at a vertex.
    pub fn get_vertex_thickness(&self, vertex_index: u32) -> f64 {
        self.vertex_thickness
            .get(&vertex_index)
            .copied()
            .unwrap_or(self.default_thickness)
    }

    /// Get the thickness at a face.
    pub fn get_face_thickness(&self, face_index: u32) -> f64 {
        self.face_thickness
            .get(&face_index)
            .copied()
            .unwrap_or(self.default_thickness)
    }

    /// Get the thickness at a face, averaging vertex values if no face value is set.
    pub fn get_face_thickness_averaged(&self, mesh: &Mesh, face_index: u32) -> f64 {
        if let Some(&t) = self.face_thickness.get(&face_index) {
            return t;
        }

        if let Some(face) = mesh.faces.get(face_index as usize) {
            let t0 = self.get_vertex_thickness(face[0]);
            let t1 = self.get_vertex_thickness(face[1]);
            let t2 = self.get_vertex_thickness(face[2]);
            return (t0 + t1 + t2) / 3.0;
        }

        self.default_thickness
    }

    /// Create a thickness map from regions with specified thicknesses.
    pub fn from_regions(regions: &[(MeshRegion, f64)], default_thickness: f64) -> Self {
        let mut map = Self::new(default_thickness);
        for (region, thickness) in regions {
            map.set_region_thickness(region, *thickness);
        }
        map
    }

    /// Create a gradient thickness map between two regions.
    pub fn gradient(
        mesh: &Mesh,
        from_region: &MeshRegion,
        from_thickness: f64,
        to_region: &MeshRegion,
        to_thickness: f64,
        default_thickness: f64,
    ) -> Self {
        let mut map = Self::new(default_thickness);

        // Set explicit thicknesses for the regions
        map.set_region_thickness(from_region, from_thickness);
        map.set_region_thickness(to_region, to_thickness);

        // For vertices not in either region, interpolate based on distance
        let from_vertices: HashSet<u32> = from_region.vertices.iter().copied().collect();
        let to_vertices: HashSet<u32> = to_region.vertices.iter().copied().collect();

        // Compute average positions of each region
        let from_centroid = compute_centroid(mesh, &from_vertices);
        let to_centroid = compute_centroid(mesh, &to_vertices);

        if let (Some(fc), Some(tc)) = (from_centroid, to_centroid) {
            let axis = tc - fc;
            let axis_len = axis.norm();

            if axis_len > 1e-10 {
                let axis_normalized = axis / axis_len;

                for (vi, vertex) in mesh.vertices.iter().enumerate() {
                    let vi = vi as u32;
                    if from_vertices.contains(&vi) || to_vertices.contains(&vi) {
                        continue;
                    }

                    // Project onto axis
                    let to_vertex = vertex.position - fc;
                    let t = to_vertex.dot(&axis_normalized) / axis_len;
                    let t_clamped = t.clamp(0.0, 1.0);

                    let thickness = from_thickness + t_clamped * (to_thickness - from_thickness);
                    map.set_vertex_thickness(vi, thickness);
                }
            }
        }

        map
    }

    /// Apply thickness values to mesh vertex offsets.
    pub fn apply_to_mesh(&self, mesh: &mut Mesh) {
        for (vi, vertex) in mesh.vertices.iter_mut().enumerate() {
            let thickness = self.get_vertex_thickness(vi as u32);
            vertex.offset = Some(thickness as f32);
        }
    }
}

/// Compute the centroid of a set of vertices.
fn compute_centroid(mesh: &Mesh, vertices: &HashSet<u32>) -> Option<Point3<f64>> {
    if vertices.is_empty() {
        return None;
    }

    let sum: Vector3<f64> = vertices
        .iter()
        .filter_map(|&vi| mesh.vertices.get(vi as usize))
        .map(|v| v.position.coords)
        .sum();

    Some(Point3::from(sum / vertices.len() as f64))
}

/// Material zone for multi-material export.
#[derive(Debug, Clone)]
pub struct MaterialZone {
    /// The region this zone covers.
    pub region: MeshRegion,

    /// Material name or ID.
    pub material_name: String,

    /// Optional material properties.
    pub properties: MaterialProperties,
}

/// Material properties for a zone.
#[derive(Debug, Clone, Default)]
pub struct MaterialProperties {
    /// Density (g/cm³).
    pub density: Option<f64>,

    /// Elastic modulus / stiffness (MPa).
    pub elastic_modulus: Option<f64>,

    /// Shore hardness (A scale).
    pub shore_hardness: Option<f64>,

    /// Flexibility factor (0-1, 0=rigid, 1=very flexible).
    pub flexibility: Option<f64>,

    /// Color (RGB, 0-255).
    pub color: Option<(u8, u8, u8)>,

    /// Additional custom properties.
    pub custom: HashMap<String, String>,
}

impl MaterialZone {
    /// Create a new material zone.
    pub fn new(region: MeshRegion, material_name: impl Into<String>) -> Self {
        Self {
            region,
            material_name: material_name.into(),
            properties: MaterialProperties::default(),
        }
    }

    /// Set density property.
    pub fn with_density(mut self, density: f64) -> Self {
        self.properties.density = Some(density);
        self
    }

    /// Set elastic modulus property.
    pub fn with_elastic_modulus(mut self, modulus: f64) -> Self {
        self.properties.elastic_modulus = Some(modulus);
        self
    }

    /// Set shore hardness property.
    pub fn with_shore_hardness(mut self, hardness: f64) -> Self {
        self.properties.shore_hardness = Some(hardness);
        self
    }

    /// Set flexibility property.
    pub fn with_flexibility(mut self, flexibility: f64) -> Self {
        self.properties.flexibility = Some(flexibility.clamp(0.0, 1.0));
        self
    }

    /// Set color property.
    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.properties.color = Some((r, g, b));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);
        mesh
    }

    #[test]
    fn test_region_from_vertices() {
        let region = MeshRegion::from_vertices("test", vec![0, 1, 2]);
        assert_eq!(region.name, "test");
        assert_eq!(region.vertex_count(), 3);
        assert!(region.contains_vertex(0));
        assert!(region.contains_vertex(1));
        assert!(region.contains_vertex(2));
        assert!(!region.contains_vertex(3));
    }

    #[test]
    fn test_region_from_faces() {
        let region = MeshRegion::from_faces("test", vec![0, 1]);
        assert_eq!(region.name, "test");
        assert_eq!(region.face_count(), 2);
        assert!(region.contains_face(0));
        assert!(region.contains_face(1));
        assert!(!region.contains_face(2));
    }

    #[test]
    fn test_selector_bounds() {
        let mesh = create_test_cube();

        // Select bottom half
        let selector = RegionSelector::bounds(
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(11.0, 11.0, 5.0),
        );

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 4); // Bottom 4 vertices
        assert!(vertices.contains(&0));
        assert!(vertices.contains(&1));
        assert!(vertices.contains(&2));
        assert!(vertices.contains(&3));
    }

    #[test]
    fn test_selector_sphere() {
        let mesh = create_test_cube();

        // Select around center
        let selector = RegionSelector::sphere(Point3::new(5.0, 5.0, 5.0), 10.0);

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 8); // All vertices within 10mm of center
    }

    #[test]
    fn test_selector_half_space() {
        let mesh = create_test_cube();

        // Select top half (z > 5)
        let selector = RegionSelector::half_space(
            Point3::new(0.0, 0.0, 5.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 4); // Top 4 vertices
        assert!(vertices.contains(&4));
        assert!(vertices.contains(&5));
        assert!(vertices.contains(&6));
        assert!(vertices.contains(&7));
    }

    #[test]
    fn test_selector_and() {
        let mesh = create_test_cube();

        // Select top half AND x > 5
        let selector = RegionSelector::half_space(Point3::new(0.0, 0.0, 5.0), Vector3::z())
            .and(RegionSelector::half_space(Point3::new(5.0, 0.0, 0.0), Vector3::x()));

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 2); // Top-right vertices (5, 6)
        assert!(vertices.contains(&5));
        assert!(vertices.contains(&6));
    }

    #[test]
    fn test_selector_or() {
        let mesh = create_test_cube();

        // Select vertex 0 OR vertex 7
        let selector =
            RegionSelector::vertices(vec![0]).or(RegionSelector::vertices(vec![7]));

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 2);
        assert!(vertices.contains(&0));
        assert!(vertices.contains(&7));
    }

    #[test]
    fn test_selector_not() {
        let mesh = create_test_cube();

        // Select all except vertex 0
        let selector = RegionSelector::vertices(vec![0]).not();

        let (vertices, _) = selector.select(&mesh);
        assert_eq!(vertices.len(), 7);
        assert!(!vertices.contains(&0));
    }

    #[test]
    fn test_region_union() {
        let r1 = MeshRegion::from_vertices("a", vec![0, 1, 2]);
        let r2 = MeshRegion::from_vertices("b", vec![2, 3, 4]);

        let union = r1.union(&r2);
        assert_eq!(union.vertex_count(), 5);
        assert!(union.contains_vertex(0));
        assert!(union.contains_vertex(4));
    }

    #[test]
    fn test_region_intersection() {
        let r1 = MeshRegion::from_vertices("a", vec![0, 1, 2]);
        let r2 = MeshRegion::from_vertices("b", vec![2, 3, 4]);

        let intersection = r1.intersection(&r2);
        assert_eq!(intersection.vertex_count(), 1);
        assert!(intersection.contains_vertex(2));
    }

    #[test]
    fn test_region_difference() {
        let r1 = MeshRegion::from_vertices("a", vec![0, 1, 2]);
        let r2 = MeshRegion::from_vertices("b", vec![2, 3, 4]);

        let diff = r1.difference(&r2);
        assert_eq!(diff.vertex_count(), 2);
        assert!(diff.contains_vertex(0));
        assert!(diff.contains_vertex(1));
        assert!(!diff.contains_vertex(2));
    }

    #[test]
    fn test_region_map() {
        let mut map = RegionMap::new();

        let r1 = MeshRegion::from_vertices("top", vec![4, 5, 6, 7]);
        let r2 = MeshRegion::from_vertices("bottom", vec![0, 1, 2, 3]);

        map.add(r1);
        map.add(r2);

        assert_eq!(map.len(), 2);
        assert!(map.contains("top"));
        assert!(map.contains("bottom"));
        assert!(!map.contains("middle"));

        let regions = map.regions_containing_vertex(0);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0], "bottom");
    }

    #[test]
    fn test_thickness_map() {
        let mut map = ThicknessMap::new(2.0);

        map.set_vertex_thickness(0, 3.0);
        map.set_vertex_thickness(1, 4.0);

        assert!((map.get_vertex_thickness(0) - 3.0).abs() < 1e-10);
        assert!((map.get_vertex_thickness(1) - 4.0).abs() < 1e-10);
        assert!((map.get_vertex_thickness(2) - 2.0).abs() < 1e-10); // default
    }

    #[test]
    fn test_thickness_map_from_regions() {
        let r1 = MeshRegion::from_vertices("thick", vec![0, 1]);
        let r2 = MeshRegion::from_vertices("thin", vec![2, 3]);

        let map = ThicknessMap::from_regions(&[(r1, 5.0), (r2, 1.0)], 2.0);

        assert!((map.get_vertex_thickness(0) - 5.0).abs() < 1e-10);
        assert!((map.get_vertex_thickness(2) - 1.0).abs() < 1e-10);
        assert!((map.get_vertex_thickness(4) - 2.0).abs() < 1e-10); // default
    }

    #[test]
    fn test_material_zone() {
        let region = MeshRegion::from_vertices("heel", vec![0, 1, 2]);
        let zone = MaterialZone::new(region, "TPU-95A")
            .with_shore_hardness(95.0)
            .with_density(1.2)
            .with_color(255, 128, 0);

        assert_eq!(zone.material_name, "TPU-95A");
        assert!((zone.properties.shore_hardness.unwrap() - 95.0).abs() < 1e-10);
        assert!((zone.properties.density.unwrap() - 1.2).abs() < 1e-10);
        assert_eq!(zone.properties.color, Some((255, 128, 0)));
    }

    #[test]
    fn test_expand_region() {
        let mesh = create_test_cube();

        let mut region = MeshRegion::from_faces("test", vec![0]); // First face [0, 2, 1]
        assert!(region.vertices.is_empty());

        region.expand_to_face_vertices(&mesh);
        assert_eq!(region.vertex_count(), 3);
        assert!(region.contains_vertex(0));
        assert!(region.contains_vertex(1));
        assert!(region.contains_vertex(2));
    }

    #[test]
    fn test_unassigned_vertices() {
        let mesh = create_test_cube();
        let mut map = RegionMap::new();

        let r1 = MeshRegion::from_vertices("partial", vec![0, 1, 2]);
        map.add(r1);

        let unassigned = map.unassigned_vertices(&mesh);
        assert_eq!(unassigned.len(), 5);
        assert!(!unassigned.contains(&0));
        assert!(unassigned.contains(&3));
    }

    #[test]
    fn test_selector_cylinder() {
        let mesh = create_test_cube();

        // Cylinder along Z axis at center with radius 10 (should get all vertices)
        let selector = RegionSelector::cylinder(
            Point3::new(5.0, 5.0, 0.0),
            Point3::new(5.0, 5.0, 10.0),
            10.0,
        );

        let (vertices, _) = selector.select(&mesh);
        // All vertices of the cube are within 10mm of the central axis
        assert!(!vertices.is_empty());
    }
}
