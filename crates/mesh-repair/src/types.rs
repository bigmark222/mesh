//! Core mesh data types.

use nalgebra::{Point3, Vector3};

/// A vertex in the mesh with optional computed attributes.
///
/// Coordinates are typically in millimeters but the library is unit-agnostic.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// 3D position.
    pub position: Point3<f64>,

    /// Unit normal vector, computed from adjacent faces.
    pub normal: Option<Vector3<f64>>,

    /// Application-specific tag (e.g., zone ID, material ID).
    pub tag: Option<u32>,

    /// Offset distance for this vertex (used by mesh-shell for variable offset).
    /// Positive = outward expansion, negative = compression.
    pub offset: Option<f32>,
}

impl Vertex {
    /// Create a new vertex with only position set.
    #[inline]
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            normal: None,
            tag: None,
            offset: None,
        }
    }

    /// Create a vertex from raw coordinates.
    #[inline]
    pub fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self::new(Point3::new(x, y, z))
    }
}

/// A triangle mesh with indexed vertices and faces.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex data.
    pub vertices: Vec<Vertex>,

    /// Triangle faces as indices into the vertex array.
    /// Each face is [v0, v1, v2] with counter-clockwise winding.
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Create a mesh with pre-allocated capacity.
    pub fn with_capacity(vertex_count: usize, face_count: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_count),
            faces: Vec::with_capacity(face_count),
        }
    }

    /// Number of vertices in the mesh.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of faces (triangles) in the mesh.
    #[inline]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if mesh is empty (no vertices or faces).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    /// Compute the axis-aligned bounding box.
    /// Returns (min_corner, max_corner) or None if mesh is empty.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0].position;
        let mut max = self.vertices[0].position;

        for vertex in &self.vertices[1..] {
            let p = &vertex.position;
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }

        Some((min, max))
    }

    /// Iterate over triangles, yielding Triangle structs with actual vertex data.
    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.faces.iter().map(|&[i0, i1, i2]| Triangle {
            v0: self.vertices[i0 as usize].position,
            v1: self.vertices[i1 as usize].position,
            v2: self.vertices[i2 as usize].position,
        })
    }

    /// Get a specific triangle by face index.
    pub fn triangle(&self, face_idx: usize) -> Option<Triangle> {
        self.faces.get(face_idx).map(|&[i0, i1, i2]| Triangle {
            v0: self.vertices[i0 as usize].position,
            v1: self.vertices[i1 as usize].position,
            v2: self.vertices[i2 as usize].position,
        })
    }

    /// Rotate mesh 90 degrees around X axis (Y becomes Z, Z becomes -Y).
    pub fn rotate_x_90(&mut self) {
        for vertex in &mut self.vertices {
            let old_y = vertex.position.y;
            let old_z = vertex.position.z;
            vertex.position.y = -old_z;
            vertex.position.z = old_y;

            if let Some(ref mut normal) = vertex.normal {
                let old_ny = normal.y;
                let old_nz = normal.z;
                normal.y = -old_nz;
                normal.z = old_ny;
            }
        }
    }

    /// Translate mesh so minimum Z is at zero.
    pub fn place_on_z_zero(&mut self) {
        if let Some((min, _)) = self.bounds() {
            let offset = -min.z;
            for vertex in &mut self.vertices {
                vertex.position.z += offset;
            }
        }
    }

    /// Translate mesh by the given vector.
    pub fn translate(&mut self, offset: Vector3<f64>) {
        for vertex in &mut self.vertices {
            vertex.position += offset;
        }
    }

    /// Scale mesh uniformly around the origin.
    pub fn scale(&mut self, factor: f64) {
        for vertex in &mut self.vertices {
            vertex.position.coords *= factor;
        }
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// A triangle with concrete vertex positions.
///
/// Utility type for geometric calculations. Winding is counter-clockwise
/// when viewed from the front (normal points toward viewer).
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Point3<f64>,
    pub v1: Point3<f64>,
    pub v2: Point3<f64>,
}

impl Triangle {
    /// Create a new triangle from three points.
    #[inline]
    pub fn new(v0: Point3<f64>, v1: Point3<f64>, v2: Point3<f64>) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute the (unnormalized) face normal via cross product.
    /// The direction follows the right-hand rule with CCW winding.
    #[inline]
    pub fn normal_unnormalized(&self) -> Vector3<f64> {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(&e2)
    }

    /// Compute the unit face normal.
    /// Returns None for degenerate triangles (zero area).
    pub fn normal(&self) -> Option<Vector3<f64>> {
        let n = self.normal_unnormalized();
        let len_sq = n.norm_squared();
        if len_sq > f64::EPSILON {
            Some(n / len_sq.sqrt())
        } else {
            None
        }
    }

    /// Compute the area of the triangle.
    #[inline]
    pub fn area(&self) -> f64 {
        self.normal_unnormalized().norm() * 0.5
    }

    /// Compute the centroid (center of mass).
    #[inline]
    pub fn centroid(&self) -> Point3<f64> {
        Point3::new(
            (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        )
    }

    /// Get the three edges as (start, end) pairs.
    pub fn edges(&self) -> [(Point3<f64>, Point3<f64>); 3] {
        [
            (self.v0, self.v1),
            (self.v1, self.v2),
            (self.v2, self.v0),
        ]
    }

    /// Check if the triangle is degenerate (zero or near-zero area).
    pub fn is_degenerate(&self, epsilon: f64) -> bool {
        self.area() < epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::from_coords(1.0, 2.0, 3.0);
        assert!(approx_eq(v.position.x, 1.0));
        assert!(approx_eq(v.position.y, 2.0));
        assert!(approx_eq(v.position.z, 3.0));
        assert!(v.normal.is_none());
        assert!(v.tag.is_none());
        assert!(v.offset.is_none());
    }

    #[test]
    fn test_triangle_normal() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );

        let normal = tri.normal().expect("non-degenerate triangle");
        assert!(approx_eq(normal.x, 0.0));
        assert!(approx_eq(normal.y, 0.0));
        assert!(approx_eq(normal.z, 1.0));
    }

    #[test]
    fn test_triangle_area() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(approx_eq(tri.area(), 0.5));
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        );
        let c = tri.centroid();
        assert!(approx_eq(c.x, 1.0));
        assert!(approx_eq(c.y, 1.0));
        assert!(approx_eq(c.z, 0.0));
    }

    #[test]
    fn test_degenerate_triangle_normal() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!(tri.normal().is_none());
    }

    #[test]
    fn test_mesh_bounds() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 5.0, 3.0));
        mesh.vertices.push(Vertex::from_coords(-2.0, 8.0, 1.0));

        let (min, max) = mesh.bounds().expect("non-empty mesh");
        assert!(approx_eq(min.x, -2.0));
        assert!(approx_eq(min.y, 0.0));
        assert!(approx_eq(min.z, 0.0));
        assert!(approx_eq(max.x, 10.0));
        assert!(approx_eq(max.y, 8.0));
        assert!(approx_eq(max.z, 3.0));
    }

    #[test]
    fn test_empty_mesh_bounds() {
        let mesh = Mesh::new();
        assert!(mesh.bounds().is_none());
    }

    #[test]
    fn test_mesh_is_empty() {
        let mesh = Mesh::new();
        assert!(mesh.is_empty());

        let mut mesh2 = Mesh::new();
        mesh2.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        assert!(mesh2.is_empty()); // no faces

        mesh2.faces.push([0, 0, 0]);
        assert!(!mesh2.is_empty());
    }
}
