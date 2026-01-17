// Collision detection shader for self-intersection testing.
//
// This shader performs parallel self-intersection detection on triangle meshes
// using a grid-based spatial hashing approach for broad-phase culling,
// followed by exact triangle-triangle intersection tests.

// Triangle structure (matches GpuTriangle in Rust)
struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
}

// AABB (Axis-Aligned Bounding Box)
struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

// Collision parameters
struct CollisionParams {
    triangle_count: u32,
    max_pairs: u32,       // Maximum intersection pairs to report
    epsilon: f32,         // Tolerance for intersection tests
    skip_adjacent: u32,   // Whether to skip adjacent triangles (1 = yes)
}

// Output intersection pair
struct IntersectionPair {
    tri_a: u32,
    tri_b: u32,
}

// Bind groups
@group(0) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(1) var<uniform> params: CollisionParams;
@group(0) @binding(2) var<storage, read_write> aabbs: array<AABB>;
@group(0) @binding(3) var<storage, read_write> intersection_pairs: array<IntersectionPair>;
@group(0) @binding(4) var<storage, read_write> pair_count: atomic<u32>;

// Compute AABB for a triangle
fn compute_aabb(tri: Triangle, epsilon: f32) -> AABB {
    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;

    var aabb: AABB;
    aabb.min = min(min(v0, v1), v2) - vec3<f32>(epsilon);
    aabb.max = max(max(v0, v1), v2) + vec3<f32>(epsilon);
    return aabb;
}

// Check if two AABBs overlap
fn aabb_overlap(a: AABB, b: AABB) -> bool {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y &&
           a.min.z <= b.max.z && a.max.z >= b.min.z;
}

// Cross product
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Project triangle onto axis and get interval
fn project_triangle(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, axis: vec3<f32>) -> vec2<f32> {
    let p0 = dot(v0, axis);
    let p1 = dot(v1, axis);
    let p2 = dot(v2, axis);
    return vec2<f32>(min(min(p0, p1), p2), max(max(p0, p1), p2));
}

// Check if two intervals overlap
fn intervals_overlap(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x <= b.y && a.y >= b.x;
}

// Triangle-triangle intersection test using Separating Axis Theorem (SAT)
fn triangles_intersect(
    a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>,
    b0: vec3<f32>, b1: vec3<f32>, b2: vec3<f32>,
    epsilon: f32
) -> bool {
    // Triangle A edges
    let ea0 = a1 - a0;
    let ea1 = a2 - a1;
    let ea2 = a0 - a2;

    // Triangle B edges
    let eb0 = b1 - b0;
    let eb1 = b2 - b1;
    let eb2 = b0 - b2;

    // Triangle normals
    let na = normalize(cross3(ea0, ea1));
    let nb = normalize(cross3(eb0, eb1));

    // Test triangle A normal
    let proj_a_na = project_triangle(a0, a1, a2, na);
    let proj_b_na = project_triangle(b0, b1, b2, na);
    if !intervals_overlap(proj_a_na, proj_b_na) {
        return false;
    }

    // Test triangle B normal
    let proj_a_nb = project_triangle(a0, a1, a2, nb);
    let proj_b_nb = project_triangle(b0, b1, b2, nb);
    if !intervals_overlap(proj_a_nb, proj_b_nb) {
        return false;
    }

    // Test 9 edge cross-product axes
    let axes = array<vec3<f32>, 9>(
        cross3(ea0, eb0), cross3(ea0, eb1), cross3(ea0, eb2),
        cross3(ea1, eb0), cross3(ea1, eb1), cross3(ea1, eb2),
        cross3(ea2, eb0), cross3(ea2, eb1), cross3(ea2, eb2)
    );

    for (var i = 0u; i < 9u; i = i + 1u) {
        let axis = axes[i];
        let len_sq = dot(axis, axis);

        // Skip degenerate axes (parallel edges)
        if len_sq < epsilon * epsilon {
            continue;
        }

        let axis_norm = axis / sqrt(len_sq);
        let proj_a = project_triangle(a0, a1, a2, axis_norm);
        let proj_b = project_triangle(b0, b1, b2, axis_norm);

        if !intervals_overlap(proj_a, proj_b) {
            return false;
        }
    }

    // No separating axis found - triangles intersect
    return true;
}

// Check if two triangles share a vertex (are adjacent)
fn triangles_share_vertex(
    a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>,
    b0: vec3<f32>, b1: vec3<f32>, b2: vec3<f32>,
    epsilon: f32
) -> bool {
    let eps_sq = epsilon * epsilon;

    // Check all 9 vertex pairs
    if distance(a0, b0) < eps_sq { return true; }
    if distance(a0, b1) < eps_sq { return true; }
    if distance(a0, b2) < eps_sq { return true; }
    if distance(a1, b0) < eps_sq { return true; }
    if distance(a1, b1) < eps_sq { return true; }
    if distance(a1, b2) < eps_sq { return true; }
    if distance(a2, b0) < eps_sq { return true; }
    if distance(a2, b1) < eps_sq { return true; }
    if distance(a2, b2) < eps_sq { return true; }

    return false;
}

fn distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return dot(d, d);
}

// Pass 1: Compute AABBs for all triangles
@compute @workgroup_size(256, 1, 1)
fn compute_aabbs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.triangle_count {
        return;
    }

    let tri = triangles[idx];
    aabbs[idx] = compute_aabb(tri, params.epsilon);
}

// Pass 2: Test triangle pairs for intersection
// This uses a simple O(n^2) approach; for production, a spatial hash would be faster
@compute @workgroup_size(256, 1, 1)
fn test_intersections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.triangle_count {
        return;
    }

    let tri_a = triangles[i];
    let aabb_a = aabbs[i];
    let a0 = tri_a.v0.xyz;
    let a1 = tri_a.v1.xyz;
    let a2 = tri_a.v2.xyz;

    // Test against all triangles with higher index
    for (var j = i + 1u; j < params.triangle_count; j = j + 1u) {
        // Check if we've hit the max pairs limit
        let current_count = atomicLoad(&pair_count);
        if current_count >= params.max_pairs {
            return;
        }

        let aabb_b = aabbs[j];

        // Broad phase: AABB overlap test
        if !aabb_overlap(aabb_a, aabb_b) {
            continue;
        }

        let tri_b = triangles[j];
        let b0 = tri_b.v0.xyz;
        let b1 = tri_b.v1.xyz;
        let b2 = tri_b.v2.xyz;

        // Skip adjacent triangles if requested
        if params.skip_adjacent == 1u {
            if triangles_share_vertex(a0, a1, a2, b0, b1, b2, params.epsilon) {
                continue;
            }
        }

        // Narrow phase: exact triangle-triangle intersection
        if triangles_intersect(a0, a1, a2, b0, b1, b2, params.epsilon) {
            // Atomically add intersection pair
            let pair_idx = atomicAdd(&pair_count, 1u);
            if pair_idx < params.max_pairs {
                intersection_pairs[pair_idx] = IntersectionPair(i, j);
            }
        }
    }
}
