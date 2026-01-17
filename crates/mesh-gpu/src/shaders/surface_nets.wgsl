// Surface Nets isosurface extraction shader.
//
// Surface Nets is an algorithm for extracting isosurfaces from volumetric data.
// It produces higher quality meshes than Marching Cubes with simpler logic.
//
// The algorithm:
// 1. Identify "active" cells (cells where the SDF changes sign)
// 2. For each active cell, compute vertex position by averaging edge crossings
// 3. Connect vertices from adjacent active cells to form quads/triangles

// Grid parameters
struct GridParams {
    origin: vec4<f32>,
    dims: vec4<u32>,      // x, y, z, padding
    voxel_size: f32,
    iso_value: f32,       // Isosurface value (typically 0.0)
    _padding: vec2<f32>,
}

// Output vertex structure
struct OutputVertex {
    position: vec4<f32>,  // xyz + padding
    normal: vec4<f32>,    // xyz + padding (computed from gradient)
}

// Bind groups
@group(0) @binding(0) var<storage, read> sdf_values: array<f32>;
@group(0) @binding(1) var<uniform> grid: GridParams;
@group(0) @binding(2) var<storage, read_write> active_cells: array<u32>;  // Per-cell flag (0 or 1)
@group(0) @binding(3) var<storage, read_write> cell_vertices: array<OutputVertex>;
@group(0) @binding(4) var<storage, read_write> vertex_count: atomic<u32>;

// Convert 3D coordinates to linear index (for SDF values, uses ZYX ordering)
fn linearize_sdf(x: u32, y: u32, z: u32) -> u32 {
    return z + y * grid.dims.z + x * grid.dims.y * grid.dims.z;
}

// Convert linear cell index to 3D coordinates
fn delinearize_cell(idx: u32) -> vec3<u32> {
    let cells_x = grid.dims.x - 1u;
    let cells_y = grid.dims.y - 1u;
    let cells_z = grid.dims.z - 1u;

    let z = idx % cells_z;
    let rem = idx / cells_z;
    let y = rem % cells_y;
    let x = rem / cells_y;
    return vec3<u32>(x, y, z);
}

// Get SDF value at grid position
fn get_sdf(x: u32, y: u32, z: u32) -> f32 {
    if x >= grid.dims.x || y >= grid.dims.y || z >= grid.dims.z {
        return 1000.0; // Outside grid, large positive value
    }
    return sdf_values[linearize_sdf(x, y, z)];
}

// Helper for saturating subtraction (WGSL doesn't have it built-in for u32)
fn saturating_sub(a: u32, b: u32) -> u32 {
    if a >= b {
        return a - b;
    }
    return 0u;
}

// Compute gradient (normal) at a point using central differences
fn compute_gradient(x: u32, y: u32, z: u32) -> vec3<f32> {
    let dx = get_sdf(x + 1u, y, z) - get_sdf(saturating_sub(x, 1u), y, z);
    let dy = get_sdf(x, y + 1u, z) - get_sdf(x, saturating_sub(y, 1u), z);
    let dz = get_sdf(x, y, z + 1u) - get_sdf(x, y, saturating_sub(z, 1u));

    let grad = vec3<f32>(dx, dy, dz);
    let len = length(grad);
    if len > 0.0001 {
        return grad / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0); // Default up normal
}

// Linear interpolation for finding edge crossing
fn interpolate_crossing(p0: vec3<f32>, p1: vec3<f32>, v0: f32, v1: f32) -> vec3<f32> {
    let iso = grid.iso_value;
    if abs(v1 - v0) < 0.0001 {
        return (p0 + p1) * 0.5;
    }
    let t = (iso - v0) / (v1 - v0);
    return p0 + t * (p1 - p0);
}

// Get world position of a voxel corner
fn corner_position(x: u32, y: u32, z: u32) -> vec3<f32> {
    return vec3<f32>(
        grid.origin.x + f32(x) * grid.voxel_size,
        grid.origin.y + f32(y) * grid.voxel_size,
        grid.origin.z + f32(z) * grid.voxel_size
    );
}

// Pass 1: Identify active cells (cells with sign change)
@compute @workgroup_size(256, 1, 1)
fn identify_active_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cells_x = grid.dims.x - 1u;
    let cells_y = grid.dims.y - 1u;
    let cells_z = grid.dims.z - 1u;
    let total_cells = cells_x * cells_y * cells_z;

    if idx >= total_cells {
        return;
    }

    let coords = delinearize_cell(idx);
    let x = coords.x;
    let y = coords.y;
    let z = coords.z;

    // Get SDF values at 8 corners of the cell
    let v000 = get_sdf(x, y, z);
    let v100 = get_sdf(x + 1u, y, z);
    let v010 = get_sdf(x, y + 1u, z);
    let v110 = get_sdf(x + 1u, y + 1u, z);
    let v001 = get_sdf(x, y, z + 1u);
    let v101 = get_sdf(x + 1u, y, z + 1u);
    let v011 = get_sdf(x, y + 1u, z + 1u);
    let v111 = get_sdf(x + 1u, y + 1u, z + 1u);

    let iso = grid.iso_value;

    // Check for sign change (at least one corner above iso, one below)
    let below = (v000 < iso) || (v100 < iso) || (v010 < iso) || (v110 < iso) ||
                (v001 < iso) || (v101 < iso) || (v011 < iso) || (v111 < iso);
    let above = (v000 >= iso) || (v100 >= iso) || (v010 >= iso) || (v110 >= iso) ||
                (v001 >= iso) || (v101 >= iso) || (v011 >= iso) || (v111 >= iso);

    // Set per-cell activity flag (1 = active, 0 = inactive)
    if below && above {
        active_cells[idx] = 1u;
    } else {
        active_cells[idx] = 0u;
    }
}

// Pass 2: Generate vertices for active cells
@compute @workgroup_size(256, 1, 1)
fn generate_vertices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cells_x = grid.dims.x - 1u;
    let cells_y = grid.dims.y - 1u;
    let cells_z = grid.dims.z - 1u;
    let total_cells = cells_x * cells_y * cells_z;

    if idx >= total_cells {
        return;
    }

    // Check if this cell is active (per-cell flag)
    if active_cells[idx] == 0u {
        return;
    }

    let coords = delinearize_cell(idx);
    let x = coords.x;
    let y = coords.y;
    let z = coords.z;

    // Get corner positions and SDF values
    let p000 = corner_position(x, y, z);
    let p100 = corner_position(x + 1u, y, z);
    let p010 = corner_position(x, y + 1u, z);
    let p110 = corner_position(x + 1u, y + 1u, z);
    let p001 = corner_position(x, y, z + 1u);
    let p101 = corner_position(x + 1u, y, z + 1u);
    let p011 = corner_position(x, y + 1u, z + 1u);
    let p111 = corner_position(x + 1u, y + 1u, z + 1u);

    let v000 = get_sdf(x, y, z);
    let v100 = get_sdf(x + 1u, y, z);
    let v010 = get_sdf(x, y + 1u, z);
    let v110 = get_sdf(x + 1u, y + 1u, z);
    let v001 = get_sdf(x, y, z + 1u);
    let v101 = get_sdf(x + 1u, y, z + 1u);
    let v011 = get_sdf(x, y + 1u, z + 1u);
    let v111 = get_sdf(x + 1u, y + 1u, z + 1u);

    let iso = grid.iso_value;

    // Find edge crossings and average them (Surface Nets algorithm)
    var crossing_sum = vec3<f32>(0.0, 0.0, 0.0);
    var crossing_count = 0.0;

    // Edge along X at (y=0, z=0)
    if (v000 < iso) != (v100 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p000, p100, v000, v100);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along X at (y=1, z=0)
    if (v010 < iso) != (v110 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p010, p110, v010, v110);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along X at (y=0, z=1)
    if (v001 < iso) != (v101 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p001, p101, v001, v101);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along X at (y=1, z=1)
    if (v011 < iso) != (v111 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p011, p111, v011, v111);
        crossing_count = crossing_count + 1.0;
    }

    // Edge along Y at (x=0, z=0)
    if (v000 < iso) != (v010 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p000, p010, v000, v010);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Y at (x=1, z=0)
    if (v100 < iso) != (v110 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p100, p110, v100, v110);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Y at (x=0, z=1)
    if (v001 < iso) != (v011 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p001, p011, v001, v011);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Y at (x=1, z=1)
    if (v101 < iso) != (v111 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p101, p111, v101, v111);
        crossing_count = crossing_count + 1.0;
    }

    // Edge along Z at (x=0, y=0)
    if (v000 < iso) != (v001 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p000, p001, v000, v001);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Z at (x=1, y=0)
    if (v100 < iso) != (v101 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p100, p101, v100, v101);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Z at (x=0, y=1)
    if (v010 < iso) != (v011 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p010, p011, v010, v011);
        crossing_count = crossing_count + 1.0;
    }
    // Edge along Z at (x=1, y=1)
    if (v110 < iso) != (v111 < iso) {
        crossing_sum = crossing_sum + interpolate_crossing(p110, p111, v110, v111);
        crossing_count = crossing_count + 1.0;
    }

    if crossing_count > 0.0 {
        let vertex_pos = crossing_sum / crossing_count;
        let normal = compute_gradient(x, y, z);

        // Allocate vertex index
        let vertex_idx = atomicAdd(&vertex_count, 1u);

        cell_vertices[idx] = OutputVertex(
            vec4<f32>(vertex_pos.x, vertex_pos.y, vertex_pos.z, f32(vertex_idx)),
            vec4<f32>(normal.x, normal.y, normal.z, 0.0)
        );
    }
}
