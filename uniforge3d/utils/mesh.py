"""Shape-only remeshing and the established 512-to-1024 surface-voxel handoff."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import cumesh
import numpy as np
import torch
import trimesh
from skimage import measure

import o_voxel

def mesh_to_occupancy(mesh: trimesh.Trimesh, grid_size: int = 64) -> torch.Tensor:
    """Convert mesh to occupancy grid.

    Args:
        mesh: trimesh.Trimesh object
        grid_size: Size of occupancy grid (default 64)

    Returns:
        Occupancy grid as tensor
    """
    try:
        # Voxelize the mesh
        voxelized = mesh.voxelized(pitch=1.0/grid_size)
        occupancy = voxelized.matrix.astype(np.float32)

        # Ensure correct shape
        if occupancy.shape[0] < grid_size:
            pad_width = [(0, grid_size - s) for s in occupancy.shape]
            occupancy = np.pad(occupancy, pad_width)
        else:
            occupancy = occupancy[:grid_size, :grid_size, :grid_size]

        return torch.from_numpy(occupancy).bool()
    except Exception as e:
        raise ValueError(f"Failed to convert mesh to occupancy: {e}")


def fill_holes_occupancy(occupancy_logits: torch.Tensor) -> torch.Tensor:
    """Fill holes in occupancy grid using cumesh mesh processing.

    Args:
        occupancy_logits: Occupancy logits tensor

    Returns:
        Occupancy logits with filled holes
    """
    occupancy = (occupancy_logits > 0).cpu().numpy().astype(np.float32)

    try:
        # Extract mesh from occupancy
        vertices, faces, _, _ = measure.marching_cubes(occupancy, level=0.5)
        if len(vertices) == 0:
            return occupancy_logits

        # Use cumesh to fill holes
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.fill_holes(max_hole_perimeter=3e-2)

        # Get filled mesh
        filled_vertices = np.asarray(mesh.vertices)
        filled_faces = np.asarray(mesh.faces)

        # Re-voxelize to occupancy grid
        filled_mesh = trimesh.Trimesh(vertices=filled_vertices, faces=filled_faces)
        grid_size = occupancy.shape[0]

        # Voxelize the mesh
        filled_occupancy = filled_mesh.voxelized(pitch=1.0/grid_size).matrix.astype(np.float32)

        # Ensure same shape
        if filled_occupancy.shape != occupancy.shape:
            # Pad or crop to match
            if filled_occupancy.shape[0] < grid_size:
                pad_width = [(0, grid_size - s) for s in filled_occupancy.shape]
                filled_occupancy = np.pad(filled_occupancy, pad_width)
            else:
                filled_occupancy = filled_occupancy[:grid_size, :grid_size, :grid_size]

        return torch.from_numpy(filled_occupancy).bool().to(occupancy_logits.device)
    except (ValueError, AttributeError):
        # If filling fails, return original
        return occupancy_logits


def dual_contour_from_occupancy(occupancy_logits: torch.Tensor, grid_size: int = 512) -> trimesh.Trimesh:
    """Extract mesh from occupancy grid using marching cubes + cumesh optimization.

    Args:
        occupancy_logits: Occupancy logits tensor
        grid_size: Target grid size for upsampling (512, 1024, 2048, etc)
    """
    occupancy = (occupancy_logits > 0).cpu().numpy().astype(np.float32)
    current_size = occupancy.shape[0]

    # Upsample if needed
    if grid_size != current_size:
        import cv2
        occupancy = cv2.resize(occupancy, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
        occupancy = np.repeat(np.repeat(occupancy[np.newaxis, :, :], grid_size, axis=0), 1, axis=0)
        occupancy = np.transpose(occupancy, (1, 2, 0))  # Make it 3D

    try:
        # Use marching cubes to extract mesh from occupancy
        vertices, faces, _, _ = measure.marching_cubes(occupancy, level=0.5)

        if len(vertices) == 0:
            return trimesh.Trimesh(vertices=[], faces=[], process=False)

        # Normalize vertices to [-0.5, 0.5]
        vertices = (vertices / grid_size) - 0.5

        # Use cumesh to optimize mesh quality
        try:
            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            # Apply cumesh optimizations
            mesh.simplify(max(len(faces) // 2, 100), verbose=False)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
        except (RuntimeError, AttributeError):
            # If cumesh optimization fails, use raw marching cubes result
            pass

        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except (ValueError, AttributeError, RuntimeError):
        # Empty or invalid mesh, return empty trimesh
        return trimesh.Trimesh(vertices=[], faces=[], process=False)


def export_glb(mesh: trimesh.Trimesh, path: Path, *, overwrite: bool) -> Path:
    """Atomically publish one complete GLB without truncating an older result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".glb", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        mesh.export(temporary, file_type="glb", extension_webp=True)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def official_shape_only_postprocess(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    grid_size: int,
    pre_simplify_target: int,
    decimation_target: int,
    remesh_band: float,
    remesh_project: float,
) -> trimesh.Trimesh:

    grid_size = 1024
    """TRELLIS.2 example.py/to_glb geometry path without UV or texture work."""
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    print("[official_shape_only_postprocess] Initialized CuMesh with vertices and faces.")
    mesh.simplify(pre_simplify_target, verbose=False)
    print("[official_shape_only_postprocess] Simplified mesh with pre_simplify_target:", pre_simplify_target)
    try:
        mesh.fill_holes(max_hole_perimeter=3e-2)
    except RuntimeError as error:
        if "[CuMesh]" not in str(error) or "invalid configuration argument" not in str(error):
            raise
        warnings.warn(
            "CuMesh fill_holes rejected this mesh; continuing from the simplified mesh.",
            RuntimeWarning,
        )
        del mesh
        torch.cuda.empty_cache()
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(pre_simplify_target, verbose=False)

 
    #filled_vertices, filled_faces = mesh.read()
    
    #bvh = cumesh.cuBVH(filled_vertices, filled_faces)
# Instead of remesh_narrow_band_dc, just clean and simplify directly:
    '''mesh.fill_holes(max_hole_perimeter=3e-2)'''
    '''mesh.simplify(decimation_target, verbose=False)'''
    '''mesh.compute_vertex_normals()'''
    

    '''
    #grid_size = 1024 
    print("[official_shape_only_postprocess] Filled mesh with vertices and faces.")
    scale = (grid_size + 3 * remesh_band) / grid_size
    mesh.init(
        *cumesh.remeshing.remesh_narrow_band_dc(
            filled_vertices,
            filled_faces,
            center=torch.zeros(3, device=vertices.device, dtype=vertices.dtype),
            scale=scale,
            resolution=grid_size,
            band=remesh_band,
            project_back=remesh_project,
            verbose=False,
            bvh=bvh,
        )
    )'''
    mesh.simplify(decimation_target, verbose=False)
    print("[official_shape_only_postprocess] Simplified mesh with decimation_target:", decimation_target)
  
    mesh.compute_vertex_normals()
    output_vertices, output_faces = mesh.read()
    output_normals = mesh.read_vertex_normals()
    
    vertices_np = output_vertices.cpu().numpy()
    faces_np = output_faces.cpu().numpy()
    normals_np = output_normals.cpu().numpy()
    
    # Model coordinates -> GLB display coordinates: (x,y,z) -> (x,z,-y).
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]

    
    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
    )


def display_to_model(vertices: np.ndarray) -> np.ndarray:
    """Undo GLB export axes: (x,y,z) -> (x,-z,y)."""
    return np.asarray(vertices, dtype=np.float64)[:, [0, 2, 1]] * [1.0, -1.0, 1.0]


def normalize_to_cube(vertices: np.ndarray, target_max_side: float) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected nonempty [N,3] vertices")
    if not np.isfinite(vertices).all() or not 0 < target_max_side < 1:
        raise ValueError("Nonfinite mesh or invalid target side")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) * 0.5
    longest_side = float((upper - lower).max())
    if longest_side <= 1e-12:
        raise ValueError("Degenerate mesh bounds")
    scale = target_max_side / longest_side
    normalized = (vertices - center) * scale
    bounds = np.stack((normalized.min(axis=0), normalized.max(axis=0)))
    if not np.allclose(bounds.mean(axis=0), 0.0, atol=1e-10):
        raise RuntimeError("Normalized mesh is not AABB centered")
    if not np.isclose(np.ptp(normalized, axis=0).max(), target_max_side, atol=1e-10):
        raise RuntimeError("Normalized mesh does not have the requested longest side")
    if np.abs(normalized).max() >= 0.5:
        raise RuntimeError("Normalized mesh is outside the open unit cube")
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    return normalized, {
        "center_model": center.tolist(),
        "uniform_scale": scale,
        "target_max_side": target_max_side,
        "source_bounds_model": [lower.tolist(), upper.tolist()],
        "normalized_bounds_model": bounds.tolist(),
        "model_to_normalized_4x4": transform.tolist(),
    }



def surface_voxelize_ovoxel(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Voxelize surface using Microsoft's native o-voxel CUDA library."""
    # Ensure tensors are on CUDA and in the right format
    v_tensor = torch.from_numpy(vertices).float().cpu()
    f_tensor = torch.from_numpy(faces).long().cpu()
    
    # Define AABB and grid size matching your canonical space
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    print("[surface_voxelize_ovoxel] Starting voxelization with resolution:", resolution)
    # Native O-Voxel conversion (handles sparse occupancy & dual grid on GPU in milliseconds)
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=v_tensor,
        faces=f_tensor,
        grid_size=resolution,
        aabb=aabb,
    )
    print("[surface_voxelize_ovoxel] Voxelization completed. Number of occupied voxels:", voxel_indices.shape[0])
    
    # Return unique voxel coordinates as a numpy array
    return voxel_indices.cpu().numpy().astype(np.int32)
'''
def surface_voxelize(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    *,
    max_faces_per_chunk: int = 100_000,
) -> np.ndarray:
    """Voxelize triangle surfaces with an exact triangle-box SAT test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if max_faces_per_chunk <= 0:
        raise ValueError("max_faces_per_chunk must be positive")

    half_size = 0.5 / resolution
    coord_chunks = []
    
    for start in range(0, len(faces), max_faces_per_chunk):
        triangles = vertices[faces[start : start + max_faces_per_chunk]]
        grid_triangles = (triangles + 0.5) * resolution
        lower = np.floor(grid_triangles.min(axis=1)).astype(np.int32)
        upper = np.floor(grid_triangles.max(axis=1)).astype(np.int32)
        lower = np.clip(lower, 0, resolution - 1)
        upper = np.clip(upper, 0, resolution - 1)
        dimensions = upper - lower + 1

        # Group triangles by dimension in a single vectorized pass
        unique_dims, inverse_indices = np.unique(dimensions, axis=0, return_inverse=True)

        for i, dimensions_key in enumerate(unique_dims):
            triangle_ids = np.flatnonzero(inverse_indices == i)
            ranges = [np.arange(int(size), dtype=np.int32) for size in dimensions_key]
            offsets = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
            
            candidate_coords = lower[triangle_ids, None, :] + offsets[None, :, :]
            candidate_coords = candidate_coords.reshape(-1, 3)
            
            candidate_triangles = np.broadcast_to(
                triangles[triangle_ids, None, :, :],
                (len(triangle_ids), len(offsets), 3, 3),
            ).reshape(-1, 3, 3)
            
            centers = -0.5 + (candidate_coords.astype(np.float64, copy=False) + 0.5) / resolution
            intersects = _triangles_intersect_axis_aligned_boxes(
                candidate_triangles,
                centers,
                half_size,
            )
            if intersects.any():
                coord_chunks.append(candidate_coords[intersects])

    coords = np.concatenate(coord_chunks, axis=0) if coord_chunks else np.empty((0, 3), dtype=np.int32)
    if coords.size == 0:
        raise RuntimeError("Surface voxelization returned no voxels")
    coords = np.unique(coords, axis=0)
    if int(coords.min()) < 0 or int(coords.max()) >= resolution:
        raise RuntimeError("Surface voxels are outside the canonical grid")
    return coords
'''
def surface_voxelize(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    *,
    max_faces_per_chunk: int = 100_000,
) -> np.ndarray:
    """Voxelize triangle surfaces with an exact triangle-box SAT test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if max_faces_per_chunk <= 0:
        raise ValueError("max_faces_per_chunk must be positive")

    half_size = 0.5 / resolution
    coord_chunks = []
    print("[surface_voxelize] Starting surface voxelization with resolution:", resolution)
    print("[surface_voxelize] Number of faces to process:", len(faces))
    print("[surface_voxelize] Number of vertices to process:", len(vertices))
    print("[surface_voxelize] Half size of voxels:", half_size)
    print("[surface_voxelize] Starting processing faces in chunks of max_faces_per_chunk:", max_faces_per_chunk)
    for start in range(0, len(faces), max_faces_per_chunk):
        triangles = vertices[faces[start : start + max_faces_per_chunk]]
        grid_triangles = (triangles + 0.5) * resolution
        lower = np.floor(grid_triangles.min(axis=1)).astype(np.int32)
        upper = np.floor(grid_triangles.max(axis=1)).astype(np.int32)
        lower = np.clip(lower, 0, resolution - 1)
        upper = np.clip(upper, 0, resolution - 1)
        dimensions = upper - lower + 1

        for dimensions_key in np.unique(dimensions, axis=0):
            triangle_ids = np.flatnonzero(np.all(dimensions == dimensions_key, axis=1))
            ranges = [np.arange(int(size), dtype=np.int32) for size in dimensions_key]
            offsets = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
            candidate_coords = lower[triangle_ids, None, :] + offsets[None, :, :]
            candidate_coords = candidate_coords.reshape(-1, 3)
            candidate_triangles = np.broadcast_to(
                triangles[triangle_ids, None, :, :],
                (len(triangle_ids), len(offsets), 3, 3),
            ).reshape(-1, 3, 3)
            centers = -0.5 + (candidate_coords.astype(np.float64) + 0.5) / resolution
            intersects = _triangles_intersect_axis_aligned_boxes(
                candidate_triangles,
                centers,
                half_size,
            )
            coord_chunks.append(candidate_coords[intersects].astype(np.int32, copy=False))

    coords = np.concatenate(coord_chunks, axis=0) if coord_chunks else np.empty((0, 3), dtype=np.int32)
    if coords.size == 0:
        raise RuntimeError("Surface voxelization returned no voxels")
    coords = np.unique(coords, axis=0)
    if int(coords.min()) < 0 or int(coords.max()) >= resolution:
        raise RuntimeError("Surface voxels are outside the canonical grid")
    return coords


def _triangles_intersect_axis_aligned_boxes(
    triangles: np.ndarray,
    centers: np.ndarray,
    half_size: float,
) -> np.ndarray:
    """Vectorized Akenine-Moller triangle/AABB overlap test."""
    local = triangles - centers[:, None, :]
    edge0 = local[:, 1] - local[:, 0]
    edge1 = local[:, 2] - local[:, 1]
    edge2 = local[:, 0] - local[:, 2]
    keep = np.ones(len(local), dtype=bool)
    epsilon = np.finfo(np.float64).eps * 32

    keep &= np.all(local.min(axis=1) <= half_size + epsilon, axis=1)
    keep &= np.all(local.max(axis=1) >= -half_size - epsilon, axis=1)

    for edge in (edge0, edge1, edge2):
        axes = (
            np.stack((np.zeros(len(edge)), edge[:, 2], -edge[:, 1]), axis=1),
            np.stack((-edge[:, 2], np.zeros(len(edge)), edge[:, 0]), axis=1),
            np.stack((edge[:, 1], -edge[:, 0], np.zeros(len(edge))), axis=1),
        )
        for axis in axes:
            projection = np.einsum("nvc,nc->nv", local, axis)
            radius = half_size * np.abs(axis).sum(axis=1) + epsilon
            keep &= projection.min(axis=1) <= radius
            keep &= projection.max(axis=1) >= -radius

    normal = np.cross(edge0, local[:, 2] - local[:, 0])
    plane_distance = np.einsum("nc,nc->n", normal, local[:, 0])
    plane_radius = half_size * np.abs(normal).sum(axis=1) + epsilon
    keep &= np.abs(plane_distance) <= plane_radius
    return keep
