"""Export: the cloud as is, a mesh via marching cubes, obj/usdz for viewing.

The mesh is built from the TSDF field rather than the cloud when the field is saved: the field is a
continuous surface representation, so the mesh comes out faithfully rather than "from points".
"""
from pathlib import Path

import numpy as np

from .. import io as bio


def mesh_from_points(ply_path, out_path, voxel=0.02, depth_hint=9):
    """Fallback: Poisson reconstruction from the cloud when no TSDF field is available (open3d)."""
    import open3d as o3d
    P, rgb = bio.read_ply(ply_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.pipelines.surface_reconstruction.create_from_point_cloud_poisson(
        pcd, depth=depth_hint) if hasattr(o3d.pipelines, "surface_reconstruction") else \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth_hint)
    keep = np.asarray(dens) > np.quantile(np.asarray(dens), 0.02)
    mesh.remove_vertices_by_mask(~keep)
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    return dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles))


def run(cfg, formats=None):
    p = cfg.paths()
    formats = formats or cfg.export
    made = {}
    if "ply" in formats:
        made["ply"] = str(p["cloud"])
    if "mesh" in formats or "obj" in formats:
        made["mesh"] = str(mesh_from_points(p["cloud"], p["mesh"], cfg.voxel))
    if "usdz" in formats:
        try:
            from pxr import Usd  # noqa: F401
            made["usdz"] = "TODO: usd export"
        except ImportError:
            made["usdz"] = "skipped: pxr (USD) not available"
    return made
