#!/usr/bin/env python3
"""Build the reconstruction template that triangulation needs, directly from database.db.

Triangulation with known poses only needs one camera plus N images whose image_id/name/camera_id agree
with database.db, each attached to a registered frame: intrinsics and poses are overwritten from the
pose prior, and triangulate_points(clear_points=True) discards any 3D points. Running the COLMAP mapper
just to obtain this skeleton is unnecessary; building it from the database takes milliseconds.
"""
import pycolmap


def images_with_pairs(dbp):
    """image_ids of frames that have at least one verified pair (two_view_geometries.rows > 0).
    A frame with no pairs at all in the template crashes pycolmap.ObservationManager ("unordered_map::at");
    such frames occur with motion blur or when the camera turns toward a blank wall."""
    import sqlite3
    con = sqlite3.connect(dbp)
    have = set()
    for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
        if rows and rows > 0:
            have.add(pid // 2147483647); have.add(pid % 2147483647)
    con.close()
    return have


def build_template(dbp, cam_params=None, only_with_pairs=True):
    db = pycolmap.Database.open(dbp)
    cams = db.read_all_cameras()
    imgs = sorted(db.read_all_images(), key=lambda i: i.image_id)
    db.close()
    keep = images_with_pairs(dbp) if only_with_pairs else None
    rec = pycolmap.Reconstruction()
    for c in cams:
        if cam_params is not None:
            c.params = cam_params
        rec.add_camera_with_trivial_rig(c)
    for im in imgs:
        if keep is not None and im.image_id not in keep:
            continue
        rec.add_image_with_trivial_frame(im)
    for fid in list(rec.frames):
        rec.frame(fid).rig_from_world = pycolmap.Rigid3d()   # identity, overwritten by VGGT poses
        if not rec.frame(fid).has_pose:
            raise RuntimeError("frame %d has no pose after init" % fid)
        rec.register_frame(fid)
    return rec
