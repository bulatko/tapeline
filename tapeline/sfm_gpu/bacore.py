"""GPU bundle adjustment core: exporting the problem from a pycolmap reconstruction into tensors,
the COLMAP OPENCV projection model, reprojection. Everything is float64.

Conventions (match COLMAP):
  Xc = R @ Xw + t          (rig_from_world: world -> camera)
  u = Xc.x/Xc.z, v = Xc.y/Xc.z
  OPENCV distortion, params = [fx, fy, cx, cy, k1, k2, p1, p2]
"""
import ctypes

import numpy as np
import torch


# ---------- problem export ----------
def export_problem(rec):
    """rec -> dict of numpy arrays.
    frames: list of frame_id in index order; poses R (F,3,3), t (F,3)
    points: xyz (P,3), pids order
    obs: pt_idx (N,), fr_idx (N,), xy (N,2)
    cam: params (8,)
    """
    frame_ids, R_l, t_l = [], [], []
    fidx = {}
    for fid, f in sorted(rec.frames.items()):
        p = f.rig_from_world
        fidx[fid] = len(frame_ids)
        frame_ids.append(fid)
        R_l.append(np.asarray(p.rotation.matrix(), float))
        t_l.append(np.asarray(p.translation, float))
    pids, X_l = [], []
    pidx = {}
    for pid, p3 in sorted(rec.points3D.items()):
        pidx[pid] = len(pids)
        pids.append(pid)
        X_l.append(np.asarray(p3.xyz, float))
    pt_i, fr_i, xy_l = [], [], []
    for iid, im in sorted(rec.images.items()):
        fid = im.frame_id() if callable(im.frame_id) else im.frame_id
        k = fidx[fid]
        for p2 in im.points2D:
            if p2.has_point3D():
                pt_i.append(pidx[p2.point3D_id])
                fr_i.append(k)
                xy_l.append(np.asarray(p2.xy, float))
    cam = list(rec.cameras.values())[0]
    return dict(frame_ids=frame_ids, pids=pids,
                R=np.asarray(R_l), t=np.asarray(t_l),
                X=np.asarray(X_l),
                pt=np.asarray(pt_i, np.int64), fr=np.asarray(fr_i, np.int64),
                xy=np.asarray(xy_l), cam=np.asarray(cam.params, float),
                camera_id=cam.camera_id)


def import_problem(rec, prob, R, t, X, cam_params):
    """Write optimized values back into the pycolmap reconstruction."""
    import pycolmap
    for k, fid in enumerate(prob["frame_ids"]):
        rec.frames[fid].rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.ascontiguousarray(R[k])), np.ascontiguousarray(t[k]))
    for k, pid in enumerate(prob["pids"]):
        rec.points3D[pid].xyz = np.ascontiguousarray(X[k])
    rec.cameras[prob["camera_id"]].params = np.asarray(cam_params, float)
    return rec


# ---------- projection model ----------
def project(Xw, R, t, cam):
    """Xw (N,3) world points, R (N,3,3), t (N,3) per-observation poses, cam (8,).
    Returns uv (N,2) and Xc (N,3)."""
    Xc = torch.einsum('nij,nj->ni', R, Xw) + t
    return distort_project(Xc, cam), Xc


def distort_project(Xc, cam):
    fx, fy, cx, cy, k1, k2, p1, p2 = [cam[i] for i in range(8)]
    z = Xc[:, 2]
    u = Xc[:, 0] / z
    v = Xc[:, 1] / z
    u2, v2 = u * u, v * v
    r2 = u2 + v2
    rad = k1 * r2 + k2 * r2 * r2
    du = u * rad + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u2)
    dv = v * rad + p1 * (r2 + 2.0 * v2) + 2.0 * p2 * u * v
    return torch.stack([fx * (u + du) + cx, fy * (v + dv) + cy], 1)


def reproj_errors(prob, R=None, t=None, X=None, cam=None, device="cpu"):
    """Per-observation reprojection error
    (observations with z<=1e-6 or non-finite values are discarded)."""
    dt = torch.float64
    R = torch.as_tensor(prob["R"] if R is None else R, dtype=dt, device=device)
    t = torch.as_tensor(prob["t"] if t is None else t, dtype=dt, device=device)
    X = torch.as_tensor(prob["X"] if X is None else X, dtype=dt, device=device)
    cam = torch.as_tensor(prob["cam"] if cam is None else cam, dtype=dt, device=device)
    pt = torch.as_tensor(prob["pt"], device=device)
    fr = torch.as_tensor(prob["fr"], device=device)
    xy = torch.as_tensor(prob["xy"], dtype=dt, device=device)
    Xc = torch.einsum('nij,nj->ni', R[fr], X[pt]) + t[fr]
    ok = torch.isfinite(Xc).all(1) & (Xc[:, 2] > 1e-6)
    uv = distort_project(Xc[ok], cam)
    e = torch.linalg.norm(uv - xy[ok], dim=1)
    e = e[torch.isfinite(e)]
    return e


def stats(e):
    e = e.detach().cpu().numpy()
    return dict(median=float(np.median(e)), mean=float(np.mean(e)),
                p90=float(np.percentile(e, 90)), p99=float(np.percentile(e, 99)),
                n_obs=int(len(e)))


class FastExporter:
    """Fast repeated export of the problem from a pycolmap reconstruction.

    Rationale: a naive Python traversal of points2D is slow, and the refinement loop inside
    triangulation needs several exports per round, so the overhead would eat the GPU speedup.
    This relies on the pixel coordinates of observations NOT changing (they are keypoints
    from the database); only the point3D_id assignment changes. xy is cached once,
    after that only the ids are read and vectorized with numpy.
    """

    INVALID = np.uint64(18446744073709551615)   # kInvalidPoint3DId

    def __init__(self, rec):
        self.xy_cache = {}
        self.fr_of = {}
        self.frame_ids = []
        fidx = {}
        for fid, f in sorted(rec.frames.items()):
            fidx[fid] = len(self.frame_ids)
            self.frame_ids.append(fid)
        self.image_ids = sorted(rec.images.keys())
        for iid in self.image_ids:
            im = rec.images[iid]
            fid = im.frame_id() if callable(im.frame_id) else im.frame_id
            self.fr_of[iid] = fidx[fid]
            n = len(im.points2D)
            self.xy_cache[iid] = np.fromiter(
                (c for p in im.points2D for c in p.xy), dtype=np.float64, count=2 * n
            ).reshape(n, 2)
        self.fidx = fidx

    def export(self, rec):
        pids = np.fromiter(rec.points3D.keys(), dtype=np.uint64)
        pids.sort()
        X = np.fromiter((c for i in pids for c in rec.points3D[i].xyz),
                        dtype=np.float64, count=3 * len(pids)).reshape(-1, 3)
        pt_l, fr_l, xy_l = [], [], []
        for iid in self.image_ids:
            im = rec.images[iid]
            n = len(im.points2D)
            ids = np.fromiter((p.point3D_id for p in im.points2D), dtype=np.uint64, count=n)
            ok = ids != self.INVALID
            if not ok.any():
                continue
            idx = np.searchsorted(pids, ids[ok])
            pt_l.append(idx)
            fr_l.append(np.full(int(ok.sum()), self.fr_of[iid], dtype=np.int64))
            xy_l.append(self.xy_cache[iid][ok])
        R = np.stack([np.asarray(rec.frames[f].rig_from_world.rotation.matrix(), float)
                      for f in self.frame_ids])
        t = np.stack([np.asarray(rec.frames[f].rig_from_world.translation, float)
                      for f in self.frame_ids])
        cam = list(rec.cameras.values())[0]
        return dict(frame_ids=self.frame_ids, pids=pids, R=R, t=t, X=X,
                    pt=np.concatenate(pt_l).astype(np.int64),
                    fr=np.concatenate(fr_l), xy=np.concatenate(xy_l),
                    cam=np.asarray(cam.params, float), camera_id=cam.camera_id)

    @staticmethod
    def write_points(rec, pids, X):
        for k in range(len(pids)):
            rec.points3D[int(pids[k])].xyz = X[k]

    @staticmethod
    def import_problem(rec, prob, R, t, X, cam_params):
        return import_problem(rec, prob, R, t, X, cam_params)


def _addr(a):
    """Data address of a numpy array that pybind returned as a REFERENCE to a C++ struct field."""
    return int(a.__array_interface__["data"][0])


class MemExporter:
    """Problem export/import WITHOUT a pybind call per element -- via numpy views of foreign memory.

    Rationale: per-element traversal of pycolmap through pybind dominates the cost of
    GPU triangulation, while C++ itself moves the whole reconstruction much faster; the
    difference is per-call overhead, not data volume.

    What it relies on (verified by probing, not assumed):
      * pybind returns `Point2D.xy` and `Point3D.xyz` BY REFERENCE (reference_internal): writes
        into the returned numpy array are visible from pycolmap => the fields have stable addresses;
      * `image.points2D` is a contiguous std::vector of POD structs (stride 32 bytes =
        Eigen::Vector2d xy + uint64 point3D_id + padding), so ALL ids and xy of a frame
        are read through a single numpy view;
      * `points3D` is a node-based std::unordered_map; references to elements are not invalidated
        by insertion/removal, so xyz addresses are cached by pid and computed only for new points.

    Safeguards (otherwise this would be blind poking at foreign memory):
      * the point3D_id column is LOCATED by matching against pybind values, not taken on faith;
      * the vector stride is checked against the address of the last element (a0 + (n-1)*stride);
      * at initialization all ids and xy of the first VERIFY_IMGS frames are compared bit-exactly;
      * on every export SPOT points are spot-checked; on mismatch the addresses are rebuilt
        from scratch and the check is repeated; a second mismatch raises an exception;
      * if the layout is not confirmed, `ok=False` and `make_exporter` falls back to FastExporter.
    """

    INVALID = np.uint64(18446744073709551615)   # kInvalidPoint3DId
    VERIFY_IMGS = 3
    SPOT = 32
    LUT_MAX = 50_000_000        # above this, the id->index table is not built (memory)

    def __init__(self, rec):
        self.frame_ids, fidx = [], {}
        for fid in sorted(rec.frames.keys()):
            fidx[fid] = len(self.frame_ids)
            self.frame_ids.append(fid)
        self.fidx = fidx
        self.image_ids = sorted(rec.images.keys())
        self.fr_of = {}
        for iid in self.image_ids:
            im = rec.images[iid]
            fid = im.frame_id() if callable(im.frame_id) else im.frame_id
            self.fr_of[iid] = fidx[fid]
        self.stride_words = None
        self.idcol = None
        self.ok = False
        self._cpids = np.zeros(0, np.uint64)
        self._caddr = np.zeros(0, np.int64)
        self._ns = None            # per-frame points2D counts at the previous export
        self._xy_all = None        # contiguous xy cache of all observations
        self._fr_all = None
        self.reason = ""
        self._detect(rec)

    # ---------- memory layout ----------
    @staticmethod
    def _words(addr, n, w):
        """(n, w) uint64 view at a foreign address. Copies and allocates nothing."""
        return np.ctypeslib.as_array(
            (ctypes.c_uint64 * (n * w)).from_address(addr)).reshape(n, w)

    def _detect(self, rec):
        cand, checked = None, 0
        for iid in self.image_ids:
            im = rec.images[iid]
            n = len(im.points2D)
            if n < 3:
                continue
            a0 = _addr(im.points2D[0].xy)
            st = _addr(im.points2D[1].xy) - a0
            if st <= 0 or st % 8 or st > 128:
                self.reason = "unsupported Point2D stride %d" % st
                return
            if _addr(im.points2D[n - 1].xy) != a0 + (n - 1) * st:
                self.reason = "points2D is not contiguous"
                return
            w = st // 8
            if self.stride_words is None:
                self.stride_words = w
            elif self.stride_words != w:
                self.reason = "Point2D stride is inconsistent"
                return
            W = self._words(a0, n, w)
            ids_py = np.fromiter((p.point3D_id for p in im.points2D), np.uint64, n)
            xy_py = np.fromiter((c for p in im.points2D for c in p.xy),
                                np.float64, 2 * n).reshape(n, 2)
            if not (W.view(np.float64)[:, :2] == xy_py).all():
                self.reason = "xy at address does not match pybind"
                return
            hit = {j for j in range(2, w) if bool((W[:, j] == ids_py).all())}
            cand = hit if cand is None else (cand & hit)
            checked += 1
            if checked >= self.VERIFY_IMGS:
                break
        if not cand or len(cand) != 1:
            self.reason = "point3D_id column is ambiguous: %s" % cand
            return
        self.idcol = cand.pop()
        self.ok = True

    # ---------- point addresses ----------
    def _addr_all(self, rec, pids):
        """Full address collection in a SINGLE items() pass -- no per-point hash lookup.
        That items() yields references rather than copies is verified right here
        on SPOT points against a regular key lookup."""
        n = len(pids)
        k_l = np.empty(n, np.uint64)
        a_l = np.empty(n, np.int64)
        i = 0
        for pid, p3 in rec.points3D.items():
            if i >= n:
                return None
            k_l[i] = pid
            a_l[i] = _addr(p3.xyz)
            i += 1
        if i != n:
            return None
        o = np.argsort(k_l)
        if not (k_l[o] == pids).all():
            return None
        adr = a_l[o]
        P = rec.points3D
        for j in range(0, n, max(1, n // self.SPOT)):
            if _addr(P[int(pids[j])].xyz) != adr[j]:
                return None            # items() returned copies -- this path is unusable
        return adr

    def _addr_for(self, rec, pids):
        """xyz addresses for pids (sorted). Cached by pid: only new points are computed."""
        if not len(self._cpids) and len(pids):
            adr = self._addr_all(rec, pids)
            if adr is not None:
                self._cpids, self._caddr = pids, adr
                return adr
        adr = np.zeros(len(pids), np.int64)
        if len(self._cpids):
            j = np.searchsorted(self._cpids, pids).clip(0, len(self._cpids) - 1)
            hit = self._cpids[j] == pids
            adr[hit] = self._caddr[j[hit]]
        else:
            hit = np.zeros(len(pids), bool)
        P = rec.points3D
        for k in np.flatnonzero(~hit):
            adr[k] = _addr(P[int(pids[k])].xyz)
        self._cpids, self._caddr = pids, adr
        return adr

    @staticmethod
    def _view(adr):
        if not len(adr):
            return np.zeros(0), np.zeros(0, np.int64)
        base = int(adr.min())
        span = int(adr.max()) - base + 24
        arr = np.ctypeslib.as_array(
            (ctypes.c_double * (span // 8 + 3)).from_address(base))
        off = (adr - base) // 8
        return arr, (off[:, None] + np.arange(3)).ravel()

    def _spot_ok(self, rec, pids, X):
        if not len(pids):
            return True
        P = rec.points3D
        for i in range(0, len(pids), max(1, len(pids) // self.SPOT)):
            if not (np.asarray(P[int(pids[i])].xyz, float) == X[i]).all():
                return False
        return True

    def _gather(self, rec, pids):
        arr, idx3 = self._view(self._addr_for(rec, pids))
        X = arr[idx3].reshape(-1, 3)
        if not self._spot_ok(rec, pids, X):          # address cache is stale -- rebuild
            self._cpids = np.zeros(0, np.uint64)
            arr, idx3 = self._view(self._addr_for(rec, pids))
            X = arr[idx3].reshape(-1, 3)
            if not self._spot_ok(rec, pids, X):
                raise RuntimeError("MemExporter: xyz addresses do not match pycolmap")
        return X

    # ---------- same as FastExporter.export, but without a per-element call ----------
    def export(self, rec):
        pids = np.fromiter(rec.points3D.keys(), dtype=np.uint64)
        pids.sort()
        X = self._gather(rec, pids)
        # id -> point index: a direct lookup table instead of searchsorted per frame;
        # COLMAP ids are dense and grow monotonically, but if they are sparse the table
        # is not built and the searchsorted path is used.
        lut = None
        if len(pids) and int(pids[-1]) < self.LUT_MAX:
            lut = np.full(int(pids[-1]) + 1, -1, np.int32)
            lut[pids] = np.arange(len(pids), dtype=np.int32)
        # Observations: a FEW large numpy operations instead of one small one per frame.
        # Observation xy never change (they are keypoints from the database) -- cached once
        # as a contiguous array; only the point3D_id assignment changes.
        W_l, n_l = [], []
        for iid in self.image_ids:
            im = rec.images[iid]
            n = len(im.points2D)
            n_l.append(n)
            W_l.append(None if n == 0
                       else self._words(_addr(im.points2D[0].xy), n, self.stride_words))
        if n_l != self._ns:                      # frame contents changed -- reset caches
            self._ns = n_l
            self._xy_all = None
            self._fr_all = np.repeat(np.array([self.fr_of[i] for i in self.image_ids],
                                              dtype=np.int64), n_l)
        ids_all = np.concatenate([W[:, self.idcol] for W in W_l if W is not None])
        if self._xy_all is None:
            self._xy_all = np.ascontiguousarray(
                np.concatenate([W.view(np.float64)[:, :2] for W in W_l if W is not None]))
        ok = ids_all != self.INVALID
        sel = ids_all[ok]
        pt = lut[sel] if lut is not None else np.searchsorted(pids, sel)
        fr = self._fr_all[ok]
        xy = self._xy_all[ok]
        R = np.stack([np.asarray(rec.frames[f].rig_from_world.rotation.matrix(), float)
                      for f in self.frame_ids])
        t = np.stack([np.asarray(rec.frames[f].rig_from_world.translation, float)
                      for f in self.frame_ids])
        cam = list(rec.cameras.values())[0]
        return dict(frame_ids=self.frame_ids, pids=pids, R=R, t=t, X=X,
                    pt=pt.astype(np.int64), fr=fr, xy=xy,
                    cam=np.asarray(cam.params, float), camera_id=cam.camera_id)

    def write_points(self, rec, pids, X):
        pids = np.asarray(pids, dtype=np.uint64)
        arr, idx3 = self._view(self._addr_for(rec, pids))
        X = np.ascontiguousarray(X, dtype=np.float64)
        arr[idx3] = X.ravel()
        if not self._spot_ok(rec, pids, X):
            raise RuntimeError("MemExporter: xyz write is not visible from pycolmap")

    def import_problem(self, rec, prob, R, t, X, cam_params):
        import pycolmap
        for k, fid in enumerate(prob["frame_ids"]):
            rec.frames[fid].rig_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(np.ascontiguousarray(R[k])), np.ascontiguousarray(t[k]))
        self.write_points(rec, prob["pids"], X)
        rec.cameras[prob["camera_id"]].params = np.asarray(cam_params, float)
        return rec


def make_exporter(rec, verbose=True):
    """MemExporter if the memory layout is confirmed, otherwise the slower FastExporter."""
    mx = MemExporter(rec)
    if mx.ok:
        return mx
    if verbose:
        print("   MemExporter NOT applicable (%s) -> FastExporter" % mx.reason, flush=True)
    return FastExporter(rec)
