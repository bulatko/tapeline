"""Thread/process counts from the container's actual CPU quota (cgroup) rather than the number of visible cores.

In a container a job may see far more cores than its quota allows (e.g. 128 visible cores with a quota of 12,
cpu.max 1200000 100000); by default torch, cv2 and the frame workers size themselves by the visible cores, which
oversubscribes the CPU several times over and slows the CPU stages down.
The module imports only os/sys so it can be called before numpy/torch (the OMP/MKL environment variables only take
effect that way).
"""
import os


def quota():
    """Integer CPU count: min(cgroup v2/v1 quota, affinity). Without a quota - affinity (or os.cpu_count())."""
    try:
        n_aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n_aff = os.cpu_count() or 1
    q = None
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        a, b = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if a != "max":
            q = int(a) / int(b)
    except (OSError, ValueError):
        try:  # cgroup v1
            a = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            b = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if a > 0 and b > 0:
                q = a / b
        except (OSError, ValueError):
            pass
    n = n_aff if q is None else min(n_aff, int(q))
    return max(1, n)


def set_env(n):
    """Before importing numpy/torch: OpenMP/MKL/OpenBLAS threads (does not override values set externally)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(n))


def apply(n):
    """After import: torch, cv2, BLAS via threadpoolctl. Returns a dict of the actual values."""
    out = {"cpu_quota": n}
    try:
        import torch
        torch.set_num_threads(n)
        out["torch"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        cv2.setNumThreads(n)
        out["cv2"] = cv2.getNumThreads()
    except Exception:  # noqa: BLE001
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
        out["blas"] = n
    except Exception:  # noqa: BLE001
        pass
    return out


def limit(cfg, want):
    """want, capped by the quota when cfg.cpu_quota is set (otherwise want unchanged)."""
    if getattr(cfg, "cpu_quota", False):
        return max(1, min(int(want), quota()))
    return int(want)
