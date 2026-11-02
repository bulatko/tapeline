"""Per-process model cache and background preloading.

get(key, fn): model from the cache (waiting for a background load if one is running), otherwise fn() and cache it.
prefetch(key, fn): run fn() in a background thread (weights load while the CPU stages run).
drop(key): evict from the cache (free the GPU).
"""
import threading

_cache, _threads, _err = {}, {}, {}
_pending = []  # deferred prefetches: started after the frames stage's fork pool (fork while a loader thread is alive = deadlock)
_lock = threading.Lock()


def _run(key, fn):
    try:
        v = fn()
        with _lock:
            _cache[key] = v
    except BaseException as e:  # background load failed - get() will reload on the main thread
        _err[key] = e


def prefetch(key, fn):
    with _lock:
        if key in _cache or key in _threads:
            return
        th = threading.Thread(target=_run, args=(key, fn), daemon=True, name=f"prefetch-{key}")
        _threads[key] = th
    th.start()


def defer(key, fn):
    """Defer a background load until run_pending() (called by frames after it creates its process pool)."""
    _pending.append((key, fn))


def run_pending():
    while _pending:
        prefetch(*_pending.pop(0))


def get(key, fn):
    for i, (k, _) in enumerate(list(_pending)):  # not started yet - load synchronously
        if k == key:
            _pending.pop(i)
            break
    th = _threads.get(key)
    if th is not None:
        th.join()
    with _lock:
        if key in _cache:
            return _cache[key]
    v = fn()
    with _lock:
        _cache[key] = v
    return v


def drop(key):
    th = _threads.pop(key, None)
    if th is not None:
        th.join()
    with _lock:
        _cache.pop(key, None)


def start_prefetch(cfg):
    """Background loads of VGGT (+dino) at process start - ALWAYS deferred: the frames stage (both the fast path and
    the default select()) creates a fork pool, and forking while a loader thread is alive deadlocks. The fast path
    starts them right after creating its pool (run_pending in select_fast), the default path after the stage (after_stage)."""
    if not getattr(cfg, "prefetch", False):
        return
    from .prior import load_vggt
    from .match import load_dino
    defer("vggt", lambda: load_vggt(cfg))
    if cfg.retrieval == "dino":
        defer("dino", lambda: load_dino(getattr(cfg, "fast_init", False)))


def after_stage(cfg, st):
    """Runner hook after stage st: after frames - the deferred loads (no more process pools follow), after prior - DA3."""
    if not getattr(cfg, "prefetch", False):
        return
    if st == "frames":
        run_pending()
    elif st == "prior":
        from .depth import _load_da3
        prefetch(("da3", cfg.depth_model), lambda: _load_da3(cfg.depth_model, getattr(cfg, "fast_init", False)))


class tf32:
    """TF32 for fp32 matmuls during a model forward pass (the VGGT camera head and fp32 parts of DA3 run outside autocast).
    Enabled only with cfg.tf32; the previous flags are restored on exit (BA/verify are unaffected)."""

    def __init__(self, cfg):
        self.on = bool(getattr(cfg, "tf32", False))

    def __enter__(self):
        if self.on:
            import torch
            self.prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return self

    def __exit__(self, *a):
        if self.on:
            import torch
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.prev
        return False
