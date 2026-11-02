"""GPU pose-refinement core (triangulation and bundle adjustment).

Inherited constraints that matter:
  * a single camera, OPENCV model (another distortion model requires changing the derivatives in gpuba);
  * a single rig; rig_from_world is used as cam_from_world;
  * pycolmap is pinned STRICTLY to 4.0.3 -- the fast exporter reads the memory layout of the C++
    structs of that specific build and silently falls back to the slow path after an upgrade;
  * GPU-BA memory grows as the sum of squared track lengths.
"""
from . import bacore, gpuba, trigpu, template

__all__ = ["bacore", "gpuba", "trigpu", "template"]
