"""Oracles. A red gate stops the run - gates are part of the pipeline, not reporting.

Each returns dict(name, ok, **numbers). The thresholds follow the offline pose protocol
and the planarity measurements; they can be changed, but then the meaning of "green" changes too.
"""
from .poses import poses_gate
from .depth_convention import depth_convention_gate
from .frame_scale import frame_scale_gate
from .cloud import cloud_gate

__all__ = ["poses_gate", "depth_convention_gate", "frame_scale_gate", "cloud_gate"]
