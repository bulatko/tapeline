"""tapeline - room reconstruction from a video or from frames with known poses.

    from tapeline import Pipeline, Config
    Pipeline(Config(video="apt.mp4", work="runs/apt")).run()

Chain: frames -> initial poses -> matches -> pose refinement -> depth -> scale ->
canonical world -> fusion -> export. Every stage writes its artifacts to the work directory
and is checked by its own oracle; a red oracle stops the run.
"""
from .config import Config
from .runner import Pipeline, ORDER

__all__ = ["Config", "Pipeline", "ORDER"]
__version__ = "0.1.0"
