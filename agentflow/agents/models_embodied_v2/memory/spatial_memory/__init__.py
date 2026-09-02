"""Spatial memory: occupancy map, world-anchored landmarks, committed targets."""

from .candidates import (
    annotate_image,
    describe_candidates,
    generate_candidates,
    project_to_pixel,
)
from .landmarks import (
    LandmarkRegistry,
)
from .occupancy_grid import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
)
from .spatial_memory import SpatialMemory
from .targets import CommittedTarget

__all__ = (
    "annotate_image",
    "describe_candidates",
    "generate_candidates",
    "project_to_pixel",
    "CommittedTarget",
    "FREE",
    "LandmarkRegistry",
    "OCCUPIED",
    "OccupancyGrid",
    "SpatialMemory",
    "UNKNOWN",
)
