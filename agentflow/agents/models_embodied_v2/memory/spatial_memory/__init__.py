"""Spatial memory: occupancy map, world-anchored landmarks, committed targets."""

from .candidates import (
    Candidate,
    annotate_image,
    describe_candidates,
    generate_candidates,
    project_to_pixel,
)
from .events import DoorwayCrossing, DoorwayCrossingDetector
from .landmarks import LandmarkRegistry, SpatialLandmark
from .occupancy_grid import FREE, OCCUPIED, UNKNOWN, Frontier, OccupancyGrid, path_length_m
from .spatial_memory import SpatialMemory
from .targets import CommittedTarget

__all__ = (
    "Candidate",
    "annotate_image",
    "describe_candidates",
    "generate_candidates",
    "project_to_pixel",
    "CommittedTarget",
    "DoorwayCrossing",
    "DoorwayCrossingDetector",
    "FREE",
    "Frontier",
    "LandmarkRegistry",
    "OCCUPIED",
    "OccupancyGrid",
    "SpatialLandmark",
    "SpatialMemory",
    "UNKNOWN",
    "path_length_m",
)
