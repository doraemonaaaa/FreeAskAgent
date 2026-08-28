"""Floor-level validation of the actor's snapped waypoint."""

from __future__ import annotations

import numpy as np
import pytest

from agentflow.agents.models_embodied_v2.actor import Actor


CAMERA_HEIGHT_M = 1.25
WALL_DEPTH_M = 3.0
INTRINSICS = np.array(
    ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
)


def _room_depth(height=24, width=32):
    """A level camera facing a wall: floor below the horizon, wall beyond."""
    depth = np.full((height, width), WALL_DEPTH_M, dtype=np.float64)
    fy, cy = INTRINSICS[1, 1], INTRINSICS[1, 2]
    for v in range(height):
        if v > cy:
            floor_range = CAMERA_HEIGHT_M * fy / (v - cy)
            if floor_range < WALL_DEPTH_M:
                depth[v, :] = floor_range
    return depth


def _camera_to_world():
    transform = np.eye(4)
    transform[1, 3] = CAMERA_HEIGHT_M
    return transform


def _actor(**kwargs):
    return Actor(engine=object(), patch_radius_px=0, **kwargs)


def test_centre_pixel_snaps_to_floor_when_camera_height_is_known():
    actor = _actor(camera_height_m=CAMERA_HEIGHT_M)
    depth = _room_depth()

    point = actor.waypoint_from_pixel(
        (16, 12), depth, INTRINSICS, _camera_to_world()
    )

    assert point.on_floor is True
    assert actor.last_waypoint_on_floor is True
    # Nearest floor-level pixel to the horizon request: same column, first
    # row that back-projects within tolerance of the floor (the wall base).
    assert abs(point.world_xyz[1]) <= actor.max_floor_offset_m
    assert point.pixel_uv[0] == 16
    assert point.pixel_uv[1] > INTRINSICS[1, 2]


def test_legacy_behaviour_without_camera_height_keeps_wall_point():
    actor = _actor()
    depth = _room_depth()

    point = actor.waypoint_from_pixel(
        (16, 12), depth, INTRINSICS, _camera_to_world()
    )

    assert point.on_floor is False
    assert point.pixel_uv == (16, 12)
    assert point.depth_m == pytest.approx(WALL_DEPTH_M)
    assert point.world_xyz[1] == pytest.approx(CAMERA_HEIGHT_M)


def test_floor_below_tolerance_is_not_walkable():
    """A basin 1 m below the floor is rejected in favour of the real floor."""
    actor = _actor(camera_height_m=CAMERA_HEIGHT_M)
    depth = _room_depth()
    fy, cy = INTRINSICS[1, 1], INTRINSICS[1, 2]
    # Carve a sunken basin into the left half: those rays hit a plane 1 m
    # lower, so their depth grows accordingly.
    for v in range(depth.shape[0]):
        if v > cy:
            depth[v, :16] = (CAMERA_HEIGHT_M + 1.0) * fy / (v - cy)

    point = actor.waypoint_from_pixel(
        (2, 23), depth, INTRINSICS, _camera_to_world()
    )

    assert point.on_floor is True
    assert point.pixel_uv[0] >= 16
    assert abs(point.world_xyz[1]) <= actor.max_floor_offset_m


def test_wall_only_view_falls_back_and_reports_off_floor():
    """Nose to a wall: no pixel reaches floor level, so the legacy snap is
    used and the point is flagged as off the floor."""
    actor = _actor(camera_height_m=CAMERA_HEIGHT_M)
    depth = np.full((24, 32), 1.0, dtype=np.float64)

    point = actor.waypoint_from_pixel(
        (16, 12), depth, INTRINSICS, _camera_to_world()
    )

    assert point.on_floor is False
    assert actor.last_waypoint_on_floor is False
    assert point.pixel_uv == (16, 12)


def test_invalid_camera_height_is_rejected():
    with pytest.raises(ValueError):
        _actor(camera_height_m=0.0)
    with pytest.raises(ValueError):
        _actor(max_floor_offset_m=0.0)
