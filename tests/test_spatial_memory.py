from __future__ import annotations

import math

import numpy as np
import pytest

from agentflow.agents.models_embodied_v2.data_models import CameraIntrinsics
from agentflow.agents.models_embodied_v2.memory.spatial_memory import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    CommittedTarget,
    LandmarkRegistry,
    OccupancyGrid,
    SpatialMemory,
)

# A wide vertical field of view so the floor is visible from ~1 m out even
# at the default pixel stride of the grid.
H, W = 96, 128
CAMERA_HEIGHT = 1.25
INTRINSICS = CameraIntrinsics(fx=40.0, fy=40.0, cx=64.0, cy=48.0)


def _pose(x: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Habitat-style camera_to_world: y up, camera looks along -z at yaw 0."""
    yaw = math.radians(yaw_deg)
    transform = np.eye(4)
    # forward = (sin yaw, 0, -cos yaw); right = (cos yaw, 0, sin yaw)
    transform[:3, 0] = (math.cos(yaw), 0.0, math.sin(yaw))
    transform[:3, 1] = (0.0, 1.0, 0.0)
    transform[:3, 2] = (-math.sin(yaw), 0.0, math.cos(yaw))
    transform[:3, 3] = (x, CAMERA_HEIGHT, z)
    return transform


def _room_depth(wall_m: float | None = 3.0) -> np.ndarray:
    """Flat floor seen by a level camera, optionally a wall straight ahead."""
    vs = np.arange(H, dtype=np.float64)[:, None]
    us = np.arange(W, dtype=np.float64)[None, :]
    below = vs - INTRINSICS.cy
    depth = np.full((H, W), 6.0)  # far ceiling/sky where no floor is visible
    floor = np.where(below > 0, CAMERA_HEIGHT * INTRINSICS.fy / np.maximum(below, 1e-6), np.inf)
    depth = np.minimum(depth, floor)
    if wall_m is not None:
        depth = np.minimum(depth, wall_m)
    return np.broadcast_to(depth, (H, W)).astype(np.float32) + 0 * us


def test_floor_becomes_free_and_wall_becomes_occupied():
    grid = OccupancyGrid(camera_height_m=CAMERA_HEIGHT, pixel_stride=1)
    pose = _pose(0.0, 0.0)
    for _ in range(3):
        grid.update(_room_depth(wall_m=3.0), INTRINSICS, pose)
    # Floor 1.5 m ahead (z = -1.5) is free; the wall face at z = -3 is occupied;
    # far behind the camera nothing was ever seen.
    assert grid.state_at(0.0, -1.5) == FREE
    assert grid.state_at(0.0, -3.0) == OCCUPIED
    assert grid.state_at(0.0, 4.0) == UNKNOWN
    assert grid.explored_area_m2() > 2.0


def test_plan_routes_around_a_known_obstacle():
    grid = OccupancyGrid(camera_height_m=CAMERA_HEIGHT)
    grid.update(_room_depth(wall_m=None), INTRINSICS, _pose(0.0, 0.0))
    # Paint a wall segment across the corridor with a gap on the right.
    for x in np.arange(-2.0, 0.6, 0.1):
        row, col = grid.world_to_cell(float(x), -2.0)
        grid._occ_hits[row, col] = 10
    path = grid.plan((0.0, 0.0), (0.0, -3.5), inflate_m=0.15)
    assert path is not None
    xs = [p[0] for p in path]
    # The route swings right (positive x) to pass the gap instead of crossing
    # the painted wall.
    assert max(xs) > 0.6
    assert all(grid.state_at(x, z) != OCCUPIED for x, z in path)


def test_frontiers_lie_at_the_edge_of_what_was_seen():
    grid = OccupancyGrid(camera_height_m=CAMERA_HEIGHT, pixel_stride=1)
    grid.update(_room_depth(wall_m=None), INTRINSICS, _pose(0.0, 0.0))
    frontiers = grid.frontiers((0.0, 0.0), 0.0)
    assert frontiers
    # Every frontier is a free cell at the border of the observed floor, and
    # the farthest ones lie ahead of the camera (negative z).
    ahead = [f for f in frontiers if f.centre_xz[1] < -1.0]
    assert ahead and all(abs(f.bearing_deg) < 90 for f in ahead)


def test_landmark_sightings_merge_into_a_running_mean():
    registry = LandmarkRegistry(merge_radius_m=1.0)
    first = registry.register("doorway", (2.0, 0.0, -3.0), step=1, subgoal_id="2")
    second = registry.register("doorway", (2.4, 0.0, -3.0), step=2, subgoal_id="2")
    far = registry.register("doorway", (8.0, 0.0, -3.0), step=3, subgoal_id="2")
    assert first is second and far is not first
    assert second.observations == 2
    assert second.world_xyz[0] == pytest.approx(2.2)
    assert registry.for_subgoal("2") is second  # best supported wins
    assert len(registry) == 2


def test_committed_target_lifecycle():
    target = CommittedTarget((0.0, 0.0, -2.0), "model_waypoint", "1", created_step=0,
                             tolerance_m=0.5, max_age_steps=12, stagnation_steps=3)
    # Walking toward it keeps it active; arriving within tolerance ends it.
    for step, z in enumerate((0.0, -0.5, -1.0, -1.4), start=1):
        assert target.update(step=step, position_xz=(0.0, z), yaw_deg=0.0) == "active"
    assert target.update(step=5, position_xz=(0.0, -1.9), yaw_deg=0.0) == "reached"

    stuck = CommittedTarget((0.0, 0.0, -2.0), "landmark", "1", created_step=0, stagnation_steps=3)
    for step in range(1, 5):
        status = stuck.update(step=step, position_xz=(0.0, 0.0), yaw_deg=0.0)
    assert status == "stagnant"

    passed = CommittedTarget((0.0, 0.0, -2.0), "preview", "1", created_step=0, tolerance_m=0.3)
    # Cutting the corner and ending 0.6 m past the point counts as done.
    assert passed.update(step=1, position_xz=(0.5, -2.6), yaw_deg=0.0) == "passed"


def test_spatial_memory_walks_the_planned_path_and_reports_state():
    memory = SpatialMemory(camera_height_m=CAMERA_HEIGHT)
    memory.observe(step=0, depth_m=_room_depth(wall_m=None), intrinsics=INTRINSICS,
                   camera_to_world=_pose(0.0, 0.0))
    memory.register_landmark("dining table", (0.5, 0.0, -3.0), subgoal_id="1", confidence=0.9)
    memory.commit_target((0.5, 0.0, -3.0), kind="landmark", subgoal_id="1", reason="test")
    waypoint, remaining, how = memory.next_waypoint()
    assert how == "path" and remaining == pytest.approx(3.04, abs=0.4)
    # The lookahead point is ~1.5 m along the path, not the far target.
    assert -1.9 < waypoint[2] < -1.1
    assert memory.active_target("1") is not None
    assert memory.active_target("2") is None  # a new subgoal drops the target
    assert memory.target is None and memory.last_release_reason == "subgoal changed"
    state = memory.diagnostics()
    assert state["landmarks"] == 1 and state["observations"] == 1 and state["target"] is None
    assert memory.summary() == "sp=-"


def test_choose_frontier_prefers_the_requested_bearing_and_skips_visited():
    memory = SpatialMemory(camera_height_m=CAMERA_HEIGHT)
    memory.observe(step=0, depth_m=_room_depth(wall_m=None), intrinsics=INTRINSICS,
                   camera_to_world=_pose(0.0, 0.0))
    ahead = memory.choose_frontier()
    assert ahead is not None and ahead[2] < 0.0
    memory.commit_target(ahead, kind="frontier", subgoal_id="1")
    memory.release_target("test")
    again = memory.choose_frontier()
    assert again is None or math.hypot(again[0] - ahead[0], again[2] - ahead[2]) >= 0.75
    assert memory.render_topdown().shape[2] == 3


def test_target_tracks_bearing_while_the_agent_turns_to_face_it():
    # Target 2 m to the right of a camera facing -z: bearing +90 while the
    # agent has not turned, ~0 once it faces +x (yaw 90).
    target = CommittedTarget((2.0, 0.0, 0.0), "model_waypoint", "1", created_step=0)
    target.update(step=1, position_xz=(0.0, 0.0), yaw_deg=0.0)
    assert target.bearings[-1] == pytest.approx(90.0)
    assert target.still_aligning(tolerance_deg=20.0)
    target.update(step=2, position_xz=(0.0, 0.0), yaw_deg=90.0)
    assert abs(target.bearings[-1]) < 1e-6
    # The earlier off-axis bearing keeps the exemption alive for the window.
    assert target.still_aligning(tolerance_deg=20.0)
    assert not CommittedTarget((0.0, 0.0, -2.0), "x", "1", created_step=0).still_aligning(tolerance_deg=20.0)
