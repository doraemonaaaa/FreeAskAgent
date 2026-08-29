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


def test_candidates_come_from_floor_sectors_landmark_and_rear_frontier():
    from agentflow.agents.models_embodied_v2.memory.spatial_memory import (
        annotate_image, generate_candidates, project_to_pixel,
    )
    from agentflow.agents.models_embodied_v2.memory.spatial_memory.occupancy_grid import Frontier

    depth = _room_depth(wall_m=3.0)
    pose = _pose(0.0, 0.0)
    # Floor mask from the same geometry the depth was built from.
    vs = np.arange(H, dtype=np.float64)[:, None]
    floor = np.broadcast_to((vs - INTRINSICS.cy) > 16.7, (H, W))  # rows whose floor depth < 3 m
    landmark = (1.0, 0.0, -2.5)
    behind = Frontier(centre_xz=(0.0, 3.0), size=12, distance_m=3.0, bearing_deg=180.0)
    candidates = generate_candidates(
        depth_m=depth, floor_mask=floor, intrinsics=INTRINSICS, camera_to_world=pose,
        image_shape=(H, W), frontiers=[behind], landmark_xyz=landmark, floor_y=0.0,
    )
    labels = [c.label for c in candidates]
    kinds = {c.label: c.kind for c in candidates}
    in_view = [c for c in candidates if c.kind != "turn"]
    assert 2 <= len(in_view) <= 5 and labels[: len(in_view)] == [str(i) for i in range(1, len(in_view) + 1)]
    assert "landmark" in kinds.values() and "opening" in kinds.values()
    assert "B" in labels and kinds["B"] == "turn"
    # Every in-view marker projects inside the frame and lies in front of the wall.
    for c in in_view:
        assert c.pixel_uv is not None and 0 <= c.pixel_uv[0] < W and 0 <= c.pixel_uv[1] < H
        assert c.world_xyz[2] > -3.0 and c.distance_m >= 0.8
    # Markers are numbered left to right.
    xs = [c.pixel_uv[0] for c in in_view]
    assert xs == sorted(xs)
    # The landmark's projection round-trips.
    lm = next(c for c in in_view if c.kind == "landmark")
    assert lm.pixel_uv == project_to_pixel(landmark, INTRINSICS, pose, (H, W))
    image = annotate_image(np.zeros((H, W, 3), dtype=np.uint8), candidates)
    assert image.shape == (H, W, 3) and image.max() == 255  # markers drawn
    text = "\n".join(c.describe() for c in candidates)
    assert "[1]" in text and "[B] turn around" in text


def test_committed_target_snaps_off_the_wall_clearance_band():
    grid = OccupancyGrid(camera_height_m=CAMERA_HEIGHT, pixel_stride=1)
    for _ in range(3):
        grid.update(_room_depth(wall_m=3.0), INTRINSICS, _pose(0.0, 0.0))
    memory = SpatialMemory(camera_height_m=CAMERA_HEIGHT)
    memory.grid = grid
    memory.observe(step=0, depth_m=_room_depth(wall_m=3.0), intrinsics=INTRINSICS,
                   camera_to_world=_pose(0.0, 0.0))
    # A point 0.15 m in front of the wall is inside the clearance band.
    target = memory.commit_target((0.0, 0.0, -2.85), kind="som", subgoal_id="1")
    assert target.world_xyz[2] > -2.85 + 0.15
    assert grid.state_at(target.world_xyz[0], target.world_xyz[2]) == FREE
    assert grid.snap_to_clear_free(0.0, -2.85) is not None
    assert grid.snap_to_clear_free(0.0, 20.0) is None  # nothing known there


def test_traversability_window_filters_and_snaps_targets():
    from agentflow.agents.models_embodied_v2.memory.spatial_memory.candidates import Candidate

    memory = SpatialMemory(camera_height_m=CAMERA_HEIGHT)
    memory.observe(step=0, depth_m=_room_depth(wall_m=None), intrinsics=INTRINSICS,
                   camera_to_world=_pose(0.0, 0.0))
    # 6 m window at 0.25 m: everything navigable except x > 1.5 (glass).
    mask = np.ones((24, 24), dtype=bool)
    mask[:, 18:] = False  # columns with x >= -3 + 18*0.25 = 1.5
    memory.set_traversability(origin_xz=(-3.0, -3.0), resolution_m=0.25, mask=mask)
    assert memory.is_navigable(0.0, -1.0) is True
    assert memory.is_navigable(2.5, -1.0) is False
    assert memory.is_navigable(10.0, 0.0) is None
    assert memory.snap_navigable(2.0, -1.0, radius_m=1.0) is not None
    assert memory.snap_navigable(2.9, -1.0, radius_m=1.0) is None
    # An unreachable target is refused; a reachable one is kept.
    assert memory.commit_target((2.9, 0.0, -1.0), kind="som", subgoal_id="1") is None
    assert memory.last_release_reason == "unreachable target"
    target = memory.commit_target((1.7, 0.0, -1.0), kind="som", subgoal_id="1")
    assert target is not None and target.world_xyz[0] < 1.5
    cands = [
        Candidate("1", (2.9, 0.0, -2.0), 3.5, 40.0, "opening", (100, 40)),
        Candidate("2", (0.0, 0.0, -2.0), 2.0, 0.0, "opening", (64, 40)),
        Candidate("3", (-1.0, 0.0, -2.0), 2.2, -20.0, "opening", (30, 40)),
        Candidate("L", (-3.0, 0.0, 0.0), 3.0, -90.0, "turn", None, "turn left"),
    ]
    kept = memory.filter_navigable(cands)
    assert [c.label for c in kept] == ["1", "2", "L"]  # renumbered left to right
    assert kept[0].world_xyz[0] == pytest.approx(-1.0)


def test_target_behind_the_agent_survives_the_turn_in_place():
    target = CommittedTarget((0.0, 0.0, 2.0), "frontier", "1", created_step=0,
                             stagnation_steps=6, max_age_steps=30)
    # Twelve 15-degree turns with no translation: bearing shrinks from 180 to 0.
    for step, yaw in enumerate(range(0, 181, 15), start=1):
        status = target.update(step=step, position_xz=(0.0, 0.0), yaw_deg=float(yaw))
        assert status == "active", (step, yaw, status)
    assert target.aligning_steps >= 10 and target.stagnant_steps <= 2
    # Facing it and still not moving does count as stagnation.
    for step in range(20, 27):
        status = target.update(step=step, position_xz=(0.0, 0.0), yaw_deg=180.0)
    assert status == "stagnant"


def test_visual_map_is_a_crisp_window_around_the_agent():
    memory = SpatialMemory(camera_height_m=CAMERA_HEIGHT)
    blank = memory.visual_map(size_px=120)
    assert blank.shape == (120, 120, 3) and int(blank[0, 0, 0]) == 128  # nothing known yet
    memory.observe(step=0, depth_m=_room_depth(wall_m=3.0), intrinsics=INTRINSICS,
                   camera_to_world=_pose(0.0, 0.0))
    memory.commit_target((0.0, 0.0, -2.0), kind="som", subgoal_id="1")
    image = memory.visual_map(window_m=8.0, size_px=160, extra_points=[(1.0, 0.0, -1.0)])
    assert image.shape == (160, 160, 3)
    colours = {tuple(c) for c in image.reshape(-1, 3)}
    assert (255, 255, 255) in colours      # free floor ahead
    assert (255, 140, 0) in colours        # the agent
    assert (230, 30, 30) in colours        # committed target
    assert (255, 220, 0) in colours        # the extra (candidate) point
