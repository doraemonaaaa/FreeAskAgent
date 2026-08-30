from __future__ import annotations

import numpy as np
import pytest

from agentflow.agents.vln_agent_4 import VLNAgent
from agentflow.agents.models_embodied_v2.data_models import PreviewView


SCENE_IN_PROGRESS = (
    '{"landmark":{"visible":false,"direction":"UNKNOWN",'
    '"proximity":"UNKNOWN","passed":false,'
    '"destination_dominant":false,"u":null,"v":null,'
    '"confidence":0.1},"door_state":"NOT_APPLICABLE",'
    '"door_camera_side":"NOT_APPLICABLE",'
    '"door_transition":"NOT_APPLICABLE",'
    '"current_room_side":"NOT_APPLICABLE","completed":false,'
    '"completion_confidence":0.1,"error_mode":"NONE",'
    '"error_confidence":0.0,"final_target":{"visible":false,'
    '"proximity":"UNKNOWN","confidence":0.1},'
    '"evidence":"open floor continues toward the pool"}'
)

WAYPOINT_FORWARD = (
    '{"action_mode":"EXECUTION","execution":{"stop":false,'
    '"intent":"FINAL_APPROACH","u":500,"v":750},'
    '"confidence":0.9,"evidence":"open floor continues ahead"}'
)

WAYPOINT_PREVIEW = (
    '{"action_mode":"PREVIEW","preview":true,'
    '"confidence":0.8,"evidence":"doorway is outside current view"}'
)

WAYPOINT_FINAL_STOP = (
    '{"action_mode":"EXECUTION","execution":{"stop":true,'
    '"intent":"STOP"},"confidence":0.99,'
    '"evidence":"The camera is positioned directly beside the pool with '
    'no other room visible."}'
)


class RoutedEngine:
    supports_image_pixel_budget = True

    def __init__(self):
        self.calls = []

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label = "plan"
            response = (
                "1|Walk forward to the pool area|"
                "The camera is directly beside the pool"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label = "scene"
            response = SCENE_IN_PROGRESS
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label = "waypoint"
            response = WAYPOINT_FORWARD
        else:
            raise AssertionError("unexpected independent VLM call")
        self.calls.append((label, content, kwargs))
        return response


class FinalStopEngine(RoutedEngine):
    def __init__(self, waypoint_response=WAYPOINT_FINAL_STOP):
        super().__init__()
        self.waypoint_response = waypoint_response

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label = "plan"
            response = (
                "1|Walk forward to the pool area|"
                "The camera is positioned directly beside the pool with no "
                "other room visible"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label = "scene"
            response = SCENE_IN_PROGRESS
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label = "waypoint"
            response = self.waypoint_response
        else:
            raise AssertionError("unexpected independent VLM call")
        self.calls.append((label, content, kwargs))
        return response


def test_vln_agent_4_uses_one_scene_call_then_one_waypoint_call():
    engine = RoutedEngine()
    agent = VLNAgent(engine=engine)
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)

    decision = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        np.array(
            (
                (16.0, 0.0, 16.0),
                (0.0, 16.0, 12.0),
                (0.0, 0.0, 1.0),
            )
        ),
        np.eye(4),
    )

    assert decision.stop is False
    assert decision.point is not None
    assert [label for label, _, _ in engine.calls] == [
        "plan",
        "scene",
        "waypoint",
    ]
    assert agent.last_caption is not None
    assert agent.last_caption.subgoal_id == "1"
    assert agent.temporal_memory.diagnostics()["frame_ids"] == [1]


def test_repeated_grounded_final_waypoint_stop_ends_episode():
    engine = FinalStopEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 2.0, dtype=np.float32)

    first = agent.act(rgb, depth, instruction, intrinsics, np.eye(4))
    second = agent.act(rgb, depth, instruction, intrinsics, np.eye(4))

    assert first.stop is False
    assert second.stop is True
    assert agent.last_waypoint_stop_disposition == (
        "accepted_repeated_visual_final"
    )
    assert "accepted repeated" in agent.last_waypoint_guard_reason
    # The independent visual guard stops navigation without forging a
    # Temporal Memory completion event.
    assert not agent.task_memory.is_task_complete()


def test_repeated_contradictory_final_waypoint_stop_remains_deferred():
    response = (
        '{"action_mode":"EXECUTION","execution":{"stop":true,'
        '"intent":"STOP"},"confidence":0.99,'
        '"evidence":"The pool is visible in the distance through the '
        'doorway, but the camera has not yet reached it."}'
    )
    engine = FinalStopEngine(response)
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 2.0, dtype=np.float32)

    for _ in range(3):
        decision = agent.act(rgb, depth, instruction, intrinsics, np.eye(4))
        assert decision.stop is False

    assert agent.last_waypoint_stop_disposition == "deferred_unverified_final"


class PreviewRoutedEngine(RoutedEngine):
    def __init__(self, waypoint_response=WAYPOINT_PREVIEW):
        super().__init__()
        self.waypoint_response = waypoint_response

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label = "plan"
            response = (
                "1|Exit through the doorway|"
                "The camera crosses the threshold"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label = "scene"
            response = SCENE_IN_PROGRESS.replace(
                '"door_state":"NOT_APPLICABLE"',
                '"door_state":"NOT_VISIBLE"',
            ).replace(
                '"door_camera_side":"NOT_APPLICABLE"',
                '"door_camera_side":"UNKNOWN"',
            ).replace(
                '"door_transition":"NOT_APPLICABLE"',
                '"door_transition":"NONE"',
            ).replace(
                '"current_room_side":"NOT_APPLICABLE"',
                '"current_room_side":"ORIGINAL_SIDE"',
            )
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label = "waypoint"
            response = self.waypoint_response
        elif system_prompt.startswith("You are judging which way"):
            label = "preview"
            response = (
                '{"view_index":2,"u":300,"v":800,"confidence":0.95,'
                '"evidence":"right-side view centers the doorway and floor"}'
            )
        else:
            raise AssertionError("unexpected VLM call")
        self.calls.append((label, content, kwargs))
        return response


def test_preview_selection_skips_second_waypoint_call_without_depth_override():
    engine = PreviewRoutedEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Exit through the doorway."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    first = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        intrinsics,
        np.eye(4),
    )
    assert first.action_mode == "PREVIEW"
    cache_size = len(agent.temporal_memory.captioner._png_cache)
    views = tuple(
        PreviewView(
            yaw_deg=yaw,
            rgb=np.full((24, 32, 3), index, dtype=np.uint8),
            depth=np.full(
                (24, 32),
                6.0 if index == 0 else 2.0,
                dtype=np.float32,
            ),
            intrinsics=intrinsics,
            camera_to_world=np.eye(4),
        )
        for index, yaw in enumerate((-45.0, 0.0, 45.0))
    )

    decision = agent.act_on_preview(views, instruction)

    assert decision.point is not None
    assert agent.last_preview_view_index == 2
    assert agent.last_preview_yaw_deg == 45.0
    assert agent.last_preview_selection.view_index == 2
    assert agent.last_preview_guard_reason is None
    assert agent.last_requested_normalized == (300, 800)
    assert "skipped redundant waypoint VLM" in agent.last_waypoint_guard_reason
    assert [label for label, _, _ in engine.calls] == [
        "plan",
        "scene",
        "waypoint",
        "preview",
    ]
    assert len(agent.temporal_memory.captioner._png_cache) == cache_size


def test_localized_doorway_world_waypoint_is_reused_without_another_waypoint_vlm():
    engine = PreviewRoutedEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Exit through the doorway."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    first = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        intrinsics,
        np.eye(4),
    )
    assert first.action_mode == "PREVIEW"
    views = tuple(
        PreviewView(
            yaw_deg=yaw,
            rgb=np.full((24, 32, 3), index, dtype=np.uint8),
            depth=np.full((24, 32), 2.0, dtype=np.float32),
            intrinsics=intrinsics,
            camera_to_world=np.eye(4),
        )
        for index, yaw in enumerate((-45.0, 0.0, 45.0))
    )
    preview_decision = agent.act_on_preview(views, instruction)
    labels_before = [label for label, _, _ in engine.calls]

    reused = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        intrinsics,
        np.eye(4),
    )

    assert reused.point == preview_decision.point
    assert [label for label, _, _ in engine.calls] == labels_before
    assert agent.temporal_memory.diagnostics()["frame_ids"] == [1, 2]
    assert agent.last_waypoint_guard_reason == (
        "reused stable doorway world waypoint; skipped waypoint VLM"
    )


def test_doorway_waypoint_does_not_expire_while_distance_improves():
    engine = PreviewRoutedEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Exit through the doorway."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        intrinsics,
        np.eye(4),
    )
    views = tuple(
        PreviewView(
            yaw_deg=yaw,
            rgb=np.full((24, 32, 3), index, dtype=np.uint8),
            depth=np.full((24, 32), 2.0, dtype=np.float32),
            intrinsics=intrinsics,
            camera_to_world=np.eye(4),
        )
        for index, yaw in enumerate((-45.0, 0.0, 45.0))
    )
    original = agent.act_on_preview(views, instruction).point
    assert original is not None

    for index in range(30):
        pose = np.eye(4)
        progress = index / 100.0
        pose[0, 3] = original.world_xyz[0] * progress
        pose[2, 3] = original.world_xyz[2] * progress
        decision = agent.act(
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.full((24, 32), 2.0, dtype=np.float32),
            instruction,
            intrinsics,
            pose,
        )
        assert decision.point == original

    assert [label for label, _, _ in engine.calls].count("waypoint") == 1


def test_unlocalized_doorway_turn_remains_a_measured_turn():
    engine = PreviewRoutedEngine(
        '{"action_mode":"EXECUTION","execution":{"stop":false,'
        '"intent":"TURN_LEFT","turn_deg":30},"confidence":0.9,'
        '"evidence":"doorway is somewhere to the left"}'
    )
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Exit through the doorway."
    agent.prepare_task(instruction)

    decision = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 2.0, dtype=np.float32),
        instruction,
        np.array(
            ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
        ),
        np.eye(4),
    )

    assert decision.action_mode == "EXECUTION"
    assert decision.point is None
    assert agent.last_requested_turn_deg == -30
    assert agent.last_waypoint_model_intent == "TURN_LEFT"
    assert agent.last_waypoint_applied_intent == "TURN_LEFT"
    assert agent.last_waypoint_guard_reason is None


SCENE_DOOR_VISIBLE_RIGHT = (
    '{"landmark":{"visible":true,"direction":"RIGHT",'
    '"proximity":"NEAR","passed":false,'
    '"destination_dominant":false,"u":760,"v":640,'
    '"confidence":0.97},"door_state":"APPROACHING",'
    '"door_camera_side":"BEFORE_DOOR",'
    '"door_transition":"APPROACHED",'
    '"current_room_side":"ORIGINAL_SIDE","completed":false,'
    '"completion_confidence":0.0,"error_mode":"NONE",'
    '"error_confidence":0.0,"final_target":{"visible":false,'
    '"proximity":"UNKNOWN","confidence":0.0},'
    '"evidence":"doorway with jambs visible on the right"}'
)

WAYPOINT_TURN_RIGHT = (
    '{"action_mode":"EXECUTION","execution":{"stop":false,'
    '"intent":"TURN_RIGHT","turn_deg":30},'
    '"confidence":0.9,"evidence":"doorway is on the right"}'
)


class DoorwayTurnEngine(RoutedEngine):
    """Captioner localizes the doorway; waypoint model only wants to turn."""

    def __init__(self, waypoint_response=WAYPOINT_TURN_RIGHT):
        super().__init__()
        self.waypoint_response = waypoint_response

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label = "plan"
            response = (
                "1|Exit through the doorway into the pool room|"
                "The camera has crossed the threshold into the pool room"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label = "scene"
            response = SCENE_DOOR_VISIBLE_RIGHT
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label = "waypoint"
            response = self.waypoint_response
        else:
            raise AssertionError("unexpected independent VLM call")
        self.calls.append((label, content, kwargs))
        return response


def _room_depth(height=24, width=32, camera_height=1.25, wall=3.0, fy=16.0, cy=12.0):
    depth = np.full((height, width), wall, dtype=np.float32)
    for v in range(height):
        if v > cy:
            floor_range = camera_height * fy / (v - cy)
            if floor_range < wall:
                depth[v, :] = floor_range
    return depth


def test_localized_landmark_overrides_model_turn_with_floor_waypoint():
    engine = DoorwayTurnEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "exit into the pool room and stop before pool."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    camera_to_world = np.eye(4)
    camera_to_world[1, 3] = 1.25

    decision = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        _room_depth(),
        instruction,
        intrinsics,
        camera_to_world,
    )

    assert decision.stop is False
    assert decision.turn_deg is None
    assert decision.point is not None
    assert decision.point.on_floor is True
    assert agent.last_requested_normalized == (760, 640)
    assert agent.last_requested_turn_deg is None
    assert "beneath the located landmark" in agent.last_waypoint_guard_reason
    # Landmark column, snapped down onto the floor in front of the doorway.
    assert decision.point.pixel_uv[0] == round(760 * 31 / 1000)
    assert abs(decision.point.world_xyz[1]) <= agent.actor.max_floor_offset_m
    # A verified floor point beneath a doorway becomes the locked target.
    assert agent._doorway_waypoint is decision.point


def test_model_turn_is_kept_when_no_landmark_is_localized():
    engine = FinalStopEngine(WAYPOINT_TURN_RIGHT)
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    camera_to_world = np.eye(4)
    camera_to_world[1, 3] = 1.25

    decision = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        _room_depth(),
        instruction,
        intrinsics,
        camera_to_world,
    )

    assert decision.turn_deg == 30
    assert decision.point is None


def test_off_floor_point_is_not_locked_as_doorway():
    engine = DoorwayTurnEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "exit into the pool room and stop before pool."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    camera_to_world = np.eye(4)
    camera_to_world[1, 3] = 1.25

    # Nose to a wall: no pixel reaches floor level.
    decision = agent.act(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.full((24, 32), 1.0, dtype=np.float32),
        instruction,
        intrinsics,
        camera_to_world,
    )

    assert decision.point is not None
    assert decision.point.on_floor is False
    assert agent._doorway_waypoint is None
    assert "not locked" in agent.last_waypoint_guard_reason


def _turned(camera_to_world, yaw_deg):
    """Rotate a camera-to-world transform about +y (positive turns right)."""
    angle = np.deg2rad(-yaw_deg)
    rotation = np.array(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    turned = np.array(camera_to_world, dtype=np.float64)
    turned[:3, :3] = rotation @ turned[:3, :3]
    return turned


class TurnThenHallwayEngine(RoutedEngine):
    """Planner emits a compound turn stage; waypoint model always wants to turn."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label = "plan"
            response = (
                "1|Turn left and proceed through the hallway|"
                "The camera is moving through the hallway"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label = "scene"
            response = SCENE_IN_PROGRESS
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label = "waypoint"
            response = WAYPOINT_TURN_RIGHT.replace(
                '"intent":"TURN_RIGHT","turn_deg":30',
                '"intent":"TURN_LEFT","turn_deg":-30',
            )
        else:
            raise AssertionError("unexpected independent VLM call")
        self.calls.append((label, content, kwargs))
        return response


def test_turn_stage_phase_ends_on_measured_rotation_not_model_requests():
    engine = TurnThenHallwayEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Turn left and go through the hallway."
    agent.prepare_task(instruction)
    # The compound stage was split: the turn is its own stage.
    assert [s.description for s in agent.subgoals] == [
        "Turn left",
        "Proceed through the hallway",
    ]
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    base = np.eye(4)
    base[1, 3] = 1.25
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)

    decision = agent.act(rgb, _room_depth(), instruction, intrinsics, base)
    assert agent._navigation_phase == "TURN_LEFT"
    assert decision.turn_deg == -30

    # Camera has physically rotated 75 degrees left: the turn is done even
    # though the model keeps asking for more, and the judge sees it too.
    for yaw in (-30.0, -60.0, -75.0):
        decision = agent.act(
            rgb, _room_depth(), instruction, intrinsics, _turned(base, yaw)
        )
    # The measured rotation both ends the turn phase and completes the turn
    # stage itself, so the hallway stage -- the last one, hence a final
    # approach -- is active now.
    assert agent._navigation_phase == "FINAL_APPROACH"
    assert agent.task_memory.get_current_subgoal().subgoal_id == "2"
    assert agent.temporal_memory.diagnostics()["turn_progress_deg"] == 0.0


def test_turn_stage_is_abandoned_after_half_a_circle():
    engine = TurnThenHallwayEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Turn left and go through the hallway."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    base = np.eye(4)
    base[1, 3] = 1.25
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    # Model asks LEFT but the measured rotation goes the wrong way; after a
    # half circle the turn phase ends regardless.
    agent.act(rgb, _room_depth(), instruction, intrinsics, base)
    for yaw in (45.0, 90.0, 135.0, 180.0):
        agent.act(rgb, _room_depth(), instruction, intrinsics, _turned(base, yaw))
    assert agent._navigation_phase == "FOLLOW_CORRIDOR"


def test_walking_past_a_locked_waypoint_counts_as_reached():
    from agentflow.agents.models_embodied_v2.data_models import NavigationPoint

    point = NavigationPoint(
        pixel_uv=(0, 0), depth_m=1.0, camera_xyz=(0.0, 0.0, -1.0),
        world_xyz=(0.0, 0.0, -1.0), on_floor=True,
    )
    ahead = np.eye(4)  # camera at origin looking down -z: point is 1 m ahead
    assert VLNAgent._waypoint_passed(point, camera_to_world=ahead) is False
    behind = np.eye(4)
    behind[2, 3] = -1.6  # walked 0.6 m past the point on the same heading
    assert VLNAgent._waypoint_passed(point, camera_to_world=behind) is True
    far_behind = np.eye(4)
    far_behind[2, 3] = -2.5
    assert VLNAgent._waypoint_passed(point, camera_to_world=far_behind) is False


def test_final_stage_is_final_approach_and_never_locks_a_landmark():
    class PoolAheadEngine(RoutedEngine):
        def __call__(self, content, *, system_prompt, **kwargs):
            if system_prompt.startswith("You are an indoor navigation task planner"):
                response = (
                    "1|Exit through the doorway|The camera has crossed the threshold\n"
                    "2|Walk forward to the pool|The pool is directly ahead and within a step of the camera"
                )
                label = "plan"
            elif system_prompt.startswith("You are the temporal scene observer"):
                response = SCENE_DOOR_VISIBLE_RIGHT
                label = "scene"
            elif system_prompt.startswith("You are an indoor navigation actor"):
                response = WAYPOINT_TURN_RIGHT
                label = "waypoint"
            else:
                raise AssertionError("unexpected independent VLM call")
            self.calls.append((label, content, kwargs))
            return response

    agent = VLNAgent(engine=PoolAheadEngine(), patch_radius_px=0, camera_height_m=1.25)
    agent.prepare_task("exit into the pool room and stop before pool.")
    final = agent.subgoals[-1]
    assert agent._phase_for_subgoal(final) == "FINAL_APPROACH"
    assert agent._phase_for_subgoal(agent.subgoals[0]) != "FINAL_APPROACH"

    # Drive the final stage directly: a localized landmark steers the step but
    # must not become a reused target that bypasses the waypoint model.
    agent.task_memory.reset(goal="stop before pool.", subgoals=(final,))
    agent.subgoals = [final]
    intrinsics = np.array(((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0)))
    camera_to_world = np.eye(4)
    camera_to_world[1, 3] = 1.25
    agent.act(np.zeros((24, 32, 3), dtype=np.uint8), _room_depth(), "stop before pool.", intrinsics, camera_to_world)
    assert agent._navigation_phase == "FINAL_APPROACH"
    assert agent._doorway_waypoint is None


def test_spin_release_keeps_committed_point_as_judge_evidence():
    from agentflow.agents.models_embodied_v2.data_models import NavigationPoint, Subgoal

    engine = DoorwayTurnEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    agent.prepare_task("exit into the pool room and stop before pool.")
    current = agent.task_memory.get_current_subgoal()
    point = NavigationPoint(
        pixel_uv=(0, 0), depth_m=4.0, camera_xyz=(0.0, 0.0, -4.0),
        world_xyz=(0.0, 0.0, -4.0), on_floor=True,
    )
    agent._doorway_waypoint = point
    agent._doorway_waypoint_subgoal_id = current.subgoal_id
    agent._doorway_waypoint_reach_tolerance_m = 1.0

    agent._clear_doorway_waypoint(keep_for_judge=True)
    assert agent._doorway_waypoint is None
    camera_to_world = np.eye(4)
    camera_to_world[1, 3] = 1.25
    assert agent._doorway_target_distance(current, camera_to_world=camera_to_world) == 4.0
    assert agent._judge_target_tolerance(current) == 1.0

    agent._clear_doorway_waypoint()
    assert agent._doorway_target_distance(current, camera_to_world=camera_to_world) is None


SCENE_SOFA_CENTRED = SCENE_DOOR_VISIBLE_RIGHT.replace('"u":760,"v":640', '"u":520,"v":640').replace(
    '"direction":"RIGHT"', '"direction":"CENTER"'
).replace('"door_state":"APPROACHING"', '"door_state":"NOT_APPLICABLE"').replace(
    '"door_camera_side":"BEFORE_DOOR"', '"door_camera_side":"NOT_APPLICABLE"'
).replace('"door_transition":"APPROACHED"', '"door_transition":"NOT_APPLICABLE"').replace(
    '"current_room_side":"ORIGINAL_SIDE"', '"current_room_side":"NOT_APPLICABLE"'
)


class TurnToSofaEngine(RoutedEngine):
    """Turn stage; the model keeps asking to turn although the sofa is centred."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            label, response = "plan", (
                "1|Turn right toward the sofa|After turning right, the sofa is centred in the view\n"
                "2|Walk to the sofa|The sofa is directly ahead within a step"
            )
        elif system_prompt.startswith("You are the temporal scene observer"):
            label, response = "scene", SCENE_SOFA_CENTRED
        elif system_prompt.startswith("You are an indoor navigation actor"):
            label, response = "waypoint", WAYPOINT_TURN_RIGHT
        else:
            raise AssertionError("unexpected independent VLM call")
        self.calls.append((label, content, kwargs))
        return response


def test_visible_landmark_ends_the_turn_and_is_walked_to_not_turned_toward():
    engine = TurnToSofaEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Turn right to the sofa."
    agent.prepare_task(instruction)
    intrinsics = np.array(((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0)))
    base = np.eye(4)
    base[1, 3] = 1.25
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)

    # Step 0: model wants TURN_RIGHT, but the sofa is already located ->
    # walked to, not turned toward.
    first = agent.act(rgb, _room_depth(), instruction, intrinsics, base)
    assert first.turn_deg is None and first.point is not None
    assert agent.last_requested_normalized == (520, 640)
    # The located landmark is committed during the turn stage as well.
    assert agent._doorway_waypoint is not None
    # The prompt no longer demands a turn once the landmark is in view.
    waypoint_prompt = engine.calls[-1][1][0]
    assert "requested turn is satisfied" in waypoint_prompt

    # After one 30-degree primitive the centred landmark ends the turn stage.
    agent.act(rgb, _room_depth(), instruction, intrinsics, _turned(base, 30.0))
    assert agent.task_memory.get_current_subgoal().subgoal_id == "2"


class SofaPreviewEngine(PreviewRoutedEngine):
    """Two-stage plan whose first stage names no door at all."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            self.calls.append(("plan", content, kwargs))
            return (
                "1|Walk toward the sofa|The sofa is directly ahead\n"
                "2|Stop beside the lamp|The lamp is beside the camera"
            )
        if system_prompt.startswith("You are judging which way"):
            self.calls.append(("preview", content, kwargs))
            return (
                '{"view_index":0,"u":300,"v":800,"confidence":0.95,'
                '"evidence":"left view shows the sofa with floor in front"}'
            )
        return super().__call__(content, system_prompt=system_prompt, **kwargs)


def _yaw_transform(yaw_deg: float) -> np.ndarray:
    transform = np.eye(4)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    transform[0, 0], transform[0, 2] = c, s
    transform[2, 0], transform[2, 2] = -s, c
    return transform


def test_captioner_selected_preview_target_is_locked_until_reached():
    engine = SofaPreviewEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Walk toward the sofa, then stop beside the lamp."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 2.0, dtype=np.float32)
    first = agent.act(rgb, depth, instruction, intrinsics, np.eye(4))
    assert first.action_mode == "PREVIEW"
    # The selected view faces 90 degrees left; its floor point lies well off
    # the agent's own camera axis.
    views = tuple(
        PreviewView(
            yaw_deg=yaw,
            rgb=rgb,
            depth=np.full((24, 32), 3.0, dtype=np.float32),
            intrinsics=intrinsics,
            camera_to_world=_yaw_transform(yaw),
        )
        for yaw in (-90.0, 0.0, 90.0)
    )
    committed = agent.act_on_preview(views, instruction)
    assert committed.point is not None
    assert agent._doorway_waypoint is committed.point
    assert abs(agent._waypoint_bearing_deg(
        committed.point, camera_to_world=np.eye(4)
    )) > 45.0
    waypoint_calls = sum(label == "waypoint" for label, _, _ in engine.calls)

    # One 15-degree primitive later the target is reused as-is: no PREVIEW,
    # no straight-ahead fallback, no second waypoint VLM call.
    again = agent.act(rgb, depth, instruction, intrinsics, _yaw_transform(15.0))
    assert again.action_mode == "EXECUTION"
    assert again.point is committed.point
    assert "reused stable doorway world waypoint" in agent.last_waypoint_guard_reason
    assert sum(label == "waypoint" for label, _, _ in engine.calls) == waypoint_calls


def test_waypoint_bearing_is_signed_right_positive():
    from agentflow.agents.models_embodied_v2.data_models import NavigationPoint

    point = NavigationPoint(
        pixel_uv=(0, 0), depth_m=1.0,
        camera_xyz=(1.0, 0.0, 0.0), world_xyz=(1.0, 0.0, 0.0),
    )
    assert VLNAgent._waypoint_bearing_deg(point, camera_to_world=np.eye(4)) == 90.0
    ahead = NavigationPoint(
        pixel_uv=(0, 0), depth_m=1.0,
        camera_xyz=(0.0, 0.0, -2.0), world_xyz=(0.0, 0.0, -2.0),
    )
    assert VLNAgent._waypoint_bearing_deg(ahead, camera_to_world=np.eye(4)) == 0.0


def test_reached_locked_waypoint_is_released_and_does_not_rearm():
    from agentflow.agents.models_embodied_v2.data_models import NavigationPoint

    engine = SofaPreviewEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    agent.prepare_task("Walk toward the sofa, then stop beside the lamp.")
    current = agent.task_memory.get_current_subgoal()
    point = NavigationPoint(
        pixel_uv=(0, 0), depth_m=3.0,
        camera_xyz=(0.0, 0.0, -3.0), world_xyz=(0.0, 0.0, -3.0),
    )
    agent._doorway_waypoint = point
    agent._doorway_waypoint_subgoal_id = current.subgoal_id
    agent._doorway_waypoint_reach_tolerance_m = 0.75

    far = np.eye(4)
    assert agent._locked_doorway_decision(current, camera_to_world=far) is not None

    # Within tolerance: control is handed back and the point is kept only as
    # judge evidence.
    near = np.eye(4)
    near[2, 3] = -2.5
    assert agent._locked_doorway_decision(current, camera_to_world=near) is None
    assert agent._doorway_waypoint is None
    assert agent._judge_target_point is point

    # Drifting past the tolerance again must not revive the old target.
    assert agent._locked_doorway_decision(current, camera_to_world=far) is None


def test_final_stop_vote_requires_measured_near_range():
    engine = FinalStopEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    # The text says "directly beside the pool" every step, but the forward
    # band measures 3.5 m: no vote, the STOP stays deferred.
    far = np.full((24, 32), 3.5, dtype=np.float32)
    for _ in range(3):
        decision = agent.act(rgb, far, instruction, intrinsics, np.eye(4))
        assert decision.stop is False
        assert agent.last_waypoint_stop_disposition == "deferred_unverified_final"
        assert "beyond 2.5m" in agent.last_waypoint_guard_reason
    # Once the frame actually measures close, two votes end the episode.
    near = np.full((24, 32), 1.0, dtype=np.float32)
    agent.act(rgb, near, instruction, intrinsics, np.eye(4))
    decision = agent.act(rgb, near, instruction, intrinsics, np.eye(4))
    assert decision.stop is True
    assert agent.last_waypoint_stop_disposition == "accepted_repeated_visual_final"


class TwoStageWalkEngine(RoutedEngine):
    """A plain approach stage followed by the final stage."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            self.calls.append(("plan", content, kwargs))
            return (
                "1|Walk forward to the sofa|The sofa is directly ahead\n"
                "2|Walk forward to the pool area|"
                "The camera is directly beside the pool"
            )
        return super().__call__(content, system_prompt=system_prompt, **kwargs)


def test_plain_walk_stage_is_judged_every_other_step_after_its_first():
    from agentflow.agents.models_embodied_v2.skiils.protocol import (
        CAPTIONER_ANALYSIS_INTERVAL_STEPS,
    )

    assert CAPTIONER_ANALYSIS_INTERVAL_STEPS == 2
    engine = TwoStageWalkEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0)
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(
        ((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0))
    )
    for step in range(4):
        camera_to_world = np.eye(4)
        camera_to_world[2, 3] = -0.25 * step  # measured forward progress
        decision = agent.act(
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.full((24, 32), 2.0, dtype=np.float32),
            instruction,
            intrinsics,
            camera_to_world,
        )
        assert decision.stop is False

    labels = [label for label, _, _ in engine.calls]
    # Step 0 (first observation of the stage) and step 2 are judged; steps 1
    # and 3 keep their frames for the next judgement but skip the Captioner.
    # The waypoint model answers once: its point becomes a Spatial Memory
    # target that steps 1-3 keep walking to without another call.
    assert labels == ["plan", "scene", "waypoint", "scene"]
    assert agent.temporal_memory.diagnostics()["frame_ids"] == [1, 2, 3, 4]
    assert agent.spatial_memory.target is not None
    assert agent.spatial_memory.target.kind == "model_waypoint"
    assert "reused spatial target" in agent.last_waypoint_guard_reason


def _room_depth_for_agent(height=96, width=128, fy=40.0, cy=48.0, camera_height=1.25):
    """Flat floor ahead of a level camera, far returns elsewhere."""
    vs = np.arange(height, dtype=np.float64)[:, None]
    below = vs - cy
    floor = np.where(below > 0, camera_height * fy / np.maximum(below, 1e-6), 6.0)
    return np.broadcast_to(np.minimum(floor, 6.0), (height, width)).astype(np.float32).copy()


def test_spatial_target_is_walked_to_then_released_and_final_stage_keeps_the_model():
    engine = TwoStageWalkEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((40.0, 0.0, 64.0), (0.0, 40.0, 48.0), (0.0, 0.0, 1.0)))
    depth = _room_depth_for_agent()
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)

    def act(z):
        pose = np.eye(4)
        pose[1, 3] = 1.25
        pose[2, 3] = z
        return agent.act(rgb, depth, instruction, intrinsics, pose)

    first = act(0.0)
    assert first.point is not None and first.point.on_floor
    target = agent.spatial_memory.target
    assert target is not None and target.kind == "model_waypoint"
    target_z = target.world_xyz[2]
    assert target_z < -0.6  # a real distance ahead, not the next step

    # Walk toward it: no waypoint-model call while the target is active, and
    # the reported point lies on the route to it.
    calls_before = len([l for l, _, _ in engine.calls if l == "waypoint"])
    z = 0.0
    for _ in range(12):
        z -= 0.25
        decision = act(z)
        if agent.spatial_memory.last_release_reason in ("reached", "passed"):
            break
        assert decision.point is not None
        assert decision.point.world_xyz[2] <= z + 0.05
        assert "reused spatial target" in agent.last_waypoint_guard_reason
    calls_after = len([l for l, _, _ in engine.calls if l == "waypoint"])
    # Exactly one more call: the re-query once the target was reached, which
    # committed the next target in the same step.
    assert calls_after == calls_before + 1
    assert agent.spatial_memory.last_release_reason in ("reached", "passed")
    assert agent.spatial_memory.target is not None
    assert agent.spatial_memory.diagnostics()["observations"] >= 2
    assert agent.spatial_memory.grid.explored_area_m2() > 1.0

    # The final stage never commits: STOP must stay a per-step model decision.
    agent.task_memory.reset(goal=instruction, subgoals=agent.subgoals[1:])
    agent.temporal_memory.reset()
    agent.spatial_memory.reset()
    agent.subgoals = list(agent.subgoals[1:])
    calls_before = len([l for l, _, _ in engine.calls if l == "waypoint"])
    for _ in range(3):
        z -= 0.25
        act(z)
    assert len([l for l, _, _ in engine.calls if l == "waypoint"]) == calls_before + 3
    assert agent.spatial_memory.target is None


def test_spatial_memory_switch_keeps_the_model_in_the_loop(monkeypatch):
    monkeypatch.setenv("VLN_SPATIAL_MEMORY", "0")
    engine = TwoStageWalkEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    assert agent.use_spatial_memory is False
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((40.0, 0.0, 64.0), (0.0, 40.0, 48.0), (0.0, 0.0, 1.0)))
    depth = _room_depth_for_agent()
    for step in range(3):
        pose = np.eye(4)
        pose[1, 3] = 1.25
        pose[2, 3] = -0.25 * step
        agent.act(np.zeros((96, 128, 3), dtype=np.uint8), depth, instruction, intrinsics, pose)
    assert agent.spatial_memory.target is None
    assert agent.last_spatial_summary == "sp=off"
    assert [l for l, _, _ in engine.calls].count("waypoint") == 3


class SetOfMarkEngine(TwoStageWalkEngine):
    """Answers the set-of-mark question with a fixed marker label."""

    def __init__(self, choice="2"):
        super().__init__()
        self.choice = choice
        self.som_prompts = []

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are choosing where an indoor navigation agent walks next"):
            self.calls.append(("som", content, kwargs))
            self.som_prompts.append(content[0])
            return ('{"choice":"%s","confidence":0.8,"evidence":"corridor continues there"}' % self.choice)
        return super().__call__(content, system_prompt=system_prompt, **kwargs)


def _floor_mask_for_agent(height=96, cy=48.0):
    vs = np.arange(height, dtype=np.float64)[:, None]
    return np.broadcast_to(vs - cy > 0, (height, 128)).copy()


def test_set_of_mark_choice_becomes_the_committed_target():
    engine = SetOfMarkEngine(choice="2")
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    assert agent.use_som
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((40.0, 0.0, 64.0), (0.0, 40.0, 48.0), (0.0, 0.0, 1.0)))
    depth = _room_depth_for_agent()
    pose = np.eye(4)
    pose[1, 3] = 1.25
    decision = agent.act(np.zeros((96, 128, 3), dtype=np.uint8), depth, instruction, intrinsics, pose)

    labels = [label for label, _, _ in engine.calls]
    assert labels == ["plan", "scene", "som"]  # no pixel-proposal call
    assert agent.last_som_choice == "2"
    chosen = next(c for c in agent.last_som_candidates if c["label"] == "2")
    target = agent.spatial_memory.target
    assert target is not None and target.kind == "som"
    # The target lies on the ray toward marker 2, at most 3 m out (a far
    # marker is walked part-way before the model is asked again), snapped
    # onto free floor within a couple of grid cells.
    to_marker = np.array(chosen["world_xyz"])[[0, 2]]
    to_target = np.array(target.world_xyz)[[0, 2]]
    assert np.linalg.norm(to_target) <= 3.0 + 0.35
    assert np.dot(to_marker, to_target) / (np.linalg.norm(to_marker) * np.linalg.norm(to_target)) > 0.98
    assert decision.point is not None and decision.stop is False
    assert "set-of-mark: model chose marker 2" in agent.last_waypoint_guard_reason
    prompt = engine.som_prompts[0]
    assert "Options:" in prompt and "[1]" in prompt and "Full route instruction" in prompt
    # The image sent carries the markers (PNG bytes), not the raw frame.
    assert isinstance(engine.calls[-1][1][1], bytes) and engine.calls[-1][1][1][:4] == b"\x89PNG"


def test_set_of_mark_falls_back_to_pixel_path_on_unusable_replies():
    engine = SetOfMarkEngine(choice="Z")  # never a listed label
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((40.0, 0.0, 64.0), (0.0, 40.0, 48.0), (0.0, 0.0, 1.0)))
    pose = np.eye(4)
    pose[1, 3] = 1.25
    agent.act(np.zeros((96, 128, 3), dtype=np.uint8), _room_depth_for_agent(), instruction, intrinsics, pose)
    labels = [label for label, _, _ in engine.calls]
    assert labels == ["plan", "scene", "som", "som", "waypoint"]
    assert agent.last_som_choice is None and "not a listed label" in agent.last_som_error
    assert agent.spatial_memory.target is not None and agent.spatial_memory.target.kind == "model_waypoint"


class SoMWithLandmarkEngine(SetOfMarkEngine):
    """Scene observer locates the stage landmark; SoM must not be asked."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are the temporal scene observer"):
            self.calls.append(("scene", content, kwargs))
            # Landmark located at (760, 640); the captioner forces the door
            # fields to NOT_APPLICABLE for this non-doorway stage.
            return SCENE_DOOR_VISIBLE_RIGHT
        return super().__call__(content, system_prompt=system_prompt, **kwargs)


def test_located_landmark_is_walked_to_and_locked_instead_of_asking_set_of_mark():
    engine = SoMWithLandmarkEngine(choice="1")
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Walk forward to the sofa, then walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((40.0, 0.0, 64.0), (0.0, 40.0, 48.0), (0.0, 0.0, 1.0)))
    pose = np.eye(4)
    pose[1, 3] = 1.25
    decision = agent.act(np.zeros((96, 128, 3), dtype=np.uint8), _room_depth_for_agent(), instruction, intrinsics, pose)
    labels = [label for label, _, _ in engine.calls]
    assert labels == ["plan", "scene"]  # neither set-of-mark nor the pixel path
    target = agent.spatial_memory.target
    assert target is not None and target.kind == "landmark"
    assert agent._doorway_waypoint is not None  # locked for the completion judge
    assert agent._doorway_waypoint.world_xyz == pytest.approx(target.world_xyz, abs=1e-6)
    assert decision.point is not None
    assert "walking to the floor beneath the located landmark" in agent.last_waypoint_guard_reason


class TurnAroundEngine(RoutedEngine):
    """Plans a 'Turn around' stage; the waypoint model keeps asking to turn."""

    def __call__(self, content, *, system_prompt, **kwargs):
        if system_prompt.startswith("You are an indoor navigation task planner"):
            self.calls.append(("plan", content, kwargs))
            return "1|Turn around|The hallway behind is now ahead\n2|Walk down the hallway to the pool area|The camera is beside the pool"
        if system_prompt.startswith("You are an indoor navigation actor"):
            self.calls.append(("waypoint", content, kwargs))
            return ('{"action_mode":"EXECUTION","execution":{"stop":false,"intent":"TURN_LEFT","turn_deg":-45},'
                    '"confidence":0.9,"evidence":"nothing useful ahead"}')
        return super().__call__(content, system_prompt=system_prompt, **kwargs)


def test_turn_around_stage_is_measured_as_a_half_turn():
    engine = TurnAroundEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Turn around and walk down the hallway to the pool area."
    agent.prepare_task(instruction)
    assert agent.subgoals[0].description == "Turn around"
    intrinsics = np.array(((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0)))
    base = np.eye(4)
    base[1, 3] = 1.25
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)

    decision = agent.act(rgb, _room_depth(), instruction, intrinsics, base)
    assert agent._navigation_phase == "TURN_LEFT"  # driven like a left turn
    assert decision.turn_deg == -45
    # 90 degrees is not a half turn yet: still stage 1, still turning.
    for yaw in (-45.0, -90.0):
        decision = agent.act(rgb, _room_depth(), instruction, intrinsics, _turned(base, yaw))
    assert agent.task_memory.get_current_subgoal().subgoal_id == "1"
    assert decision.turn_deg == -45
    # Past 150 degrees the stage completes by measurement, whatever the model says.
    for yaw in (-135.0, -165.0):
        agent.act(rgb, _room_depth(), instruction, intrinsics, _turned(base, yaw))
    assert agent.task_memory.get_current_subgoal().subgoal_id == "2"
    assert agent._navigation_phase == "FINAL_APPROACH"


def test_turn_around_accepts_either_direction():
    engine = TurnAroundEngine()
    agent = VLNAgent(engine=engine, patch_radius_px=0, camera_height_m=1.25)
    instruction = "Turn around and walk down the hallway to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0)))
    base = np.eye(4)
    base[1, 3] = 1.25
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    agent.act(rgb, _room_depth(), instruction, intrinsics, base)
    for yaw in (45.0, 90.0, 135.0, 165.0):  # the follower happened to turn right
        agent.act(rgb, _room_depth(), instruction, intrinsics, _turned(base, yaw))
    assert agent.task_memory.get_current_subgoal().subgoal_id == "2"
