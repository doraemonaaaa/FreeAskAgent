from __future__ import annotations

import numpy as np

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
    assert "overrode model TURN" in agent.last_waypoint_guard_reason
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
