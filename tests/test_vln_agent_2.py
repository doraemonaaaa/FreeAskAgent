import numpy as np
import pytest

from agentflow.agents.vln_agent_2 import (
    Actor,
    CameraIntrinsics,
    Subgoal,
)
from agentflow.agents.models_embodied_v2.memory import TaskMemory


class FakeEngine:
    def __init__(self, response: str):
        self.response = response

    def __call__(self, content, **kwargs):
        return self.response


class SequenceFakeEngine:
    def __init__(self, *responses: str):
        self.responses = list(responses)

    def __call__(self, content, **kwargs):
        return self.responses.pop(0)


def _actor(
    response='{"stop":false,"u":2,"v":3}',
    *,
    goal="go forward",
):
    return Actor(
        engine=FakeEngine(response),
        task_memory=TaskMemory(goal),
        min_depth_m=0.1,
        max_depth_m=5.0,
        patch_radius_px=0,
    )


def _act(
    actor,
    *,
    depth=None,
    intrinsics=None,
    camera_to_world=None,
    normalized_depth=False,
    depth_min_m=None,
    depth_max_m=None,
):
    return actor.act(
        np.zeros((6, 6, 3), dtype=np.uint8),
        (
            np.full((6, 6), 2.0, dtype=np.float32)
            if depth is None
            else depth
        ),
        "go forward",
        (
            CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0)
            if intrinsics is None
            else intrinsics
        ),
        np.eye(4) if camera_to_world is None else camera_to_world,
        normalized_depth=normalized_depth,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )


def test_actor_back_projects_habitat_camera_point_into_world_waypoint():
    actor = _actor()
    transform = np.array(
        (
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    decision = _act(actor, camera_to_world=transform)

    assert not decision.stop
    assert decision.point.pixel_uv == (2, 3)
    assert decision.point.camera_xyz == (0.0, -1.0, -2.0)
    assert decision.point.world_xyz == (10.0, 19.0, 28.0)


def test_actor_searches_nearby_when_requested_pixel_has_invalid_depth():
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    depth[3, 2] = 0.0
    actor = _actor()

    decision = _act(actor, depth=depth, intrinsics=np.eye(3))

    assert not decision.stop
    assert decision.point.pixel_uv != (2, 3)
    assert decision.point.depth_m == 2.0


def test_actor_can_explicitly_stop_without_reading_a_waypoint():
    actor = _actor('{"stop":true}', goal="stop at the table")

    decision = _act(actor)

    assert decision.stop
    assert decision.point is None


def test_actor_requires_sensor_range_to_decode_normalized_depth():
    actor = _actor()
    with pytest.raises(ValueError, match="depth_min_m"):
        _act(
            actor,
            depth=np.full((6, 6, 1), 0.5, dtype=np.float32),
            intrinsics=np.eye(3),
            normalized_depth=True,
        )


def test_prepare_task_decomposes_and_stores_ordered_subgoals():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals":['
            '{"subgoal_id":"1","description":"Exit the room.",'
            '"completion_criteria":"The camera crosses the doorway."},'
            '{"subgoal_id":"2","description":"Stop beside the sofa.",'
            '"completion_criteria":"The sofa is close and the agent stops."}'
            "]}"
        ),
        task_memory=TaskMemory(
            "Leave the room and go to the sofa."
        ),
    )

    subgoals = actor.prepare_task(
        "Leave the room and go to the sofa."
    )

    assert subgoals == (
        Subgoal(
            "1",
            "Exit the room.",
            "The camera crosses the doorway.",
        ),
        Subgoal(
            "2",
            "Stop beside the sofa.",
            "The sofa is close and the agent stops.",
        ),
    )
    assert (
        actor.task_memory.get_current_subgoal().description
        == "Exit the room."
    )


def test_prepare_task_rejects_invalid_response_without_replacing_state():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals":[{"subgoal_id":"1",'
            '"description":"Move forward.",'
            '"completion_criteria":"The doorway is reached."}]}',
            '{"subgoals":[{"description":"Missing fields."}]}',
        ),
        task_memory=TaskMemory("Go forward."),
    )
    actor.prepare_task("Go forward.")

    with pytest.raises(ValueError, match="invalid subgoal JSON"):
        actor.prepare_task("Replacement task.")

    assert actor.task_memory.get_task() == "Go forward."
    assert (
        actor.task_memory.get_current_subgoal().description
        == "Move forward."
    )
