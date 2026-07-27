import numpy as np
import pytest

from agentflow.agents.vln_agent_2 import Actor, CameraIntrinsics, Subgoal
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


def _actor(response='{"stop": false, "u": 2, "v": 3}', *, goal="go forward"):
    return Actor(
        engine=FakeEngine(response),
        task_memory=TaskMemory(goal),
        min_depth_m=0.1,
        max_depth_m=5.0,
        patch_radius_px=0,
    )


def test_actor_back_projects_habitat_camera_point_into_world_waypoint():
    actor = _actor()
    actor.task_memory.record_input(
        np.zeros((6, 6, 3), dtype=np.uint8),
        depth=np.full((6, 6), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0),
        camera_to_world=np.array(
            (
                (1.0, 0.0, 0.0, 10.0),
                (0.0, 1.0, 0.0, 20.0),
                (0.0, 0.0, 1.0, 30.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    decision = actor.act()

    assert not decision.stop
    assert decision.point.pixel_uv == (2, 3)
    assert decision.point.camera_xyz == (0.0, -1.0, -2.0)
    assert decision.point.world_xyz == (10.0, 19.0, 28.0)


def test_actor_searches_nearby_when_the_requested_pixel_has_invalid_depth():
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    depth[3, 2] = 0.0
    actor = _actor()
    actor.task_memory.record_input(
        np.zeros((6, 6, 3), dtype=np.uint8),
        depth=depth,
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
    )
    decision = actor.act()

    assert not decision.stop
    assert decision.point.pixel_uv != (2, 3)
    assert decision.point.depth_m == 2.0


def test_actor_can_explicitly_stop_without_reading_a_waypoint():
    actor = _actor('{"stop": true}', goal="stop at the table")
    actor.task_memory.record_input(
        np.zeros((6, 6, 3), dtype=np.uint8),
        depth=np.full((6, 6), 2.0, dtype=np.float32),
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
    )
    decision = actor.act()

    assert decision.stop
    assert decision.point is None


def test_actor_requires_sensor_range_to_decode_normalized_depth():
    with pytest.raises(ValueError, match="depth_min_m"):
        actor = _actor()
        actor.task_memory.record_input(
            np.zeros((6, 6, 3), dtype=np.uint8),
            depth=np.full((6, 6, 1), 0.5, dtype=np.float32),
            intrinsics=np.eye(3),
            camera_to_world=np.eye(4),
            normalized_depth=True,
        )
        actor.act()


def test_prepare_task_decomposes_and_stores_ordered_subgoals():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals": ["Exit the room through the visible doorway.", "Reach and stop beside the sofa."]}'
        ),
        task_memory=TaskMemory("Leave the room and go to the sofa."),
    )

    subgoals = actor.prepare_task()

    assert subgoals == (
        Subgoal("Exit the room through the visible doorway."),
        Subgoal("Reach and stop beside the sofa."),
    )
    assert actor.task_memory.get_current_subgoal().description == subgoals[0].description


def test_prepare_task_rejects_invalid_subgoal_response_without_replacing_state():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals": ["Move forward until the doorway is reached."]}',
            '{"subgoals": [{"description": "Not a string."}]}',
        ),
        task_memory=TaskMemory("Go forward."),
    )
    actor.prepare_task()

    with pytest.raises(ValueError, match="invalid subgoal JSON"):
        actor.prepare_task()

    assert actor.task_memory.get_task() == "Go forward."
    assert (
        actor.task_memory.get_current_subgoal().description
        == "Move forward until the doorway is reached."
    )
