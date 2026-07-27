import numpy as np
import pytest

from agentflow.agents.vln_agent_2 import Actor, CameraIntrinsics, Subgoal


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


def _actor(response='{"stop": false, "u": 2, "v": 3}'):
    return Actor(
        engine=FakeEngine(response),
        min_depth_m=0.1,
        max_depth_m=5.0,
        patch_radius_px=0,
    )


def test_actor_back_projects_habitat_camera_point_into_world_waypoint():
    decision = _actor().act(
        np.zeros((6, 6, 3), dtype=np.uint8),
        np.full((6, 6), 2.0, dtype=np.float32),
        "go forward",
        CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0),
        np.array(
            (
                (1.0, 0.0, 0.0, 10.0),
                (0.0, 1.0, 0.0, 20.0),
                (0.0, 0.0, 1.0, 30.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )

    assert not decision.stop
    assert decision.point.pixel_uv == (2, 3)
    assert decision.point.camera_xyz == (0.0, -1.0, -2.0)
    assert decision.point.world_xyz == (10.0, 19.0, 28.0)


def test_actor_searches_nearby_when_the_requested_pixel_has_invalid_depth():
    depth = np.full((6, 6), 2.0, dtype=np.float32)
    depth[3, 2] = 0.0
    decision = _actor().act(
        np.zeros((6, 6, 3), dtype=np.uint8),
        depth,
        "go forward",
        np.eye(3),
        np.eye(4),
    )

    assert not decision.stop
    assert decision.point.pixel_uv != (2, 3)
    assert decision.point.depth_m == 2.0


def test_actor_can_explicitly_stop_without_reading_a_waypoint():
    decision = _actor('{"stop": true}').act(
        np.zeros((6, 6, 3), dtype=np.uint8),
        np.full((6, 6), 2.0, dtype=np.float32),
        "stop at the table",
        np.eye(3),
        np.eye(4),
    )

    assert decision.stop
    assert decision.point is None


def test_actor_requires_sensor_range_to_decode_normalized_depth():
    with pytest.raises(ValueError, match="depth_min_m"):
        _actor().act(
            np.zeros((6, 6, 3), dtype=np.uint8),
            np.full((6, 6, 1), 0.5, dtype=np.float32),
            "go forward",
            np.eye(3),
            np.eye(4),
            normalized_depth=True,
        )


def test_prepare_task_decomposes_and_stores_ordered_subgoals():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals": ["Exit the room through the visible doorway.", "Reach and stop beside the sofa."]}'
        )
    )

    subgoals = actor.prepare_task("Leave the room and go to the sofa.")

    assert subgoals == (
        Subgoal("Exit the room through the visible doorway."),
        Subgoal("Reach and stop beside the sofa."),
    )
    assert actor.subgoals == list(subgoals)
    assert actor.task_instruction == "Leave the room and go to the sofa."


def test_prepare_task_rejects_invalid_subgoal_response_without_replacing_state():
    actor = Actor(
        engine=SequenceFakeEngine(
            '{"subgoals": ["Move forward until the doorway is reached."]}',
            '{"subgoals": [{"description": "Not a string."}]}',
        )
    )
    actor.prepare_task("Go forward.")

    with pytest.raises(ValueError, match="invalid subgoal JSON"):
        actor.prepare_task("Turn left.")

    assert actor.task_instruction == "Go forward."
    assert actor.subgoals == [Subgoal("Move forward until the doorway is reached.")]
