from __future__ import annotations

import json

import numpy as np
import pytest

from agentflow.agents.models_embodied_v2 import Subgoal, TaskMemory
from agentflow.agents.vln_agent_3 import (
    Actor,
    GrowingCompletionMemory,
)


PLAN = json.dumps(
    {
        "subgoals": [
            {
                "subgoal_id": "1",
                "description": "Exit the room.",
                "completion_criteria": "The camera is beyond the doorway.",
            },
            {
                "subgoal_id": "2",
                "description": "Stop beside the sofa.",
                "completion_criteria": "The sofa is close and the agent stops.",
            },
        ]
    }
)

NON_DOORWAY_PLAN = json.dumps(
    {
        "subgoals": [
            {
                "subgoal_id": "1",
                "description": "Reach the hall marker.",
                "completion_criteria": "The camera is at the hall marker.",
            },
            {
                "subgoal_id": "2",
                "description": "Stop beside the sofa.",
                "completion_criteria": "The sofa is close and the agent stops.",
            },
        ]
    }
)

LANDMARK_UNKNOWN = (
    '{"visible":false,"direction":"UNKNOWN","proximity":"UNKNOWN",'
    '"passed":false,"confidence":0.2,"evidence":"target not visible"}'
)
LANDMARK_NEAR = (
    '{"visible":true,"direction":"LEFT","proximity":"NEAR",'
    '"passed":false,"confidence":0.91,"evidence":"doorway fills left side"}'
)
LANDMARK_CENTER_NEAR = (
    '{"visible":true,"direction":"CENTER","proximity":"NEAR",'
    '"passed":false,"confidence":0.91,"evidence":"doorway is centered and near"}'
)
LANDMARK_RIGHT_NEAR = (
    '{"visible":true,"direction":"RIGHT","proximity":"NEAR",'
    '"passed":false,"confidence":0.91,"evidence":"doorway fills right side"}'
)
LANDMARK_AT = (
    '{"visible":true,"direction":"CENTER","proximity":"AT",'
    '"passed":false,"confidence":0.91,"evidence":"camera is at threshold"}'
)
LANDMARK_DESTINATION_DOMINANT = (
    '{"visible":true,"direction":"CENTER","proximity":"NEAR",'
    '"passed":false,"destination_dominant":true,"confidence":0.95,'
    '"evidence":"pool and pool-room interior fill the center of the image; '
    'old doorway is only at the right edge"}'
)
LANDMARK_RIGHT_DESTINATION_DOMINANT = (
    '{"visible":true,"direction":"RIGHT","proximity":"NEAR",'
    '"passed":false,"destination_dominant":true,"confidence":0.95,'
    '"evidence":"pool room dominates while doorway remains on the right"}'
)
LANDMARK_LEFT_DESTINATION_DOMINANT = (
    '{"visible":true,"direction":"LEFT","proximity":"NEAR",'
    '"passed":false,"destination_dominant":true,"confidence":0.95,'
    '"evidence":"pool room dominates while doorway has moved to the left"}'
)


class SequenceEngine:
    supports_image_pixel_budget = True

    def __init__(
        self,
        *responses: str,
        landmark_responses: tuple[str, ...] = (),
    ):
        self.responses = list(responses)
        self.landmark_responses = list(landmark_responses)
        self.calls = []

    def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        if "visual landmark tracker" in kwargs.get(
            "system_prompt", ""
        ):
            if self.landmark_responses:
                return self.landmark_responses.pop(0)
            return LANDMARK_UNKNOWN
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def _rgb(value: int = 0) -> np.ndarray:
    return np.full((8, 10, 3), value, dtype=np.uint8)


def _act(
    actor: Actor,
    value: int = 0,
    camera_to_world: np.ndarray | None = None,
):
    return actor.act(
        _rgb(value),
        np.full((8, 10), 2.0, dtype=np.float32),
        "Exit the room and stop beside the sofa.",
        np.array(
            (
                (5.0, 0.0, 4.5),
                (0.0, 5.0, 3.5),
                (0.0, 0.0, 1.0),
            )
        ),
        np.eye(4) if camera_to_world is None else camera_to_world,
    )


def _pose(*, x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    radians = np.radians(yaw_deg)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        )
    )
    transform[0, 3] = x
    return transform


WAYPOINT = (
    '{"stop":false,"intent":"FOLLOW_CORRIDOR","u":556,"v":857,'
    '"confidence":0.9,"evidence":"open floor continues ahead"}'
)
STOP = (
    '{"stop":true,"intent":"STOP","confidence":0.95,'
    '"evidence":"final destination reached"}'
)
NO_ERROR = (
    '{"error":false,"error_mode":"NONE","confidence":0.96,'
    '"evidence":"measured translation shows normal progress"}'
)
TURN_ERROR = (
    '{"error":true,"error_mode":"TURN_OSCILLATION","confidence":0.93,'
    '"evidence":"yaw alternates while translation remains zero"}'
)
WALL_ERROR = (
    '{"error":true,"error_mode":"WALL_STUCK","confidence":0.95,'
    '"evidence":"translation and yaw remain zero against a wall"}'
)


def test_prepare_task_uses_strict_schema_and_growing_memory():
    engine = SequenceEngine(PLAN)
    actor = Actor(engine=engine)

    subgoals = actor.prepare_task(
        "Exit the room and stop beside the sofa."
    )

    assert [item.subgoal_id for item in subgoals] == ["1", "2"]
    assert "reaches that named landmark or decision point" in (
        engine.calls[0][1]["system_prompt"]
    )
    assert isinstance(actor.temporal_memory, GrowingCompletionMemory)
    assert actor.temporal_memory.diagnostics()[
        "error_detection_enabled"
    ] is True


def test_prepare_task_retries_invalid_plan_without_replacing_state():
    valid_one = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Reach the doorway.",
                    "completion_criteria": "The doorway is directly ahead.",
                }
            ]
        }
    )
    invalid_ids = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "2",
                    "description": "Wrong first ID.",
                    "completion_criteria": "Invalid order.",
                }
            ]
        }
    )
    engine = SequenceEngine(invalid_ids, valid_one)
    actor = Actor(engine=engine)

    result = actor.prepare_task("Reach the doorway.")

    assert [item.subgoal_id for item in result] == ["1"]
    assert len(engine.calls) == 2


def test_prepare_task_repairs_only_unquoted_known_json_keys():
    response = (
        '{"subgoals":['
        '{"subgoal_id":"1","description":"Reach door.",'
        '"completion_criteria":"The camera is at the door."},'
        '{"subgoal_id":"2","description":"Turn left.",'
        'completion_criteria":"The camera faces left."}]}'
    )
    actor = Actor(engine=SequenceEngine(response))

    result = actor.prepare_task("Reach the door.")

    assert len(result) == 2
    assert result[1].completion_criteria == "The camera faces left."


def test_prepare_task_removes_stop_only_from_intermediate_criteria():
    response = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Reach the glass door.",
                    "completion_criteria": (
                        "The camera has reached the glass door and is "
                        "stopped at that landmark."
                    ),
                },
                {
                    "subgoal_id": "2",
                    "description": "Stop beside the table.",
                    "completion_criteria": (
                        "The camera is stopped beside the table."
                    ),
                },
            ]
        }
    )
    actor = Actor(engine=SequenceEngine(response))

    result = actor.prepare_task("Reach the glass door, then the table.")

    assert result[0].completion_criteria == (
        "The camera has reached the glass door."
    )
    assert result[1].completion_criteria == (
        "The camera is stopped beside the table."
    )


def test_prepare_task_repairs_invented_full_circuit_requirement():
    response = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Walk clockwise around the bed.",
                    "completion_criteria": (
                        "Return to the starting point after completing a "
                        "full clockwise circuit."
                    ),
                },
                {
                    "subgoal_id": "2",
                    "description": "Exit through the hallway door.",
                    "completion_criteria": "The hallway is the central view.",
                },
            ]
        }
    )
    actor = Actor(engine=SequenceEngine(response))

    result = actor.prepare_task(
        "Walk clockwise around the bed and exit through the hallway door."
    )

    assert "full clockwise circuit" not in result[0].completion_criteria
    assert "starting point" not in result[0].completion_criteria
    assert "Exit through the hallway door" in (
        result[0].completion_criteria
    )


def test_prepare_task_rewrites_pre_turn_endpoint_as_route_geometry():
    response = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Walk straight down the hallway.",
                    "completion_criteria": "Reach the glass door.",
                },
                {
                    "subgoal_id": "2",
                    "description": "Turn left at the glass door.",
                    "completion_criteria": "Complete the left turn.",
                },
            ]
        }
    )
    actor = Actor(engine=SequenceEngine(response))

    result = actor.prepare_task("Walk ahead, then turn left.")

    assert result[0].description.endswith(
        "until reaching the next left-turn decision point."
    )
    assert "glass" not in result[0].completion_criteria.lower()
    assert "where the next left turn can now be executed" in (
        result[0].completion_criteria
    )
    assert result[1].description == "Turn left at the glass door."


def test_consecutive_turns_include_the_intervening_corridor_leg():
    response = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Turn left at the glass door.",
                    "completion_criteria": "Complete the left turn.",
                },
                {
                    "subgoal_id": "2",
                    "description": "Turn right at the hallway.",
                    "completion_criteria": "Complete the right turn.",
                },
            ]
        }
    )
    actor = Actor(engine=SequenceEngine(response))

    result = actor.prepare_task("Turn left, then turn right.")

    assert result[0].description.endswith(
        "then follow the new corridor until reaching the next "
        "right-turn decision point."
    )
    actor._sync_navigation_phase(result[0])
    assert actor._navigation_phase == "TURN_LEFT"
    for _ in range(4):
        actor._update_navigation_progress(
            result[0],
            yaw_delta_deg=-15.0,
        )
    assert actor._navigation_phase == "FOLLOW_CORRIDOR"
    assert actor._turn_follow_phase_started is True


def test_new_corridor_is_centered_before_heading_lock_is_established():
    response = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Turn left at the glass door.",
                    "completion_criteria": "Complete the left turn.",
                },
                {
                    "subgoal_id": "2",
                    "description": "Turn right at the hallway.",
                    "completion_criteria": "Complete the right turn.",
                },
            ]
        }
    )
    side_point = (
        '{"stop":false,"intent":"FOLLOW_CORRIDOR","u":200,"v":400,'
        '"confidence":0.8,"evidence":"side opening looks traversable"}'
    )
    actor = Actor(
        engine=SequenceEngine(response, side_point),
        patch_radius_px=0,
    )
    subgoal = actor.prepare_task("Turn left, then turn right.")[0]
    actor._sync_navigation_phase(subgoal)
    for _ in range(4):
        actor._update_navigation_progress(
            subgoal,
            yaw_delta_deg=-15.0,
        )

    pixel = actor._select_pixel(_rgb(), "Turn left, then turn right.")

    assert actor._navigation_phase == "FOLLOW_CORRIDOR"
    assert actor._corridor_heading_yaw_deg is None
    assert actor.last_requested_normalized == (500, 400)
    assert pixel == (5, 3)
    assert "must remain centered" in actor.last_waypoint_guard_reason


@pytest.mark.parametrize(
    "payload",
    [
        {"subgoals": []},
        {
            "subgoals": [
                {
                    "subgoal_id": 1,
                    "description": "Numeric ID is rejected.",
                    "completion_criteria": "Strict types.",
                }
            ]
        },
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Extra field.",
                    "completion_criteria": "Strict shape.",
                    "unexpected": True,
                }
            ]
        },
    ],
)
def test_prepare_task_rejects_invalid_plan_after_retry(payload):
    response = json.dumps(payload)
    actor = Actor(engine=SequenceEngine(response, response))

    with pytest.raises(ValueError, match="after 2 attempts"):
        actor.prepare_task("Go forward.")


def test_landmark_tracker_feeds_completion_and_waypoint_context():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        landmark_responses=(LANDMARK_NEAR,),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor).stop is False

    assert actor.last_landmark is not None
    assert actor.last_landmark.visible is True
    assert actor.last_landmark.direction == "LEFT"
    assert actor.last_landmark.proximity == "NEAR"
    assert actor.last_landmark_error is None
    completion_content = engine.calls[2][0]
    assert "landmark_visible=True" in completion_content[1]
    assert "landmark_proximity=NEAR" in completion_content[1]
    waypoint_prompt = engine.calls[-1][0][0]
    assert "Current landmark state: visible=True" in waypoint_prompt
    assert "proximity=NEAR" in waypoint_prompt


def test_waypoint_receives_measured_behavior_history():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, 1, _pose(x=0.0)).stop is False
    assert _act(actor, 2, _pose(x=0.25)).stop is False

    history = actor.behavior_history()
    assert history[-1]["behavior"] == "MOVE_FORWARD"
    assert history[-1]["translation_m"] == pytest.approx(0.25)
    assert history[-1]["requested_waypoint"] == (556, 857)
    waypoint_prompt = engine.calls[-1][0][0]
    assert "behavior=MOVE_FORWARD" in waypoint_prompt
    assert "previous_normalized_waypoint=(556,857)" in waypoint_prompt
    assert "Measured path length in active subgoal: 0.250 m" in (
        engine.calls[-3][0][0]
    )


def test_corridor_heading_lock_blocks_premature_side_turn():
    straight_plan = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Walk straight down the hallway.",
                    "completion_criteria": "Reach the glass door.",
                }
            ]
        }
    )
    side_turn = (
        '{"stop":false,"intent":"TURN_RIGHT","u":800,"v":600,'
        '"confidence":0.9,"evidence":"open side doorway"}'
    )
    engine = SequenceEngine(
        straight_plan,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        side_turn,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, 1, _pose(x=0.0)).stop is False
    assert _act(actor, 2, _pose(x=0.25)).stop is False
    decision = _act(actor, 3, _pose(x=0.50))

    assert decision.stop is False
    assert actor.last_waypoint_model_intent == "TURN_RIGHT"
    assert actor.last_waypoint_applied_intent == "FOLLOW_CORRIDOR"
    assert actor.last_requested_normalized == (500, 600)
    assert decision.point.pixel_uv == (5, 4)
    assert "blocked side turn" in (
        actor.last_waypoint_guard_reason
    )
    assert "Corridor heading lock: active" in engine.calls[-1][0][0]


def test_measured_turn_closes_unrenderable_pre_turn_decision_point():
    route_plan = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Walk straight down the hallway.",
                    "completion_criteria": "Reach the glass door.",
                },
                {
                    "subgoal_id": "2",
                    "description": "Turn left at the glass door.",
                    "completion_criteria": "Complete the left turn.",
                },
            ]
        }
    )
    responses = [route_plan]
    for _ in range(6):
        responses.extend(('{"completed":false}', WAYPOINT))
    actor = Actor(engine=SequenceEngine(*responses), patch_radius_px=0)
    actor.prepare_task("Walk down the hall, then turn left.")

    poses = [
        _pose(x=0.0),
        _pose(x=0.5),
        _pose(x=1.0),
        _pose(x=1.5),
        _pose(x=2.0),
        _pose(x=2.0, yaw_deg=15.0),
    ]
    for pose in poses[:-1]:
        assert _act(actor, camera_to_world=pose).stop is False
        assert actor.task_memory.get_current_subgoal().subgoal_id == "1"

    assert _act(actor, camera_to_world=poses[-1]).stop is False

    assert actor.last_caption.completed is True
    assert actor.last_subgoal_before == "1"
    assert actor.last_subgoal_after == "2"
    raw = json.loads(actor.last_caption.raw_response)
    assert "path=2.00m" in raw["completion_guard"]
    assert "measured left yaw=-15.0deg" in raw["completion_guard"]


def test_measured_forward_motion_after_doorway_at_closes_exit_stage():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        landmark_responses=(
            LANDMARK_NEAR,
            LANDMARK_AT,
            LANDMARK_DESTINATION_DOMINANT,
        ),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, camera_to_world=_pose(x=0.0)).stop is False
    assert _act(actor, camera_to_world=_pose(x=0.50)).stop is False
    decision = _act(actor, camera_to_world=_pose(x=0.75))

    assert decision.stop is False
    assert actor.last_caption.completed is True
    assert actor.last_subgoal_before == "1"
    assert actor.last_subgoal_after == "2"
    raw = json.loads(actor.last_caption.raw_response)
    assert "tracker passed" in raw["completion_guard"]
    assert "verified doorway crossing from AT" in (
        actor.last_landmark.evidence
    )


def test_two_dominant_destination_views_close_doorway_stage():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        landmark_responses=(
            LANDMARK_CENTER_NEAR,
            LANDMARK_DESTINATION_DOMINANT,
        ),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, camera_to_world=_pose(x=0.0)).stop is False
    decision = _act(actor, camera_to_world=_pose(x=0.50))

    assert decision.stop is False
    assert actor.last_landmark is not None
    assert actor.last_landmark.destination_dominant is True
    assert actor.last_landmark.passed is True
    assert actor.last_caption.completed is True
    assert actor.last_subgoal_before == "1"
    assert actor.last_subgoal_after == "2"
    raw = json.loads(actor.last_caption.raw_response)
    assert "tracker passed" in raw["completion_guard"]


def test_dominant_destination_without_threshold_sequence_is_rejected():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":true}',
        WAYPOINT,
        landmark_responses=(
            LANDMARK_UNKNOWN,
            LANDMARK_DESTINATION_DOMINANT,
        ),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, camera_to_world=_pose(x=0.0)).stop is False
    decision = _act(actor, camera_to_world=_pose(x=0.50))

    assert decision.stop is False
    assert actor.last_landmark is not None
    assert actor.last_landmark.destination_dominant is True
    assert actor.last_landmark.passed is False
    assert actor.last_caption.completed is False
    assert actor.task_memory.get_current_subgoal().subgoal_id == "1"
    raw = json.loads(actor.last_caption.raw_response)
    assert "rejected doorway completion before sufficient" in (
        raw["completion_guard"]
    )


def test_translated_dominant_side_transition_proves_oblique_crossing():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        landmark_responses=(
            LANDMARK_RIGHT_DESTINATION_DOMINANT,
            LANDMARK_RIGHT_DESTINATION_DOMINANT,
            LANDMARK_RIGHT_DESTINATION_DOMINANT,
            LANDMARK_LEFT_DESTINATION_DOMINANT,
        ),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    for x in (0.0, 0.25, 0.50):
        assert _act(actor, camera_to_world=_pose(x=x)).stop is False
    decision = _act(actor, camera_to_world=_pose(x=0.75))

    assert decision.stop is False
    assert actor.last_landmark is not None
    assert actor.last_landmark.passed is True
    assert actor.last_caption.completed is True
    assert actor.task_memory.get_current_subgoal().subgoal_id == "2"
    assert "side transition RIGHT->LEFT" in actor.last_landmark.evidence


def test_doorway_side_change_alone_does_not_fake_a_crossing():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        landmark_responses=(
            LANDMARK_RIGHT_NEAR,
            LANDMARK_RIGHT_NEAR,
            LANDMARK_NEAR,
        ),
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert _act(actor, camera_to_world=_pose(x=0.0)).stop is False
    assert _act(actor, camera_to_world=_pose(x=0.25)).stop is False
    decision = _act(
        actor,
        camera_to_world=_pose(x=0.50),
    )

    assert decision.stop is False
    assert actor.last_caption.completed is False
    assert actor.task_memory.get_current_subgoal().subgoal_id == "1"
    assert "completion_guard" not in json.loads(
        actor.last_caption.raw_response
    )


def test_measured_net_rotation_closes_pure_turn_subgoal():
    turn_plan = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Turn left and align with the hallway.",
                    "completion_criteria": "Complete the left turn.",
                }
            ]
        }
    )
    responses = [turn_plan]
    for _ in range(4):
        responses.extend(('{"completed":false}', WAYPOINT))
    responses.append('{"completed":false}')
    actor = Actor(engine=SequenceEngine(*responses), patch_radius_px=0)
    actor.prepare_task("Turn left.")

    for yaw in (0.0, 15.0, 30.0, 45.0):
        assert _act(
            actor,
            camera_to_world=_pose(yaw_deg=yaw),
        ).stop is False
    decision = _act(actor, camera_to_world=_pose(yaw_deg=60.0))

    assert decision.stop is True
    assert actor.task_memory.is_task_complete()
    assert actor.last_caption.completed is True
    raw = json.loads(actor.last_caption.raw_response)
    assert "net left yaw=60.0deg" in raw["completion_guard"]


def test_repeated_near_field_model_evidence_stops_at_final_target():
    final_plan = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Walk to the table on the right and stop.",
                    "completion_criteria": (
                        "Reach the table and stop beside it."
                    ),
                }
            ]
        }
    )
    final_waypoint = (
        '{"stop":false,"intent":"FINAL_APPROACH","u":500,"v":500,'
        '"confidence":0.7,"evidence":"The console table is visible and '
        'appears to be the target landmark."}'
    )
    engine = SequenceEngine(
        final_plan,
        '{"completed":false}',
        final_waypoint,
        '{"completed":false}',
        final_waypoint,
        '{"completed":false}',
        final_waypoint,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Walk to the table on the right and stop.")

    assert _act(actor, camera_to_world=_pose(x=0.0)).stop is False
    assert _act(actor, camera_to_world=_pose(x=0.25)).stop is False
    decision = _act(actor, camera_to_world=_pose(x=0.50))

    assert decision.stop is True
    assert "final target evidence guard" in decision.raw_response
    assert actor.last_waypoint_stop_disposition == (
        "accepted_final_evidence_guard"
    )
    assert actor.last_waypoint_evidence == (
        "The console table is visible and appears to be the target landmark."
    )
    assert not actor.task_memory.is_task_complete()


def test_completion_history_grows_but_model_evidence_is_bounded():
    responses = [PLAN]
    for index in range(20):
        responses.append('{"completed":false}')
        if index >= 7:
            responses.append(NO_ERROR)
        responses.append(WAYPOINT)
    engine = SequenceEngine(*responses)
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    for index in range(20):
        decision = _act(actor, index)
        assert decision.stop is False

    memory = actor.temporal_memory
    diagnostics = memory.diagnostics()
    assert diagnostics["completion_history_size"] == 20
    # One anchor plus the recent eight-frame error window keeps temporal
    # evidence bounded after error detection becomes active.
    assert diagnostics["completion_window_size"] == 9
    assert diagnostics["completion_frame_ids"][0] == 1
    assert diagnostics["completion_frame_ids"][-8:] == list(
        range(13, 21)
    )
    assert [frame.frame_id for frame in memory.recent_frames()] == list(
        range(1, 21)
    )
    # Planning plus landmark, completion, optional error and waypoint calls.
    assert len(engine.calls) == 74
    final_completion_content = engine.calls[-3][0]
    assert sum(
        isinstance(item, bytes) for item in final_completion_content
    ) == 9
    assert "landmark_visible=False" in final_completion_content[1]
    final_error_content = engine.calls[-2][0]
    assert sum(isinstance(item, bytes) for item in final_error_content) == 8
    assert "translation_m=0.000" in final_error_content[1]
    assert "yaw_delta_deg=+0.0" in final_error_content[1]
    for _, kwargs in engine.calls[-4:]:
        assert kwargs["image_min_pixels"] == 64**2
    assert [
        kwargs["max_tokens"] for _, kwargs in engine.calls[-4:]
    ] == [64, 128, 128, 64]
    assert engine.calls[-4][1]["image_max_pixels"] == 224**2
    assert engine.calls[-3][1]["image_max_pixels"] == 160**2
    assert engine.calls[-2][1]["image_max_pixels"] == 160**2
    assert engine.calls[-1][1]["image_max_pixels"] == 224**2
    assert actor.last_caption.error is False
    assert actor.last_caption.error_mode == "NONE"


def test_completed_subgoal_clears_history_and_final_completion_stops():
    engine = SequenceEngine(
        NON_DOORWAY_PLAN,
        '{"completed":true}',
        WAYPOINT,
        '{"completed":false}',
        WAYPOINT,
        '{"completed":true}',
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    first = _act(actor, 1)
    assert first.stop is False
    assert actor.task_memory.get_current_subgoal().subgoal_id == "2"

    second = _act(actor, 2)
    assert second.stop is False
    assert [
        frame.subgoal_id for frame in actor.temporal_memory.recent_frames()
    ] == ["2"]

    third = _act(actor, 3)
    assert third.stop is True
    assert actor.task_memory.is_task_complete()
    assert actor.last_model_response == "all subgoals complete"


def test_rgbd_waypoint_back_projection_remains_compatible():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        WAYPOINT,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = _act(actor)

    assert decision.stop is False
    assert decision.point.pixel_uv == (5, 6)
    assert actor.last_requested_normalized == (556, 857)
    assert decision.point.depth_m == 2.0
    assert decision.point.camera_xyz == pytest.approx(
        (0.2, -1.0, -2.0)
    )


def test_waypoint_model_cannot_stop_before_completion_memory():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        STOP,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = _act(actor)

    assert decision.stop is False
    assert decision.point.pixel_uv == (5, 5)
    assert "[STOP deferred]" in decision.raw_response
    assert actor.last_waypoint_stop_disposition == "ignored_nonfinal"
    assert actor.last_waypoint_raw_response == STOP
    assert actor.last_requested_pixel == (5, 5)
    assert actor.last_subgoal_before == "1"
    assert actor.last_subgoal_after == "1"
    assert not actor.task_memory.is_task_complete()


def test_nonfinal_stop_with_unused_coordinates_is_ignored():
    engine = SequenceEngine(
        PLAN,
        '{"completed":false}',
        '{"stop":true,"intent":"STOP","u":480,"v":490,'
        '"confidence":0.9,"evidence":"destination reached"}',
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = _act(actor)

    assert decision.stop is False
    assert decision.point.pixel_uv == (5, 5)
    assert actor.last_waypoint_stop_disposition == "ignored_nonfinal"
    assert actor.last_requested_normalized == (500, 750)


def test_waypoint_stop_on_final_subgoal_waits_for_verification():
    engine = SequenceEngine(
        NON_DOORWAY_PLAN,
        '{"completed":true}',
        WAYPOINT,
        '{"completed":false}',
        STOP,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")
    assert _act(actor, 1).stop is False
    assert actor.task_memory.get_current_subgoal().subgoal_id == "2"

    decision = _act(actor, 2)

    assert decision.stop is False
    assert "[STOP deferred]" in decision.raw_response
    assert actor.last_waypoint_stop_disposition == (
        "deferred_unverified_final"
    )
    assert actor.last_waypoint_raw_response == STOP
    assert actor.last_requested_pixel is not None
    assert not actor.task_memory.is_task_complete()


def test_visible_final_destination_cannot_bypass_active_subgoal():
    three_stage_plan = json.dumps(
        {
            "subgoals": [
                {
                    "subgoal_id": "1",
                    "description": "Reach the hallway.",
                    "completion_criteria": "The hallway is reached.",
                },
                {
                    "subgoal_id": "2",
                    "description": "Follow the hallway.",
                    "completion_criteria": "Reach the next junction.",
                },
                {
                    "subgoal_id": "3",
                    "description": "Stop beside the table.",
                    "completion_criteria": "The table is beside the camera.",
                },
            ]
        }
    )
    near_table_stop = (
        '{"stop":true,"intent":"STOP","confidence":0.95,'
        '"evidence":"the final table is beside the camera at immediate '
        'stopping distance"}'
    )
    engine = SequenceEngine(
        three_stage_plan,
        '{"completed":true}',
        WAYPOINT,
        '{"completed":false}',
        near_table_stop,
    )
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Follow the hallway and stop beside the table.")

    assert _act(actor, 1).stop is False
    assert actor.task_memory.get_current_subgoal().subgoal_id == "2"

    decision = _act(actor, 2)

    assert decision.stop is False
    assert actor.last_waypoint_stop_disposition == "ignored_nonfinal"
    assert not actor.task_memory.is_task_complete()
    assert actor.last_subgoal_before == "2"
    assert actor.last_subgoal_after == "2"


def test_error_requires_four_confident_votes_before_action_level_recovery():
    responses = [PLAN]
    for index in range(11):
        responses.append('{"completed":false}')
        if index >= 7:
            responses.append(TURN_ERROR)
        responses.append(WAYPOINT)
    engine = SequenceEngine(*responses)
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    for value in range(8):
        assert _act(
            actor,
            value,
            _pose(yaw_deg=15.0 if value % 2 else 0.0),
        ).stop is False
    first_error_prompt = engine.calls[-1][0][0]
    assert "Confirmed recent error" not in first_error_prompt

    assert _act(actor, 8, _pose(yaw_deg=0.0)).stop is False
    assert actor.last_recovery_mode is None
    assert _act(actor, 9, _pose(yaw_deg=15.0)).stop is False
    assert actor.last_recovery_mode is None
    decision = _act(actor, 10, _pose(yaw_deg=0.0))

    assert decision.stop is False
    assert actor.last_caption.error is True
    assert actor.last_caption.error_mode == "TURN_OSCILLATION"
    assert actor.last_caption.error_confidence == pytest.approx(0.93)
    assert actor.task_memory.temporal_status == (
        "ERROR=True; mode=TURN_OSCILLATION"
    )
    assert actor.last_error_candidate == "TURN_OSCILLATION"
    assert actor.last_recovery_mode == "TURN_OSCILLATION"
    assert "both yaw signs" in actor.last_error_guard_reason
    assert actor.last_requested_normalized == (500, 750)
    assert actor.last_waypoint_guard_reason == (
        "forced deterministic action waypoint for TURN_OSCILLATION"
    )
    assert "action-level TURN_OSCILLATION recovery" in (
        actor.last_waypoint_evidence
    )
    assert engine.responses == [WAYPOINT]
    assert json.loads(actor.last_caption.raw_response) == {
        "completion": '{"completed":false}',
        "error": TURN_ERROR,
    }


def test_confirmed_wall_stuck_forces_a_left_turn_waypoint():
    responses = [PLAN]
    for index in range(11):
        responses.append('{"completed":false}')
        if index >= 7:
            responses.append(WALL_ERROR)
        responses.append(WAYPOINT)
    engine = SequenceEngine(*responses)
    actor = Actor(engine=engine, patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = None
    for value in range(11):
        decision = _act(actor, value, _pose())
        assert decision.stop is False

    assert decision is not None and decision.point is not None
    assert actor.last_recovery_mode == "WALL_STUCK"
    assert actor.last_requested_normalized == (250, 500)
    assert decision.point.world_xyz[0] == pytest.approx(-1.5)
    assert decision.point.world_xyz[2] == pytest.approx(0.0)
    assert "force-left-turn waypoint" in decision.raw_response
    assert engine.responses == [WAYPOINT]


def test_force_forward_flag_does_not_leak_into_the_next_act():
    actor = Actor(
        engine=SequenceEngine(
            PLAN,
            '{"completed":false}',
            WAYPOINT,
        ),
        patch_radius_px=0,
    )
    actor.prepare_task("Exit the room and stop beside the sofa.")
    actor._force_forward_this_step = True
    actor._force_left_turn_this_step = True

    decision = _act(actor, 1, _pose())

    assert decision.stop is False
    assert actor._force_forward_this_step is False
    assert actor._force_left_turn_this_step is False
    assert "force-forward waypoint" not in decision.raw_response
    assert "force-left-turn waypoint" not in decision.raw_response


def test_no_valid_depth_returns_turn_recovery_instead_of_raising():
    actor = Actor(
        engine=SequenceEngine(
            PLAN,
            '{"completed":false}',
            WAYPOINT,
        ),
        patch_radius_px=0,
    )
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = actor.act(
        _rgb(),
        np.zeros((8, 10), dtype=np.float32),
        "Exit the room and stop beside the sofa.",
        np.array(
            (
                (5.0, 0.0, 4.5),
                (0.0, 5.0, 3.5),
                (0.0, 0.0, 1.0),
            )
        ),
        _pose(),
    )

    assert decision.stop is False
    assert decision.point is not None
    assert actor.last_recovery_mode == "NO_VALID_DEPTH"
    assert actor.last_requested_normalized == (250, 500)
    assert "no-valid-depth recovery" in decision.raw_response
    assert "force-left-turn waypoint" in decision.raw_response


def test_motion_guard_rejects_error_while_agent_is_progressing():
    progressing_error = (
        '{"error":true,"error_mode":"GET_NOWHERE","confidence":1.0,'
        '"evidence":"incorrectly claims no progress"}'
    )
    responses = [PLAN]
    for index in range(9):
        responses.append('{"completed":false}')
        if index >= 7:
            responses.append(progressing_error)
        responses.append(WAYPOINT)
    actor = Actor(engine=SequenceEngine(*responses), patch_radius_px=0)
    actor.prepare_task("Exit the room and stop beside the sofa.")

    for value in range(9):
        decision = _act(
            actor,
            value,
            _pose(x=value * 0.25),
        )
        assert decision.stop is False

    assert actor.last_caption.error_mode == "GET_NOWHERE"
    assert actor.last_error_candidate == "NONE"
    assert actor.last_recovery_mode is None
    assert "recent translation 1.00m" in actor.last_error_guard_reason


def test_invalid_waypoint_uses_safe_fallback_without_aborting_episode():
    invalid = (
        '{"stop":false,"intent":"FOLLOW_CORRIDOR","u":500,'
        '"v":1001,"confidence":0.9,"evidence":"open floor"}'
    )
    actor = Actor(
        engine=SequenceEngine(
            PLAN,
            '{"completed":false}',
            invalid,
            invalid,
        )
    )
    actor.prepare_task("Exit the room and stop beside the sofa.")

    decision = _act(actor)

    assert decision.stop is False
    assert actor.last_requested_normalized == (500, 750)
    assert "safe fallback" in decision.raw_response
    assert "invalid waypoint JSON" in actor.last_waypoint_guard_reason


def test_injected_task_memory_is_reset_with_validated_subgoals():
    task = TaskMemory(
        "Old task.",
        subgoals=(Subgoal("1", "Old.", "Old proof."),),
    )
    actor = Actor(engine=SequenceEngine(PLAN), task_memory=task)

    actor.prepare_task("Exit the room and stop beside the sofa.")

    assert actor.task_memory is task
    assert actor.task_memory.get_task() == (
        "Exit the room and stop beside the sofa."
    )
    assert actor.task_memory.get_current_subgoal().description == (
        "Exit the room."
    )
