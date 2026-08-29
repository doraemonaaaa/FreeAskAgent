from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from agentflow.agents.models_embodied_v2.skiils.protocol import (
    MAX_COMPLETION_EVIDENCE_FRAMES,
)

from agentflow.agents.models_embodied_v2 import (
    CaptionResult,
    Subgoal,
)
from agentflow.agents.models_embodied_v2.memory import (
    TaskMemory,
    TemporalEventKind,
    TemporalMemory,
    TemporalMemoryConfig,
    TemporalStateError,
)
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    FinalTargetEvidence,
    SceneAnalysisResult,
    SceneLandmark,
)


np = pytest.importorskip("numpy")


class FakeCaptioner:
    def __init__(self, *outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def analyze(self, request):
        self.calls.append(request)
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        return output


class SceneCaptioner:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.calls = []

    def analyze_scene(self, request):
        self.calls.append(request)
        return self.result_factory(request)


def _scene_result(
    request,
    *,
    completed=False,
    destination_dominant=False,
    final_visible=False,
    proximity="NEAR",
    door_state="NOT_APPLICABLE",
    door_camera_side="NOT_APPLICABLE",
):
    evidence = "bounded unified scene evidence"
    return SceneAnalysisResult(
        subgoal_id=request.subgoal.subgoal_id,
        landmark=SceneLandmark(
            visible=True,
            direction="CENTER",
            proximity=proximity,
            passed=completed,
            destination_dominant=destination_dominant,
            confidence=0.95,
            evidence=evidence,
            u=500,
            v=450,
        ),
        completed=completed,
        completion_confidence=0.95 if completed else 0.0,
        completion_evidence=evidence,
        door_state=door_state,
        door_camera_side=door_camera_side,
        error=False,
        error_mode="NONE",
        error_confidence=0.0,
        error_evidence=evidence,
        final_target=FinalTargetEvidence(
            visible=final_visible,
            proximity=proximity if final_visible else "UNKNOWN",
            confidence=0.95 if final_visible else 0.0,
            evidence=evidence,
        ),
        raw_response="{}",
        latency_ms=1.0,
    )


def _result(
    *,
    subgoal_id="1",
    completed=False,
    error=False,
    error_mode="NONE",
):
    return CaptionResult(
        subgoal_id=subgoal_id,
        completed=completed,
        error=error,
        error_mode=error_mode,
        raw_response=(
            f'{{"completed":{str(completed).lower()},'
            f'"error":{str(error).lower()}}}'
        ),
        latency_ms=10.0,
    )


def _subgoals():
    return (
        Subgoal("1", "Exit the room", "Cross the doorway threshold."),
        Subgoal("2", "Stop before the pool", "Reach the pool edge."),
    )


def _memory(
    *outputs,
    enable_error_detection=False,
):
    task = TaskMemory(
        "Exit into the pool room and stop before the pool.",
        subgoals=_subgoals(),
    )
    captioner = FakeCaptioner(*outputs)
    memory = TemporalMemory(
        captioner=captioner,
        task_memory=task,
        config=TemporalMemoryConfig(
            enable_error_detection=enable_error_detection
        ),
    )
    return memory, captioner, task


def _frame(index):
    return np.random.default_rng(index).integers(
        0,
        256,
        (24, 32, 3),
        dtype=np.uint8,
    )


def _push(memory, count=8, *, start=0):
    for index in range(start, start + count):
        memory.append_observation(_frame(index))


def test_eight_images_are_copied_and_analyzed_without_actions():
    memory, captioner, _ = _memory(_result())
    mutable = np.zeros((24, 32, 3), dtype=np.uint8)
    memory.append_observation(mutable)
    mutable[:] = 255
    _push(memory, 7, start=1)

    assert memory.analyze_if_ready() is not None
    request = captioner.calls[0]
    assert request.subgoal == _subgoals()[0]
    assert [frame.frame_id for frame in request.frames] == list(range(1, 9))
    assert [frame.frame_id for frame in memory.recent_frames()] == list(
        range(1, 9)
    )
    assert int(np.asarray(memory.recent_frames()[0].image).max()) == 0
    assert not hasattr(memory, "append_step")
    assert not hasattr(memory, "recent_actions")


def test_task_memory_rgb_is_consumed_once_and_analyzed_on_each_frame():
    memory, captioner, task = _memory(
        *(_result() for _ in range(8))
    )

    for index in range(8):
        task.record_input(_frame(index))
        result = memory.update_from_task_memory()
        assert result is not None
        assert len(captioner.calls[-1].frames) == index + 1

    assert len(captioner.calls) == 8
    assert len(memory.recent_frames()) == 8
    assert memory.update_from_task_memory() is None
    assert len(memory.recent_frames()) == 8


def test_each_analysis_publishes_only_completion_event():
    memory, _, task = _memory(
        _result(error=True, error_mode="WALL_STUCK")
    )
    _push(memory)
    memory.analyze()

    events = memory.drain_events()
    assert [event.kind for event in events] == [
        TemporalEventKind.SUBGOAL_COMPLETED,
    ]
    assert events[0].value is False
    assert list(task.temporal_events) == [
        event.to_dict() for event in events
    ]
    assert task.temporal_status == ""


def test_captioner_error_is_normalized_when_detection_is_disabled():
    memory, _, _ = _memory(
        _result(error=True, error_mode="IN_PLACE_SPIN")
    )
    _push(memory)

    result = memory.analyze()

    assert result.error is False
    assert result.error_mode == "NONE"
    assert memory.latest_result == result


def test_enabled_error_is_published_without_local_rule_override():
    memory, _, task = _memory(
        _result(error=True, error_mode="IN_PLACE_SPIN"),
        enable_error_detection=True,
    )
    same = np.zeros((24, 32, 3), dtype=np.uint8)
    for _ in range(8):
        memory.append_observation(same)

    result = memory.analyze()

    assert result.error is True
    assert result.error_mode == "IN_PLACE_SPIN"
    events = memory.drain_events()
    assert [event.kind for event in events] == [
        TemporalEventKind.ERROR,
        TemporalEventKind.SUBGOAL_COMPLETED,
    ]
    assert task.temporal_status == "ERROR=True; mode=IN_PLACE_SPIN"


def test_enabled_error_requires_consistent_fields():
    memory, _, _ = _memory(
        _result(error=False, error_mode="GET_NOWHERE"),
        enable_error_detection=True,
    )
    _push(memory)

    with pytest.raises(TemporalStateError, match="inconsistent"):
        memory.analyze()


def test_visual_rule_does_not_add_error_when_detection_is_disabled():
    memory, _, task = _memory(_result())
    same = np.zeros((24, 32, 3), dtype=np.uint8)
    for _ in range(8):
        memory.append_observation(same)

    result = memory.analyze()

    assert result.error is False
    assert result.error_mode == "NONE"
    assert task.temporal_status == ""


def test_completion_advances_subgoal_and_next_frame_clears_old_window():
    memory, _, task = _memory(_result(completed=True))
    _push(memory)
    memory.analyze()

    assert task.get_current_subgoal().subgoal_id == "2"
    memory.append_observation(_frame(9))

    frames = memory.recent_frames()
    assert [frame.frame_id for frame in frames] == [9]
    assert frames[0].subgoal_id == "2"


def test_task_reset_automatically_resets_temporal_memory():
    memory, _, task = _memory(_result())
    task.record_input(_frame(0))
    memory.update_from_task_memory()
    old_generation = task.get_reset_generation()
    assert [frame.frame_id for frame in memory.recent_frames()] == [1]

    # The first frame of the new episode also has observation_count == 1.
    # Temporal Memory must detect the Task Memory generation change before
    # applying duplicate-observation filtering.
    task.reset(
        goal="A new episode with the same subgoal ID.",
        subgoals=(Subgoal("1", "Find the stairs", "See the stairs."),),
    )
    assert task.get_reset_generation() == old_generation + 1
    assert memory.recent_frames() == ()
    assert memory.current_subgoal.description == "Find the stairs"

    new_image = _frame(99)
    task.record_input(new_image)
    memory.update_from_task_memory()

    frames = memory.recent_frames()
    assert [frame.frame_id for frame in frames] == [1]
    assert np.array_equal(frames[0].image, new_image)
    assert memory.current_subgoal.description == "Find the stairs"
    assert memory.latest_result is None
    assert memory.drain_events() == ()


def test_reset_episode_resets_both_memories_in_one_call():
    memory, _, task = _memory(_result())
    task.record_input(_frame(0))
    memory.update_from_task_memory()
    old_generation = task.get_reset_generation()

    memory.reset_episode(
        goal="Navigate to the kitchen.",
        task_guidance="Look for kitchen counters.",
        subgoals=(
            Subgoal("1", "Enter the kitchen", "Kitchen counters are nearby."),
        ),
    )

    assert task.get_reset_generation() == old_generation + 1
    assert task.get_task() == "Navigate to the kitchen."
    assert task.observation_count == 0
    assert memory.recent_frames() == ()
    assert memory.current_subgoal.description == "Enter the kitchen"
    assert memory.latest_result is None


def test_manual_reset_clears_temporal_state():
    memory, _, _ = _memory(_result())
    _push(memory, 7)
    memory.reset()

    assert memory.recent_frames() == ()
    assert memory.latest_result is None
    assert memory.drain_events() == ()
    assert memory.diagnostics()["frame_ids"] == []


def test_model_failure_keeps_window_and_emits_no_event():
    memory, _, task = _memory(RuntimeError("model unavailable"))
    _push(memory)

    assert memory.analyze_if_ready() is None
    assert len(memory.recent_frames()) == 8
    assert "model unavailable" in memory.last_analysis_error
    assert memory.drain_events() == ()
    assert list(task.temporal_events) == []


def test_diagnostics_serializes_dataclass_without_action_fields():
    memory, _, _ = _memory(_result())
    _push(memory)
    memory.analyze()

    diagnostics = memory.diagnostics()

    assert diagnostics["frame_ids"] == list(range(1, 9))
    assert "actions" not in diagnostics
    assert "raw_response" not in diagnostics["latest_result"]
    assert diagnostics["latest_result"]["completed"] is False


@dataclass(frozen=True)
class AgentSubgoal:
    task: str
    guidance: str


def test_task_memory_accepts_vln_agent_subgoal_shape():
    task = TaskMemory(
        "Find the table",
        subgoals=[
            AgentSubgoal(
                task="Reach the hallway",
                guidance="The camera is inside the hallway.",
            )
        ],
    )

    current = task.get_current_subgoal()
    assert current.subgoal_id == "1"
    assert current.description == "Reach the hallway"
    assert current.completion_criteria == "The camera is inside the hallway."


def test_missing_subgoal_and_window_size_invariants():
    task = TaskMemory("Find the table")
    memory = TemporalMemory(
        captioner=FakeCaptioner(),
        task_memory=task,
    )

    with pytest.raises(TemporalStateError, match="current subgoal"):
        memory.append_observation(_frame(1))
    with pytest.raises(ValueError, match="fixed eight-frame"):
        TemporalMemoryConfig(window_size=3)


def test_unified_scene_call_is_once_per_step_and_history_stays_bounded():
    task = TaskMemory(
        "Reach two landmarks.",
        subgoals=(
            Subgoal("1", "Reach the hall marker", "See the marker nearby."),
            Subgoal("2", "Reach the table", "See the table nearby."),
        ),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(request))
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    for index in range(20):
        memory.set_motion_evidence(translation_m=0.25, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        memory.analyze()

    assert len(captioner.calls) == 20
    assert len(memory.recent_frames()) == MAX_COMPLETION_EVIDENCE_FRAMES
    assert [frame.frame_id for frame in memory.recent_frames()] == list(
        range(21 - MAX_COMPLETION_EVIDENCE_FRAMES, 21)
    )
    assert [frame.frame_id for frame in captioner.calls[-1].frames] == list(
        range(21 - MAX_COMPLETION_EVIDENCE_FRAMES, 21)
    )
    assert memory.recent_frames()[-1].subgoal_path_length_m == pytest.approx(5.0)


def test_deferred_analysis_keeps_frame_for_next_captioner_window():
    task = TaskMemory(
        "Reach two landmarks.",
        subgoals=(
            Subgoal("1", "Reach the hall marker", "See the marker nearby."),
        ),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(request))
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    task.record_input(_frame(0))
    assert memory.update_from_task_memory(analyze=False) is None
    assert len(captioner.calls) == 0
    task.record_input(_frame(1))
    result = memory.update_from_task_memory()

    assert result is not None
    assert len(captioner.calls) == 1
    assert [frame.frame_id for frame in captioner.calls[0].frames] == [1, 2]


def test_doorway_completion_is_owned_by_captioner_crossing_judgement():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(
        request,
        completed=len(request.frames) >= 2,
        destination_dominant=len(request.frames) >= 2,
        door_state=("CROSSED" if len(request.frames) >= 2 else "APPROACHING"),
        door_camera_side=(
            "AFTER_DOOR" if len(request.frames) >= 2 else "BEFORE_DOOR"
        ),
    ))
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    memory.set_motion_evidence(translation_m=0.0, yaw_delta_deg=0.0)
    memory.append_observation(_frame(0))
    assert memory.analyze().completed is False
    memory.set_motion_evidence(translation_m=0.0, yaw_delta_deg=0.0)
    memory.append_observation(_frame(1))
    result = memory.analyze()

    assert result.completed is True
    assert task.get_current_subgoal().subgoal_id == "2"
    assert result.door_state == "CROSSED"
    assert result.door_camera_side == "AFTER_DOOR"
    assert len(captioner.calls) == 2


def test_doorway_cannot_jump_from_unseen_directly_to_crossed():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            door_state="CROSSED",
            door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    memory.append_observation(_frame(0))
    result = memory.analyze()

    assert result.completed is False
    assert task.get_current_subgoal().subgoal_id == "1"
    assert "camera has not reached" in memory.diagnostics()[
        "completion_guard"
    ]


def test_model_crossing_is_accepted_at_model_localized_doorway():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            door_state="CROSSED",
            door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    memory.set_doorway_target_distance(0.80)
    memory.append_observation(_frame(0))
    assert memory.analyze().completed is False
    memory.set_doorway_target_distance(0.30)
    memory.append_observation(_frame(1))
    result = memory.analyze()

    assert result.completed is True
    assert task.get_current_subgoal().subgoal_id == "2"
    assert memory.diagnostics()["doorway_target_distance_m"] is None


def test_final_subgoal_requires_two_stable_model_owned_at_observations():
    task = TaskMemory(
        "Walk to the pool.",
        subgoals=(
            Subgoal(
                "1",
                "Walk forward to the pool area",
                "The camera is directly beside the pool.",
            ),
        ),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            final_visible=True,
            proximity="AT",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    results = []
    for index in range(2):
        memory.set_motion_evidence(translation_m=0.25, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        results.append(memory.analyze())

    assert [result.completed for result in results] == [False, True]
    assert task.is_task_complete() is True
    assert len(captioner.calls) == 2


def test_sustained_confident_crossing_is_accepted_after_walking():
    """Four confident CROSSED judgements plus real walking release the guard.

    This is the escape hatch for a mislocalized doorway point that the
    camera can never reach, combined with a model that never reported the
    approach stages.
    """
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            door_state="CROSSED",
            door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    for index in range(3):
        memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
        assert memory.diagnostics()["doorway_crossed_streak"] == index + 1
        assert task.get_current_subgoal().subgoal_id == "1"

    memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
    memory.append_observation(_frame(3))
    result = memory.analyze()

    assert result.completed is True
    assert task.get_current_subgoal().subgoal_id == "2"
    assert "accepted sustained doorway crossing" in memory.diagnostics()[
        "completion_guard"
    ] or memory.diagnostics()["completion_guard"] is None


def test_sustained_crossing_without_walking_stays_rejected():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            door_state="CROSSED",
            door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    for index in range(6):
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False

    assert task.get_current_subgoal().subgoal_id == "1"
    assert "crossed streak 6/4" in memory.diagnostics()["completion_guard"]


def test_crossed_streak_resets_on_a_contradicting_judgement():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    states = iter(
        ["CROSSED", "CROSSED", "NOT_VISIBLE", "CROSSED", "CROSSED", "CROSSED"]
    )

    def factory(request):
        state = next(states)
        return _scene_result(
            request,
            completed=state == "CROSSED",
            destination_dominant=True,
            door_state=state,
            door_camera_side="AFTER_DOOR" if state == "CROSSED" else "UNKNOWN",
        )

    memory = TemporalMemory(captioner=SceneCaptioner(factory), task_memory=task)
    for index in range(6):
        memory.set_motion_evidence(translation_m=0.5, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        memory.analyze()

    # Streak restarted at the NOT_VISIBLE frame: 3 < 4, still held.
    assert task.get_current_subgoal().subgoal_id == "1"
    assert memory.diagnostics()["doorway_crossed_streak"] == 3


def test_far_localized_doorway_blocks_crossing_claims_until_reached():
    """A CROSSED claim while the localized doorway is still 2 m ahead is a
    measurement contradiction; once the camera has passed within reach it is
    latched and a later claim goes through even after walking on."""
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request,
            completed=True,
            destination_dominant=True,
            door_state="CROSSED",
            door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    for index, distance in enumerate((2.4, 2.0, 1.6, 1.2, 1.1)):
        memory.set_doorway_target_distance(distance)
        memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
        assert "still" in memory.diagnostics()["completion_guard"]
    assert task.get_current_subgoal().subgoal_id == "1"

    memory.set_doorway_target_distance(0.45)
    memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
    memory.append_observation(_frame(5))
    assert memory.analyze().completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def test_reached_doorway_is_latched_for_later_crossing_claims():
    task = TaskMemory(
        "Exit the room and reach the pool.",
        subgoals=_subgoals(),
    )
    states = iter(["NOT_VISIBLE", "NOT_VISIBLE", "NOT_VISIBLE", "CROSSED"])

    def factory(request):
        state = next(states)
        return _scene_result(
            request,
            completed=state == "CROSSED",
            destination_dominant=True,
            door_state=state,
            door_camera_side="AFTER_DOOR" if state == "CROSSED" else "UNKNOWN",
        )

    memory = TemporalMemory(captioner=SceneCaptioner(factory), task_memory=task)
    # First observation binds the subgoal (no doorway localized yet), then the
    # camera passes through the doorway point and walks 1.5 m beyond it
    # before the model first reports the crossing.
    for index, distance in enumerate((None, 0.40, 0.90)):
        memory.set_doorway_target_distance(distance)
        memory.set_motion_evidence(translation_m=0.5, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
    assert memory.diagnostics()["doorway_reached"] is True

    memory.set_doorway_target_distance(1.50)
    memory.set_motion_evidence(translation_m=0.5, yaw_delta_deg=0.0)
    memory.append_observation(_frame(3))
    result = memory.analyze()

    assert result.completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def test_turn_subgoal_completion_waits_for_measured_rotation():
    task = TaskMemory(
        "Turn left and go through the hallway.",
        subgoals=(
            Subgoal("1", "Turn left", "After turning left, the hallway is centred in the view."),
            Subgoal("2", "Proceed through the hallway", "The camera is moving through the hallway."),
        ),
    )
    verdicts = iter([False, True, True])
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request, completed=next(verdicts), destination_dominant=True
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()  # binds subgoal 1

    # The fake landmark is centred, so only a rotation below one primitive
    # (15 deg) is still an unfinished turn.
    memory.set_turn_progress(10.0)
    memory.append_observation(_frame(1))
    assert memory.analyze().completed is False
    assert "measured left turn is 10 deg" in memory.diagnostics()["completion_guard"]

    memory.set_turn_progress(75.0)
    memory.append_observation(_frame(2))
    assert memory.analyze().completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def test_non_doorway_completion_is_deferred_while_committed_target_is_ahead():
    task = TaskMemory(
        "Go beside the bed and stop.",
        subgoals=(
            Subgoal("1", "Go beside the bed", "The bed is beside the camera."),
            Subgoal("2", "Stop", "The camera has stopped."),
        ),
    )
    verdicts = iter([False, True, True])
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request, completed=next(verdicts), destination_dominant=True
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()  # binds the subgoal; no target yet

    memory.set_committed_target_distance(0.82)
    memory.append_observation(_frame(1))
    assert memory.analyze().completed is False
    assert "committed waypoint is still 0.82 m ahead" in memory.diagnostics()["completion_guard"]
    assert task.get_current_subgoal().subgoal_id == "1"

    memory.set_committed_target_distance(0.30)
    memory.append_observation(_frame(2))
    assert memory.analyze().completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def test_turn_subgoal_completes_on_measured_rotation_without_model_consent():
    task = TaskMemory(
        "Turn left and go through the hallway.",
        subgoals=(
            Subgoal("1", "Turn left", "After turning left, the hallway is centred in the view."),
            Subgoal("2", "Proceed through the hallway", "The camera is moving through the hallway."),
        ),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(request, completed=False))
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()

    memory.set_turn_progress(65.0)
    memory.append_observation(_frame(1))
    result = memory.analyze()

    assert result.completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def test_stairs_subgoal_completes_on_measured_rise_that_levels_off():
    task = TaskMemory(
        "Go up the stairs and enter the bedroom.",
        subgoals=(
            Subgoal("1", "Go up the stairs", "The stairs are below and behind the camera."),
            Subgoal("2", "Enter the bedroom", "The camera has crossed the bedroom threshold."),
        ),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(request, completed=False))
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()

    # Climbing: height rising, not levelled yet.
    for index, rise in enumerate((0.15, 0.30, 0.42), start=1):
        memory.set_elevation_progress(rise)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
    # Levelled off at +0.42 m for three observations.
    for index, rise in enumerate((0.42, 0.43), start=4):
        memory.set_elevation_progress(rise)
        memory.append_observation(_frame(index))
        result = memory.analyze()
    assert result.completed is True
    assert "accepted measured stairs up" in memory.diagnostics()["completion_guard"] or task.get_current_subgoal().subgoal_id == "2"
    assert task.get_current_subgoal().subgoal_id == "2"


def test_landmark_stage_completes_on_arrival_at_committed_point():
    task = TaskMemory(
        "Go beside the bed and turn left.",
        subgoals=(
            Subgoal("1", "Go beside the bed", "The bed is beside the camera."),
            Subgoal("2", "Turn left", "After turning left, the hallway is centred."),
        ),
    )
    captioner = SceneCaptioner(lambda request: _scene_result(request, completed=False))
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()

    # Localized from 3.2 m away: tolerance 0.8 m.
    for index, distance in enumerate((3.2, 2.4, 1.6, 1.0), start=1):
        memory.set_committed_target_distance(distance, reach_tolerance_m=0.8)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
    memory.set_committed_target_distance(0.7, reach_tolerance_m=0.8)
    memory.append_observation(_frame(5))
    assert memory.analyze().completed is True
    assert task.get_current_subgoal().subgoal_id == "2"


def _route_subgoals():
    return tuple(
        Subgoal(str(i), d, c)
        for i, (d, c) in enumerate(
            (
                ("Turn left", "After turning left, the stairs are centred."),
                ("Go up the stairs", "The stairs are behind the camera."),
                ("Enter the bedroom", "The camera has crossed the bedroom threshold."),
                ("Go beside the bed", "The bed is beside the camera."),
                ("Go through the hallway", "The camera has passed the hallway."),
                ("Go just inside the bathroom doorway", "The camera is just inside the bathroom doorway."),
            ),
            start=1,
        )
    )


def _walk(memory, frames, *, at, step_m=0.4, start=0):
    """Feed frames with the destination reported AT (or not)."""
    for index in range(start, start + frames):
        memory.set_motion_evidence(translation_m=step_m, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        memory.analyze()


def _at_captioner(at_from_frame):
    def factory(request):
        frame_id = request.frames[-1].frame_id
        return _scene_result(
            request,
            completed=False,
            final_visible=frame_id >= at_from_frame,
            proximity="AT",
        )
    return SceneCaptioner(factory)


def test_stuck_stage_is_skipped_when_destination_is_verified_at():
    task = TaskMemory("Route to the bathroom.", subgoals=_route_subgoals())
    # Frame ids are assigned by the memory, 1-based and consecutive.
    memory = TemporalMemory(captioner=_at_captioner(at_from_frame=26), task_memory=task)
    # Stuck on stage 4 with 3 stages remaining: not near the end, so the
    # skip needs the stall condition (20 observations) plus the AT streak.
    task._current_subgoal_index = 3
    _walk(memory, 25, at=False)                       # frames 1..25, no AT
    assert task.get_current_subgoal().subgoal_id == "4"
    _walk(memory, 2, at=True, start=25)               # frames 26,27: AT x2, not enough
    assert task.get_current_subgoal().subgoal_id == "4"
    _walk(memory, 1, at=True, start=27)               # frame 28: AT x3
    assert task.get_current_subgoal().subgoal_id == "6"
    assert memory.diagnostics()["stage_skip"]["skipped"] == ["4", "5"]
    assert "skipped stages 4, 5" in memory.diagnostics()["completion_guard"]
    # The final stage is only activated, not completed.
    assert task.is_task_complete() is False


def test_destination_look_alike_early_in_the_route_does_not_skip():
    task = TaskMemory("Route to the bathroom.", subgoals=_route_subgoals())
    memory = TemporalMemory(captioner=_at_captioner(at_from_frame=0), task_memory=task)
    # AT from the very first frame, but barely any distance walked and
    # five stages remain: neither gate opens.
    _walk(memory, 6, at=True, step_m=0.2)
    assert task.get_current_subgoal().subgoal_id == "1"
    assert memory.diagnostics()["stage_skip"] is None


def test_skip_near_the_end_needs_no_stall():
    task = TaskMemory("Route to the bathroom.", subgoals=_route_subgoals())
    memory = TemporalMemory(captioner=_at_captioner(at_from_frame=11), task_memory=task)
    task._current_subgoal_index = 4                   # stage 5: one stage remains
    _walk(memory, 10, at=False)                       # frames 1..10, 4 m walked, no AT
    _walk(memory, 3, at=True, start=10)               # frames 11,12,13: AT x3
    assert task.get_current_subgoal().subgoal_id == "6"
    assert memory.diagnostics()["stage_skip"]["skipped"] == ["5"]


def test_single_at_report_is_not_enough():
    task = TaskMemory("Route to the bathroom.", subgoals=_route_subgoals())
    states = iter([False] * 10 + [True, False, True, False, True])

    def factory(request):
        return _scene_result(request, completed=False, final_visible=next(states), proximity="AT")

    memory = TemporalMemory(captioner=SceneCaptioner(factory), task_memory=task)
    task._current_subgoal_index = 4
    _walk(memory, 15, at=None)
    assert task.get_current_subgoal().subgoal_id == "5"


def test_final_stage_accepts_at_streak_with_independent_stop_proposal():
    task = TaskMemory(
        "Walk to the pool.",
        subgoals=(Subgoal("1", "Walk forward to the pool", "The camera is beside the pool."),),
    )
    # The scene model keeps completed=false while reporting AT.
    captioner = SceneCaptioner(
        lambda request: _scene_result(request, completed=False, final_visible=True, proximity="AT")
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)

    memory.append_observation(_frame(0))
    assert memory.analyze().completed is False          # streak 1
    memory.set_stop_proposed(True)
    memory.append_observation(_frame(1))
    result = memory.analyze()                            # streak 2 + STOP proposal
    assert result.completed is True
    assert task.is_task_complete()


def test_final_stage_at_streak_of_three_suffices_without_stop_proposal():
    task = TaskMemory(
        "Walk to the pool.",
        subgoals=(Subgoal("1", "Walk forward to the pool", "The camera is beside the pool."),),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(request, completed=False, final_visible=True, proximity="AT")
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    results = []
    for index in range(3):
        memory.append_observation(_frame(index))
        results.append(memory.analyze().completed)
    assert results == [False, False, True]


def test_final_stage_near_is_never_accepted():
    task = TaskMemory(
        "Walk to the pool.",
        subgoals=(Subgoal("1", "Walk forward to the pool", "The camera is beside the pool."),),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(request, completed=False, final_visible=True, proximity="NEAR")
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.set_stop_proposed(True)
    for index in range(5):
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False


def test_landmark_still_far_in_depth_map_vetoes_completion():
    import numpy as np

    task = TaskMemory("Exit the room and reach the pool.", subgoals=_subgoals())
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request, completed=True, destination_dominant=True,
            door_state="CROSSED", door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()  # bind subgoal (rejected: no approach seen)

    # The fake landmark sits at (500, 450); the depth map says 4 m there.
    far = np.full((24, 32), 4.0, dtype=np.float32)
    memory.set_depth_observation(far)
    for index in range(1, 6):
        memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
    assert "still 4.00 m away in the depth map" in memory.diagnostics()["completion_guard"]
    assert task.get_current_subgoal().subgoal_id == "1"

    # Same claims with the landmark 0.8 m away: the streak fallback applies.
    memory.set_depth_observation(np.full((24, 32), 0.8, dtype=np.float32))
    memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
    memory.append_observation(_frame(6))
    assert memory.analyze().completed is True


def test_final_at_is_vetoed_while_destination_landmark_is_far():
    import numpy as np

    task = TaskMemory(
        "Walk to the pool.",
        subgoals=(Subgoal("1", "Walk forward to the pool", "The camera is beside the pool."),),
    )
    captioner = SceneCaptioner(
        lambda request: _scene_result(request, completed=False, final_visible=True, proximity="AT")
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.set_depth_observation(np.full((24, 32), 4.3, dtype=np.float32))
    memory.set_stop_proposed(True)
    for index in range(4):
        memory.append_observation(_frame(index))
        assert memory.analyze().completed is False
    memory.set_depth_observation(np.full((24, 32), 0.9, dtype=np.float32))
    memory.append_observation(_frame(4))
    assert memory.analyze().completed is True


def test_range_veto_is_lifted_once_committed_point_was_reached():
    import numpy as np

    task = TaskMemory("Exit the room and reach the pool.", subgoals=_subgoals())
    captioner = SceneCaptioner(
        lambda request: _scene_result(
            request, completed=True, destination_dominant=True,
            door_state="CROSSED", door_camera_side="AFTER_DOOR",
        )
    )
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()
    memory.set_depth_observation(np.full((24, 32), 3.0, dtype=np.float32))
    # Landmark far, but the camera passed the committed doorway point.
    memory.set_committed_target_distance(0.3)
    memory.set_motion_evidence(translation_m=0.4, yaw_delta_deg=0.0)
    memory.append_observation(_frame(1))
    assert memory.analyze().completed is True


def test_turn_stage_completes_when_landmark_is_centred_after_a_short_turn():
    task = TaskMemory(
        "Turn right to the sofa and go to it.",
        subgoals=(
            Subgoal("1", "Turn right toward the sofa", "After turning right, the sofa is centred in the view."),
            Subgoal("2", "Walk to the sofa", "The sofa is directly ahead within a step."),
        ),
    )
    # _scene_result reports the landmark at u=500 (centred), completed=False.
    captioner = SceneCaptioner(lambda request: _scene_result(request, completed=False))
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    memory.append_observation(_frame(0))
    memory.analyze()

    memory.set_turn_progress(0.0)                 # centred but not turned yet
    memory.append_observation(_frame(1))
    assert memory.analyze().completed is False
    memory.set_turn_progress(30.0)                # one or two primitives later
    memory.append_observation(_frame(2))
    assert memory.analyze().completed is True
    assert task.get_current_subgoal().subgoal_id == "2"
