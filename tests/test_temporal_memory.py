from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from agentflow.agents.models_embodied_v2.TemporalCaptioner import CameraMotion
from agentflow.agents.models_embodied_v2.memory import (
    CumulativeErrorMode,
    CumulativeErrorPhase,
    StepExecution,
    TemporalMemory,
    TemporalMemoryConfig,
    TemporalObservation,
    TemporalStateError,
)


np = pytest.importorskip("numpy")


def _image(value: int):
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    image[:, :8, 0] = (value + 25) % 255
    return image


def _observation(
    index: int,
    *,
    episode_id: str = "episode",
    position=None,
    yaw=None,
    distance=None,
    landmarks=None,
    timestamp=None,
):
    return TemporalObservation(
        image=_image(index * 11),
        episode_id=episode_id,
        timestamp_seconds=float(index if timestamp is None else timestamp),
        position_xyz=position,
        yaw_degrees=yaw,
        distance_to_goal_meters=distance,
        landmark_ids=landmarks,
    )


@dataclass
class _Record:
    label: str
    raw_response: str = field(
        default="raw-model-response",
        init=False,
    )
    model_latency_ms: float = field(default=10.0, init=False)
    latency_budget_met: bool = field(default=True, init=False)

    def to_memory_text(self):
        return self.label

    def to_memory_dict(self):
        return {"label": self.label}


@dataclass
class _StepCaption:
    step_id: int
    visible_landmarks: tuple[str, ...]


@dataclass
class _LandmarkRecord(_Record):
    step_captions: tuple[_StepCaption, ...]


class _FakeCaptioner:
    def __init__(self):
        self.requests = []
        self.fail = False
        self.last_raw_response = None

    def analyze_steps(self, request):
        self.requests.append(request)
        if self.fail:
            self.last_raw_response = "malformed-model-output"
            raise RuntimeError("model unavailable")
        self.last_raw_response = "raw-model-response"
        return _Record(f"record-{len(self.requests)}")


class _LandmarkCaptioner(_FakeCaptioner):
    def analyze_steps(self, request):
        self.requests.append(request)
        return _LandmarkRecord(
            f"record-{len(self.requests)}",
            tuple(
                _StepCaption(step.step_id, ("同一扇门",))
                for step in request.steps
            ),
        )


def _complete(
    memory: TemporalMemory,
    step_id: int,
    pre: TemporalObservation,
    post: TemporalObservation,
    action: str = "FORWARD",
    *,
    collision=None,
    before_status="1. Subtask: navigate | Completion status: IN PROGRESS",
    after_status="1. Subtask: navigate | Completion status: IN PROGRESS",
):
    memory.stage_action(pre, action, before_status)
    return memory.complete_pending_step(
        post,
        StepExecution(
            step_id=step_id,
            commanded_action=action,
            collision=collision,
        ),
        after_status,
    )


def test_transition_uses_first_image_after_action():
    memory = TemporalMemory("goal", episode_id="episode")
    pre = _observation(0, position=(0.0, 0.0, 0.0), yaw=0)
    post = _observation(1, position=(0.1, 0.0, 0.0), yaw=0)

    step = _complete(memory, 1, pre, post)

    assert step.pre_observation is pre
    assert step.post_observation is post
    assert step.commanded_action == "FORWARD"
    assert step.observed_action == "FORWARD"
    assert step.action_match == "MATCH"
    assert memory.recent_actions() == ("FORWARD",)


def test_pending_action_must_be_completed_before_next_stage():
    memory = TemporalMemory("goal", episode_id="episode")
    memory.stage_action(_observation(0), "LEFT", "")

    assert memory.pending_step_id == 1
    assert memory.pending_selected_action == "TURN_LEFT"
    inferred = memory.infer_pending_execution()
    assert inferred == StepExecution(1, "TURN_LEFT")
    assert memory.recent_actions() == ("TURN_LEFT",)
    with pytest.raises(TemporalStateError, match="pending"):
        memory.stage_action(_observation(1), "FORWARD", "")


@pytest.mark.parametrize("invalid_step_id", [True, 1.0, 1.9, "1"])
def test_step_execution_rejects_non_integer_step_ids(invalid_step_id):
    with pytest.raises(ValueError, match="integer"):
        StepExecution(invalid_step_id, "FORWARD")


def test_terminal_discard_still_validates_episode_and_step_id():
    memory = TemporalMemory("goal", episode_id="episode-a")
    pre = _observation(1, episode_id="episode-a", timestamp=1.0)
    memory.stage_action(pre, "FORWARD", "")

    with pytest.raises(TemporalStateError, match="active episode"):
        memory.finish_episode(
            _observation(1, episode_id="episode-b", timestamp=1.0),
            StepExecution(1, "FORWARD", terminal=True),
            "",
        )
    assert memory.pending_step_id == 1

    with pytest.raises(
        TemporalStateError,
        match="consecutive step_id 1",
    ):
        memory.finish_episode(
            _observation(1, episode_id="episode-a", timestamp=1.0),
            StepExecution(2, "FORWARD", terminal=True),
            "",
        )
    assert memory.pending_step_id == 1


def test_four_observations_create_first_three_step_model_request():
    captioner = _FakeCaptioner()
    memory = TemporalMemory(
        "find the kitchen",
        episode_id="episode",
        captioner=captioner,
    )
    observations = [
        _observation(
            index,
            position=(index * 0.1, 0.0, 0.0),
            yaw=0,
            distance=10.0 - index * 0.1,
            landmarks=(),
        )
        for index in range(4)
    ]

    for index in range(3):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )
        result = memory.analyze_if_ready()
        if index < 2:
            assert result is None

    assert result == _Record("record-1")
    assert len(captioner.requests) == 1
    request = captioner.requests[0]
    assert len(request.steps) == 3
    assert [step.step_id for step in request.steps] == [1, 2, 3]
    assert all(
        step.image is observations[index + 1].image
        for index, step in enumerate(request.steps)
    )
    assert "最近走的三步发生了什么" in request.notes[0]

    diagnostics = memory.diagnostics(include_raw_response=True)
    assert diagnostics["completed_step_ids"] == [1, 2, 3]
    assert diagnostics["last_analyzed_step_id"] == 3
    assert diagnostics["latest_analysis"]["label"] == "record-1"
    assert (
        diagnostics["latest_analysis"]["raw_response"]
        == "raw-model-response"
    )
    assert (
        diagnostics["timing"]["temporal_memory"]["inference_count"]
        == 1
    )
    assert (
        diagnostics["timing"]["video_understanding"][
            "average_inference_ms"
        ]
        == 10.0
    )
    json.dumps(diagnostics)


def test_three_step_window_analyzes_only_at_default_stride_boundaries():
    captioner = _FakeCaptioner()
    memory = TemporalMemory("goal", episode_id="episode", captioner=captioner)
    observations = [
        _observation(
            index,
            position=(index * 0.1, 0.0, 0.0),
            yaw=0,
            distance=20 - index * 0.1,
            landmarks=(),
        )
        for index in range(9)
    ]

    request_counts = []
    for index in range(8):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )
        memory.analyze_if_ready()
        request_counts.append(len(captioner.requests))

    assert request_counts == [0, 0, 1, 1, 1, 2, 2, 2]
    assert len(captioner.requests) == 2
    assert [
        step.step_id for step in captioner.requests[0].steps
    ] == [1, 2, 3]
    assert [
        step.step_id for step in captioner.requests[1].steps
    ] == [4, 5, 6]
    assert [step.step_id for step in memory.recent_steps()] == [6, 7, 8]

    memory.stage_action(observations[8], "STOP", "")
    assert len(memory.recent_actions()) == 3
    assert memory.recent_actions()[-1] == "STOP"


def test_model_failure_keeps_previous_record_and_rule_state():
    captioner = _FakeCaptioner()
    memory = TemporalMemory("goal", episode_id="episode", captioner=captioner)
    observations = [
        _observation(
            index,
            position=(index * 0.1, 0.0, 0.0),
            yaw=0,
            distance=20 - index * 0.1,
            landmarks=(),
        )
        for index in range(7)
    ]
    for index in range(3):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )
        first = memory.analyze_if_ready()
    assert first == _Record("record-1")

    captioner.fail = True
    for index in range(3, 6):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )
        second = memory.analyze_if_ready()

    assert second is first
    assert memory.latest_record is first
    assert "model unavailable" in memory.last_analysis_error
    assert memory.latest_rule_status.action_execution_mismatch == "ABSENT"
    assert "record-1" in memory.context()
    timing = memory.timing_summary()
    assert timing["temporal_memory"]["inference_count"] == 2
    assert timing["temporal_memory"]["success_count"] == 1
    assert timing["temporal_memory"]["failure_count"] == 1
    diagnostics = memory.diagnostics(include_raw_response=True)
    assert (
        diagnostics["last_failed_raw_response"]
        == "malformed-model-output"
    )


def test_model_landmarks_are_stored_as_lightweight_episode_evidence():
    captioner = _LandmarkCaptioner()
    memory = TemporalMemory("goal", episode_id="episode", captioner=captioner)
    tracker = "1. Subtask: navigate | Completion status: IN PROGRESS"
    observations = [
        _observation(
            index,
            position=(0, 0, 0),
            yaw=0,
            distance=10,
            # R2R supplies no structured landmark IDs.
            landmarks=None,
        )
        for index in range(7)
    ]
    for index in range(3):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            "STOP",
            before_status=tracker,
            after_status=tracker,
        )
    memory.analyze_if_ready()

    assert memory.known_landmarks == ("同一扇门",)
    assert all(
        step.post_observation.landmark_ids == ("同一扇门",)
        for step in memory.recent_steps()
    )
    assert sum(
        len(step.newly_discovered_landmarks)
        for step in memory.recent_steps()
    ) == 1

    for step_id in range(4, 7):
        _complete(
            memory,
            step_id,
            observations[step_id - 1],
            observations[step_id],
            "STOP",
            before_status=tracker,
            after_status=tracker,
        )
        memory.analyze_if_ready()

    # In the second rolling window the same model landmark is no longer a new
    # discovery, allowing the three-step no-progress rule to use it as known
    # negative evidence instead of silently treating missing R2R metadata as 0.
    assert all(
        step.post_observation.landmark_ids == ("同一扇门",)
        for step in memory.recent_steps()
    )
    assert memory.latest_rule_status.get_nowhere == "PRESENT"


def test_environment_pose_has_priority_for_motion_and_mismatch():
    memory = TemporalMemory("goal", episode_id="episode")
    step = _complete(
        memory,
        1,
        _observation(0, position=(0, 0, 0), yaw=0),
        _observation(1, position=(0, 0, 0), yaw=15),
        "FORWARD",
    )

    assert step.motion.source == "environment_odometry"
    assert step.motion.camera_motion == CameraMotion.TURN_LEFT
    assert step.action_match == "MISMATCH"
    assert memory.latest_rule_status.action_execution_mismatch == "PRESENT"


def test_missing_motion_metadata_stays_unknown_when_flow_is_unusable():
    memory = TemporalMemory("goal", episode_id="episode")
    pre = TemporalObservation("not-an-image", "episode", 0.0)
    post = TemporalObservation("also-not-an-image", "episode", 1.0)

    step = _complete(memory, 1, pre, post)

    assert step.observed_action == "UNKNOWN"
    assert step.action_match == "UNCERTAIN"
    assert step.motion.source == "optical_flow_unavailable"


def test_collision_sensor_and_action_mismatch_are_preserved():
    memory = TemporalMemory("goal", episode_id="episode")
    step = _complete(
        memory,
        1,
        _observation(0, position=(0, 0, 0), yaw=0),
        _observation(1, position=(0, 0, 0), yaw=0),
        "FORWARD",
        collision=True,
    )

    assert step.motion.collision is True
    assert step.action_match == "MISMATCH"
    assert memory.latest_rule_status.collision == "PRESENT"


def test_revisit_requires_leaving_gap_and_visual_similarity():
    memory = TemporalMemory("goal", episode_id="episode")
    frames = [
        TemporalObservation(_image(40), "episode", 0, (0, 0, 0), 0, 5),
        TemporalObservation(_image(40), "episode", 1, (0, 0, 0), 0, 5),
        TemporalObservation(_image(80), "episode", 2, (2, 0, 0), 0, 5),
        TemporalObservation(_image(90), "episode", 3, (4, 0, 0), 0, 5),
        TemporalObservation(_image(40), "episode", 4, (0.1, 0, 0), 0, 5),
    ]
    first = _complete(memory, 1, frames[0], frames[1], "STOP")
    _complete(memory, 2, frames[1], frames[2])
    _complete(memory, 3, frames[2], frames[3])
    returned = _complete(memory, 4, frames[3], frames[4])

    assert first.is_new_node is False
    assert returned.topology_node_id == first.topology_node_id
    assert returned.is_revisit is True
    assert memory.latest_rule_status.repeated_visit == "PRESENT"


def test_revisit_is_unknown_when_required_visual_evidence_is_missing():
    memory = TemporalMemory("goal", episode_id="episode")
    positions = [(0, 0, 0), (0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 0, 0)]
    observations = [
        TemporalObservation(
            f"invalid-image-{index}",
            "episode",
            float(index),
            position,
            0,
            5,
        )
        for index, position in enumerate(positions)
    ]
    for index in range(4):
        returned = _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            "STOP",
        )

    assert returned.is_revisit is None
    assert memory.latest_rule_status.repeated_visit == "UNCERTAIN"

def test_image_only_calls_build_visual_topology_inside_memory():
    memory = TemporalMemory("goal", episode_id="episode")
    rng = np.random.default_rng(7)
    first_view = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    second_view = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    third_view = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    images = [
        first_view,
        first_view.copy(),
        second_view,
        third_view,
        first_view.copy(),
    ]
    observations = [
        TemporalObservation(
            image=image,
            episode_id="episode",
            timestamp_seconds=float(index),
        )
        for index, image in enumerate(images)
    ]

    steps = [
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            "STOP",
        )
        for index in range(4)
    ]

    assert steps[0].topology_node_id == "visual-node-0000"
    assert steps[0].is_new_node is False
    assert steps[1].is_new_node is True
    assert steps[2].is_new_node is True
    assert steps[3].topology_node_id == "visual-node-0000"
    assert steps[3].is_revisit is True
    assert memory.latest_rule_status.repeated_visit == "PRESENT"


def test_get_nowhere_requires_complete_negative_progress_coverage():
    memory = TemporalMemory("goal", episode_id="episode")
    tracker = "1. Subtask: navigate | Completion status: IN PROGRESS"
    observations = [
        TemporalObservation(
            _image(30),
            "episode",
            float(index),
            (0, 0, 0),
            0,
            10,
            (),
        )
        for index in range(5)
    ]
    # Establish the initial node outside the final three-step window.
    _complete(
        memory,
        1,
        observations[0],
        observations[1],
        "STOP",
        before_status=tracker,
        after_status=tracker,
    )
    for step_id in range(2, 5):
        _complete(
            memory,
            step_id,
            observations[step_id - 1],
            observations[step_id],
            "STOP",
            before_status=tracker,
            after_status=tracker,
        )

    assert len(memory.recent_steps()) == 3
    assert all(step.is_new_node is False for step in memory.recent_steps())
    assert memory.latest_rule_status.get_nowhere == "PRESENT"
    assert memory.latest_rule_status.overall_progress == "STALLED"


def test_missing_landmark_or_goal_signals_do_not_become_false_zeroes():
    memory = TemporalMemory("goal", episode_id="episode")
    for index in range(3):
        _complete(
            memory,
            index + 1,
            _observation(index, position=(0, 0, 0), yaw=0),
            _observation(index + 1, position=(0, 0, 0), yaw=0),
            "STOP",
        )

    progress = memory._build_progress_signals(memory.recent_steps())
    assert progress.new_landmarks_count is None
    assert progress.no_progress_steps is None
    assert memory.latest_rule_status.get_nowhere == "UNCERTAIN"


def test_two_turn_reversals_and_visual_retrace_flag_oscillation():
    memory = TemporalMemory("goal", episode_id="episode")
    yaws = [0, 15, 0, 15]
    actions = [
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
    ]
    observations = [
        TemporalObservation(
            _image(55),
            "episode",
            float(index),
            (0, 0, 0),
            yaws[index],
            10,
            (),
        )
        for index in range(4)
    ]
    for index, action in enumerate(actions):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            action,
        )

    assert [step.step_id for step in memory.recent_steps()] == [1, 2, 3]
    assert memory.latest_rule_status.motion_oscillation == "PRESENT"


def test_finish_episode_allows_same_timestamp_only_for_stop():
    memory = TemporalMemory("goal", episode_id="episode")
    observation = _observation(0, position=(0, 0, 0), yaw=0)
    memory.stage_action(observation, "STOP", "")

    result = memory.finish_episode(
        observation,
        StepExecution(1, "STOP", terminal=True),
        "",
    )

    assert result is None
    assert memory.pending_selected_action is None
    assert len(memory.recent_steps()) == 1
    assert (
        memory.recent_steps()[0].post_observation.timestamp_seconds
        > observation.timestamp_seconds
    )

    memory.stage_action(
        memory.recent_steps()[0].post_observation,
        "FORWARD",
        "",
    )
    memory.finish_episode(
        memory.recent_steps()[0].post_observation,
        StepExecution(2, "FORWARD", terminal=True),
        "",
    )
    assert memory.pending_selected_action is None
    assert len(memory.recent_steps()) == 1
    assert "discarded" in memory.last_analysis_error


def test_reset_clears_episode_state_but_keeps_captioner():
    captioner = _FakeCaptioner()
    memory = TemporalMemory("old", episode_id="old", captioner=captioner)
    memory.stage_action(_observation(0, episode_id="old"), "FORWARD", "")

    memory.reset(episode_id="new", goal="new goal")

    assert memory.captioner is captioner
    assert memory.goal == "new goal"
    assert memory.recent_steps() == ()
    assert memory.pending_step_id is None
    assert memory.latest_record is None
    with pytest.raises(TemporalStateError, match="active episode"):
        memory.stage_action(_observation(1, episode_id="old"), "FORWARD", "")


def test_legacy_window_contains_only_post_action_frames():
    memory = TemporalMemory("goal", episode_id="episode")
    observations = [
        _observation(
            index,
            position=(index * 0.1, 0, 0),
            yaw=0,
            distance=10 - index * 0.1,
            landmarks=(),
        )
        for index in range(4)
    ]
    for index in range(3):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )

    window = memory._build_legacy_window(memory.recent_steps())

    assert len(window.frames) == 3
    assert [frame.step_id for frame in window.frames] == [1, 2, 3]
    assert all(
        frame.image is observations[index + 1].image
        for index, frame in enumerate(window.frames)
    )
    assert [action.step_id for action in window.actions] == [1, 2, 3]


def test_eight_step_window_remains_available_with_explicit_config():
    captioner = _FakeCaptioner()
    memory = TemporalMemory(
        "goal",
        episode_id="episode",
        captioner=captioner,
        config=TemporalMemoryConfig(
            window_size=8,
            analysis_stride=1,
            get_nowhere_steps=8,
            oscillation_retrace_min_step_gap=2,
        ),
    )
    observations = [
        _observation(
            index,
            position=(index * 0.1, 0, 0),
            yaw=0,
            distance=10 - index * 0.1,
            landmarks=(),
        )
        for index in range(9)
    ]

    for index in range(8):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
        )
        result = memory.analyze_if_ready()

    assert result == _Record("record-1")
    assert len(captioner.requests) == 1
    assert [
        step.step_id for step in captioner.requests[0].steps
    ] == list(range(1, 9))


def test_cumulative_wall_stuck_requires_two_three_step_windows():
    memory = TemporalMemory("goal", episode_id="episode")
    image = _image(44)
    observations = [
        TemporalObservation(image.copy(), "episode", float(index))
        for index in range(7)
    ]

    for index in range(3):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            "FORWARD",
        )

    assert memory.cumulative_error_state.mode == CumulativeErrorMode.WALL_STUCK
    assert (
        memory.cumulative_error_state.phase
        == CumulativeErrorPhase.SUSPECTED
    )
    assert memory.pending_go_back_request is None

    for index in range(3, 6):
        _complete(
            memory,
            index + 1,
            observations[index],
            observations[index + 1],
            "FORWARD",
        )

    assert (
        memory.cumulative_error_state.phase
        == CumulativeErrorPhase.CONFIRMED
    )
    assert memory.pending_go_back_request is not None
    assert memory.latest_rule_status.overall_progress == "STALLED"
    assert all(
        not hasattr(item, "image") for item in memory.recent_evidence()
    )


def test_recovery_action_ack_and_final_post_frame_exit_recovering():
    memory = TemporalMemory("goal", episode_id="episode")
    image = _image(44)
    current = TemporalObservation(image.copy(), "episode", 0.0)
    for step_id in range(1, 7):
        post = TemporalObservation(
            image.copy(),
            "episode",
            float(step_id),
        )
        _complete(memory, step_id, current, post, "FORWARD")
        current = post

    request = memory.begin_go_back_recovery()
    assert request is not None
    assert memory.cumulative_error_state.phase == CumulativeErrorPhase.RECOVERING

    step_id = 7
    while (action := memory.next_recovery_action()) is not None:
        post = TemporalObservation(
            image.copy(),
            "episode",
            float(step_id),
        )
        memory.stage_action(current, action, "")
        # Peek is stable until successful staging is acknowledged.
        assert memory.next_recovery_action() == action
        memory.ack_recovery_action(action)
        memory.complete_pending_step(
            post,
            StepExecution(step_id, action),
            "",
        )
        current = post
        step_id += 1

    assert memory.active_go_back_request is None
    assert memory.cumulative_error_state.phase == CumulativeErrorPhase.COOLDOWN
    assert "recovery failed" in memory.cumulative_error_state.reason
    assert memory.next_recovery_action() is None


def test_cumulative_turn_oscillation_and_in_place_spin():
    image = _image(66)

    oscillation = TemporalMemory("goal", episode_id="episode")
    oscillation_actions = ("TURN_LEFT", "TURN_RIGHT") * 4
    oscillation_yaws = (0, 15, 0, 15, 0, 15, 0, 15, 0)
    observations = [
        TemporalObservation(
            image.copy(),
            "episode",
            float(index),
            (0, 0, 0),
            yaw,
        )
        for index, yaw in enumerate(oscillation_yaws)
    ]
    for index, action in enumerate(oscillation_actions):
        _complete(
            oscillation,
            index + 1,
            observations[index],
            observations[index + 1],
            action,
        )
    assert (
        oscillation.cumulative_error_state.mode
        == CumulativeErrorMode.TURN_OSCILLATION
    )
    assert (
        oscillation.cumulative_error_state.phase
        == CumulativeErrorPhase.CONFIRMED
    )

    spin = TemporalMemory("goal", episode_id="episode")
    observations = [
        TemporalObservation(
            image.copy(),
            "episode",
            float(index),
            (0, 0, 0),
            index * 15,
        )
        for index in range(21)
    ]
    for index in range(20):
        _complete(
            spin,
            index + 1,
            observations[index],
            observations[index + 1],
            "TURN_LEFT",
        )
    assert spin.cumulative_error_state.mode == CumulativeErrorMode.IN_PLACE_SPIN
    assert spin.cumulative_error_state.phase == CumulativeErrorPhase.CONFIRMED
