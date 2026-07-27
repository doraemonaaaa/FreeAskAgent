from __future__ import annotations

from collections import deque

import pytest

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
    CaptionResult,
    StepUnderstanding,
    Subgoal,
    SubgoalStatus,
)
from agentflow.agents.models_embodied_v2.memory.temporal_memory import (
    TemporalEventKind,
    TemporalMemory,
    TemporalMemoryConfig,
    TemporalStateError,
)


np = pytest.importorskip("numpy")


class FakeTaskMemory:
    def __init__(self):
        self.task = "Exit into the pool room and stop before the pool."
        self.guidance = "Use post-action visual evidence only."
        self.subgoal = Subgoal(
            "1",
            "Exit the room",
            "Cross the doorway threshold.",
        )
        self.events = []

    def get_task(self):
        return self.task

    def get_task_guidance(self):
        return self.guidance

    def get_current_subgoal(self):
        return self.subgoal

    def publish_temporal_event(self, kind, value):
        self.events.append((kind, value))


class FakeCaptioner:
    def __init__(self, *outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def analyze(self, request):
        self.calls.append(request)
        if not self.outputs:
            raise AssertionError("No fake CaptionResult remains")
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        return output


def _result(
    step_ids=range(1, 9),
    *,
    subgoal_id="1",
    completed=False,
    persistent_error=False,
    error_mode="NONE",
    error_ids=(),
):
    ids = list(step_ids)
    return CaptionResult(
        steps=[
            StepUnderstanding(
                step_id=step_id,
                caption=f"scene {step_id}",
                visual_change="FORWARD",
                error_clue=None,
            )
            for step_id in ids
        ],
        subgoals=[
            SubgoalStatus(
                subgoal_id=subgoal_id,
                completed=completed,
                evidence="visible evidence",
                evidence_step_ids=[ids[-1]],
            )
        ],
        persistent_error=persistent_error,
        error_mode=error_mode,
        error_evidence=(
            "persistent visual pattern" if persistent_error else "none"
        ),
        error_evidence_step_ids=list(error_ids),
        confidence=0.8,
        raw_response="{}",
        latency_ms=10.0,
    )


def _memory(*outputs):
    port = FakeTaskMemory()
    captioner = FakeCaptioner(*outputs)
    memory = TemporalMemory(captioner=captioner, task_memory=port)
    memory.reset()
    return memory, captioner, port


def _push(memory, count=8, *, start=0):
    actions = ("FORWARD", "TURN_LEFT", "TURN_RIGHT")
    for offset in range(count):
        index = start + offset
        memory.append_step(
            actions[index % len(actions)],
            np.full((4, 5, 3), index, dtype=np.uint8),
            float(index),
        )


def test_eight_step_request_comes_from_task_memory_and_stores_understanding():
    result = _result()
    memory, captioner, port = _memory(result)
    mutable_image = np.zeros((4, 5, 3), dtype=np.uint8)

    memory.append_step("FORWARD", mutable_image, 0.0)
    mutable_image[:] = 255
    _push(memory, 7, start=1)
    returned = memory.analyze_if_ready()

    assert returned is result
    assert len(captioner.calls) == 1
    request = captioner.calls[0]
    assert request.task == port.task
    assert request.task_guidance == port.guidance
    assert request.subgoals == (port.subgoal,)
    assert [step.step_id for step in request.steps] == list(range(1, 9))
    assert [step.action for step in request.steps] == list(
        memory.recent_actions()
    )
    assert int(np.asarray(memory.recent_steps()[0].post_image).max()) == 0
    understandings = memory.recent_understandings()
    assert [item.step_id for item in understandings] == list(range(1, 9))
    assert understandings[-1].caption == "scene 8"


def test_ninth_step_evicts_first_and_produces_next_chronological_window():
    memory, captioner, _ = _memory(
        _result(range(1, 9)),
        _result(range(2, 10)),
    )
    _push(memory)
    assert memory.analyze_if_ready() is not None

    memory.append_step("FORWARD", np.zeros((2, 2, 3), dtype=np.uint8), 8.0)
    assert memory.analyze_if_ready() is not None

    assert [step.step_id for step in memory.recent_steps()] == list(
        range(2, 10)
    )
    assert [
        step.step_id for step in captioner.calls[-1].steps
    ] == list(range(2, 10))


def test_incomplete_window_reports_current_subgoal_is_still_continuing():
    memory, _, port = _memory(
        _result(range(1, 9)),
        _result(range(2, 10)),
    )
    _push(memory)

    memory.analyze_if_ready()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]

    memory.append_step(
        "FORWARD",
        np.zeros((2, 2, 3), dtype=np.uint8),
        8.0,
    )
    memory.analyze_if_ready()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]
    assert port.events == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
    ]


def test_subgoal_completion_publishes_one_boolean_event_per_subgoal():
    memory, _, port = _memory(
        _result(completed=True),
        _result(completed=True),
        _result(range(9, 17), subgoal_id="2", completed=True),
    )
    _push(memory)

    memory.analyze()
    events = memory.drain_events()
    assert [(event.kind, event.value) for event in events] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, True)
    ]
    assert type(events[0].value) is bool
    assert port.events == [(TemporalEventKind.SUBGOAL_COMPLETED, True)]

    memory.analyze()
    assert memory.drain_events() == ()

    port.subgoal = Subgoal(
        "2",
        "Stop before the pool",
        "Reach the pool edge and stop.",
    )
    memory.append_step("STOP", np.zeros((2, 2, 3), dtype=np.uint8), 8.0)
    _push(memory, 7, start=9)
    memory.analyze()
    assert [event.kind for event in memory.drain_events()] == [
        TemporalEventKind.SUBGOAL_COMPLETED
    ]
    assert port.events[-1] == (TemporalEventKind.SUBGOAL_COMPLETED, True)


def test_go_back_requires_sustained_evidence_and_is_edge_triggered():
    memory, _, port = _memory(
        _result(
            persistent_error=True,
            error_mode="WALL_STUCK",
            error_ids=(1, 2),
        ),
        _result(
            persistent_error=True,
            error_mode="WALL_STUCK",
            error_ids=(1, 2, 3),
        ),
        _result(
            persistent_error=True,
            error_mode="WALL_STUCK",
            error_ids=(2, 3, 4),
        ),
        _result(),
        _result(
            persistent_error=True,
            error_mode="TURN_OSCILLATION",
            error_ids=(3, 4, 5),
        ),
    )
    _push(memory)

    memory.analyze()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]
    memory.analyze()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.GO_BACK_TO_ACTION, True),
    ]
    memory.analyze()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]
    memory.analyze()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]
    memory.analyze()
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.GO_BACK_TO_ACTION, True),
    ]
    assert port.events == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.GO_BACK_TO_ACTION, True),
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.SUBGOAL_COMPLETED, False),
        (TemporalEventKind.GO_BACK_TO_ACTION, True),
    ]


def test_completed_subgoal_suppresses_static_arrival_go_back_event():
    memory, _, _ = _memory(
        _result(
            completed=True,
            persistent_error=True,
            error_mode="GET_NOWHERE",
            error_ids=(5, 6, 7, 8),
        )
    )
    _push(memory)

    memory.analyze()

    assert [event.kind for event in memory.drain_events()] == [
        TemporalEventKind.SUBGOAL_COMPLETED
    ]


def test_model_failure_keeps_history_and_does_not_add_a_status_event():
    valid = _result()
    memory, _, port = _memory(valid, RuntimeError("model unavailable"))
    _push(memory)
    assert memory.analyze_if_ready() is valid
    assert [(event.kind, event.value) for event in memory.drain_events()] == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]

    memory.append_step(
        "FORWARD",
        np.zeros((2, 2, 3), dtype=np.uint8),
        8.0,
    )
    assert memory.analyze_if_ready() is None

    assert memory.latest_result is valid
    assert [step.step_id for step in memory.recent_steps()] == list(
        range(2, 10)
    )
    assert "model unavailable" in memory.last_analysis_error
    assert memory.drain_events() == ()
    assert port.events == [
        (TemporalEventKind.SUBGOAL_COMPLETED, False)
    ]


def test_subgoal_switch_starts_a_fresh_eight_step_window():
    memory, _, port = _memory()
    _push(memory, 4)
    port.subgoal = Subgoal(
        "2",
        "Stop before the pool",
        "Reach the pool edge and stop.",
    )
    _push(memory, 4, start=4)

    assert [step.subgoal_id for step in memory.recent_steps()] == [
        "2",
        "2",
        "2",
        "2",
    ]
    assert [step.step_id for step in memory.recent_steps()] == [5, 6, 7, 8]
    assert memory.current_subgoal.subgoal_id == "2"
    assert memory.analyze_if_ready() is None

    _push(memory, 4, start=8)
    assert [step.step_id for step in memory.recent_steps()] == list(
        range(5, 13)
    )


def test_not_ready_reset_and_input_invariants():
    memory, captioner, _ = _memory(_result(completed=True))
    _push(memory, 7)
    assert memory.analyze_if_ready() is None
    assert captioner.calls == []
    with pytest.raises(TemporalStateError, match="Unsupported action"):
        memory.append_step("GO_BACK", _image := np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="fixed eight-step"):
        TemporalMemoryConfig(window_size=3)

    memory.append_step("FORWARD", np.zeros((2, 2, 3)), 7.0)
    memory.analyze()
    assert memory.latest_result is not None
    memory.reset()

    assert memory.recent_steps() == ()
    assert memory.latest_result is None
    assert memory.last_analysis_error is None
    assert memory.drain_events() == ()
    assert memory.captioner is captioner
    assert _image.shape == (2, 2, 3)
