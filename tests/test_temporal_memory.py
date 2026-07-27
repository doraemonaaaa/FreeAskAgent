from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from agentflow.agents.models_embodied_v2.TemporalCaptioner import (
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


def _result(*, subgoal_id="1", completed=False):
    return CaptionResult(
        subgoal_id=subgoal_id,
        completed=completed,
        raw_response=str(completed).lower(),
        latency_ms=10.0,
    )


def _subgoals():
    return (
        Subgoal("1", "Exit the room", "Cross the doorway threshold."),
        Subgoal("2", "Stop before the pool", "Reach the pool edge."),
    )


def _memory(*outputs):
    task = TaskMemory(
        "Exit into the pool room and stop before the pool.",
        subgoals=_subgoals(),
    )
    captioner = FakeCaptioner(*outputs)
    memory = TemporalMemory(captioner=captioner, task_memory=task)
    return memory, captioner, task


def _frame(index):
    return np.random.default_rng(index).integers(
        0, 256, (24, 32, 3), dtype=np.uint8
    )


def _push(memory, count=8, *, start=0):
    actions = ("FORWARD", "TURN_LEFT", "TURN_RIGHT")
    for index in range(start, start + count):
        memory.append_step(actions[index % 3], _frame(index))


def test_eight_action_image_pairs_are_stored_and_analyzed():
    memory, captioner, _ = _memory(_result())
    mutable = np.zeros((24, 32, 3), dtype=np.uint8)
    memory.append_step("FORWARD", mutable)
    mutable[:] = 255
    _push(memory, 7, start=1)

    assert memory.analyze_if_ready() is not None
    request = captioner.calls[0]
    assert request.subgoal == _subgoals()[0]
    assert [step.step_id for step in request.steps] == list(range(1, 9))
    assert [step.action for step in request.steps] == list(
        memory.recent_actions()
    )
    assert int(np.asarray(memory.recent_steps()[0].post_image).max()) == 0


def test_incomplete_event_keeps_current_subgoal():
    memory, _, task = _memory(_result())
    _push(memory)
    memory.analyze_if_ready()

    events = memory.drain_events()
    assert len(events) == 1
    assert events[0].kind is TemporalEventKind.SUBGOAL_COMPLETED
    assert events[0].value is False
    assert events[0].to_dict() == {
        "kind": "SUBGOAL_COMPLETED",
        "value": False,
        "subgoal_id": "1",
        "error_mode": "NONE",
    }
    assert task.get_current_subgoal().subgoal_id == "1"
    assert task.temporal_events[-1] == events[0].to_dict()


def test_completion_advances_subgoal_and_clears_old_images():
    memory, _, task = _memory(_result(completed=True))
    _push(memory)
    memory.analyze_if_ready()

    assert memory.drain_events()[0].value is True
    assert task.get_current_subgoal().subgoal_id == "2"

    memory.append_step("FORWARD", _frame(9))
    assert [step.step_id for step in memory.recent_steps()] == [9]
    assert memory.recent_steps()[0].subgoal_id == "2"


def test_rule_based_error_modes_publish_go_back():
    same = np.zeros((24, 32, 3), dtype=np.uint8)
    alternating = [
        np.full((24, 32, 3), value, dtype=np.uint8)
        for value in (0, 255, 0, 255, 0, 255, 0, 255)
    ]
    cases = (
        (["FORWARD"] * 8, [same] * 8, "WALL_STUCK"),
        (["TURN_LEFT", "TURN_RIGHT"] * 4, alternating, "TURN_OSCILLATION"),
        (["TURN_LEFT"] * 8, [same] * 8, "IN_PLACE_SPIN"),
        (["STOP"] * 8, [same] * 8, "GET_NOWHERE"),
    )

    for actions, frames, expected in cases:
        memory, _, task = _memory(_result())
        for action, frame in zip(actions, frames):
            memory.append_step(action, frame)
        memory.analyze()
        events = memory.drain_events()
        assert [event.kind for event in events] == [
            TemporalEventKind.SUBGOAL_COMPLETED,
            TemporalEventKind.GO_BACK_TO_ACTION,
        ]
        assert events[-1].error_mode == expected
        assert task.temporal_events[-1] == events[-1].to_dict()


def test_go_back_is_edge_triggered_and_rearms(monkeypatch):
    memory, _, _ = _memory(
        _result(),
        _result(),
        _result(),
        _result(),
    )
    modes = iter(("WALL_STUCK", "WALL_STUCK", "NONE", "WALL_STUCK"))
    monkeypatch.setattr(memory, "_detect_error_mode", lambda: next(modes))
    _push(memory)

    memory.analyze()
    assert memory.drain_events()[-1].kind is TemporalEventKind.GO_BACK_TO_ACTION

    for index, expect_go_back in ((9, False), (10, False), (11, True)):
        memory.append_step("FORWARD", _frame(index))
        memory.analyze()
        kinds = [event.kind for event in memory.drain_events()]
        assert (TemporalEventKind.GO_BACK_TO_ACTION in kinds) is expect_go_back


def test_completed_subgoal_suppresses_error_detection(monkeypatch):
    memory, _, _ = _memory(_result(completed=True))
    monkeypatch.setattr(
        memory,
        "_detect_error_mode",
        lambda: pytest.fail("completion must suppress error detection"),
    )
    for _ in range(8):
        memory.append_step("FORWARD", np.zeros((24, 32, 3), dtype=np.uint8))
    memory.analyze()
    assert [event.kind for event in memory.drain_events()] == [
        TemporalEventKind.SUBGOAL_COMPLETED
    ]


def test_model_failure_keeps_window_and_emits_no_event():
    memory, _, task = _memory(RuntimeError("model unavailable"))
    _push(memory)

    assert memory.analyze_if_ready() is None
    assert len(memory.recent_steps()) == 8
    assert "model unavailable" in memory.last_analysis_error
    assert memory.drain_events() == ()
    assert list(task.temporal_events) == []


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


def test_reset_and_input_invariants():
    memory, captioner, _ = _memory(_result())
    _push(memory, 7)
    assert memory.analyze_if_ready() is None
    assert captioner.calls == []
    with pytest.raises(TemporalStateError, match="Unsupported action"):
        memory.append_step("GO_BACK", _frame(9))
    with pytest.raises(ValueError, match="fixed eight-step"):
        TemporalMemoryConfig(window_size=3)

    memory.reset()
    assert memory.recent_steps() == ()
    assert memory.latest_result is None
    assert memory.drain_events() == ()
