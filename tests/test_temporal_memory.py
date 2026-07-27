from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

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


def test_task_memory_rgb_is_consumed_once_and_analyzed_at_eight_frames():
    memory, captioner, task = _memory(_result())

    for index in range(8):
        task.record_input(_frame(index))
        result = memory.update_from_task_memory()
        if index < 7:
            assert result is None

    assert result is not None
    assert len(captioner.calls) == 1
    assert len(memory.recent_frames()) == 8
    assert memory.update_from_task_memory() is None
    assert len(memory.recent_frames()) == 8


def test_each_analysis_publishes_error_and_completion_events():
    memory, _, task = _memory(
        _result(error=True, error_mode="WALL_STUCK")
    )
    _push(memory)
    memory.analyze()

    events = memory.drain_events()
    assert [event.kind for event in events] == [
        TemporalEventKind.ERROR,
        TemporalEventKind.SUBGOAL_COMPLETED,
    ]
    assert events[0].to_dict() == {
        "kind": "ERROR",
        "value": True,
        "subgoal_id": "1",
        "error_mode": "WALL_STUCK",
    }
    assert events[1].value is False
    assert list(task.temporal_events) == [
        event.to_dict() for event in events
    ]
    assert task.temporal_status == "ERROR=True; mode=WALL_STUCK"


def test_captioner_error_is_not_discarded():
    memory, _, _ = _memory(
        _result(error=True, error_mode="IN_PLACE_SPIN")
    )
    _push(memory)

    result = memory.analyze()

    assert result.error is True
    assert result.error_mode == "IN_PLACE_SPIN"
    assert memory.latest_result == result


def test_visual_rule_can_add_error_when_model_reports_none():
    memory, _, task = _memory(_result())
    same = np.zeros((24, 32, 3), dtype=np.uint8)
    for _ in range(8):
        memory.append_observation(same)

    result = memory.analyze()

    assert result.error is True
    assert result.error_mode == "GET_NOWHERE"
    assert task.temporal_status == "ERROR=True; mode=GET_NOWHERE"


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
