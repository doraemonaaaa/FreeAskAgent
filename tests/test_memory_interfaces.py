import json
from dataclasses import dataclass

import pytest

pytest.skip(
    "legacy AsyncThinkActVLN memory interfaces are no longer exported",
    allow_module_level=True,
)

from agentflow.agents.models_embodied_v2.memory import (
    CompositeMemory,
    TaskMemory,
    TaskMemoryInterface,
    TemporalMemory,
    TemporalMemoryInterface,
)


np = pytest.importorskip("numpy")


def _image(index):
    rng = np.random.default_rng(index)
    return rng.integers(0, 256, (20, 28, 3), dtype=np.uint8)


@dataclass
class _Record:
    label: str

    def to_memory_text(self):
        return self.label

    def to_memory_dict(self):
        return {"label": self.label}


class _Captioner:
    def __init__(self):
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        return _Record(f"window-{len(self.requests)}")


def test_temporal_interface_runs_standalone_with_images_and_actions_only():
    captioner = _Captioner()
    core = TemporalMemory(
        "reach the door",
        episode_id="episode",
        captioner=captioner,
    )
    ticks = iter(index * 0.1 for index in range(20))
    interface = TemporalMemoryInterface(core, clock=lambda: next(ticks))
    frames = [_image(index) for index in range(4)]

    interface.observe(frames[0])
    for index in range(3):
        interface.stage_action("FORWARD", "subgoal in progress")
        interface.observe(
            frames[index + 1],
            subgoal_snapshot="subgoal in progress",
        )

    assert len(captioner.requests) == 1
    request = captioner.requests[0]
    assert [step.step_id for step in request.steps] == [1, 2, 3]
    assert all(
        step.image is not frames[index + 1]
        and np.array_equal(step.image, frames[index + 1])
        for index, step in enumerate(request.steps)
    )
    assert interface.latest_record == _Record("window-1")
    timing = interface.timing_summary()
    assert timing["interface"]["inference_count"] == 4
    assert timing["temporal_memory"]["inference_count"] == 1
    assert timing["temporal_memory"]["average_inference_ms"] is not None


def test_temporal_interface_snapshots_a_reused_mutable_rgb_buffer():
    captioner = _Captioner()
    core = TemporalMemory(
        "reach the door",
        episode_id="episode",
        captioner=captioner,
    )
    interface = TemporalMemoryInterface(core)
    shared_frame = np.zeros((20, 28, 3), dtype=np.uint8)

    interface.observe(
        shared_frame,
        metadata={"timestamp_seconds": 0.0},
    )
    for index in range(3):
        interface.stage_action("FORWARD")
        shared_frame.fill(index + 1)
        interface.observe(
            shared_frame,
            metadata={"timestamp_seconds": float(index + 1)},
        )

    request = captioner.requests[0]
    assert [
        int(step.image[0, 0, 0]) for step in request.steps
    ] == [1, 2, 3]
    assert all(step.image is not shared_frame for step in request.steps)


def test_composite_memory_modes_are_explicit_and_json_safe():
    task = TaskMemoryInterface(TaskMemory("goal"))
    temporal = TemporalMemoryInterface(
        TemporalMemory("goal", episode_id="episode")
    )
    memory = CompositeMemory(
        "goal",
        episode_id="episode",
        mode="task+temporal",
        task=task,
        temporal=temporal,
    )

    memory.record_input(_image(0))
    memory.close_previous_action("tracker")
    memory.stage_action("TURN_LEFT", "tracker")

    diagnostics = memory.diagnostics()
    assert diagnostics["mode"] == "task+temporal"
    assert set(diagnostics["modules"]) == {
        "task_memory",
        "temporal_memory",
    }
    assert diagnostics["recent_actions"] == ["TURN_LEFT"]
    json.dumps(diagnostics)
    assert "[Task Memory]" in memory.context()
    assert "[Temporal Memory]" in memory.context()


def test_composite_reset_clears_both_modules_and_timing():
    memory = CompositeMemory(
        "old",
        episode_id="old",
        mode="task+temporal",
    )
    memory.record_input(_image(0))
    memory.close_previous_action("tracker")
    memory.stage_action("STOP", "tracker")

    memory.reset(episode_id="new", goal="new goal")

    assert memory.episode_id == "new"
    assert memory.task_memory.goal == "new goal"
    assert memory.task_memory.observation_count == 0
    assert memory.temporal_memory.episode_id == "new"
    assert memory.temporal_memory.pending_step_id is None
    assert memory.recent_actions() == ()
    assert (
        memory.diagnostics()["timing"]["task_memory"]["inference_count"]
        == 0
    )


@pytest.mark.parametrize("mode", ["none", "task", "temporal", "task+temporal"])
def test_all_ablation_modes_construct(mode):
    memory = CompositeMemory("goal", mode=mode)

    assert (memory.task is not None) is ("task" in mode)
    assert (memory.temporal is not None) is ("temporal" in mode)
