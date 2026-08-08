from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

pytest.skip(
    "legacy AsyncThinkActVLN integration is superseded by vln_agent_3",
    allow_module_level=True,
)

from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.Actor import Actor
from workspace.FreeAskAgent.agentflow.agents.models_embodied_v2.deprecated.Thinker import Thinker
from agentflow.agents.models_embodied_v2.memory import (
    TaskMemory,
    TemporalMemory,
    TemporalMemoryConfig,
)
from agentflow.agents.vln_agent import AsyncThinkActVLN


np = pytest.importorskip("numpy")


def _frame(index: int):
    image = np.full((18, 24, 3), index * 13, dtype=np.uint8)
    image[:, :5, 1] = (index * 29) % 255
    return image


@dataclass
class _Record:
    label: str

    def to_memory_text(self) -> str:
        return self.label


class _Captioner:
    def __init__(self):
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        return _Record(f"three-step-record-{len(self.requests)}")


class _Actor:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    @staticmethod
    def rgb_to_bytes(image):
        return np.asarray(image).tobytes()

    def act(self, image, directive):
        self.calls.append((image, directive))
        return self.actions.pop(0)


class _Thinker:
    def __init__(self):
        self.task_memory = None
        self.subtask_tracker = None
        self.contexts = []
        self.planner_engine = object()
        self.closed = False

    def submit_observation(
        self,
        image,
        *,
        wait_for_completion,
        tracker_updated_callback,
    ):
        assert wait_for_completion is True
        self.subtask_tracker = (
            "1. Subtask: navigate | Completion status: IN PROGRESS"
        )
        tracker_updated_callback(self.subtask_tracker)
        # This context is read after the temporal callback. At the first
        # three-step boundary it must already contain the new model record.
        self.contexts.append(self.task_memory.context())
        return "follow temporal memory"

    def update_tracker_only(self, image):
        self.subtask_tracker = (
            "1. Subtask: navigate | Completion status: COMPLETE"
        )
        return self.subtask_tracker

    def reset(self, goal):
        self.subtask_tracker = None

    def close(self, timeout=None):
        self.closed = True


class _PlannerEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, content, *, system_prompt, **kwargs):
        self.calls.append((content, system_prompt))
        if "decompose" in system_prompt:
            return "1. Subtask: navigate | Completion status: NOT STARTED"
        if "update only each subtask" in system_prompt:
            return "1. Subtask: navigate | Completion status: IN PROGRESS"
        return "Continue toward the visible doorway."


def _metadata(state_id: int, episode_id: str = "episode"):
    return {
        "episode_id": episode_id,
        # The wire protocol carries observation state IDs, but the agent maps
        # only fields owned by TemporalObservation.
        "step_id": state_id,
        "timestamp_seconds": float(state_id),
        "position_xyz": [state_id * 0.1, 0.0, 0.0],
        "yaw_degrees": 0.0,
        "distance_to_goal_meters": 10.0 - state_id * 0.1,
        "landmark_ids": [],
    }


def test_image_only_agent_closes_three_pairs_before_next_directive():
    captioner = _Captioner()
    memory = TemporalMemory(
        "find the kitchen",
        episode_id="episode",
        captioner=captioner,
    )
    actor = _Actor(["FORWARD"] * 4)
    thinker = _Thinker()
    agent = AsyncThinkActVLN(
        "find the kitchen",
        actor=actor,
        thinker=thinker,
        temporal_memory=memory,
        temporal_captioner=captioner,
        episode_id="episode",
    )
    # Four observations close three action transitions. The fourth call also
    # proves that the next directive can already consume the fresh record.
    frames = [_frame(index) for index in range(4)]

    for frame in frames:
        action = agent.act(frame)
        assert action == "FORWARD"

    assert agent.task_memory is None
    assert agent.temporal_memory is memory
    assert agent.memory.temporal_memory is memory
    assert thinker.memory is agent.memory
    assert len(captioner.requests) == 1
    request = captioner.requests[0]
    assert [step.step_id for step in request.steps] == [1, 2, 3]
    assert all(
        step.image is not frames[index + 1]
        and np.array_equal(step.image, frames[index + 1])
        for index, step in enumerate(request.steps)
    )
    assert "three-step-record-1" in thinker.contexts[-1]
    assert memory.latest_record == _Record("three-step-record-1")


def test_terminal_finish_closes_pending_without_selecting_another_action():
    captioner = _Captioner()
    memory = TemporalMemory(
        "stop safely",
        episode_id="episode",
        captioner=captioner,
    )
    actor = _Actor(["STOP"])
    thinker = _Thinker()
    agent = AsyncThinkActVLN(
        "stop safely",
        actor=actor,
        thinker=thinker,
        temporal_memory=memory,
        temporal_captioner=captioner,
        episode_id="episode",
    )
    pre = _frame(0)
    post = _frame(1)

    assert agent.act(pre) == "STOP"
    assert agent._step_execution(
        {
            "step_id": 1,
            "commanded_action": "STOP",
            "collision": False,
            "terminal": False,
        },
        terminal=True,
    ).terminal is True
    result = agent.finish_episode(post)

    assert result is None
    assert memory.pending_step_id is None
    assert len(memory.recent_steps()) == 1
    stored_post = memory.recent_steps()[0].post_observation.image
    assert stored_post is not post
    assert np.array_equal(stored_post, post)
    assert len(actor.calls) == 1


def test_real_thinker_callback_closes_previous_action_before_directive():
    captioner = _Captioner()
    memory = TemporalMemory(
        "find the door",
        episode_id="episode",
        captioner=captioner,
    )
    actor = _Actor(["FORWARD", "TURN_LEFT"])
    planner = _PlannerEngine()
    thinker = Thinker(
        "find the door",
        actor,
        planner_engine=planner,
        memory=memory,
        show_output=False,
    )
    agent = AsyncThinkActVLN(
        "find the door",
        actor=actor,
        thinker=thinker,
        temporal_memory=memory,
        temporal_captioner=captioner,
        episode_id="episode",
    )
    first = _frame(0)
    second = _frame(1)
    try:
        assert agent.act(first, _metadata(0)) == "FORWARD"
        assert agent.act(
            second,
            _metadata(1),
            {
                "step_id": 1,
                "commanded_action": "FORWARD",
                "collision": False,
                "terminal": False,
            },
        ) == "TURN_LEFT"
    finally:
        agent.close(timeout=1)

    step = memory.recent_steps()[0]
    assert step.step_id == 1
    assert step.pre_observation.image is not first
    assert step.post_observation.image is not second
    assert np.array_equal(step.pre_observation.image, first)
    assert np.array_equal(step.post_observation.image, second)
    assert memory.pending_step_id == 2
    assert memory.recent_actions() == ("FORWARD", "TURN_LEFT")
    assert any(
        "step 1: selected=FORWARD" in str(content)
        for content, system_prompt in planner.calls
        if "next directive" in system_prompt
    )


def test_reset_clears_episode_state_and_keeps_temporal_module_instance():
    captioner = _Captioner()
    memory = TemporalMemory("old goal", episode_id="old", captioner=captioner)
    agent = AsyncThinkActVLN(
        "old goal",
        actor=_Actor(["FORWARD"]),
        thinker=_Thinker(),
        temporal_memory=memory,
        temporal_captioner=captioner,
        episode_id="old",
    )
    agent.act(_frame(0), _metadata(0, "old"))
    assert memory.pending_step_id == 1

    agent.reset(goal="new goal", episode_id="new")

    assert agent.task_memory is None
    assert agent.temporal_memory is memory
    assert agent.memory.temporal_memory is memory
    assert memory.episode_id == "new"
    assert memory.goal == "new goal"
    assert memory.pending_step_id is None
    assert memory.recent_steps() == ()
    assert memory.latest_record is None


def test_task_plus_temporal_mode_composes_distinct_memory_modules():
    captioner = _Captioner()
    temporal = TemporalMemory(
        "find the kitchen",
        episode_id="episode",
        captioner=captioner,
        config=TemporalMemoryConfig(
            cumulative_history_size=32,
            wall_stuck_confirm_steps=30,
        ),
    )
    task = TaskMemory("find the kitchen")
    actor = _Actor(["FORWARD"] * 9)
    thinker = _Thinker()
    agent = AsyncThinkActVLN(
        "find the kitchen",
        memory_mode="task+temporal",
        actor=actor,
        thinker=thinker,
        task_memory=task,
        temporal_memory=temporal,
        temporal_captioner=captioner,
        episode_id="episode",
    )

    for index in range(9):
        assert agent.act(_frame(index)) == "FORWARD"

    assert agent.task_memory is task
    assert agent.temporal_memory is temporal
    assert agent.task_memory is not agent.temporal_memory
    assert task.observation_count == 9
    assert len(captioner.requests) == 2
    assert [
        [step.step_id for step in request.steps]
        for request in captioner.requests
    ] == [[1, 2, 3], [4, 5, 6]]
    assert "[Task Memory]" in thinker.contexts[-1]
    assert "[Temporal Memory]" in thinker.contexts[-1]
    diagnostics = agent.memory_diagnostics()
    assert diagnostics["mode"] == "task+temporal"
    assert set(diagnostics["modules"]) == {
        "task_memory",
        "temporal_memory",
    }
    assert (
        diagnostics["timing"]["task_memory"]["inference_count"]
        == 9
    )
    assert (
        diagnostics["timing"]["temporal_memory"]["inference_count"]
        == 2
    )
    json.dumps(diagnostics)


def test_confirmed_cumulative_error_runs_go_back_before_actor():
    captioner = _Captioner()
    temporal = TemporalMemory(
        "leave the wall",
        episode_id="episode",
        captioner=captioner,
        config=TemporalMemoryConfig(
            recovery_turn_degrees=15,
            recovery_forward_steps=1,
        ),
    )
    task = TaskMemory("leave the wall")
    actor = _Actor(["FORWARD"] * 6)
    thinker = _Thinker()
    agent = AsyncThinkActVLN(
        "leave the wall",
        memory_mode="task+temporal",
        actor=actor,
        thinker=thinker,
        task_memory=task,
        temporal_memory=temporal,
        temporal_captioner=captioner,
        episode_id="episode",
    )
    frame = _frame(1)

    assert [agent.act(frame.copy()) for _ in range(6)] == ["FORWARD"] * 6
    recovery_action = agent.act(frame.copy())

    assert recovery_action == "TURN_LEFT"
    assert len(actor.calls) == 6
    assert temporal.pending_selected_action == "TURN_LEFT"
    assert temporal.active_go_back_request is not None
    assert temporal.next_recovery_action() == "FORWARD"
    assert "phase=RECOVERING" in task.temporal_status
    assert "go_back=ACTIVE" in task.temporal_status
    assert any("Temporal cumulative status" in event for event in task.events)
    assert any("Go Back Action" in event for event in task.events)


def test_task_only_ablation_does_not_create_temporal_memory():
    task = TaskMemory("find the door")
    thinker = _Thinker()
    agent = AsyncThinkActVLN(
        "find the door",
        memory_mode="task",
        actor=_Actor(["TURN_LEFT"]),
        thinker=thinker,
        task_memory=task,
        episode_id="episode",
    )

    assert agent.act(_frame(0)) == "TURN_LEFT"

    assert agent.task_memory is task
    assert agent.temporal_memory is None
    assert task.observation_count == 1
    assert agent.memory.recent_actions() == ("TURN_LEFT",)
    assert "[Task Memory]" in thinker.contexts[-1]
    assert "[Temporal Memory]" not in thinker.contexts[-1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MOVE_FORWARD", "FORWARD"),
        ("MOVE_FORWARD_0.25_METERS", "FORWARD"),
        ("TURN_LEFT_15_DEGREES", "TURN_LEFT"),
        ("TURN_RIGHT_15_DEGREES", "TURN_RIGHT"),
    ],
)
def test_actor_normalizes_extended_action_without_losing_pending_step(
    raw,
    expected,
):
    assert Actor.parse_action(raw) == expected
