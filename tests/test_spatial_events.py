"""Doorway-crossing detection and its completion-judge integration."""
import numpy as np
import pytest

from agentflow.agents.models_embodied_v2.memory.spatial_memory.events import (
    DoorwayCrossingDetector,
)
from agentflow.agents.models_embodied_v2.memory.spatial_memory.occupancy_grid import (
    FREE,
    OCCUPIED,
    OccupancyGrid,
)


class _FakeGrid:
    """A hand-painted state map with the OccupancyGrid coordinate API."""

    def __init__(self, state: np.ndarray, resolution_m: float = 0.10) -> None:
        self._state = state
        self.resolution_m = resolution_m

    def state_map(self) -> np.ndarray:
        return self._state

    def world_to_cell(self, x: float, z: float):
        return int(round(z / self.resolution_m)), int(round(x / self.resolution_m))

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self._state.shape[0] and 0 <= col < self._state.shape[1]


def _two_rooms(door_x=4.0, door_half_width_m=0.45, wall_z=4.0):
    """10 m x 8 m map: two free rooms split by a wall with one doorway."""
    state = np.full((80, 100), FREE, dtype=np.uint8)
    wall_row = int(wall_z / 0.10)
    state[wall_row - 1 : wall_row + 2, :] = OCCUPIED
    door_col = int(door_x / 0.10)
    half = int(door_half_width_m / 0.10)
    state[wall_row - 1 : wall_row + 2, door_col - half : door_col + half + 1] = FREE
    return _FakeGrid(state)


def _walk(detector, grid, points):
    events = []
    trail = []
    for step, xz in enumerate(points):
        trail.append(xz)
        event = detector.update(step=step, grid=grid, trail=trail)
        if event is not None:
            events.append(event)
    return events


def test_walking_through_the_door_fires_exactly_one_event():
    grid = _two_rooms()
    points = [(4.0, 1.0 + 0.3 * i) for i in range(20)]  # straight through z=4
    events = _walk(DoorwayCrossingDetector(), grid, points)
    assert len(events) == 1
    assert abs(events[0].position_xz[0] - 4.0) < 0.4
    assert abs(events[0].position_xz[1] - 4.0) < 0.7
    assert events[0].width_m <= 1.2


def test_walking_inside_one_room_fires_nothing():
    grid = _two_rooms()
    points = [(1.0 + 0.3 * i, 2.0) for i in range(10)] + [
        (3.7, 2.0 + 0.3 * i) for i in range(5)
    ]
    assert _walk(DoorwayCrossingDetector(), grid, points) == []


def test_stepping_into_a_dead_end_closet_fires_nothing():
    grid = _two_rooms()
    # Carve a 1 m x 1 m closet off the lower room behind a narrow gap.
    state = grid._state
    state[10:22, 18:20] = OCCUPIED
    state[10:22, 30:32] = OCCUPIED
    state[10:12, 18:32] = OCCUPIED
    state[20:22, 18:32] = OCCUPIED
    state[20:22, 24:26] = FREE  # the narrow entrance
    points = [(2.5, 3.4), (2.5, 3.0), (2.5, 2.6), (2.5, 2.2), (2.5, 1.8), (2.5, 1.5)]
    assert _walk(DoorwayCrossingDetector(), grid, points) == []


def test_lingering_at_the_threshold_without_passing_fires_nothing():
    grid = _two_rooms()
    forward = [(4.0, 1.0), (4.0, 1.6), (4.0, 2.2), (4.0, 2.8), (4.0, 3.4), (4.0, 3.9)]
    back = [(4.0, 3.4), (4.0, 2.8), (4.0, 2.2)]
    assert _walk(DoorwayCrossingDetector(), grid, forward + back) == []


# ---------------------------------------------------------------- judge level
from agentflow.agents.models_embodied_v2 import Subgoal  # noqa: E402
from agentflow.agents.models_embodied_v2.memory import TaskMemory, TemporalMemory  # noqa: E402
from test_temporal_memory import SceneCaptioner, _frame, _scene_result  # noqa: E402


def _doorway_task():
    return TaskMemory(
        "Leave the bedroom and walk to the couch.",
        subgoals=(
            Subgoal("1", "Leave the bedroom", "The camera is outside the bedroom door."),
            Subgoal("2", "Walk to the couch", "The couch is directly ahead."),
        ),
    )


def _uncooperative_captioner():
    # The model never reports the crossing: door not crossed, stage not complete.
    return SceneCaptioner(lambda request: _scene_result(
        request, completed=False, door_state="NOT_VISIBLE", door_camera_side="BEFORE_DOOR",
    ))


def test_measured_crossing_completes_doorway_stage_without_model_consent():
    memory = TemporalMemory(captioner=_uncooperative_captioner(), task_memory=_doorway_task())
    for index in range(3):
        memory.set_motion_evidence(translation_m=0.6, yaw_delta_deg=0.0)
        memory.set_doorway_crossing(index == 2)
        memory.append_observation(_frame(index))
        result = memory.analyze()
    assert result.completed is True
    assert "measured doorway crossing" in memory.diagnostics()["completion_guard"]


def test_measured_crossing_is_ignored_while_committed_target_is_ahead():
    memory = TemporalMemory(captioner=_uncooperative_captioner(), task_memory=_doorway_task())
    for index in range(3):
        memory.set_motion_evidence(translation_m=0.6, yaw_delta_deg=0.0)
        memory.set_doorway_target_distance(3.0)
        memory.set_doorway_crossing(index == 2)
        memory.append_observation(_frame(index))
        result = memory.analyze()
    assert result.completed is False


def test_measured_crossing_does_not_touch_non_doorway_stages():
    task = TaskMemory(
        "Walk to the couch.",
        subgoals=(Subgoal("1", "Walk to the couch", "The couch is directly ahead."),),
    )
    memory = TemporalMemory(captioner=_uncooperative_captioner(), task_memory=task)
    for index in range(3):
        memory.set_motion_evidence(translation_m=0.6, yaw_delta_deg=0.0)
        memory.set_doorway_crossing(index == 2)
        memory.append_observation(_frame(index))
        result = memory.analyze()
    assert result.completed is False


def test_spatial_facts_reach_the_scene_prompt():
    captioner = _uncooperative_captioner()
    memory = TemporalMemory(captioner=captioner, task_memory=_doorway_task())
    memory.set_motion_evidence(translation_m=1.2, yaw_delta_deg=-30.0)
    memory.append_observation(_frame(0))
    memory.set_doorway_crossing(True)
    memory.append_observation(_frame(1))
    memory.analyze()
    facts = captioner.calls[-1].spatial_facts
    assert facts is not None and "crossed 1 doorway-width constriction" in facts
    assert "net heading change" in facts


# ------------------------------------------------------------- walk budgets
from agentflow.agents.models_embodied_v2.memory.spatial_memory.targets import (  # noqa: E402
    CommittedTarget,
)


def test_target_walking_far_past_its_budget_is_released_as_overrun():
    target = CommittedTarget(
        world_xyz=(0.0, 0.0, -2.0), kind="som", subgoal_id="1", created_step=0,
        max_age_steps=500, stagnation_steps=500,
    )
    # Circle around the target at 2 m radius: distance never improves enough
    # to reach it, yet every step moves, so stagnation never fires.
    import math as m
    status = "active"
    for k in range(80):
        a = 0.35 * k
        status = target.update(step=k, position_xz=(2*m.sin(a), -2.0+2*m.cos(a)-2.0*0+0.0), yaw_deg=0.0)
        if status != "active":
            break
    assert status == "overrun"
    assert target.walked_m > target.walk_budget_m


def test_target_reached_directly_stays_within_budget():
    target = CommittedTarget(
        world_xyz=(0.0, 0.0, -2.0), kind="som", subgoal_id="1", created_step=0,
    )
    status = "active"
    for k, z in enumerate((0.0, -0.5, -1.0, -1.5, -1.8)):
        status = target.update(step=k, position_xz=(0.0, z), yaw_deg=0.0)
    assert status == "reached"


def test_stage_overrun_forces_a_preview_reorientation():
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_vln_agent_4 import RoutedEngine
    from agentflow.agents.vln_agent_4 import VLNAgent

    agent = VLNAgent(engine=RoutedEngine())
    instruction = "Walk forward to the pool area."
    agent.prepare_task(instruction)
    intrinsics = np.array(((16.0, 0.0, 16.0), (0.0, 16.0, 12.0), (0.0, 0.0, 1.0)))
    pose = np.eye(4)
    previews = 0
    for _ in range(24):
        pose = pose.copy()
        pose[2, 3] -= 0.6  # 0.6 m forward each step; > 10 m total
        decision = agent.act(
            np.zeros((24, 32, 3), np.uint8),
            np.full((24, 32), 2.0, np.float32),
            instruction, intrinsics, pose,
        )
        if decision.action_mode == "PREVIEW":
            previews += 1
    assert previews >= 1
    assert agent._overrun_count >= 1
