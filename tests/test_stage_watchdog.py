"""Stage force-advance watchdog semantics (task memory side)."""
from agentflow.agents.models_embodied_v2.memory.task_memory import TaskMemory


def make(n):
    tm = TaskMemory(goal="test goal")
    tm.set_subgoals([
        {"subgoal_id": i + 1, "description": f"step {i+1}", "completion_criteria": "reach it"}
        for i in range(n)])
    return tm


def test_force_advance_moves_to_next():
    tm = make(3)
    assert tm.get_current_subgoal().subgoal_id == "1"
    assert tm.force_advance("walked 12.5 m") is True
    assert tm.get_current_subgoal().subgoal_id == "2"
    assert any("SUBGOAL_FORCED" in e for e in tm.events)


def test_force_advance_never_passes_final_subgoal():
    tm = make(2)
    assert tm.force_advance("r") is True      # 1 -> 2
    assert tm.force_advance("r") is False     # 2 is final: refused
    assert tm.get_current_subgoal().subgoal_id == "2"
    assert not tm.is_task_complete()
