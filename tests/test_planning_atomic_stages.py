"""Compound turn stages are split so each clause gets its own endpoint."""

from __future__ import annotations

from agentflow.agents.models_embodied_v2.skiils.planning import (
    landmark_phrase,
    parse_subgoal_plan,
    split_compound_turns,
)
from agentflow.agents.models_embodied_v2.data_models import Subgoal


def test_turn_clauses_become_their_own_stages():
    plan = parse_subgoal_plan(
        "1|Enter the bedroom and turn left beside the bed|The camera is beside the bed\n"
        "2|Turn left and proceed through the hallway|The camera is moving through the hallway\n",
        instruction="Go into the bedroom and turn left, go beside the bed, turn left and go through a hallway.",
    )
    assert [s.subgoal_id for s in plan] == ["1", "2", "3", "4"]
    assert [s.description for s in plan] == [
        "Enter the bedroom",
        "Turn left beside the bed",
        "Turn left",
        "Proceed through the hallway",
    ]
    # Entering a room is a crossing; the model's own criterion stays with
    # the last clause of the stage it described.
    assert "crossed the threshold of the bedroom" in plan[0].completion_criteria
    assert plan[1].completion_criteria == "The camera is beside the bed."
    # A bare turn ends when the next stage's landmark is centred.
    assert plan[2].completion_criteria == (
        "After turning left, the hallway is centred in the view."
    )
    assert plan[3].completion_criteria == "The camera is moving through the hallway"


def test_walk_before_turn_is_grounded_without_inventing_a_doorway():
    plan = parse_subgoal_plan(
        "1|Walk down the hall|The camera reaches the end of the hall\n"
        "2|Turn right|The camera faces the office\n",
        instruction="Walk down the hall and turn right.",
    )
    # "walk down the hall" traverses the hall: its endpoint is having passed
    # it, not seeing it ahead.
    assert plan[0].completion_criteria == (
        "The camera has passed the hall: it is behind or below the camera "
        "and the space beyond it fills the view; the camera has not yet "
        "turned right."
    )
    assert "doorway" not in plan[0].completion_criteria.lower()


def test_atomic_stages_are_left_alone():
    stages = (
        Subgoal("1", "Walk forward to the pool area", "The camera is beside the pool"),
    )
    assert split_compound_turns(stages) == list(stages)


def test_landmark_phrase_strips_verbs_and_prepositions():
    assert landmark_phrase("go beside the bed") == "the bed"
    assert landmark_phrase("Proceed through the hallway") == "the hallway"
    assert landmark_phrase("ascend the stairs") == "the stairs"
    assert landmark_phrase("Walk down the hall to the marked doorway") == (
        "the hall to the marked doorway"
    )
    assert landmark_phrase("turn left") == "the route ahead"
