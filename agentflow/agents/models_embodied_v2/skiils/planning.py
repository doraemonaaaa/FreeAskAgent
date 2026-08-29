"""Strict subgoal-plan parsing and normalization helpers."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Sequence

from agentflow.agents.models_embodied_v2.data_models import Subgoal
from .protocol import SubgoalPlanOutput


# One stage per line: "id|description|completion criterion".  Anything that
# does not open with an integer and a pipe is not a stage line.
STAGE_LINE = re.compile(r"^\s*(\d+)\s*\|([^|]*)\|([^|]*)$")


def parse_subgoal_plan(
    response: str,
    *,
    instruction: str = "",
) -> list[Subgoal]:
    """Parse, strictly validate, and normalize a model-generated plan."""
    payload = parse_plan_lines(response)
    validated = SubgoalPlanOutput.model_validate(payload)
    last_index = len(validated.subgoals) - 1
    subgoals = [
        Subgoal(
            subgoal_id=item.subgoal_id,
            description=item.description,
            completion_criteria=(
                item.completion_criteria
                if index == last_index
                else remove_intermediate_stop(item.completion_criteria)
            ),
        )
        for index, item in enumerate(validated.subgoals)
    ]
    subgoals = repair_plan_fidelity(instruction, subgoals)
    validate_plan_fidelity(instruction, subgoals)
    return ground_pre_turn_stages(split_compound_turns(subgoals))


def repair_plan_fidelity(
    instruction: str,
    subgoals: Sequence[Subgoal],
) -> list[Subgoal]:
    """Remove invented loop/return requirements without dropping the route."""
    source = instruction.lower()
    repaired = list(subgoals)
    patterns = _invented_route_patterns()
    for index, item in enumerate(repaired):
        item_text = f"{item.description} {item.completion_criteria}".lower()
        if not any(
            re.search(pattern, item_text)
            and not re.search(pattern, source)
            for pattern in patterns
        ):
            continue
        next_stage = (
            repaired[index + 1].description
            if index + 1 < len(repaired)
            else "the next instruction endpoint"
        )
        repaired[index] = replace(
            item,
            completion_criteria=(
                "The camera has progressed along the instructed side of "
                "the landmark and is positioned to begin the next route "
                f"stage: {next_stage}, without adding any extra route "
                "requirement."
            ),
        )
    return repaired


def _invented_route_patterns() -> tuple[str, ...]:
    return (
        r"\b(?:full|complete)\s+(?:clockwise\s+|counterclockwise\s+)?"
        r"(?:circle|circuit|loop)\b",
        r"\breturn(?:ing)?\s+to\s+(?:the\s+)?"
        r"(?:start|starting point|original position)\b",
        r"\bback\s+to\s+(?:the\s+)?"
        r"(?:start|starting point|original position)\b",
    )


def validate_plan_fidelity(
    instruction: str,
    subgoals: Sequence[Subgoal],
) -> None:
    """Reject route requirements invented by the generated plan."""
    source = instruction.lower()
    generated = " ".join(
        f"{item.description} {item.completion_criteria}"
        for item in subgoals
    ).lower()
    for pattern in _invented_route_patterns():
        if re.search(pattern, generated) and not re.search(pattern, source):
            raise ValueError(
                "subgoal plan invented a route requirement absent from "
                "the instruction"
            )


def parse_plan_lines(response: str) -> dict[str, Any]:
    """Read the one-stage-per-line plan format into the plan schema's shape.

    The line format exists because nested JSON arrays are what the planning
    checkpoint gets wrong; a line carries no structure that can be malformed
    beyond its two separators.  Parsing stays strict: a line that opens like a
    stage but does not hold exactly three fields is an error, not something to
    repair, and ``SubgoalPlanOutput`` still enforces the rest.
    """
    stages: list[dict[str, Any]] = []
    for line in response.splitlines():
        if not line.strip():
            continue
        match = STAGE_LINE.match(line)
        if match is None:
            # Tolerate a preamble or trailing prose, never a broken stage line.
            if re.match(r"^\s*\d+\s*\|", line):
                raise ValueError(
                    f"stage line must hold exactly three fields: {line!r}"
                )
            continue
        stages.append(
            {
                "subgoal_id": match.group(1),
                "description": match.group(2).strip(),
                "completion_criteria": match.group(3).strip(),
            }
        )
    if not stages:
        raise ValueError("response contained no 'id|description|criterion' line")
    return {"subgoals": stages}


def remove_intermediate_stop(criteria: str) -> str:
    """Remove stopping requirements that apply only to final goals."""
    normalized = re.sub(
        r"\s*(?:,\s*)?\band\b[^.;]*\b"
        r"(?:stop(?:ped|ping)?|come\s+to\s+a\s+"
        r"(?:complete\s+)?stop)\b[^.;]*",
        "",
        criteria,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"\b(?:is|has)\s+(?:come\s+to\s+a\s+"
        r"(?:complete\s+)?stop|stop(?:ped|ping)?)\b",
        "is positioned",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.rstrip(" ,;")
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized or criteria


TURN_CLAUSE = re.compile(
    r"\bturn\s+(?:to\s+the\s+)?(left|right)\b", flags=re.IGNORECASE
)
_CLAUSE_SPLIT = re.compile(
    r"(?:\s*[,;]\s*|\s+)(?:and\s+then|and|then)\s+", flags=re.IGNORECASE
)
_ENTER_VERB = re.compile(
    r"^(?:enter|go\s+(?:in|into)|walk\s+(?:in|into)|exit|leave|step\s+(?:in|into))\b",
    flags=re.IGNORECASE,
)
# Stages that pass through or over something end with it behind the camera,
# not "directly ahead": stairs, hallways, rooms crossed, thresholds.
_TRAVERSE_CLAUSE = re.compile(
    r"^(?:(?:go|walk|climb|head|move|proceed|continue|run)\s+"
    r"(?:up|down|through|across|along|over)\b|(?:ascend|descend|cross)\b)",
    flags=re.IGNORECASE,
)
_LEADING_VERBS = re.compile(
    r"^(?:(?:walk|go|move|head|proceed|continue|enter|exit|leave|ascend|"
    r"climb|descend|follow|pass|approach|reach|step|come|cross|turn\s+"
    r"(?:to\s+the\s+)?(?:left|right))\b\s*)+",
    flags=re.IGNORECASE,
)
_LEADING_PREPOSITIONS = re.compile(
    r"^(?:(?:up|down|into|in|out\s+of|to|toward|towards|through|along|"
    r"past|beside|at|near|by|onto|over|until|until\s+you\s+reach)\b\s*)+",
    flags=re.IGNORECASE,
)


def landmark_phrase(clause: str) -> str:
    """The thing a clause is about: "go beside the bed" -> "the bed"."""
    text = clause.strip().rstrip(" .!?")
    text = _LEADING_VERBS.sub("", text).strip()
    text = _LEADING_PREPOSITIONS.sub("", text).strip()
    text = text.rstrip(" .!?")
    if not text:
        return "the route ahead"
    if not re.match(r"^(?:the|a|an|this|that|your)\b", text, re.IGNORECASE):
        text = f"the {text}"
    return text


def split_compound_turns(subgoals: Sequence[Subgoal]) -> list[Subgoal]:
    """Give every turn its own stage.

    A stage such as "Enter the bedroom and turn left beside the bed" is
    judged by one question, and whichever clause the judge latches onto
    (usually the doorway) completes the whole stage before the rest of it
    happened. Splitting at the turn makes each clause a stage with its own
    endpoint; the model's original criterion stays with the last clause,
    whose endpoint it described.
    """
    result: list[Subgoal] = []
    for subgoal in subgoals:
        clauses = [
            part.strip()
            for part in _CLAUSE_SPLIT.split(subgoal.description)
            if part and part.strip()
        ]
        if len(clauses) < 2 or not any(TURN_CLAUSE.search(c) for c in clauses):
            result.append(subgoal)
            continue
        # Only the turn is isolated; consecutive movement clauses stay
        # together so a stage keeps whatever route detail the planner wrote.
        groups: list[str] = []
        for clause in clauses:
            if (
                groups
                and not TURN_CLAUSE.search(clause)
                and not TURN_CLAUSE.search(groups[-1])
            ):
                groups[-1] = f"{groups[-1]} and {clause}"
            else:
                groups.append(clause)
        for index, clause in enumerate(groups):
            description = clause[0].upper() + clause[1:]
            following = groups[index + 1] if index + 1 < len(groups) else None
            if following is None:
                criterion = subgoal.completion_criteria
            else:
                criterion = _stage_criterion(clause, following)
            result.append(
                Subgoal(
                    subgoal_id=str(len(result) + 1),
                    description=description,
                    completion_criteria=criterion,
                )
            )
    return [
        replace(item, subgoal_id=str(index + 1))
        for index, item in enumerate(result)
    ]


def _stage_criterion(clause: str, following: str) -> str:
    turn = TURN_CLAUSE.search(clause)
    if turn is not None:
        return (
            f"After turning {turn.group(1).lower()}, "
            f"{landmark_phrase(following)} is centred in the view."
        )
    landmark = landmark_phrase(clause)
    if _ENTER_VERB.match(clause.strip()):
        return (
            f"The camera has crossed the threshold of {landmark} and its "
            "interior is central in the view."
        )
    if _TRAVERSE_CLAUSE.match(clause.strip()):
        return (
            f"The camera has passed {landmark}: it is behind or below the "
            "camera and the space beyond it fills the view."
        )
    return (
        f"{landmark[0].upper()}{landmark[1:]} is beside or directly ahead of "
        "the camera, within a step."
    )


def ground_pre_turn_stages(subgoals: Sequence[Subgoal]) -> list[Subgoal]:
    """A walking stage before a turn ends at its landmark, not after the turn.

    The criterion is grounded in the stage's own landmark and says what has
    not happened yet; it deliberately mentions no doorway, so a stage that is
    not a doorway crossing is not judged as one.
    """
    grounded = list(subgoals)
    for index in range(len(grounded) - 1):
        current = grounded[index]
        following = grounded[index + 1]
        turn = TURN_CLAUSE.search(following.description)
        if turn is None or TURN_CLAUSE.search(current.description):
            continue
        if not re.search(
            r"\b(walk|go|move|continue|proceed|head|follow|climb|ascend|"
            r"descend|pass)\b",
            current.description,
            flags=re.IGNORECASE,
        ):
            continue
        landmark = landmark_phrase(current.description)
        if _TRAVERSE_CLAUSE.match(current.description.strip()):
            criterion = (
                f"The camera has passed {landmark}: it is behind or below "
                "the camera and the space beyond it fills the view; the "
                f"camera has not yet turned {turn.group(1).lower()}."
            )
        else:
            criterion = (
                f"{landmark[0].upper()}{landmark[1:]} is beside or directly "
                "ahead of the camera, within a step; the camera has not yet "
                f"turned {turn.group(1).lower()}."
            )
        grounded[index] = replace(current, completion_criteria=criterion)
    return grounded
