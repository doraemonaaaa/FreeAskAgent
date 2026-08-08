"""Strict subgoal-plan parsing and normalization helpers."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Callable, Sequence

from agentflow.agents.models_embodied_v2.data_models import Subgoal
from agentflow.agents.vln_agent_3_protocol import SubgoalPlanOutput


JsonExtractor = Callable[[str], dict[str, Any]]


def parse_subgoal_plan(
    response: str,
    *,
    extract_json: JsonExtractor,
    instruction: str = "",
) -> list[Subgoal]:
    """Parse, strictly validate, and normalize a model-generated plan."""
    payload = extract_plan_json(response, extract_json=extract_json)
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
    return ground_turn_decision_points(subgoals)


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


def extract_plan_json(
    response: str,
    *,
    extract_json: JsonExtractor,
) -> dict[str, Any]:
    """Repair only missing quotes on known keys."""
    try:
        payload = extract_json(response)
        if "subgoals" in payload:
            return payload
        raise ValueError("response contained only a nested JSON object")
    except ValueError as original_error:
        repaired = re.sub(
            r"(?<=[{,])\s*"
            r"(subgoals|subgoal_id|description|completion_criteria)"
            r'"?\s*:',
            r'"\1":',
            response,
        )
        if repaired == response:
            raise original_error
        return extract_json(repaired)


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


def ground_turn_decision_points(
    subgoals: Sequence[Subgoal],
) -> list[Subgoal]:
    """Ground a pre-turn endpoint in route geometry, not its material."""
    grounded = list(subgoals)
    for index in range(len(grounded) - 1):
        current = grounded[index]
        following = grounded[index + 1]
        turn = re.search(
            r"\bturn\s+(left|right)\b",
            following.description,
            flags=re.IGNORECASE,
        )
        movement = re.search(
            r"\b(walk|go|move|continue|proceed|head)\b",
            current.description,
            flags=re.IGNORECASE,
        )
        current_turn = re.search(
            r"\bturn\s+(left|right)\b",
            current.description,
            flags=re.IGNORECASE,
        )
        if turn is None or (movement is None and current_turn is None):
            continue
        direction = turn.group(1).lower()
        description = current.description.rstrip(" .!?")
        if current_turn is None:
            description += (
                f" until reaching the next {direction}-turn decision point."
            )
        else:
            description += (
                ", then follow the new corridor until reaching the next "
                f"{direction}-turn decision point."
            )
        criterion = (
            "After sustained progress along the instructed route, the camera "
            f"is at the doorway/junction where the next {direction} turn can "
            f"now be executed. Seeing an earlier {direction}-side doorway at "
            "a distance is not sufficient; exact material appearance need "
            "not be visually distinguishable."
        )
        grounded[index] = replace(
            current,
            description=description,
            completion_criteria=criterion,
        )
    return grounded
