"""Task-level state and the receiving side of Temporal Memory events."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional, Sequence


@dataclass(frozen=True, slots=True)
class TaskInput:
    observation: Any
    goal: str


class TaskMemory:
    """Own the current subgoal and expose temporal events to the planner."""

    def __init__(
        self,
        goal: str,
        *,
        task_guidance: str = "",
        subgoals: Sequence[Any] = (),
        max_events: int = 8,
    ) -> None:
        self.max_events = int(max_events)
        if self.max_events < 1:
            raise ValueError("max_events must be positive")
        self.reset(
            goal=goal,
            task_guidance=task_guidance,
            subgoals=subgoals,
        )

    def reset(
        self,
        *,
        goal: str,
        task_guidance: str = "",
        subgoals: Sequence[Any] = (),
    ) -> None:
        self.goal = str(goal or "").strip()
        if not self.goal:
            raise ValueError("goal must not be empty")
        self.task_guidance = str(task_guidance or "").strip()
        self.latest_input: Optional[TaskInput] = None
        self.observation_count = 0
        self.subgoal_completion_status = ""
        self.temporal_status = ""
        self.events: Deque[str] = deque(maxlen=self.max_events)
        self.temporal_events: Deque[dict[str, Any]] = deque(
            maxlen=self.max_events
        )
        self._subgoals: tuple[Any, ...] = ()
        self._current_subgoal_index = 0
        self.set_subgoals(subgoals)

    def get_task(self) -> str:
        return self.goal

    def get_task_guidance(self) -> str:
        return self.task_guidance

    def get_current_subgoal(self) -> Optional[Any]:
        if self._current_subgoal_index >= len(self._subgoals):
            return None
        return self._subgoals[self._current_subgoal_index]

    def set_subgoals(self, subgoals: Sequence[Any]) -> None:
        """Install a task plan and activate its first subgoal.

        Native temporal ``Subgoal`` values are accepted directly. Objects from
        ``vln_agent_2`` with ``task`` and ``guidance`` fields are normalized
        here so the Actor does not need to know the memory schema.
        """
        from ..TemporalCaptioner import Subgoal

        normalized = []
        for index, item in enumerate(subgoals, start=1):
            if isinstance(item, Subgoal):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                subgoal_id = item.get("subgoal_id", index)
                description = item.get("description", item.get("task"))
                criteria = item.get(
                    "completion_criteria",
                    item.get("guidance"),
                )
            else:
                subgoal_id = getattr(item, "subgoal_id", index)
                description = getattr(
                    item,
                    "description",
                    getattr(item, "task", None),
                )
                criteria = getattr(
                    item,
                    "completion_criteria",
                    getattr(item, "guidance", None),
                )
            normalized.append(
                Subgoal(
                    subgoal_id=str(subgoal_id),
                    description=str(description or ""),
                    completion_criteria=str(criteria or ""),
                )
            )
        self._subgoals = tuple(normalized)
        self._current_subgoal_index = 0

    def publish_temporal_event(self, event: Any) -> None:
        """Receive one typed TemporalEvent and update task-level state."""
        payload = event.to_dict()
        self.temporal_events.append(payload)
        kind = payload["kind"]
        value = bool(payload["value"])
        subgoal_id = str(payload["subgoal_id"])

        if kind == "SUBGOAL_COMPLETED":
            state = "COMPLETE" if value else "IN_PROGRESS"
            self.subgoal_completion_status = f"Subgoal {subgoal_id}: {state}"
            if payload["error_mode"] == "NONE":
                self.temporal_status = ""
            self.record_event(kind, self.subgoal_completion_status)
            current = self.get_current_subgoal()
            if (
                value
                and current is not None
                and str(current.subgoal_id) == subgoal_id
            ):
                self._current_subgoal_index += 1
            return

        if kind == "GO_BACK_TO_ACTION" and value:
            mode = payload["error_mode"]
            self.temporal_status = f"GO_BACK_TO_ACTION requested: {mode}"
            self.record_event(kind, self.temporal_status)
            return

        self.record_event(kind, str(value))

    def record_input(self, observation: Any) -> None:
        self.latest_input = TaskInput(
            observation=observation,
            goal=self.goal,
        )
        self.observation_count += 1

    def update_subgoal_status(self, status: str) -> None:
        self.subgoal_completion_status = str(status or "")

    def update_temporal_status(self, status: str) -> None:
        self.temporal_status = str(status or "")

    def record_event(self, event: str, content: str) -> None:
        self.events.append(f"{event}: {content}")

    def context(self) -> str:
        current = self.get_current_subgoal()
        current_text = (
            "None (all subgoals complete)"
            if current is None
            else (
                f"{current.subgoal_id}: {current.description} "
                f"(proof: {current.completion_criteria})"
            )
        )
        recent = list(self.events) or ["None"]
        return (
            f"Goal: {self.goal}\n"
            f"Observation: {self._observation_reference()}\n"
            f"Current subgoal: {current_text}\n"
            f"Subgoal status: "
            f"{self.subgoal_completion_status or 'Not analyzed'}\n"
            f"Temporal status: {self.temporal_status or 'No error'}\n"
            f"Recent events: {recent}"
        )

    def diagnostics(self) -> dict[str, Any]:
        current = self.get_current_subgoal()
        return {
            "goal": self.goal,
            "observation_count": self.observation_count,
            "current_subgoal_id": (
                current.subgoal_id if current is not None else None
            ),
            "subgoal_completion_status": self.subgoal_completion_status,
            "temporal_status": self.temporal_status,
            "events": list(self.events),
            "temporal_events": list(self.temporal_events),
            "latest_observation": self._observation_reference(),
        }

    def _observation_reference(self) -> str:
        if self.latest_input is None:
            return "None"
        observation = self.latest_input.observation
        if isinstance(observation, (str, Path)):
            return f"frame {self.observation_count}: {observation}"
        if isinstance(observation, bytes):
            digest = hashlib.sha256(observation).hexdigest()[:12]
            return (
                f"frame {self.observation_count}: bytes sha256={digest}"
            )
        return (
            f"frame {self.observation_count}: "
            f"{type(observation).__name__}"
        )


__all__ = ("TaskInput", "TaskMemory")
