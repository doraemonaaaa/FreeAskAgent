"""Task-level state and the receiving side of Temporal Memory events."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional, Sequence

from ..data_models import Subgoal, TaskInput


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
        self._reset_generation = 0
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
        self._reset_generation += 1

    def get_latest_observation(self) -> Any | None:
        if self.latest_input is None:
            return None
        return self.latest_input.observation

    def get_task(self) -> str:
        return self.goal

    def get_reset_generation(self) -> int:
        """Return a monotonically increasing episode-reset version."""
        return self._reset_generation

    def get_task_guidance(self) -> str:
        return self.task_guidance

    def get_current_subgoal(self) -> Optional[Any]:
        if self._current_subgoal_index >= len(self._subgoals):
            return None
        return self._subgoals[self._current_subgoal_index]

    def get_next_subgoal(self) -> Optional[Any]:
        """Return the stage after the active one without advancing state."""
        index = self._current_subgoal_index + 1
        if index >= len(self._subgoals):
            return None
        return self._subgoals[index]

    def get_final_subgoal(self) -> Optional[Any]:
        """Return the last stage of the installed plan, if any."""
        return self._subgoals[-1] if self._subgoals else None

    def skip_to_final(self, reason: str) -> tuple[str, ...]:
        """Jump the plan to its last stage; returns the ids skipped over.

        Used when the destination itself has been verified in view while an
        intermediate stage is stuck. The final stage is activated rather than
        the task ended, so the destination still has to pass the ordinary
        final-approach stop protocol.
        """
        if not self._subgoals:
            return ()
        last_index = len(self._subgoals) - 1
        if self._current_subgoal_index >= last_index:
            return ()
        skipped = tuple(
            str(item.subgoal_id)
            for item in self._subgoals[self._current_subgoal_index:last_index]
        )
        self._current_subgoal_index = last_index
        self.subgoal_completion_status = (
            f"Skipped subgoals {', '.join(skipped)}: {reason}"
        )
        self.record_event("STAGE_SKIP", self.subgoal_completion_status)
        return skipped

    def is_current_subgoal_final(self) -> bool:
        """Whether the active stage is the last stage in the installed plan."""
        return bool(self._subgoals) and self._current_subgoal_index == (
            len(self._subgoals) - 1
        )

    def set_subgoals(self, subgoals: Sequence[Any]) -> None:
        """Install a task plan and activate its first subgoal.

        Native temporal ``Subgoal`` values are accepted directly. Objects from
        ``vln_agent_2`` with ``task`` and ``guidance`` fields are normalized
        here so the Actor does not need to know the memory schema.
        """
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
            self.record_event(kind, self.subgoal_completion_status)
            current = self.get_current_subgoal()
            if (
                value
                and current is not None
                and str(current.subgoal_id) == subgoal_id
            ):
                self._current_subgoal_index += 1
            return

        if kind == "ERROR":
            mode = payload["error_mode"] if value else "NONE"
            self.temporal_status = f"ERROR={value}; mode={mode}"
            self.record_event(kind, self.temporal_status)
            return

        self.record_event(kind, str(value))

    def has_next_subgoal(self) -> bool:
        return self._current_subgoal_index < len(self._subgoals) - 1

    def force_advance(self, reason: str) -> bool:
        """Advance past a stalled subgoal (stage watchdog).

        Never advances past the final subgoal: completing that one is the
        episode's stop condition and stays with the judge.
        """
        current = self.get_current_subgoal()
        if current is None or not self.has_next_subgoal():
            return False
        self.subgoal_completion_status = (
            f"Subgoal {current.subgoal_id}: COMPLETE (forced: {reason})")
        self.record_event("SUBGOAL_FORCED", self.subgoal_completion_status)
        self._current_subgoal_index += 1
        return True

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

    def is_task_complete(self) -> bool:
        """True once every planned subgoal has been reported complete.

        An empty plan is not completion: it means no plan was ever installed,
        which must not be mistaken for having finished one.
        """
        return bool(self._subgoals) and self._current_subgoal_index >= len(
            self._subgoals
        )

    def current_subgoal_context(self) -> str:
        """Return only the active subgoal, which is all the Actor steers by.

        ``context`` stays the broader planner-facing view; the Actor gets this
        narrower one so a finished stage cannot pull its waypoint backwards.
        """
        current = self.get_current_subgoal()
        if current is None:
            return "All subgoals are complete."
        return (
            "Current subgoal ({} of {}): {}\n"
            "Completion evidence: {}".format(
                self._current_subgoal_index + 1,
                len(self._subgoals),
                current.description,
                current.completion_criteria,
            )
        )

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
            "reset_generation": self._reset_generation,
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
