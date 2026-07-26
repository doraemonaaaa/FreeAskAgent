"""In-memory task state shared by the visual navigation components."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional


@dataclass(frozen=True)
class TaskInput:
    """The input represented in the diagram: current observation plus goal."""

    observation: Any
    goal: str


class TaskMemory:
    """Keep the latest observation, subtask state, and recent events."""

    def __init__(self, goal: str, *, max_events: int = 8):
        self.goal = goal
        self.latest_input: Optional[TaskInput] = None
        self.observation_count = 0
        self.subgoal_completion_status = ""
        self.events: Deque[str] = deque(maxlen=max_events)

    def record_input(self, observation: Any) -> None:
        self.latest_input = TaskInput(observation=observation, goal=self.goal)
        self.observation_count += 1

    def update_subgoal_status(self, status: str) -> None:
        self.subgoal_completion_status = status

    def record_event(self, event: str, content: str) -> None:
        self.events.append(f"{event}: {content}")

    def context(self) -> str:
        """Textual memory for the planner; the RGB observation is supplied separately."""
        observation = self._observation_reference()
        events = list(self.events) or ["None"]
        status = self.subgoal_completion_status or "Not initialized"
        return (
            f"Goal: {self.goal}\n"
            f"Observation: {observation}\n"
            f"Subgoal / Completion status:\n{status}\n"
            f"Event content: {events}"
        )

    def _observation_reference(self) -> str:
        if self.latest_input is None:
            return "None"
        observation = self.latest_input.observation
        if isinstance(observation, (str, Path)):
            return f"frame {self.observation_count}: {observation}"
        if isinstance(observation, bytes):
            digest = hashlib.sha256(observation).hexdigest()[:12]
            return f"frame {self.observation_count}: bytes sha256={digest}"
        return f"frame {self.observation_count}: {type(observation).__name__}"
