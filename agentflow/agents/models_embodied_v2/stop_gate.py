"""The single place that decides whether STOP is legal this step.

STOP used to be whatever the policy model emitted, which meant one over-eager
frame ended the episode. Here it becomes a gated action: the policy may only
choose it when every independent precondition agrees, and the gate result is
also used to mask STOP out of the policy's action space entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .memory.subtask_tracker import SubtaskTracker


@dataclass(frozen=True)
class StopDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class StopGate:
    """Track arrival evidence over time and rule on STOP requests."""

    def __init__(
        self,
        *,
        arrival_radius_m: float = 2.0,
        min_steps_since_target_seen: int = 4,
        depth_probe: Optional[Callable[[], Optional[float]]] = None,
    ):
        if min_steps_since_target_seen < 0:
            raise ValueError("min_steps_since_target_seen must be non-negative.")
        self.arrival_radius_m = arrival_radius_m
        self.min_steps_since_target_seen = min_steps_since_target_seen
        # Hook for the depth/segmentation range check; returns meters to the
        # target, or None when no measurement is available this step.
        self.depth_probe = depth_probe
        self.steps = 0
        self.step_target_first_seen: Optional[int] = None
        self.last_distance_m: Optional[float] = None

    def reset(self) -> None:
        self.steps = 0
        self.step_target_first_seen = None
        self.last_distance_m = None

    def observe(self, *, target_visible: bool, distance_m: Optional[float] = None) -> None:
        """Record this step's arrival evidence before the gate is consulted."""
        self.steps += 1
        if target_visible and self.step_target_first_seen is None:
            self.step_target_first_seen = self.steps
        if distance_m is not None:
            self.last_distance_m = distance_m

    @property
    def steps_since_target_seen(self) -> int:
        if self.step_target_first_seen is None:
            return 0
        return self.steps - self.step_target_first_seen

    def evaluate(self, tracker: SubtaskTracker) -> StopDecision:
        """Decide whether STOP may be offered to the policy this step."""
        if not tracker.all_complete:
            active = tracker.active_subtask
            detail = f"subtask {active.index} ({active.text}) is {active.status.value}" if active else "plan incomplete"
            return StopDecision(False, f"plan not finished: {detail}")

        if self.step_target_first_seen is None:
            return StopDecision(False, "the final target has never been confirmed visible")

        if self.steps_since_target_seen < self.min_steps_since_target_seen:
            return StopDecision(
                False,
                f"only {self.steps_since_target_seen} steps since the target became visible "
                f"(need {self.min_steps_since_target_seen})",
            )

        measured = self.depth_probe() if self.depth_probe is not None else None
        if measured is None:
            measured = self.last_distance_m
        if measured is None:
            return StopDecision(False, "no distance measurement to the final target")
        if measured > self.arrival_radius_m:
            return StopDecision(
                False, f"target is {measured:.1f} m away, beyond the {self.arrival_radius_m:.1f} m arrival radius"
            )

        return StopDecision(True, f"all subtasks complete and target measured at {measured:.1f} m")
